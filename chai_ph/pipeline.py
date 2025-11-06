import os
import numpy as np
import yaml
import torch
import csv
import gc
import re
import sys
import py2Dmol
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from chai_lab.chai1 import _bin_centers
from typing import Optional
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from chai_ph.predict import ChaiFolder
from chai_ph.helpers import (
    sample_seq,
    extract_sequence_from_pdb,
    clean_protein_sequence,
    is_smiles,
    get_backbone_coords_from_result,
    prepare_refinement_coords,
    compute_ca_rmsd,
)
from LigandMPNN.wrapper import LigandMPNNWrapper

def optimize_protein_design(
    folder: ChaiFolder,
    designer: LigandMPNNWrapper,
    initial_seq: str,
    binder_chain: str = "A",
    target_seq: Optional[str] = None,
    target_pdb: Optional[str] = None,
    target_chain: Optional[str] = "B",
    target_pdb_chain: Optional[str] = None,
    binder_mode: str = "protein",
    prefix: str = "test",
    n_steps: int = 5,
    num_trunk_recycles: int = 3,
    num_diffn_timesteps: int = 200,
    num_diffn_samples: int = 1,
    temperature: float = 0.1,
    use_esm: bool = False,
    use_esm_target: bool = False,
    use_alignment: bool = True,
    align_to: str = "all",
    scale_temp_by_plddt: bool = False,
    partial_diffusion: float = 0.0,
    pde_cutoff_intra: float = 1.5,
    pde_cutoff_inter: float = 3.0,
    high_iptm_threshold: float = 0.8,
    omit_AA: Optional[str] = None,
    bias_AA: Optional[str] = None,
    randomize_template_sequence: bool = True,
    cyclic: bool = False,
    final_validation: bool = True,
    verbose: bool = False,
    viewer: Optional[py2Dmol.view] = None,
    render_freq: int = 1,
    plot: bool = False,
):

    """
    Optimize protein design through iterative folding and sequence design.
    Now also saves all cycle metrics to summary_all_runs.csv (outside run folder, one row per run).
    """

    high_iptm_dir = os.path.join(os.path.dirname(str(prefix)), "high_iptm_yaml")

    # Extract target sequence from PDB if not provided
    if target_pdb is not None and target_seq is None:
        if target_pdb_chain is None:
            raise ValueError("target_pdb_chain must be specified when using target_pdb")
        target_seq = extract_sequence_from_pdb(target_pdb, target_pdb_chain)
        if verbose:
            print(
                f"{prefix} | Extracted target sequence from {target_pdb} chain {target_pdb_chain}: {target_seq[:60]}..."
            )

    # Detect target type
    is_ligand_target = False
    if target_seq is not None:
        is_ligand_target = is_smiles(target_seq)

    is_binder_design = target_seq is not None
    mpnn_model_type = "ligand_mpnn" if is_ligand_target else "soluble_mpnn"
    target_entity_type = "ligand" if is_ligand_target else "protein"
    if is_binder_design:
        use_entity_names = [binder_chain, target_chain]
    else:
        use_entity_names = [binder_chain]

    # Calculate PDE bins
    bin_centers = _bin_centers(0.0, 32.0, 64)
    pde_bins_intra = (bin_centers <= pde_cutoff_intra).sum().item()
    pde_bins_inter = (bin_centers <= pde_cutoff_inter).sum().item()

    if is_binder_design:
        if is_ligand_target:
            rmsd_mode = "binder_align_ligand_com_rmsd"
        else:
            rmsd_mode = "target_align_binder_rmsd"
    else:
        rmsd_mode = "all"

    def compute_pae_metrics(pae, n_target):
        """Compute PAE and iPAE"""
        if not is_binder_design:
            return {'pae': pae.mean().item(), 'ipae': None}

        mean_pae = pae.mean().item()
        if is_ligand_target:
            target_to_binder = pae[:n_target, n_target:].min(1).values.mean().item()
            binder_to_target = pae[n_target:, :n_target].min(0).values.mean().item()
        else:
            target_to_binder = pae[:n_target, n_target:].mean().item()
            binder_to_target = pae[n_target:, :n_target].mean().item()

        ipae = (target_to_binder + binder_to_target) / 2
        return {'pae': mean_pae, 'ipae': ipae}

    def compute_template_weight(prev_pde, n_target):
        """Compute PDE-based template weight"""
        if not is_binder_design:
            weight = prev_pde[..., :pde_bins_intra].sum(-1)

        else:
            n_total = prev_pde.shape[0]
            weight = torch.ones(n_total, n_total)
            weight[n_target:, n_target:] = prev_pde[n_target:, n_target:, :pde_bins_intra].sum(-1)
            weight[:n_target, n_target:] = prev_pde[:n_target, n_target:, :pde_bins_inter].sum(-1)
            weight[n_target:, :n_target] = prev_pde[n_target:, :n_target, :pde_bins_inter].sum(-1)

        return weight

    def fold_sequence(seq, prev=None, is_first_iteration=False):
        """Fold sequence and return metrics"""
        chains = []
        if is_binder_design:
            # Binder chain
            align_binder_weight = 10.0 if align_to == "binder" else 1.0
            binder_opts = {
                "use_esm": use_esm,
                "cyclic": cyclic,
                "align": align_binder_weight,
            }
            if not is_first_iteration:
                binder_opts.update(
                    {
                        "template_pdb": prev["pdb"],
                        "template_chain_id": binder_chain,
                        "randomize_template_sequence": randomize_template_sequence,
                    }
                )
            # Target chain
            align_target_weight = 10.0 if align_to in ["target", "ligand"] else 1.0
            if is_first_iteration:
                if target_pdb is not None and not is_ligand_target:
                    # Protein target with template
                    target_opts = {
                        "use_esm": use_esm_target,
                        "template_pdb": target_pdb,
                        "template_chain_id": target_pdb_chain,
                        "align": align_target_weight,
                    }
                else:
                    # Protein target without template OR ligand target
                    target_opts = {"use_esm": use_esm_target and not is_ligand_target}
            else:
                if is_ligand_target:
                    # Ligand: no ESM
                    target_opts = {
                        "use_esm": False,
                        "align": align_target_weight,
                    }
                else:
                    # Protein target: use template + optional ESM
                    target_opts = {
                        "use_esm": use_esm_target,
                        "template_pdb": prev["pdb"],
                        "template_chain_id": target_chain,
                        "randomize_template_sequence": False,
                        "align": align_target_weight,
                    }
            chains.append([target_seq, target_chain, target_entity_type, target_opts])
            chains.append([seq, binder_chain, "protein", binder_opts])

        else:
            # Unconditional
            opts = {"use_esm": use_esm, "cyclic": cyclic}
            if not is_first_iteration:
                opts.update(
                    {
                        "template_pdb": prev["pdb"],
                        "template_chain_id": binder_chain,
                        "randomize_template_sequence": randomize_template_sequence,
                    }
                )
            chains.append([seq, binder_chain, "protein", opts])

        # Fold
        if is_first_iteration:
            template_weight = None
        else:
            if prev is None or "state" not in prev or prev["state"].result is None:
                raise ValueError(
                    "Previous state or result is missing for template weight calculation."
                )
            template_weight = compute_template_weight(
                prev["state"].result["pde"], prev["n_target"]
            )
        folder.prep_inputs(chains)
        folder.get_embeddings()
        folder.run_trunk(
            num_trunk_recycles=num_trunk_recycles, template_weight=template_weight
        )

        # Partial diffusion
        refine_coords = None
        refine_step = None
        if partial_diffusion > 0 and prev is not None:
            if (
                "state" not in prev
                or prev["state"].result is None
                or prev["state"].batch_inputs is None
            ):
                print(
                    "Warning: Skipping partial diffusion due to missing previous state data."
                )
            else:
                refine_coords = prepare_refinement_coords(
                    folder,
                    prev["state"].result,
                    prev["state"].batch_inputs,
                )
                refine_step = int(num_diffn_timesteps * partial_diffusion)

        folder.sample(
            num_diffn_timesteps=num_diffn_timesteps,
            num_diffn_samples=num_diffn_samples,
            use_alignment=use_alignment,
            refine_from_coords=refine_coords,
            refine_from_step=refine_step,
            viewer=viewer,
            render_freq=render_freq,
        )
        return folder.save_state()

    def design_sequence(step, prev, design_chain):
        """Design sequence with MPNN"""
        temp_per_residue = None
        if (
            scale_temp_by_plddt
            and prev
            and "state" in prev
            and prev["state"].result
            and "plddt" in prev["state"].result
        ):
            plddt_per_token = prev["state"].result["plddt"].numpy()
            if is_binder_design:
                n_binder = len(prev["seq"])
                if "n_target" not in prev:
                    if prev["state"].batch_inputs:
                        token_exists = prev["state"].batch_inputs["token_exists_mask"][
                            0
                        ]
                        n_total = token_exists.sum().item()
                        prev["n_target"] = n_total - n_binder
                    else:
                        print("Warning: Cannot determine n_target for plddt scaling.")
                        plddt_per_token = np.array([])
                if "n_target" in prev:
                    plddt_binder = plddt_per_token[prev["n_target"] :]
                    inv_plddt = np.square(1 - plddt_binder)
                else:
                    inv_plddt = np.array([])
            else:
                inv_plddt = np.square(1 - plddt_per_token)

            temp_per_residue = {
                f"{design_chain}{i + 1}": float(v) for i, v in enumerate(inv_plddt)
            }

        extra_args = {"--batch_size": 1}
        BASE_DIR = os.path.dirname(os.path.abspath(designer.run_py))
        MODEL_DIR = os.path.join(BASE_DIR, "model_params")
        if mpnn_model_type == "ligand_mpnn":
            extra_args["--checkpoint_ligand_mpnn"] = os.path.join(
                MODEL_DIR, "ligandmpnn_v_32_010_25.pt"
            )
        else:
            extra_args["--checkpoint_soluble_mpnn"] = os.path.join(
                MODEL_DIR, "solublempnn_v_48_020.pt"
            )
        if omit_AA:
            extra_args["--omit_AA"] = omit_AA
        if bias_AA:
            extra_args["--bias_AA"] = bias_AA

        CHAIN_TO_NUMBER = {
            "A": 0,"B": 1,"C": 2,"D": 3,"E": 4,"F": 5,"G": 6,"H": 7,"I": 8,"J": 9,
        }

        sequences, _ = designer.run(
            model_type=mpnn_model_type,
            pdb_path=prev["pdb"],
            seed=111 + step,
            chains_to_design=design_chain,
            temperature=temperature,
            temperature_per_residue=temp_per_residue,
            extra_args=extra_args,
        )
        seq = sequences[0]
        if is_binder_design:
            if ":" in seq:
                seq = seq.split(":")[CHAIN_TO_NUMBER[design_chain]]
        return seq

    def format_metrics(prev, rmsd=None):
        """Format metrics for printing and CSV"""
        if (
            not prev
            or "state" not in prev
            or not prev["state"]
            or not prev["state"].result
        ):
            print("Warning: Cannot format metrics, missing state or result.")
            return "Metrics unavailable", {}

        if prev["state"].batch_inputs:
            token_exists = prev["state"].batch_inputs["token_exists_mask"][0]
            n_total = token_exists.sum().item()
            if "seq" in prev:
                prev["n_target"] = n_total - len(prev["seq"])
            else:
                prev["n_target"] = None
                print("Warning: Cannot calculate n_target for metrics, 'seq' missing.")
        else:
            prev["n_target"] = None
            print("Warning: Cannot calculate n_target for metrics, 'batch_inputs' missing.")

        result = prev["state"].result
        pae_metrics = compute_pae_metrics(
            result["pae"], prev["n_target"] if prev["n_target"] is not None else 0
        )

        plddt_val = result["plddt"].numpy().mean()
        iptm_val = result["iptm"].item() if is_binder_design else 0.0
        ptm_val = result["ptm"].item()
        pae_val = pae_metrics["pae"]
        ipae_val = pae_metrics["ipae"]
        ranking_score_val = result["ranking_score"]

        alanine_count = None
        if is_binder_design and "seq" in prev and prev["n_target"] is not None:
            binder_seq = prev["seq"]
            alanine_count = binder_seq.count("A")
        elif not is_binder_design and "seq" in prev:
            alanine_count = prev["seq"].count("A")

        out = {
            "plddt": plddt_val,
            "ptm": ptm_val, 
            "iptm": iptm_val,
            "pae": pae_val,
            "ipae": ipae_val,
            "ranking_score": ranking_score_val,
            "alanine_count": alanine_count,
        }
        if rmsd is not None:
            out["rmsd"] = rmsd
        msg = f"score={out['ranking_score']:.3f} plddt={out['plddt']:.1f} ptm={out['ptm']:.3f}"
        if is_binder_design:
            msg += f" iptm={out['iptm']:.3f} ipae={out['ipae']:.2f} ala={out['alanine_count']}"
        else:
            msg += f" pae={out['pae']:.2f} ala={out['alanine_count']}"
        if rmsd is not None:
            msg += f" rmsd={rmsd:.2f}"
        return msg, out

    def copy_prev(prev):
        x = {**prev}
        x["state"] = x["state"].copy()
        x["bb"] = x.get("bb", torch.empty(0, 3, 3)).clone()  # Add safety check for bb
        return x

    # === MAIN LOOP ===

    iptm_per_cycle = []
    plddt_per_cycle = []
    alanine_count_per_cycle = []
    ipae_per_cycle = []
    metrics_list = []
    seq_per_cycle = []

    # Step 0
    print(f"Initial seq: {initial_seq}")
    prev = {"seq": initial_seq}
    try:
        prev["state"] = fold_sequence(prev["seq"], is_first_iteration=True)
        if prev["state"] is None or prev["state"].result is None:
            raise RuntimeError("Initial folding failed to produce results.")

        prev["bb"] = get_backbone_coords_from_result(prev["state"])
        prev["pdb"] = f"{prefix}/cycle_0.cif"
        folder.save(
            prev["pdb"],
            use_entity_names=use_entity_names,
        )
        msg, metric_dict = format_metrics(prev)
        iptm0 = metric_dict.get("iptm") if is_binder_design else None
        plddt0 = metric_dict.get("plddt")
        alanine0 = metric_dict.get("alanine_count")
        ipae0 = metric_dict.get("ipae")
        seq0 = prev.get("seq")
        iptm_per_cycle.append(iptm0)
        plddt_per_cycle.append(plddt0)
        alanine_count_per_cycle.append(alanine0)
        ipae_per_cycle.append(ipae0)
        seq_per_cycle.append(seq0)
        metric_entry = {"cycle": 0}
        metric_entry.update(metric_dict)
        metrics_list.append(metric_entry)
        print(f"{prefix} | Step 0: {msg}")

    except Exception as e:
        print(f"Error during initial folding (Step 0): {e}")
        return None

    best_step = 0
    best = copy_prev(prev)
    for step in range(n_steps):
        try:
            new_seq = design_sequence(step, prev, binder_chain)
            new = {"seq": new_seq}
            new["state"] = fold_sequence(new["seq"], prev)
            if new["state"] is None or new["state"].result is None:
                print(
                    f"Warning: Folding failed for step {step + 1}, skipping evaluation."
                )
                gc.collect()
                torch.cuda.empty_cache()
                iptm_per_cycle.append(None if is_binder_design else None)
                plddt_per_cycle.append(None)
                alanine_count_per_cycle.append(None)
                ipae_per_cycle.append(None)
                seq_per_cycle.append(new_seq)
                metrics_list.append({"cycle": step + 1})
                continue

            new["bb"] = get_backbone_coords_from_result(new["state"])
            new["pdb"] = f"{prefix}/cycle_{step + 1}.cif"
            folder.save(new["pdb"], use_entity_names=use_entity_names)
            msg, metric_dict = format_metrics(
                new,
                compute_ca_rmsd(
                    prev["bb"], new["bb"], mode=rmsd_mode, n_target=prev.get("n_target")
                )
                if prev.get("n_target") is not None
                else None,
            )

            iptm_per_cycle.append(metric_dict.get("iptm") if is_binder_design else None)
            plddt_per_cycle.append(metric_dict.get("plddt"))
            alanine_count_per_cycle.append(metric_dict.get("alanine_count"))
            ipae_per_cycle.append(metric_dict.get("ipae"))
            seq_per_cycle.append(new_seq)
            metric_entry = {"cycle": step + 1}
            metric_entry.update(metric_dict)
            metrics_list.append(metric_entry)

            print(f"{prefix} | Step {step + 1}: {msg}")
            if (
                is_binder_design
                and metric_dict.get("iptm", 0.0) > high_iptm_threshold
                and "seq" in new
            ):
                sequences=[]
                sequence_entry = {
                    "protein": {
                        "id": [binder_chain],
                        "sequence": new["seq"],
                        "msa": "empty",
                        "cyclic": cyclic
                    }
                }
                sequences.append(sequence_entry)
                if binder_mode == "protein":
                    target_sequence_entry = {
                        "protein": {
                            "id": [target_chain],
                            "sequence": target_seq,
                            "msa": "empty",
                            "cyclic": cyclic
                        }
                    }
                elif binder_mode == "ligand":
                    target_sequence_entry = {
                        "ligand": {
                            "id": [target_chain],     
                            "smiles": target_seq,
                        }
                    }
                sequences.append(target_sequence_entry)
                yaml_path = os.path.join(high_iptm_dir, os.path.basename(os.path.normpath(str(prefix)))+"_cycle_"+str(step+1)+".yaml")
                os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
                with open(yaml_path, "w") as f:
                    yaml.dump({"sequences": sequences}, f)
                print(f"Saved high-confidence binder sequences to {yaml_path}")
            if (
                new["state"].result["ranking_score"]
                > best["state"].result["ranking_score"]
            ):
                best = copy_prev(new)
                best_step = step + 1

            prev = new

        except Exception as e:
            print(f"Error during optimization step {step + 1}: {e}")
            iptm_per_cycle.append(None if is_binder_design else None)
            plddt_per_cycle.append(None)
            alanine_count_per_cycle.append(None)
            ipae_per_cycle.append(None)
            seq_per_cycle.append(None)
            metrics_list.append({"cycle": step + 1})
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    # --- SAVE SUMMARY CSV outside the run folder: summary_all_runs.csv ---
    summary_csv_path = os.path.abspath(os.path.join(os.path.dirname(str(prefix)), "summary_all_runs.csv"))
    os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)

    # Compose summary row for this run
    run_id = os.path.basename(os.path.normpath(str(prefix)))
    summary_row = {"run_id": run_id}
    for cyclenum in range(len(seq_per_cycle)):
        summary_row[f"cycle_{cyclenum}_iptm"] = iptm_per_cycle[cyclenum]
        summary_row[f"cycle_{cyclenum}_plddt"] = plddt_per_cycle[cyclenum]
        summary_row[f"cycle_{cyclenum}_ipae"] = ipae_per_cycle[cyclenum]
        summary_row[f"cycle_{cyclenum}_alanine"] = alanine_count_per_cycle[cyclenum]
        summary_row[f"cycle_{cyclenum}_seq"] = seq_per_cycle[cyclenum]

    # Write/append to summary_all_runs.csv
    # Keep all columns across runs
    if os.path.exists(summary_csv_path):
        # Read existing header
        with open(summary_csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            existing_fields = reader.fieldnames
        # Update union of fieldnames for header
        all_fields = set(existing_fields) if existing_fields is not None else set()
        all_fields.update(summary_row.keys())
        # Keep order: run_id first, rest by sorted name
        all_fields = ["run_id"] + sorted(set(all_fields) - {"run_id"})
        # Update rows to new header if needed
        updated_rows = []
        for r in existing_rows:
            for k in all_fields:
                if k not in r:
                    r[k] = ""
            updated_rows.append(r)
        # Add the new summary row with default values for missing columns
        for k in all_fields:
            if k not in summary_row:
                summary_row[k] = ""
        updated_rows.append(summary_row)
        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()
            for row in updated_rows:
                writer.writerow(row)
    else:
        all_fields = ["run_id"] + sorted([k for k in summary_row.keys() if k != "run_id"])
        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()
            writer.writerow(summary_row)

    # Make metrics graphs and save (iptm, plddt, alanine count, ipae)
    xvals = np.arange(len(iptm_per_cycle))

    if plot:
        y_iptm = np.array([v if v is not None else np.nan for v in iptm_per_cycle])
        y_plddt = np.array([v if v is not None else np.nan for v in plddt_per_cycle])
        y_ipae = np.array([v if v is not None else np.nan for v in ipae_per_cycle])
        y_alacount = np.array(
            [v if v is not None else np.nan for v in alanine_count_per_cycle]
        )
        fig, axs = plt.subplots(1, 4, figsize=(13, 3))
        colors = ["#9B59B6", "#FFA500", "#1F77B4", "#2ECC71"]
        titles = ["iPTM per cycle", "pLDDT per cycle", "iPAE per cycle", "Alanine count per cycle"]
        ylabels = ["iPTM", "pLDDT", "iPAE", "Alanine count (binder)"]
        y_datas = [y_iptm, y_plddt, y_ipae, y_alacount]

        for i, (ax, yvals, color, title, ylabel) in enumerate(
            zip(axs, y_datas, colors, titles, ylabels)
        ):
            ax.plot(
                xvals,
                yvals,
                "o-",
                color=color,
                linewidth=2,
                markersize=6,
                markerfacecolor="white",
                markeredgewidth=2,
            )
            ax.set(
                xlabel="Cycle",
                ylabel=ylabel,
                title=title,
                xticks=xvals,
            )
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.spines[["top", "right"]].set_visible(False)
            for x, y in zip(xvals, yvals):
                label_str = ""
                if not np.isnan(y):
                    if i in [0, 2]:  # iPTM or iPAE
                        label_str = f"{y:.3f}"
                    elif i == 1:  # pLDDT
                        label_str = f"{y:.1f}"
                    else:  # alanine count
                        label_str = f"{int(round(y))}"
                ax.annotate(
                    label_str,
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=9,
                )
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        fig_path = f"{prefix}/run_{run_id}_design_cycle_results.png"
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        plt.savefig(fig_path, dpi=300)
        plt.show(block=False)

    if best and best.get("state"):
        try:
            folder.restore_state(best["state"])
            folder.save(f"{prefix}/best.cif", use_entity_names=use_entity_names)
        except Exception as e:
            print(f"Error restoring/saving best state: {e}")
    else:
        print("Warning: No valid 'best' state found to restore.")
        return None

    if final_validation and best and best.get("state"):
        try:
            chains = []
            if is_binder_design:
                binder_opts = {"use_esm": use_esm, "cyclic": cyclic}
                if is_ligand_target:
                    target_opts = {"use_esm": False}
                elif target_pdb is not None:
                    target_opts = {
                        "use_esm": use_esm_target,
                        "template_pdb": target_pdb,
                        "template_chain_id": target_pdb_chain,
                    }
                else:
                    target_opts = {"use_esm": use_esm_target}
                chains.append([target_seq, target_chain, target_entity_type, target_opts])
                chains.append([best["seq"], binder_chain, "protein", binder_opts])

            else:
                opts = {"use_esm": use_esm, "cyclic": cyclic}
                chains.append([best["seq"], binder_chain, "protein", opts])

            folder.prep_inputs(chains)
            folder.get_embeddings()
            folder.run_trunk(
                num_trunk_recycles=num_trunk_recycles, template_weight=None
            )
            folder.sample(
                num_diffn_timesteps=num_diffn_timesteps,
                num_diffn_samples=num_diffn_samples,
                use_alignment=use_alignment,
            )

            if folder.state is None or folder.state.result is None:
                print("Warning: Final validation sampling failed to produce results.")
            else:
                val = {
                    "seq": best["seq"],
                    "state": folder.save_state(),
                    "pdb": f"{prefix}/final_validation.cif",
                }
                folder.save(val["pdb"], use_entity_names=use_entity_names)
                val["bb"] = get_backbone_coords_from_result(val["state"])

                if best.get("n_target") is not None:
                    val_rmsd = compute_ca_rmsd(
                        best["bb"], val["bb"], mode=rmsd_mode, n_target=best["n_target"]
                    )
                else:
                    val_rmsd = None

                msg, _ = format_metrics(val, val_rmsd)
                print(f"{prefix} | Final Validation: {msg}")

                best["val"] = val

        except Exception as e:
            print(f"Error during final validation: {e}")

        finally:
            gc.collect()
            torch.cuda.empty_cache()

    return best



class ProteinHunter_Chai:
    def __init__(self, args):
        # Unpack arguments to variables (for easier translation)
        self.args = args
        self.jobname = args.jobname
        self.length = args.length
        self.percent_X = args.percent_X
        self.seq = args.seq
        self.binder_chain = args.binder_chain
        self.target_seq = args.target_seq
        self.target_pdb = args.target_pdb
        self.target_chain = args.target_chain
        self.target_pdb_chain = args.target_pdb_chain
        self.cyclic = args.cyclic
        self.n_trials = args.n_trials
        self.n_cycles = args.n_cycles
        self.n_recycles = args.n_recycles
        self.n_diff_steps = args.n_diff_steps
        self.hysteresis_mode = args.hysteresis_mode
        self.repredict = args.repredict
        self.omit_aa = args.omit_aa
        self.bias_aa = args.bias_aa
        self.temperature = args.temperature
        self.scale_temp_by_plddt = args.scale_temp_by_plddt
        self.show_visual = args.show_visual
        self.render_freq = args.render_freq
        self.gpu_id = args.gpu_id
        self.jobname = re.sub(r"\W+", "", self.jobname)
        self.use_alphafold3_validation = args.use_alphafold3_validation
        self.alphafold_dir = args.alphafold_dir
        self.af3_docker_name = args.af3_docker_name
        self.af3_database_settings = args.af3_database_settings
        self.hmmer_path = args.hmmer_path
        self.use_msa_for_af3 = args.use_msa_for_af3
        self.work_dir = args.work_dir
        self.high_iptm_threshold = args.high_iptm_threshold
        self.plot = args.plot

        def check(folder):
            return os.path.exists(folder)

        if check(self.jobname):
            n = 0
            while check(f"{self.jobname}_{n}"):
                n += 1
            self.jobname = f"{self.jobname}_{n}"

        self.binder_mode = "none"
        if is_smiles(self.target_seq):
            self.binder_mode = "ligand"
        else:
            self.target_seq = clean_protein_sequence(self.target_seq)
            if self.target_seq == "":
                self.target_seq = None
            else:
                self.binder_mode = "protein"

        self.seq_clean = clean_protein_sequence(self.seq)
        self.omit_AA = clean_protein_sequence(self.omit_aa)
        self.bias_AA = self.bias_aa
        if self.omit_AA == "":
            self.omit_AA = None
        if self.bias_AA == "":
            self.bias_AA = None

        if self.show_visual: 
            self.viewer = py2Dmol.view((600, 400), color="plddt")
            self.viewer.show()
        else:
            self.viewer = None

        self.opts = dict(
            use_esm=False,
            use_esm_target=self.binder_mode == "protein",
            pde_cutoff_intra=0.0,
            pde_cutoff_inter=0.0,
            partial_diffusion=0.0,
        )
        if self.hysteresis_mode == "templates":
            self.opts["pde_cutoff_intra"] = 1.5
            self.opts["pde_cutoff_inter"] = 3.0
        elif self.hysteresis_mode == "esm":
            self.opts["use_esm"] = True
            self.opts["use_esm_target"] = True
        elif self.hysteresis_mode == "partial_diffusion":
            self.opts["partial_diffusion"] = 0.5

        # Setup shared folder and designer (class-static)
        global folder, designer
        if "folder" not in globals() or folder is None:
            folder = ChaiFolder(
                device=f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"
            )
        if "designer" not in globals() or designer is None:
            designer = LigandMPNNWrapper()
        self.folder = folder
        self.designer = designer



        # Detect target type
        is_ligand_target = False
        if self.target_seq is not None:
            is_ligand_target = is_smiles(self.target_seq)

        self.target_entity_type = "ligand" if is_ligand_target else "protein"


    def run_pipeline(self):
        X = []
        for t in range(self.n_trials):
            if self.seq_clean == "":
                trial_seq = sample_seq(self.length, frac_X=self.percent_X / 100)
            else:
                trial_seq = self.seq_clean

            if self.viewer is not None:
                self.viewer.new_obj()


            prefix = f"./results_chai/{self.jobname}"
            x = optimize_protein_design(
                self.folder,
                self.designer,
                initial_seq=trial_seq,
                binder_chain=self.binder_chain,
                target_seq=self.target_seq,
                target_pdb=self.target_pdb,
                target_chain=self.target_chain,
                target_pdb_chain=self.target_pdb_chain,
                binder_mode=self.binder_mode,
                prefix=f"{prefix}/run_{t}",
                n_steps=self.n_cycles,
                num_trunk_recycles=self.n_recycles,
                num_diffn_timesteps=self.n_diff_steps,
                num_diffn_samples=1,
                temperature=self.temperature,
                scale_temp_by_plddt=self.scale_temp_by_plddt,
                use_alignment=True,
                align_to="all",
                high_iptm_threshold=self.high_iptm_threshold,
                randomize_template_sequence=True,
                omit_AA=self.omit_AA,
                bias_AA=self.bias_AA,
                cyclic=self.cyclic,
                verbose=False,
                viewer=self.viewer,
                render_freq=self.render_freq,
                final_validation=self.repredict,
                plot=self.plot,
                **self.opts,
            )
            X.append(x)
            self.folder.full_cleanup()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        best_sco = 0
        best_n = None
        if X:
            for n, x in enumerate(X):
                if (
                    x
                    and x.get("state")
                    and x["state"].result
                    and "ranking_score" in x["state"].result
                ):
                    if x["state"].result["ranking_score"] > best_sco:
                        best_sco = x["state"].result["ranking_score"]
                        best_n = n
                else:
                    print(
                        f"Warning: Skipping trial {n} for best selection due to missing data."
                    )
            if best_n is not None:
                self.folder.restore_state(X[best_n]["state"])
        else:
            print("Warning: No successful trials completed.")

        if self.use_alphafold3_validation:
            sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
            from utils.alphafold_utils import run_alphafold_step
            from utils.pyrosetta_utils import run_rosetta_step
            high_iptm_yaml_dir = os.path.join(prefix, "high_iptm_yaml")
            if os.path.exists(high_iptm_yaml_dir) and len(os.listdir(high_iptm_yaml_dir)) > 0:
                success_dir = os.path.join(prefix, "1_af3_rosetta_validation")
                af_output_dir, af_output_apo_dir, af_pdb_dir, af_pdb_dir_apo = run_alphafold_step(
                    high_iptm_yaml_dir,
                    self.alphafold_dir,
                    self.af3_docker_name,
                    self.af3_database_settings,
                    self.hmmer_path,
                    success_dir,
                    os.path.expanduser(self.work_dir) or os.getcwd(),
                    binder_id=self.binder_chain,
                    gpu_id=self.gpu_id,
                    high_iptm=True,
                    use_msa_for_af3=self.use_msa_for_af3,
                )
                if self.target_entity_type == "protein":
                    # --- Rosetta Step ---
                    run_rosetta_step(
                        success_dir,
                        af_pdb_dir,
                        af_pdb_dir_apo,
                        binder_id=self.binder_chain,
                        target_type=self.target_entity_type,
                    )