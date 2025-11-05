import argparse
import os

from pipeline import ProteinHunter_Boltz


# Keep parse_args() here for CLI functionality
def parse_args():
    parser = argparse.ArgumentParser(
        description="Boltz protein design with cycle optimization"
    )
    # --- Existing Arguments (omitted for brevity, keep all original args) ---
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--grad_enabled", action="store_true", default=False)
    parser.add_argument("--name", default="target_name_is_missing", type=str)
    parser.add_argument(
        "--mode", default="binder", choices=["binder", "unconditional"], type=str
    )
    parser.add_argument("--num_designs", default=50, type=int)
    parser.add_argument("--num_cycles", default=5, type=int)
    parser.add_argument("--binder_chain", default="A", type=str)
    parser.add_argument("--min_design_protein_length", default=100, type=int)
    parser.add_argument("--max_design_protein_length", default=150, type=int)
    parser.add_argument("--protein_ids", default="B", type=str)
    parser.add_argument(
        "--protein_seqs",
        default="",
        type=str,
    )
    parser.add_argument("--protein_msas", default="empty", type=str)
    parser.add_argument("--cyclics", default="", type=str)
    parser.add_argument("--ligand_id", default="B", type=str)
    parser.add_argument("--ligand_smiles", default="", type=str)
    parser.add_argument("--ligand_ccd", default="", type=str)
    parser.add_argument(
        "--nucleic_type", default="dna", choices=["dna", "rna"], type=str
    )
    parser.add_argument("--nucleic_id", default="B", type=str)
    parser.add_argument("--nucleic_seq", default="", type=str)
    parser.add_argument(
        "--template_path", default="", type=str
    )  # can be "2VSM", or path(s) to .cif/.pdb, multiple allowed separated by comma
    parser.add_argument(
        "--template_chain_id", default="", type=str
    )  # for prediction, the chain id to use for the template
    parser.add_argument(
        "--template_cif_chain_id", default="", type=str
    )  # for mmCIF files, the chain id to use for the template (for alignment)
    parser.add_argument("--no_potentials", action="store_true")
    parser.add_argument("--diffuse_steps", default=200, type=int)
    parser.add_argument("--recycling_steps", default=3, type=int)
    parser.add_argument("--boltz_model_version", default="boltz2", type=str)
    parser.add_argument(
        "--boltz_model_path",
        default="~/.boltz/boltz2_conf.ckpt",
        type=str,
    )
    parser.add_argument("--ccd_path", default="~/.boltz/mols", type=str)
    parser.add_argument("--randomly_kill_helix_feature", action="store_true")
    parser.add_argument("--negative_helix_constant", default=0.2, type=float)
    parser.add_argument("--logmd", action="store_true")
    parser.add_argument("--save_dir", default="", type=str)
    parser.add_argument("--add_constraints", action="store_true")
    parser.add_argument("--contact_residues", default="", type=str)
    parser.add_argument("--omit_AA", default="C", type=str)
    parser.add_argument("--exclude_P", action="store_true", default=False)
    parser.add_argument("--frac_X", default=0.5, type=float)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot cycles figs per run (requires matplotlib)",
    )

    # NEW: Add constraint_target_chain argument
    parser.add_argument(
        "--constraint_target_chain",
        default="B",
        type=str,
        help="Target chain for constraints and contact calculations",
    )

    # NEW: Add no_contact_filter argument
    parser.add_argument(
        "--no_contact_filter",
        action="store_true",
        help="Do not filter or restart for unbound contact residues at cycle 0",
    )
    parser.add_argument("--max_contact_filter_retries", default=6, type=int)
    parser.add_argument("--contact_cutoff", default=15.0, type=float)

    parser.add_argument(
        "--alphafold_dir", default=os.path.expanduser("~/alphafold3"), type=str
    )
    parser.add_argument("--af3_docker_name", default="alphafold3_yc", type=str)
    parser.add_argument(
        "--af3_database_settings", default="~/alphafold3/alphafold3_data_save", type=str
    )
    parser.add_argument(
        "--hmmer_path",
        default="~/.conda/envs/alphafold3_venv",
        type=str,
    )
    parser.add_argument("--use_msa_for_af3", action="store_true")
    parser.add_argument("--work_dir", default="", type=str)

    # temp and bias params
    parser.add_argument("--temperature_start", default=0.05, type=float)
    parser.add_argument("--temperature_end", default=0.001, type=float)
    parser.add_argument("--alanine_bias_start", default=-0.5, type=float)
    parser.add_argument("--alanine_bias_end", default=-0.2, type=float)
    parser.add_argument("--alanine_bias", action="store_true")

    parser.add_argument("--high_iptm_threshold", default=0.8, type=float)
    # --- End Existing Arguments ---

    return parser.parse_args()


def main():
    args = parse_args()
    # Instantiate the main class and run the pipeline
    protein_hunter = ProteinHunter_Boltz(args)
    protein_hunter.run_pipeline()


if __name__ == "__main__":
    main()
