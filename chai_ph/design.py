import os
import argparse
from pipeline import ProteinHunter_Chai


def parse_args():
    """Parses command-line arguments for ProteinHunter_Chai."""
    parser = argparse.ArgumentParser(
        description="ProteinHunter input options for structure prediction and design cycles."
    )

    # --- Job and Sequence Settings ---
    sequence_group = parser.add_argument_group("Sequence and Job Settings")
    sequence_group.add_argument(
        "--jobname", type=str, default="test", help="Job name for output files and folders."
    )
    sequence_group.add_argument(
        "--length", type=int, default=150, help="Length of the designed protein chain."
    )
    sequence_group.add_argument(
        "--percent_X",
        type=int,
        default=50,
        help="Percentage of 'X' residues in the initial sequence (0, 50, or 100).",
    )
    sequence_group.add_argument(
        "--seq", type=str, default="", help="Input sequence for the binder chain (optional). If empty, sequence is randomly sampled."
    )
    sequence_group.add_argument(
        "--target_seq",
        type=str,
        default="",
        help="Target sequence (protein) or SMILES (ligand) for binder design (optional).",
    )
    sequence_group.add_argument(
        "--cyclic",
        action="store_true",
        default=False,
        help="Enable cyclic topology for the designed chain.",
    )

    # --- Optimization Options ---
    opt_group = parser.add_argument_group("Optimization and Folding Parameters")
    opt_group.add_argument(
        "--n_trials",
        type=int,
        default=1,
        help="Number of independent optimization trials.",
    )
    opt_group.add_argument(
        "--n_cycles",
        type=int,
        default=5,
        help="Number of folding/design optimization cycles (steps).",
    )
    opt_group.add_argument(
        "--n_recycles",
        type=int,
        default=3,
        help="Number of trunk recycles per fold step.",
    )
    opt_group.add_argument(
        "--n_diff_steps",
        type=int,
        default=200,
        help="Diffusion steps for structure sampling.",
    )
    opt_group.add_argument(
        "--hysteresis_mode",
        type=str,
        default="esm",
        choices=["templates", "esm", "partial_diffusion", "none"],
        help="Strategy for template/feature reuse to guide folding (hysteresis).",
    )
    opt_group.add_argument(
        "--repredict",
        action="store_true",
        default=True,
        help="Re-predict final best structure without templates for validation.",
    )

    # --- MPNN Options ---
    mpnn_group = parser.add_argument_group("ProteinMPNN Settings")
    mpnn_group.add_argument(
        "--omit_aa", type=str, default="", help="Amino acid types to omit from design (e.g., 'C')."
    )
    mpnn_group.add_argument(
        "--bias_aa",
        type=str,
        default="",
        help="Amino acid types to bias (e.g., 'A:-2.0,P:-1.0').",
    )
    mpnn_group.add_argument(
        "--temperature",
        type=float,
        default=0.01,
        help="ProteinMPNN sampling temperature.",
    )
    mpnn_group.add_argument(
        "--scale_temp_by_plddt",
        action="store_true",
        default=False,
        help="Scale MPNN temperature inversely by pLDDT for focused design.",
    )

    # --- Visualization and Hardware Options ---
    vis_group = parser.add_argument_group("Hardware and Visualization")
    vis_group.add_argument(
        "--show_visual",
        action="store_true",
        default=False,
        help="Show interactive py3Dmol visualization during the run.",
    )
    vis_group.add_argument(
        "--render_freq",
        type=int,
        default=100,
        help="Visualization refresh frequency (diffusion steps).",
    )
    vis_group.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use.")


    af_group = parser.add_argument_group("AlphaFold Settings")
    af_group.add_argument(
        "--alphafold_dir", default=os.path.expanduser("~/alphafold3"), type=str
    )
    af_group.add_argument("--af3_docker_name", default="alphafold3_yc", type=str)
    af_group.add_argument(
        "--af3_database_settings", default="~/alphafold3/alphafold3_data_save", type=str
    )
    af_group.add_argument(
        "--hmmer_path",
        default="~/.conda/envs/alphafold3_venv",
        type=str,
    )
    af_group.add_argument("--use_msa_for_af3", action="store_true")
    af_group.add_argument("--work_dir", default="", type=str)
    af_group.add_argument("--high_iptm_threshold", default=0.8, type=float)

    return parser.parse_args()


def main():
    """Main function to run the ProteinHunter pipeline."""
    args = parse_args()
    protein_hunter = ProteinHunter_Chai(args)
    protein_hunter.run_pipeline()


if __name__ == "__main__":
    main()