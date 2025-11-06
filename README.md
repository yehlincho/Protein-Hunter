# Protein-Hunter 😈

<p align="center">
  <img src="./protein_hunter.png" alt="Protein Hunter" width="500"/>
</p>

<p align="center" style="font-size:90%">
  <em>DAlphaBall.gcc
    <strong>Note:</strong> Logo is a ChatGPT-modified version of the Netflix animation and is for illustration only.
  </em>
</p>

> 📄 **Paper**: [Protein Hunter: exploiting structure hallucination
within diffusion for protein design](https://www.biorxiv.org/content/10.1101/2025.10.10.681530v2.full.pdf)  
> 🚀 **Colab**: https://colab.research.google.com/drive/1JBP7iMPLKiJrhjUlFfi0ShQcu8vHn2gI#scrollTo=CzE1iBF-ZCI0

---


## 📝 Colab Notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yehlincho/Protein-Hunter/blob/main/protein_hunter_chai_colab.ipynb)

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yehlincho/Protein-Hunter.git
   cd Protein-Hunter
   ```

2. **Run the automated setup**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

> ⚠️ **Note**: AlphaFold3 setup is not included. Please install it separately by following the [official instructions](https://github.com/google-deepmind/alphafold3).

The setup script will automatically:
- ✅ Create a conda environment with Python 3.10
- ✅ Install all required dependencies
- ✅ Set up a Jupyter kernel for notebooks
- ✅ Download Boltz and Chai weights
- ✅ Configure LigandMPNN and ProteinMPNN
- ✅ Optionally install PyRosetta
- ❌ AF3 must be installed separately

---

We have implemented two different AF3-style models in our Protein Hunter pipeline (more models will be added in the future):
- Boltz1/2
- Chai1


## Run Code End-to-End

## 1️⃣ End-to-end structure and sequence generation  
👉 See example usage in `run_protein_hunter.py` for reference. 🐍✨  
This will take you from an initial input to final designed protein structures and sequences, all in one pipeline!

> 💡 **Tips:** The original evaluation in the paper used an all-X sequence for initial design. However, to increase the diversity of generated folds, you can mix random amino acids with X residues by setting the `percent_X` parameter (e.g., `--percent_X 50` for 50% X and 50% random AAs). Adjusting this ratio helps explore a broader design space.


⚠️ **Warning**: To run the AlphaFold3 cross-validation pipeline, you need to specify your AlphaFold3 directory, Docker name, database settings, and conda environment in the configuration. These can be set using the following arguments:
- `--alphafold_dir`: Path to your AlphaFold3 installation (default: ~/alphafold3)
- `--af3_docker_name`: Name of your AlphaFold3 Docker container
- `--af3_database_settings`: Path to AlphaFold3 database
- `--af3_hmmer_path`: Path to HMMER
- `--use_alphafold3_validation`: Add this flag to enable AlphaFold3-based validation. 

## Protein Hunter (Boltz Edition ⚡) 
To use AlphaFold3 validation, make sure your AlphaFold3 Docker is installed, specify the correct AlphaFold3 directory, and turn on `--use_alphafold3_validation`.
- **Protein-protein design:**  
  To design a protein-protein complex, run:  
  ```
  python boltz_ph/design.py --num_designs 3 --num_cycles 7 --protein_seqs AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE --protein_ids B --protein_msas "" --gpu_id 2 --name PDL1_mix_aa --min_design_protein_length 90 --max_design_protein_length 150 --high_iptm_threshold 0.7 --use_msa_for_af3 --plot
  ```

- **Small molecule binder design:**  
  For designing a protein binder for a small molecule (e.g., SAM), use:  
  ```
  python boltz_ph/design.py --num_designs 5 --num_cycles 7 --ligand_ccd SAM --ligand_id B --gpu_id 2 --name SAM_binder --min_design_protein_length 130 --max_design_protein_length 150 --high_iptm_threshold 0.7 --use_msa_for_af3 --plot
  ```

- **DNA/RNA PDB design:**  
  To design a protein binder for a nucleic acid (e.g., an RNA sequence), run:
  ```
  python boltz_ph/design.py --num_designs 5 --num_cycles 7 --nucleic_seq AGAGAGAGA --nucleic_id B --nucleic_type rna --gpu_id 0 --name RNA_bind --min_design_protein_length 130 --max_design_protein_length 150 --high_iptm_threshold 0.7 --use_msa_for_af3 --plot
  ```

- **Designs with multiple/heterogeneous target types:**  
  Want to target multiple types of molecules (e.g., a protein with a ligand and a template)? Run:
  ```
  python boltz_ph/design.py --num_designs 5 --num_cycles 7 --protein_seqs AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE --protein_ids B --protein_msas "" --ligand_ccd SAM --ligand_id C --gpu_id 0 --name PDL1_SAM --min_design_protein_length 90 --max_design_protein_length 150 --high_iptm_threshold 0.8 --use_msa_for_af3 --plot
  ```

## Protein Hunter (Chai Edition ☕) 

> ⚠️ **Caution:** The Chai version is under active development
- [x] Support for multiple targets
- [x] Full AlphaFold (AF3) validation

- **Unconditional protein design:**  
  Generate de novo proteins of a desired length:
  ```
  python chai_ph/design.py --jobname unconditional_design --length 120 --percent_X 0 --seq "" --target_seq ACDEFGHIKLMNPQRSTVWY --cyclic --n_trials 1 --n_cycles 5 --n_recycles 3 --n_diff_steps 200 --hysteresis_mode templates --repredict --omit_aa "" --temperature 0.1 --scale_temp_by_plddt --render_freq 100 --gpu_id 2 --plot

- **Protein binder design:**  
  To design a binder for a specific protein target (e.g., PDL1):
  ```
  python chai_ph/design.py --jobname PDL1_binder --length 120 --percent_X 50 --seq "" --target_seq AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE --n_trials 1 --n_cycles 5 --n_recycles 3 --n_diff_steps 200 --hysteresis_mode templates --repredict --omit_aa "" --temperature 0.1 --scale_temp_by_plddt --render_freq 100 --gpu_id 2 --use_msa_for_af3 --plot
  ```

- **Cyclic protein binder design:**  
  Design a cyclic peptide binder for a specific protein target:
  ```
  python chai_ph/design.py --jobname PDL1_cyclic_binder --length 120 --percent_X 50 --seq "" --cyclic --target_seq AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE --n_trials 1 --n_cycles 5 --n_recycles 3 --n_diff_steps 200 --hysteresis_mode templates --repredict --omit_aa "" --temperature 0.1 --scale_temp_by_plddt --render_freq 100 --gpu_id 2 --use_msa_for_af3 --plot
  ```

- **Small molecule (ligand) binder design:**  
  To design a protein binder for a small molecule or ligand (SMILES string as target):
  ```
  python chai_ph/design.py --jobname ligand_binder --length 120 --percent_X 0 --seq "" --target_seq O=C(NCc1cocn1)c1cnn(C)c1C(=O)Nc1ccn2cc(nc2n1)c1ccccc1 --n_trials 1 --n_cycles 5 --n_recycles 3 --n_diff_steps 200 --hysteresis_mode esm --repredict --omit_aa "" --temperature 0.01 --scale_temp_by_plddt --render_freq 100 --gpu_id 2 --plot
  ```

## 2️⃣ Refine your own designs!
🛠️ You can provide your initial designs as input and further improve their structures by iteratively redesigning and predicting them. Repeat as needed for optimal results!

See the code in `refiner.ipynb` for example usage.

For example, you can generate a design using Boltzgen, take the final output, and refine it further using the iterative pipeline. 

---




## 🎥 Trajectory Visualization
We have implemented trajectory visualization using LogMD and py2Dmol (developed by Sergey Ovchinnikov).

## ✅ Structure Validation

### Primary Evaluation: AlphaFold3
Final structures are validated using **AlphaFold3** for:
- Structure quality assessment 
- Confidence scoring
- Cross-validation against design targets

## 🎯 Successful Designs

After running the pipeline with `run_protein_hunter.py`, high-confidence designs can be found in:

`your_output_folder/high_iptm_yaml`

After running AlphaFold3, the validated structures are saved in:

`your_output_folder/03_af_pdb_success`


## 📝 To-Do List

- [ ] Add cross-validation between Boltz and Chai (both directions) without using AlphaFold3 as an option
- [ ] Implement multi-timer support for Protein Hunter Chai edition
- [ ] Explore other cool applications

Collaboration is always welcome! Email me. Let's chat.

## 📄 License & Citation

**License**: MIT License - See LICENSE file for details  
**Citation**: If you use Protein Hunter in your research, please cite:
```
@article{cho2025protein,
  title={Protein Hunter: exploiting structure hallucination within diffusion for protein design},
  author={Cho, Yehlin and Rangel, Griffin and Bhardwaj, Gaurav and Ovchinnikov, Sergey},
  journal={bioRxiv},
  pages={2025--10},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```
---

## 📧 Contact & Support

**Questions or Collaboration**: yehlin@mit.edu  
**Issues**: Please report bugs and feature requests via GitHub Issues

---

## ⚠️ Important Disclaimer

> **EXPERIMENTAL SOFTWARE**: This pipeline is under active development and has **NOT been experimentally validated** in laboratory settings. We release this code to enable community contributions and collaborative development. Use at your own discretion and validate results independently.

