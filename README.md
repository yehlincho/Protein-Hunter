# Protein-Hunter 😈


<p align="center">
  <img src="./protein_hunter.png" alt="Protein Hunter" width="500"/>
</p>

> 📄 **Paper**: [Protein Hunter: exploiting structure hallucination
within diffusion for protein design](https://www.biorxiv.org/content/10.1101/2025.10.10.681530v2.full.pdf)  
> 🚀 **Colab**: https://colab.research.google.com/drive/1JBP7iMPLKiJrhjUlFfi0ShQcu8vHn2gI#scrollTo=CzE1iBF-ZCI0

---

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
- ✅ Download Boltz model weights
- ✅ Configure LigandMPNN and ProteinMPNN
- ✅ Optionally install PyRosetta
- ❌ AF3 must be installed separately

---

We have implemented two different AF3-style models in our Protein Hunter pipeline (more models will be added in the future):
- Boltz1/2
- Chai1



## Run Code End-to-End
- **Protein-protein design:**  
  To design a protein-protein complex, run:  
  ```
  python boltz/protein_hunter.py --num_designs 3 --num_cycles 7 --protein_seqs AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE --protein_ids B --protein_msas "" --gpu_id 2 --name PDL1_mix_aa --min_design_protein_length 90 --max_design_protein_length 150 --high_iptm_threshold 0.7 --use_msa_for_af3 --plot
  ```

- **Small molecule binder design:**  
  For designing a protein binder for a small molecule (e.g. SAM), use:  
  ```
  python boltz/protein_hunter.py --num_designs 5 --num_cycles 7 --ligand_ccd SAM --ligand_id B --gpu_id 2 --name SAM_binder --min_design_protein_length 130 --max_design_protein_length 150 --high_iptm_threshold 0.7 --use_msa_for_af3 --plot
  ```

- **DNA/RNA PDB design:**  
  To design a protein binder for a nucleic acid (e.g. an RNA sequence), run this in Python:
  ```
  python boltz/protein_hunter.py --num_designs 5 --num_cycles 7 --nucleic_seq AGAGAGAGA --nucleic_id B --nucleic_type rna --gpu_id 0 --name RNA_bind --min_design_protein_length 130 --max_design_protein_length 150 --high_iptm_threshold 0.7 --use_msa_for_af3 --plot
  ```

- **Designs with multiple/heterogeneous target types:**  
  If you want to target multiple, different types of molecules (for example, a protein with a ligand and a template), run:
  ```
  python one_shot_diff/protein_hunter.py --num_designs 5 --num_cycles 7 --protein_seqs AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE --protein_ids B --protein_msas "" --ligand_ccd SAM --ligand_id C --gpu_id 0 --name PDL1_SAM --min_design_protein_length 90 --max_design_protein_length 150 --high_iptm_threshold 0.8 --use_msa_for_af3 --plot
  ```


⚠️ **Warning**: To run the AlphaFold3 cross-validation pipeline, you need to specify your AlphaFold3 directory, Docker name, database settings, and conda environment in the configuration. These can be set using the following arguments:
- `--alphafold_dir`: Path to your AlphaFold3 installation (default: ~/alphafold3)
- `--af3_docker_name`: Name of your AlphaFold3 Docker container
- `--af3_database_settings`: Path to AlphaFold3 database
- `--af3_hmmer_path`: Path to HMMER


## 🎥 Trajectory Visualization
We installed trajectory visualization based on LogMD


### ProteinMPNN
- **Use case**: Protein-protein interface design

### LigandMPNN  
- **Use case**: Protein-ligand and non-protein biomolecule interfaces

## ✅ Structure Validation

### Primary Evaluation: AlphaFold3
Final structures are validated using **AlphaFold3** for:
- Structure quality assessment 
- Confidence scoring
- Cross-validation against design targets

## 🎯 Successful Designs

After running the pipeline in `run_protein_hunter.py`, high-confidence designs can be found in:

`your_output_folder/03_af_pdb_success`


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

