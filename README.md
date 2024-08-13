# Benchmarking text-integrated protein language models embeddings and embedding fusion on diverse downstream tasks

:sparkles:In this repository, we have the datasets, models, and code used in our study!:sparkles:
## :hammer_and_wrench: Installation and environment set-up
First, please clone this repository and create a corresponding conda environment :snake:.\
:exclamation: NOTE: For the PyTorch installation, please install the version appropriate for your hardware: [see here](https://pytorch.org/get-started/previous-versions/)
```bash
conda create -n tplm python=3.10
conda activate tplm
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install scikit-learn==1.3.1
pip install -U "huggingface_hub[cli]"
```
We provide the `environment.yml` but recommend running the commands above instead of installing from the yml file.

## :computer: Reproducing Results 
:exclamation: NOTE: These experiments are performed with an NVIDIA A6000 GPU with CUDA 12.3. Please note exact reproducibility is not guaranteed across device: [see here](https://pytorch.org/docs/stable/notes/randomness.html)

To reproduce the results from our study in sequential order, please follow the steps listed below.
1. `download_data_embs.sh`
2. `run_tplm_benchmarks.sh`
3. `run_embedding_fusion_benchmarks.sh`
4. `run_ppi.sh`
5. `run_cath.sh`
---
### :one: Downloading Data and Embeddings
The data and embeddings are stored in HuggingFace and our `download_data_embs.sh` uses `huggingface-cli` to download the necessary files. 

:exclamation: NOTE: Before running `download_data_embs.sh`, please add your HuggingFace token after the `--token` flag. Once added, run `download_data_embs.sh`.

**Dataset Details**

The datasets used in this study are created by the following authors:
  - AAV, GB1, and Meltome: https://github.com/J-SNACKKB/FLIP
  - GFP and Stability:  https://github.com/songlab-cal/tape
  - Location: https://github.com/HannesStark/protein-localization
  - PPI: https://github.com/daisybio/data-leakage-ppi-prediction
  - CATH/Homologous sequence recovery: https://www.cathdb.info/

**Generating New Embeddings**

We have provided sample scripts for generating embeddings for each protein language model (pLM) in the `embedding_generation/` directory. To generate your own embeddings using the pLMs from this study, follow these steps:
1. Clone the Repository:
   - Clone the repository of the respective pLM you intend to use. Please follow the specific setup and environment setup instructions detailed in each pLM's repository.
2. Generate Embeddings:
   - Copy the embedding generation script we provided in `embedding_generation/` into the cloned pLM's directory. Each pLM has a different embedding generation script, so please make sure you use the appropriate one.  
   - Execute these scripts within the pLM's environment and directory to generate new embeddings. Ensure that the outputs are directed to the appropriate location. 
---
### :two: Benchmarking text-integrated protein language models against ESM2 3B
Run `run_tplm_benchmarks.sh` to train models for benchmarking tpLMs against ESM2 3B on AAV, GB1, GFP, Location, Meltome, and Stability. 

---
### :three: Evaluating embedding fusion
Run `run_embedding_fusion_benchmarks.sh` to train models for benchmarking embedding fusion with tpLMs on AAV, GB1, GFP, Location, Meltome, and Stability. 

---
### :four: Identifying optimal combinations and evaluating performance on protein-protein interaction prediction
Run `run_ppi.sh` to use the greedy heuristic to identify a promising combination of embeddings, then train models with all possible combinations of embeddings to identify the true best combination.

---
### :five: Identifying optimal combinations and evaluating performance on homologous sequence recovery
Run `run_cath.sh` to use the greedy heuristic to identify a promising combination of embeddings, then evaluate all possible combinations of embeddings to identify the true best combination.

---
