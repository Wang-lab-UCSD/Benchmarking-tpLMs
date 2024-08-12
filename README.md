# Benchmarking text-integrated protein language models embeddings and their combinations

:sparkles:In this repository, we have the datasets, models, and code used in our study!:sparkles:
## :hammer_and_wrench: Installation and environment set-up
First, please clone this repository and a corresponding conda environment :snake:.\
:exclamation: NOTE: For the PyTorch installation, please install the version appropriate for your hardware: [see here](https://pytorch.org/get-started/previous-versions/)
```
conda create -n tplm python=3.10
conda activate tplm
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install scikit-learn==1.3.1
```
We provide the `environment.yml` but recommend running the commands above instead of installing from the yml file.

## :computer: Reproducing Results 
:exclamation: NOTE: These experiments are performed with an NVIDIA A6000 GPU with CUDA 12.3. Please note exact reproducibility is not guaranteed across device: [see here](https://pytorch.org/docs/stable/notes/randomness.html)

To reproduce the results from our study in sequential order, please follow the steps listed below.
1. `download_embeddings.sh`
2. `tplm_benchmarks.sh`
3. `embedding_fusion_benchmarks.sh`
4. `run_ppi.sh`
5. `run_cath.sh`

### :one: Downloading Embeddings
After running `download_embeddings.sh`, ensure that the `embeddings` directory is structured as follows, with subdirectories for each dataset, which contain subdirectories for each protein language model.
```
embeddings/
├── aav/
│ ├── esm2/
│ ├── esm3/
│ ├── ...
│ └── protst/
├── cath/
├── gb1/
...
└── stability/
```
**Generating New Embeddings**

We have provided sample scripts for generating embeddings for each protein language model (pLM) in the `data/embedding_generation` directory. To generate your own embeddings using the pLMs from this study, follow these steps:
1. Clone the Repository:
   - Clone the repository of the respective pLM you intend to use. Please follow the specific setup and environment setup instructions detailed in each pLM's repository.
2. Generate Embeddings:
   - Copy the embedding generation script we provided in `data/embedding_generation` into the cloned pLM's directory. Each pLM has a different embedding generation script, so please make sure you use the appropriate one.  
   - Execute these scripts within the pLM's environment and directory to generate new embeddings. Ensure that the outputs are directed to the appropriate location. 

### :two: Benchmarking text-integrated protein language models against ESM2 3B
Run `tplm_benchmarks.sh` to train models for benchmarking tpLMs against ESM2 3B on AAV, GB1, GFP, Location, Meltome, and Stability. 
### :three: Evaluating embedding fusion
Run `embedding_fusion_benchmarks.sh` to train models for benchmarking embedding fusion with tpLMs on AAV, GB1, GFP, Location, Meltome, and Stability. 
### :four: Identifying optimal combinations and evaluating performance on protein-protein interaction prediction
Run `run_ppi.sh` to use the greedy heuristic to identify a promising combination of embeddings, then train models with all possible combinations of embeddings to identify the true best combination.
### :five: Identifying optimal combinations and evaluating performance on homologous sequence recovery
Run `run_cath.sh` to use the greedy heuristic to identify a promising combination of embeddings, then evaluate all possible combinations of embeddings to identify the true best combination.
