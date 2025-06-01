# Benchmarking text-integrated protein language model embeddings and embedding fusion on diverse downstream tasks

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
:exclamation: NOTE: These experiments are performed with an NVIDIA A6000 GPU with CUDA 12.3. Please note exact reproducibility is not guaranteed across devices: [see here](https://pytorch.org/docs/stable/notes/randomness.html)

To reproduce the results from our study in sequential order, please follow the steps listed below.
1. `download_data_embs.sh`
2. `run_tplm_benchmarks.sh`
3. `run_embedding_fusion_benchmarks.sh`
4. `benchmark_selection_algs.ipynb`
5. `run_ppi.sh`
6. `run_cath.sh`
---
### :one: Downloading Data and Embeddings
The data and embeddings are stored in HuggingFace and our `download_data_embs.sh` uses `huggingface-cli` to download the necessary files. 

:exclamation: NOTE: Before running `download_data_embs.sh`, please add your HuggingFace token after the `--token` flag. Once added, run `download_data_embs.sh`.
```bash
huggingface-cli login --add-to-git-credential --token # Add your Huggingface token here 
```

<details>
  <summary><strong>Dataset Details</strong></summary>
  <br>
The datasets used in this study are created by the following authors:
  
  - AAV, GB1, and Meltome: https://github.com/J-SNACKKB/FLIP
  
  - GFP and Stability:  https://github.com/songlab-cal/tape

  - Location: https://github.com/HannesStark/protein-localization
  
  - PPI: https://github.com/daisybio/data-leakage-ppi-prediction
  
  - CATH/Homologous sequence recovery: https://www.cathdb.info/

</details>


<details>
  <summary><strong>Generating New Embeddings</strong></summary>
  <br>

We have provided sample scripts for generating embeddings for each protein language model (pLM) in the `embedding_generation/` directory. To generate your own embeddings using the pLMs from this study, follow these steps:

1. **Clone the Repository**:
   - Clone the repository of the respective pLM you intend to use. Please follow the specific setup and environment setup instructions detailed in each pLM's repository.

2. **Generate Embeddings**:
   - Copy the embedding generation script we provided in `embedding_generation/` into the cloned pLM's directory. Each pLM has a different embedding generation script, so please make sure you use the appropriate one.  
   - Execute these scripts within the pLM's environment and directory to generate new embeddings. Ensure that the outputs are directed to the appropriate location.

</details>

---
### :two: Benchmarking text-integrated protein language models against Ankh, ESM2 3B, and ProtT5.
Run `run_tplm_benchmarks.sh` to train models for benchmarking tpLMs against large pLMs on AAV, GB1, GFP, Location, Meltome, and Stability. 

---
### :three: Evaluating embedding fusion
Run `run_embedding_fusion_benchmarks.sh` to train models for benchmarking embedding fusion with tpLMs on AAV, GB1, GFP, Location, Meltome, and Stability. 

After steps 2 and 3, we can run `analysis/benchmark_analysis.ipynb`. This notebook will read the output files of benchmarking and aggregate all the results, calculate the mean and 95% CIs of all single and combined embedding performances.

---
### :four: Evaluating different feature selection algorithms
Run the `analysis/benchmark_selection_algs.ipynb` notebook to evaluate three different feature selection algorithms against the exhaustive search on AAV, GB1, GFP, Location, Meltome, and Stability. This step requires the results from steps 2 and 3 to properly run. 

This notebook contains a python-version of the selection algorithms, and uses the results of steps 3 to run through the selection algorithms, recording the total time used. It also contains the code to generate Figure 3C. Please consider the term reverse elimination interchangeable with the term backward selection used in the paper. (This will be updated soon to remove any confusion.)

---
### :five: Identifying optimal combinations and evaluating performance on homologous sequence recovery
Run `run_cath.sh` to use embedding fusion + greedier forward selection to identify a promising combination of embeddings, then evaluate all possible combinations of embeddings to identify the true best combination. Note, for this task, we can afford to check combinations including pLMs.

In `analysis/homologous_sequence_vis.ipynb` we will visualize the embeddings of all the CATH dataset proteins with t-SNE, using first ProteinCLIP, the embedding that enabled the previous state-of-the-art results. Next, we will visualize the combination of embeddings found by GFS that enabled the new state-of-the-art results.


---
### :six: Identifying optimal combinations and evaluating performance on protein-protein interaction prediction
Run `run_ppi.sh` to use embedding fusion + greedier forward selection to identify a promising combination of embeddings, then train models with all possible combinations of embeddings to identify the true best combination. We also train the original NaderiAlizadeh classifier with ESM2 650M, and also run inference to get the predicted probabilities of this model. Note, for inference, we need to load the model checkpoints. If reproducing the results, please ensure that you have first run training to generate the checkpoint. If you do not wish to train the model, we will upload the exact checkpoints used to the Huggingface repo above.

Next, run `analysis/ppi_analysis.ipynb` For further analysis of our best embedding fusion combination, we will calculate the precision and number of false positives made in the top-k predictions of the previous best NaderiAlizadeh classifier, and compare them with our embedding fusion + GFS enabled model. 

---
