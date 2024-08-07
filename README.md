# Benchmarking text-integrated protein language models embeddings and their combinations

In this repository, we have the datasets, models, and other code used in our study!

## Installation and environment set-up
These experiments are performed with an NVIDIA A6000 GPU with CUDA 12.3. Please note exact reproducibility is not guaranteed across device: [see here](https://pytorch.org/docs/stable/notes/randomness.html)

```
git clone repo
cd repo
```

## Reproducing Results
To reproduce the results from our study in sequential order, please follow the steps listed below.
1. download_embeddings.sh
2. tplm_benchmarks.sh
3. embedding_fusion_benchmarks.sh
4. run_ppi.sh
5. run_cath.sh


### Downloading Embeddings
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
### Benchmarking text-integrated protein language models against ESM2 3B
### Evaluating embedding fusion
### Identifying optimal combinations and evaluating performance on protein-protein interaction prediction
### Identifying optimal combinations and evaluating performance on homologous sequence recovery
