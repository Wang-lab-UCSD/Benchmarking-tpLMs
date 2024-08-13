:exclamation: NOTE: These scripts cannot be run directly from this repository. For the pLM of interest, clone the original repo. Follow their instructions on environment setup, then copy the appropriate generation.py script into the corresponding pLM repo, then run from within.

1. [ESM2](https://github.com/facebookresearch/esm) (Commit 2b36991)
2. [ESM3](https://github.com/evolutionaryscale/esm) (Commit dd9fb13)
3. [OntoProtein](https://github.com/zjunlp/OntoProtein) (Commit 6360f45)
4. [ProteinDT](https://github.com/chao1224/ProteinDT) (Commit 369c3ce)
5. [ProtST](https://github.com/DeepGraphLearning/ProtST) (Commit db53a76)
6. [ProteinCLIP](https://github.com/wukevin/proteinclip) (Commit c732a40)
7. [ProTrek](https://github.com/westlake-repl/ProTrek) (Commit ba433f0)

---
The main function is formatted as such:
```python
def main():
    datasets = ['stability_sequences']
    max_length = 5800
    base_dir = "../../Benchmarking-tpLMs/data/stability"
    output_dir = '../../Benchmarking-tpLMs/embeddings/stability/protrek'
    protein_dictionary = {}
    combined_data_list = []

    device = torch.device('cuda')
    model= load_model(device)

    for dataset in datasets:
        data_file = os.path.join(base_dir, f"{dataset}.tsv")
        with open(data_file, "r") as f:
            combined_data_list.extend(f.read().strip().split('\n'))

    process_sequences(model, combined_data_list, max_length, device, protein_dictionary)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(protein_dictionary, os.path.join(output_dir, 'protein_dictionary.pt'))
```
datasets, base_dir, and output_dir should be changed accordingly.
