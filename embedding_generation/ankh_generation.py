import torch
from transformers import AutoTokenizer, T5EncoderModel
import os
import re

def load_model_and_tokenizer(device):
    tokenizer = AutoTokenizer.from_pretrained("ElnaggarLab/ankh-large")
    model = T5EncoderModel.from_pretrained("ElnaggarLab/ankh-large")
    model.to(device)
    return model, tokenizer

def get_protein_embeddings(model, tokenizer, sequence, device):
    
    sequence_examples = [sequence]

    ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids,attention_mask=attention_mask)
        base_embedding = torch.mean(embedding_repr.last_hidden_state[0,:-1], dim=0).to('cpu')

        return base_embedding

def process_sequences(model, tokenizer, data_list, max_length, device, protein_dictionary):

    for no, data in enumerate(data_list, 1):
        print(f"{no}/{len(data_list)}", flush=True)
        uniprot_id, sequence = data.strip().split("\t")

        if len(sequence) > max_length:
            sequence = sequence[:max_length//2] + sequence[-max_length//2:]


        protein_dictionary[uniprot_id] = get_protein_embeddings(model, tokenizer, sequence, device)


def main():
    datasets = ['aav', 'gb1', 'gfp', 'location', 'meltome', 'stability']
    max_length = 5800

    device = torch.device('cuda')
    model,  tokenizer = load_model_and_tokenizer(device)
    model.eval()

    for dataset in datasets:
        base_dir = f"../../Benchmarking-tpLMs/data/{dataset}"
        output_dir = f'../../Benchmarking-tpLMs/embeddings/{dataset}/ankh'
        protein_dictionary = {}
        data_list = []
        data_file = os.path.join(base_dir, f"{dataset}_sequences.tsv")
        with open(data_file, "r") as f:
            data_list.extend(f.read().strip().split('\n'))
        process_sequences(model, tokenizer, data_list, max_length, device, protein_dictionary)
        os.makedirs(output_dir, exist_ok=True)
        torch.save(protein_dictionary, os.path.join(output_dir, 'protein_dictionary.pt'))

if __name__ == "__main__":
    main()
