import torch
import torch.nn.functional as F
import argparse
import csv
from main_utils import (setup_cath_logs, load_embeddings, concatenate_embeddings, load_superfamily_mapping, normalize_embeddings)

def main():
    ##################### Setup #####################
    parser = argparse.ArgumentParser(description="Homologous sequence recovery")
    parser.add_argument("--dataset", choices=['cath'], required=True, help="Specify which dataset to use for training and evaluation")
    parser.add_argument("--embeddings", type=str, required=True, help="Select pLM embeddings to combine.")
    parser.add_argument("--device", default="cpu", help="Set device")
    args = parser.parse_args()

    # Setup logging file
    cath_log_path = setup_cath_logs()

    # Try to embedding 
    try:
        dictionaries = load_embeddings(args.embeddings, args.dataset)
        normalized_dictionaries = [normalize_embeddings(dict_emb, args.device) for dict_emb in dictionaries]
        embedding_dictionary, _ = concatenate_embeddings(normalized_dictionaries)

    except ValueError as e:
        print(f"Error: {e}")
        return
    
    query_domains = load_superfamily_mapping("/new-stg/home/young/plm-fusion-for-ppi/data/cath/cath_queries.tsv")
    all_domains = load_superfamily_mapping("/new-stg/home/young/plm-fusion-for-ppi/data/cath/cath_superfamilies.tsv")

    # Convert embedding dictionary to matrix for batch processing
    all_domain_ids, all_embeddings = zip(*[(id, embedding_dictionary[id]) for id in all_domains])
    all_embeddings_tensor = torch.stack(all_embeddings).to(args.device)
    
    correct = 0
    total = len(query_domains)
    with open(cath_log_path, 'a', newline='') as log_file:
        log_writer = csv.writer(log_file, delimiter='\t')
        for query_id, query_superfamily_id in query_domains.items():
            query_embedding = embedding_dictionary[query_id].unsqueeze(0).to(args.device)

            # Compute cosine similarities for all domain embeddings excluding the query embedding itself
            similarities = F.cosine_similarity(query_embedding, all_embeddings_tensor, dim=1)
            similarities[all_domain_ids.index(query_id)] = float('-inf')  # We set similarity between self to be negative infinity so its not counted

            # Get the index of the maximum similarity
            max_index = torch.argmax(similarities)
            retrieved_domain_id = all_domains[all_domain_ids[max_index]]

            # Check if the retrieved domain is from the same superfamily
            if query_superfamily_id == retrieved_domain_id:
                correct += 1

        # Calculate top-1 accuracy
        accuracy = correct / total
        log_writer.writerow([args.embeddings, accuracy])

if __name__ == "__main__":
    main()
