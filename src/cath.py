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
    
    query_domains = load_superfamily_mapping("data/cath/cath_queries.tsv")
    all_domains = load_superfamily_mapping("data/cath/cath_superfamilies.tsv")
    
    # Convert embedding dictionary to matrix for batch processing
    all_domain_ids, all_embeddings = zip(*[(id, embedding_dictionary[id]) for id in all_domains])
    all_embeddings_tensor = torch.stack(all_embeddings).to(args.device)
    
    correct = 0
    total = len(query_domains)

    # Dynamically determine C classes in your dataset
    c_classes_in_data = sorted(set(sf.split('.')[0] for sf in query_domains.values()))
    per_class_correct = {c: 0 for c in c_classes_in_data}
    per_class_total = {c: 0 for c in c_classes_in_data}


    with open(cath_log_path, 'a', newline='') as log_file:
        log_writer = csv.writer(log_file, delimiter='\t')

        for query_id, query_superfamily_id in query_domains.items():
            query_embedding = embedding_dictionary[query_id].unsqueeze(0).to(args.device)

            # Compute cosine similarities with all domain embeddings
            similarities = F.cosine_similarity(query_embedding, all_embeddings_tensor, dim=1)
            similarities[all_domain_ids.index(query_id)] = float('-inf')  # exclude self

            max_index = torch.argmax(similarities)
            retrieved_domain_id = all_domains[all_domain_ids[max_index]]

            # Parse C class (first number of CATH)
            query_C = query_superfamily_id.split('.')[0]
            per_class_total[query_C] += 1

            if query_superfamily_id == retrieved_domain_id:
                correct += 1
                per_class_correct[query_C] += 1

        # Overall accuracy
        accuracy = correct / total

        # Prepare log row
        log_row = [args.embeddings, accuracy]
        for c in sorted(per_class_correct):
            correct_c = per_class_correct[c]
            total_c = per_class_total[c]
            log_row.append(f"{correct_c}/{total_c}" if total_c > 0 else "0/0")

        # Write: embeddings, overall accuracy, then per-Class accuracies (C=1 to C=6)
        log_writer.writerow(log_row)

if __name__ == "__main__":
    main()
