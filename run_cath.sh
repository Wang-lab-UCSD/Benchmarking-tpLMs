embeddings=(B C D E F G)

# Location of the logs
log_directory="results/cath_results"

# Store the accuracies for individual embeddings
declare -A accs

# Loop over each embedding
for emb in "${embeddings[@]}"; do
    python3 src/cath.py --dataset 'cath' --embeddings "$emb" --device 'cuda'
    acc=$(awk -F'\t' -v emb="$emb" '($1 == emb) {print $2}' "${log_directory}/cath_log.tsv")
    accs[$emb]=$acc
done

# Sort embeddings by accuracy (highest to lowest)
IFS=$'\n' sorted_embeddings=($(for emb in "${!accs[@]}"; do echo "$emb ${accs[$emb]}"; done | sort -k2 -nr | awk '{print $1}'))
unset IFS

# Now we do a greedy search
best_combination=()
current_best_acc=0

for emb in "${sorted_embeddings[@]}"; do
    if [ -z "${best_combination[*]}" ]; then
        best_combination+=("$emb")
        current_best_acc="${accs[$emb]}"
        echo "Initial best combination set to $emb with Accuracy: $current_best_acc"
    else
        # Concatenate new embedding to the current combination
        current_combination=("${best_combination[@]}" "$emb")
        IFS=$'\n' sorted_combination=($(printf "%s\n" "${current_combination[@]}" | sort))
        unset IFS
        new_combination_str=$(printf "%s" "${sorted_combination[@]}")
        
        # Evaluate homologous sequence recovery with current combination of embeddings
        python3 src/cath.py --dataset 'cath' --embeddings "$new_combination_str" --device 'cuda'

        # Calculate accuracy for the new combination
        acc_new=$(awk -F'\t' -v emb="$new_combination_str" '($1 == emb) {print $2}' "${log_directory}/cath_log.tsv")
        echo "Accuracy for combination $new_combination_str: $acc_new"

        # If the accuracy of current combination > accuracy of current best combination, current combination becomes the current best.
        if awk -v new="$acc_new" -v best="$current_best_acc" 'BEGIN { if (new > best) exit 0; else exit 1 }'; then
            current_best_acc=$acc_new
            echo "Updated best combination: ${sorted_combination[*]}"
            best_combination=("${sorted_combination[@]}") # Update best combination to sorted one
        fi
    fi
done
echo "Best combination found: ${best_combination[*]}"

# Now do homologous sequence recovery with all combinations to identify the true best combination
embeddings=(BC BD BE BF BG CD CE CF CG DE DF DG EF EG FG BCD BCE BCF BCG BDE BDF BDG BEF BEG BFG CDE CDF CDG CEF CEG CFG DEF DEG DFG EFG BCDE BCDF BCDG BCEF BCEG BCFG BDEF BDEG BDFG BEFG CDEF CDEG CDFG CEFG DEFG BCDEF BCDEG BCDFG BCEFG BDEFG CDEFG BCDEFG)
for emb in "${embeddings[@]}"; do
    python3 src/cath.py --dataset 'cath' --embeddings "$emb" --device 'cuda'
done
