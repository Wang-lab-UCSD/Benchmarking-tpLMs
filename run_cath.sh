# Location of the logs
log_directory="results/cath_results"
# Clear any previous logging
if [ -f "${log_directory}/cath_log.tsv" ]; then
    rm "${log_directory}/cath_log.tsv"
fi

embeddings=("0" "1" "2" B C D E F G)

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
elements=("0" "1" "2" "B" "C" "D" "E" "F" "G")
embeddings=()

num_elements=${#elements[@]}
total=$((1 << num_elements))  # 2^n subsets

for ((i=1; i<total; i++)); do
    subset=""
    for ((j=0; j<num_elements; j++)); do
        if (( (i >> j) & 1 )); then
            subset+="${elements[j]}"
        fi
    done
    if (( ${#subset} > 1 )); then
        embeddings+=("$subset")
    fi
done

IFS=$'\n' sorted=($(printf "%s\n" "${embeddings[@]}" | sort))
embeddings=("${sorted[@]}")


for emb in "${embeddings[@]}"; do
    python3 src/cath.py --dataset 'cath' --embeddings "$emb" --device 'cuda'
done
