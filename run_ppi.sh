
# Location of the logs
log_directory="results/ppi_results"

# Clear any previous logging
if [ -f "${log_directory}/ppi_log.tsv" ]; then
    rm "${log_directory}/ppi_log.tsv"
fi

embeddings=(B C D E F G)
seeds=(2 4 8 16 32)

# Store the average auprc for individual embeddings
declare -A auprc_averages

# Loop over each embedding
for emb in "${embeddings[@]}"; do
    total_auprc=0
    for seed in "${seeds[@]}"; do
        # Train model using that embedding for 5 seeds
        python3 src/train_ppi.py --dataset 'ppi' --embeddings "$emb" --hidden_dimension 1024 --device 'cuda' --seed "$seed" --evaluate
        auprc=$(awk -F'\t' -v emb="$emb" -v seed="$seed" '($3 == seed && $1 == emb) {print $13}' "${log_directory}/ppi_log.tsv")
        total_auprc=$(awk -v total="$total_auprc" -v add="$auprc" 'BEGIN { print total + add }')
    done
    # Calculate the average AUPRC for this embedding
    average_auprc=$(awk -v total="$total_auprc" -v count="${#seeds[@]}" 'BEGIN { print total / count }')
    auprc_averages[$emb]=$average_auprc
done

# Sort embeddings by average AUPRC (highest to lowest)
IFS=$'\n' sorted_embeddings=($(for emb in "${!auprc_averages[@]}"; do echo "$emb ${auprc_averages[$emb]}"; done | sort -k2 -nr | awk '{print $1}'))
unset IFS

# Now we do a greedy search
best_combination=()
current_best_auprc=0


for emb in "${sorted_embeddings[@]}"; do
    if [ -z "${best_combination[*]}" ]; then
        best_combination+=("$emb")
        current_best_auprc="${auprc_averages[$emb]}"
        # Highest single embedding is always part of combination
        echo "Initial best combination set to $emb with AUPRC: $current_best_auprc"
    else
        # Concatenate new embedding to the current combination
        current_combination=("${best_combination[@]}" "$emb")
        IFS=$'\n' sorted_combination=($(printf "%s\n" "${current_combination[@]}" | sort))
        unset IFS
        new_combination_str=$(printf "%s" "${sorted_combination[@]}")
        
        # Train model using current combination of embeddings
        for seed in "${seeds[@]}"; do
            python3 src/train_ppi.py --dataset 'ppi' --embeddings "$new_combination_str" --hidden_dimension 1024 --device 'cuda' --seed "$seed" --evaluate
        done

        # Calculate average AUPRC for the new combination
        total_auprc=0
        for seed in "${seeds[@]}"; do
            auprc=$(awk -F'\t' -v seed="$seed" -v emb="$new_combination_str" '($3 == seed && $1 == emb) {print $13}' "${log_directory}/ppi_log.tsv")
            total_auprc=$(awk -v total="$total_auprc" -v add="$auprc" 'BEGIN { print total + add }')
        done
        average_auprc_new=$(awk -v total="$total_auprc" -v count="${#seeds[@]}" 'BEGIN { print total / count }')
        echo "Average AUPRC for combination $new_combination_str: $average_auprc_new"


        # If the avg AUPRC of current combination > avg AUPRC of current best combination, current combination becomes the current best.
        if awk -v new="$average_auprc_new" -v best="$current_best_auprc" 'BEGIN { if (new > best) exit 0; else exit 1 }'; then
            current_best_auprc=$average_auprc_new
            echo "Updated best combination: ${sorted_combination[*]}"
            best_combination=("${sorted_combination[@]}") # Update best combination to sorted one
        fi
    fi
done
echo "Best combination found: ${best_combination[*]}"


# Now train models with all combinations to identify the true best combination
embeddings=(BC BD BE BF BG CD CE CF CG DE DF DG EF EG FG BCD BCE BCF BCG BDE BDF BDG BEF BEG BFG CDE CDF CDG CEF CEG CFG DEF DEG DFG EFG BCDE BCDF BCDG BCEF BCEG BCFG BDEF BDEG BDFG BEFG CDEF CDEG CDFG CEFG DEFG BCDEF BCDEG BCDFG BCEFG BDEFG CDEFG BCDEFG)
for emb in "${embeddings[@]}"; do
    for seed in "${seeds[@]}"; do
        python3 src/train_ppi.py --dataset 'ppi' --embeddings "$emb" --hidden_dimension 1024 --device 'cuda' --seed "$seed" --evaluate
    done
done
