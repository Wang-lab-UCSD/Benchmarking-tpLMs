# 0: ankh
# 1: esm2 (3B)
# 2: protT5

# B: esm3
# C: ontoprotein
# D: proteinclip
# E: protst
# F: protrek
# G: proteindt

# Train models for comparison of tpLMs with large pLMs.
embeddings=("0" "1" "2" B C D E F G)
seeds=(2 4 8 16 32)
datasets=("aav" "gb1" "gfp" "location" "meltome" "stability")

for dataset in "${datasets[@]}"; do
    for emb in "${embeddings[@]}"; do
        for seed in "${seeds[@]}"; do
            python3 src/train_benchmarks.py --mode train --dataset "$dataset" --embeddings "$emb" --hidden_dimension 32 --device 'cuda' --dropout 0.2 --seed "$seed" --evaluate
        done
    done
done
