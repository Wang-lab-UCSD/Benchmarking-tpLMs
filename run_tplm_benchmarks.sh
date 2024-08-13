# A: esm2
# B: esm3
# C: ontoprotein
# D: proteinclip
# E: protst
# F: protrek
# G: proteindt

# Train models for comparison of tpLMs with ESM2 3B

embeddings=(A B C D E F G)
seeds=(2 4 8 16 32)

# AAV
for emb in "${embeddings[@]}"; do
    for seed in "${seeds[@]}"; do
        python3 src/train_benchmarks.py --dataset 'aav' --embeddings "$emb" --hidden_dimension 32 --device 'cuda' --dropout 0.2 --seed "$seed" --evaluate
    done
done

# GB1
for emb in "${embeddings[@]}"; do
    for seed in "${seeds[@]}"; do
        python3 src/train_benchmarks.py --dataset 'gb1' --embeddings "$emb" --hidden_dimension 32 --device 'cuda' --dropout 0.2 --seed "$seed" --evaluate
    done
done

# GFP
for emb in "${embeddings[@]}"; do
    for seed in "${seeds[@]}"; do
        python3 src/train_benchmarks.py --dataset 'gfp' --embeddings "$emb" --hidden_dimension 32 --device 'cuda' --dropout 0.2 --seed "$seed" --evaluate
    done
done

# Location
for emb in "${embeddings[@]}"; do
    for seed in "${seeds[@]}"; do
        python3 src/train_benchmarks.py --dataset 'location' --embeddings "$emb" --hidden_dimension 32 --device 'cuda' --dropout 0.2 --seed "$seed" --evaluate
    done
done

# Meltome
for emb in "${embeddings[@]}"; do
    for seed in "${seeds[@]}"; do
        python3 src/train_benchmarks.py --dataset 'meltome' --embeddings "$emb" --hidden_dimension 32 --device 'cuda' --dropout 0.2 --seed "$seed" --evaluate
    done
done

# Stability
for emb in "${embeddings[@]}"; do
    for seed in "${seeds[@]}"; do
        python3 src/train_benchmarks.py --dataset 'stability' --embeddings "$emb" --hidden_dimension 32 --device 'cuda' --dropout 0.2 --seed "$seed" --evaluate
    done
done


