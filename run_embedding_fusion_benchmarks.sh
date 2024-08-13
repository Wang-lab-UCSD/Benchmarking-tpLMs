# Train models for embedding fusion with tpLMs

embeddings=(BC BD BE BF BG CD CE CF CG DE DF DG EF EG FG BCD BCE BCF BCG BDE BDF BDG BEF BEG BFG CDE CDF CDG CEF CEG CFG DEF DEG DFG EFG BCDE BCDF BCDG BCEF BCEG BCFG BDEF BDEG BDFG BEFG CDEF CDEG CDFG CEFG DEFG BCDEF BCDEG BCDFG BCEFG BDEFG CDEFG BCDEFG)
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

# Now, train models with repeated single embeddings for comparison (5 seeds as before)
# AAV
for seed in "${seeds[@]}"; do
    python3 src/train_benchmarks.py --dataset 'aav' --embeddings "CCCC" --hidden_dimension 32 --device 'cuda' --dropout 0.2 --seed "$seed" --evaluate
done
# GB1
for seed in "${seeds[@]}"; do
    python3 src/train_benchmarks.py --dataset 'gb1' --embeddings "FFFFFF" --hidden_dimension 32 --device 'cuda' --dropout 0.2 --seed "$seed" --evaluate
done
# GFP
for seed in "${seeds[@]}"; do
    python3 src/train_benchmarks.py --dataset 'gfp' --embeddings "CC" --hidden_dimension 32 --device 'cuda' --dropout 0.2 --seed "$seed" --evaluate
done
# Location
for seed in "${seeds[@]}"; do
    python3 src/train_benchmarks.py --dataset 'location' --embeddings "FF" --hidden_dimension 32 --device 'cuda' --dropout 0.2 --seed "$seed" --evaluate
done
# Meltome
for seed in "${seeds[@]}"; do
    python3 src/train_benchmarks.py --dataset 'meltome' --embeddings "FFFFFF" --hidden_dimension 32 --device 'cuda' --dropout 0.2 --seed "$seed" --evaluate
done
# Stability
for seed in "${seeds[@]}"; do
    python3 src/train_benchmarks.py --dataset 'stability' --embeddings "BB" --hidden_dimension 32 --device 'cuda' --dropout 0.2 --seed "$seed" --evaluate
done