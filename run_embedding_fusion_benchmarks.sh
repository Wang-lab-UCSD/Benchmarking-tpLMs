# Train models for embedding fusion with tpLMs

embeddings=(BC BD BE BF BG CD CE CF CG DE DF DG EF EG FG BCD BCE BCF BCG BDE BDF BDG BEF BEG BFG CDE CDF CDG CEF CEG CFG DEF DEG DFG EFG BCDE BCDF BCDG BCEF BCEG BCFG BDEF BDEG BDFG BEFG CDEF CDEG CDFG CEFG DEFG BCDEF BCDEG BCDFG BCEFG BDEFG CDEFG BCDEFG)
seeds=(2 4 8 16 32)
datasets=("aav" "gb1" "gfp" "location" "meltome" "stability")
for dataset in "${datasets[@]}"; do
    for emb in "${embeddings[@]}"; do
        for seed in "${seeds[@]}"; do
            python3 src/train_benchmarks.py --mode train --dataset "$dataset" --embeddings "$emb" --hidden_dimension 32 --device 'cuda' --dropout 0.2 --seed "$seed" --evaluate
        done
    done
done


#Now, train models with repeated single embeddings for comparison (5 seeds as before)
best_embeddings=("CCCC" "FFFFFF" "CC" "FF" "FFFFFF" "BB")

for i in "${!datasets[@]}"; do
    dataset="${datasets[$i]}"
    emb="${best_embeddings[$i]}"
    for seed in "${seeds[@]}"; do
        python3 src/train_benchmarks.py --mode train --dataset "$dataset" --embeddings "$emb" --hidden_dimension 32 --device 'cuda' --dropout 0.2 --seed "$seed" --evaluate
    done
done
