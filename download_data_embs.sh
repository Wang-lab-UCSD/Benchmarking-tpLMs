
huggingface-cli login --add-to-git-credential --token # Add your Huggingface token here 
download_dir="$PWD"
huggingface-cli download yk0/Benchmarking_tpLMs_data  data_and_embeddings.tar.gz --repo-type dataset --local-dir "$download_dir"

# Unzip the file
if [ -f "$download_dir/data_and_embeddings.tar.gz" ]; then
    tar -xzvf "$download_dir/data_and_embeddings.tar.gz" -C "$download_dir"
    echo "File unzipped successfully in $download_dir"
else
    echo "File to unzip not found. Check if file is in correct location."
    exit 1
fi