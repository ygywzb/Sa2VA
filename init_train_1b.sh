#!/bin/bash
# Sa2VA-1B的训练数据的准备，不包括uv创建的环境

data_dir="$1"
sa2va_dir="$2"

is_abs_path() {
    [[ "$1" == /* ]]
}

echo "Starting environment preparation for Sa2VA-1B ..."
if ! is_abs_path "$data_dir"; then
    echo "Please provide an absolute path for data directory."
    exit 1
fi
if ! is_abs_path "$sa2va_dir"; then
    echo "Please provide an absolute path for Sa2VA code directory."
    exit 1
fi

# prepare data directories
echo "Preparing data directories under $data_dir ..."
cd "$data_dir" || { echo "Data directory not found!"; exit 1; }
mkdir "sa2va_data" && cd "sa2va_data" || { echo "Failed to create or access sa2va_data directory!"; exit 1; }
mkdir "data" || { echo "Failed to create data directory!"; exit 1; }
mkdir "pretrained" || { echo "Failed to create pretrained directory!"; exit 1; }

download_from_hf() {
    local repo_id="$1"
    local local_dir="$2"
    local is_dataset="$3" # "true" for dataset, "false" for model

    # check if hf CLI is installed
    if ! command -v hf &> /dev/null; then
        echo "hf CLI could not be found, please install it first."
        exit 1
    fi

    if [ "$is_dataset" == "true" ]; then
        echo "Downloading dataset from Hugging Face: $repo_id to $local_dir"
        hf download --repo-type dataset "$repo_id" --local-dir "$local_dir"
    else
        echo "Downloading model from Hugging Face: $repo_id to $local_dir"
        hf download "$repo_id" --local-dir "$local_dir"
    fi
}

unzip_all_files() {
    mkdir -p extracted

    files=($(ls -S *.zip))
    total=${#files[@]}

    [ $total -eq 0 ] && { echo "无zip文件" >> unzip.log; exit 0; }

    i=0
    for f in "${files[@]}"; do
        ((i++))
        echo "$(date) [$i/$total] 解压 $f 到 extracted/" >> unzip.log
        unzip -q "$f" -d extracted/
        echo "$(date) [$i/$total] 完成 $f" >> unzip.log
    done
}

export HF_ENDPOINT=https://hf-mirror.com

echo "Downloading datasets for Sa2VA-Training..."
mkdir "tmp"  || { echo "Failed to create tmp directory!"; exit 1; }
# download dataset: $data_dir/sa2va_data/tmp 
download_from_hf "Dense-World/Sa2VA-Training" "./tmp" "true"
cd "tmp"
# unzip dataset to: $data_dir/sa2va_data/tmp/extracted
unzip_all_files
# mv to $data_dir/sa2va_data/data
mv extracted/* ../data/
cd ..
rm -rf "tmp"

echo "Downloading pretrained models for Sa2VA-Training..."
cd "pretrained"
# OpenGVLab/InternVL2_5-1B
download_from_hf "OpenGVLab/InternVL2_5-1B" "./InternVL2_5-1B" "false"
# facebook/sam2-hiera-large
download_from_hf "facebook/sam2-hiera-large" "./sam2" "false"

echo "Environment preparation for Sa2VA-1B completed successfully."

echo "create links to sa2va code directory ..."
cd "$sa2va_dir" || { echo "Failed to access sa2va code directory!"; exit 1; }
ln -s "$data_dir/sa2va_data/pretrained" "./pretrained"
ln -s "$data_dir/sa2va_data/data" "./data"
echo "Links created successfully."

echo "All done."
