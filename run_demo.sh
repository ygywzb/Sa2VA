#!/bin/bash

model_name="$1"
text="$2"
img_path="$3"

# Set CUDA_VISIBLE_DEVICES from the last argument
if [ $# -ge 4 ]; then
    export CUDA_VISIBLE_DEVICES="${!#}"
else
    export CUDA_VISIBLE_DEVICES=1
fi

# check img_path
if [[ "$img_path" != ./demo/images/* ]]; then
    echo "Error: img_path must be under ./demo/images"
    exit 1
fi

echo "CUDA_VISIBLE_DEVICES is set to $CUDA_VISIBLE_DEVICES"

. .venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }
python ./demo/demo_1B.py --model_path "./pretrained/$model_name" --work-dir ./demo/output --text "<image>$text" --single_image "$img_path"