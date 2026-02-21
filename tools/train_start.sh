#!/usr/bin/env bash

set -x

# 默认值
DEFAULT_CUDA_DEVICES="0,1,2,3"

CONFIG_FILE="${1:?please specify config file}"
CUDA_DEVICES="${2:-$DEFAULT_CUDA_DEVICES}"
NUM_GPUS="${3:-4}"  # 默认使用4张GPU

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export CUDA_PATH="/usr/local/cuda-12.4"

. .venv/bin/activate || { echo "activate failed, exit."; exit 1; }
echo "activate success."

bash tools/dist.sh train "$CONFIG_FILE" "$NUM_GPUS"
