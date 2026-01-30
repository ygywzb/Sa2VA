#!/usr/bin/env bash

set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_PATH="/workspace/cuda-11.8"

. .venv/bin/activate || { echo "activate failed, exit."; exit 1; }
echo "activate success."

bash tools/dist.sh train projects/sa2va/configs/sa2va_in25_1b_dev.py 4
