#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_PATH="$ROOT_DIR/pretrained/sa2va-models/Sa2VA-1B"
DATA_ROOT="$ROOT_DIR/data/ROS-Sa2VA"
OUTPUT_ROOT="$ROOT_DIR/work_dirs/ros_refseg_eval"
RUN_NAME=""
DATASET="all"
SPLIT="all"
MODE="smoke"
SMOKE_STEPS=8
NUM_GPUS=1
GPU_IDS="0"
NUM_WORKERS=4
USE_THINK="false"
SAVE_PRED_MASK=0
DEVICE="cuda"
DRY_RUN=0

print_help() {
  cat <<EOF
Usage: tools/eval_ros_refseg.sh [options]

Options:
  --model-path PATH         Model path (default: pretrained/sa2va-models/Sa2VA-1B)
  --data-root PATH          Data root (default: data/ROS-Sa2VA)
  --output-root PATH        Output root (default: work_dirs/ros_refseg_eval)
  --run-name NAME           Explicit run directory name
  --dataset NAME            RRSISD | ris_lad | all (default: all)
  --split NAME              val | test | all (default: all)
  --mode MODE               smoke | full (default: smoke)
  --smoke-steps N           Max steps for smoke mode (default: 8)
  --num-gpus N              Number of GPUs (default: 1)
  --gpu-ids IDS             CUDA visible ids, e.g. 0 or 0,1,2,3 (default: 0)
  --num-workers N           Dataloader workers per process (default: 4)
  --use-think BOOL          true | false (default: false)
  --save-pred-mask          Save predicted masks
  --device NAME             cuda | cpu | auto (default: cuda)
  --dry-run                 Print command only, do not execute
  -h, --help                Show this help

Examples:
  Single-GPU smoke:
    tools/eval_ros_refseg.sh --mode smoke --num-gpus 1 --gpu-ids 0

  Single-GPU full:
    tools/eval_ros_refseg.sh --mode full --num-gpus 1 --gpu-ids 0

  4-GPU smoke:
    tools/eval_ros_refseg.sh --mode smoke --num-gpus 4 --gpu-ids 0,1,2,3

  4-GPU full:
    tools/eval_ros_refseg.sh --mode full --num-gpus 4 --gpu-ids 0,1,2,3
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)
      MODEL_PATH="$2"; shift 2 ;;
    --data-root)
      DATA_ROOT="$2"; shift 2 ;;
    --output-root)
      OUTPUT_ROOT="$2"; shift 2 ;;
    --run-name)
      RUN_NAME="$2"; shift 2 ;;
    --dataset)
      DATASET="$2"; shift 2 ;;
    --split)
      SPLIT="$2"; shift 2 ;;
    --mode)
      MODE="$2"; shift 2 ;;
    --smoke-steps)
      SMOKE_STEPS="$2"; shift 2 ;;
    --num-gpus)
      NUM_GPUS="$2"; shift 2 ;;
    --gpu-ids)
      GPU_IDS="$2"; shift 2 ;;
    --num-workers)
      NUM_WORKERS="$2"; shift 2 ;;
    --use-think)
      USE_THINK="$2"; shift 2 ;;
    --save-pred-mask)
      SAVE_PRED_MASK=1; shift 1 ;;
    --device)
      DEVICE="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift 1 ;;
    -h|--help)
      print_help; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      print_help
      exit 1 ;;
  esac
done

if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "--mode must be smoke or full" >&2
  exit 1
fi

if [[ "$NUM_GPUS" -lt 1 ]]; then
  echo "--num-gpus must be >= 1" >&2
  exit 1
fi

MAX_STEP=0
if [[ "$MODE" == "smoke" ]]; then
  MAX_STEP="$SMOKE_STEPS"
fi

if [[ -z "$RUN_NAME" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  RUN_NAME="${MODE}_${DATASET}_${SPLIT}_${NUM_GPUS}gpu_${TS}"
fi

RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
mkdir -p "$OUTPUT_ROOT"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TOKENIZERS_PARALLELISM=false

BASE_ARGS=(
  "projects/sa2va/evaluation/ros_refseg/eval_sa2va_ros_refseg.py"
  "$MODEL_PATH"
  "--dataset" "$DATASET"
  "--split" "$SPLIT"
  "--data_root" "$DATA_ROOT"
  "--output_root" "$OUTPUT_ROOT"
  "--run_name" "$RUN_NAME"
  "--num_workers" "$NUM_WORKERS"
  "--max_step" "$MAX_STEP"
  "--use_think" "$USE_THINK"
  "--device" "$DEVICE"
)

if [[ "$SAVE_PRED_MASK" -eq 1 ]]; then
  BASE_ARGS+=("--save_pred_mask")
fi

cd "$ROOT_DIR"

echo "ROOT_DIR=$ROOT_DIR"
echo "RUN_DIR=$RUN_DIR"
echo "MODE=$MODE NUM_GPUS=$NUM_GPUS GPU_IDS=$GPU_IDS"

if [[ "$NUM_GPUS" -eq 1 ]]; then
  CMD=(python "${BASE_ARGS[@]}" "--launcher" "none")
else
  CMD=(torchrun --standalone --nnodes=1 --nproc-per-node "$NUM_GPUS" "${BASE_ARGS[@]}" "--launcher" "pytorch")
fi

echo "Command: ${CMD[*]}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

"${CMD[@]}"

echo "Done. Artifacts are in $RUN_DIR"
