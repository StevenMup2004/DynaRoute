#!/usr/bin/env bash
set -uo pipefail

# Run all LPG checkpoints on DynaBench sequentially.
#
# Usage:
#   nohup bash run_all_lpg_checkpoints.sh > logs/run_all_lpg_checkpoints.out 2>&1 &
#
# Optional overrides:
#   CHECKPOINT_ROOT=/path/to/seed_11 bash run_all_lpg_checkpoints.sh
#   GPU=0 MAX_NEW_TOKENS=160 bash run_all_lpg_checkpoints.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LPG_REPO_PATH="${LPG_REPO_PATH:-/home/user04/VHDang/Guardrail/fix-ver/Latent_Policy_Guard}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${LPG_REPO_PATH}/training/outputs/lpg_qwen3_4b/Qwen3-4B/ep_3/lr_5e-05/seed_11}"
DATASET_PATH="${DATASET_PATH:-montehoover/DynaBench}"
SUBSET="${SUBSET:-DynaBench}"
SPLIT="${SPLIT:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-log/lpg_dynabench}"
SUMMARY_CSV="${SUMMARY_CSV:-log/summary_lpg_all_checkpoints.csv}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-1024}"
GPU="${GPU:-}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-logs/lpg_all_checkpoints_${RUN_ID}}"
mkdir -p "$LOG_DIR" "$(dirname "$SUMMARY_CSV")"

echo "LPG repo:        $LPG_REPO_PATH"
echo "Checkpoint root: $CHECKPOINT_ROOT"
echo "Dataset:         $DATASET_PATH / $SUBSET / $SPLIT"
echo "Log dir:         $LOG_DIR"
echo "Summary CSV:     $SUMMARY_CSV"

mapfile -t CHECKPOINTS < <(
  find "$CHECKPOINT_ROOT" -maxdepth 1 -type d -name 'checkpoint-*' \
    | sort -V
)

if [[ "${#CHECKPOINTS[@]}" -eq 0 ]]; then
  echo "No checkpoints found under: $CHECKPOINT_ROOT" >&2
  exit 1
fi

echo "Found ${#CHECKPOINTS[@]} checkpoints."

FAILURES=()

for CKPT_DIR in "${CHECKPOINTS[@]}"; do
  CKPT_NAME="$(basename "$CKPT_DIR")"
  LOG_FILE="${LOG_DIR}/${CKPT_NAME}.log"

  echo
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $CKPT_NAME"
  echo "  ckpt: $CKPT_DIR"
  echo "  log:  $LOG_FILE"

  CMD=(
    python eval_lpg_dynabench.py
    --lpg_repo_path "$LPG_REPO_PATH"
    --model_path "$MODEL_PATH"
    --ckpt_dir "$CKPT_DIR"
    --dataset_path "$DATASET_PATH"
    --subset "$SUBSET"
    --split "$SPLIT"
    --output_dir "$OUTPUT_DIR"
    --summary_csv "$SUMMARY_CSV"
    --run_name "lpg_${CKPT_NAME}"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --model_max_length "$MODEL_MAX_LENGTH"
  )

  if [[ -n "$GPU" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}" > "$LOG_FILE" 2>&1
    STATUS=$?
  else
    "${CMD[@]}" > "$LOG_FILE" 2>&1
    STATUS=$?
  fi

  if [[ "$STATUS" -eq 0 ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE   $CKPT_NAME"
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED $CKPT_NAME (exit $STATUS)"
    FAILURES+=("$CKPT_NAME")
  fi
done

echo
echo "Summary CSV: $SUMMARY_CSV"
echo "Logs:        $LOG_DIR"

if [[ "${#FAILURES[@]}" -gt 0 ]]; then
  echo "Failed checkpoints: ${FAILURES[*]}" >&2
  exit 1
fi

echo "All checkpoints finished successfully."
