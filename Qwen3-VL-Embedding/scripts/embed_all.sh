#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/hyang/BigML_sharding_dataset/Qwen3-VL-Embedding"
DATA_DIR="${ROOT_DIR}/data/consistent_chat"
OUT_DIR="${ROOT_DIR}/outputs/embeddings"

# Optional overrides:
#   BATCH_SIZE=32 MAX_LENGTH=8192 DEVICE=cuda bash scripts/embed_all.sh
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
DEVICE="${DEVICE:-cuda}"

MODELS=(
  "Qwen/Qwen3-Embedding-0.6B"
  "Qwen/Qwen3-Embedding-4B"
)
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29600}"
job_idx=0

mkdir -p "${OUT_DIR}"

for input_path in "${DATA_DIR}"/*.jsonl; do
  file_name="$(basename "${input_path}")"

  # Skip pair files.
  if [[ "${file_name}" == *pair* ]]; then
    continue
  fi

  dataset_name="${file_name%.jsonl}"

  for model_name in "${MODELS[@]}"; do
    model_tag="$(basename "${model_name}" | tr '[:upper:]' '[:lower:]')"
    out_path="${OUT_DIR}/${dataset_name}_${model_tag}.pt"
    master_port="$((MASTER_PORT_BASE + job_idx))"

    echo "[INFO] Embedding ${file_name} with ${model_name} (master_port=${master_port})"
    torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port "${master_port}" "${ROOT_DIR}/embed.py" \
      --input-file "${input_path}" \
      --text-field conversations \
      --model "${model_name}" \
      --batch-size "${BATCH_SIZE}" \
      --max-length "${MAX_LENGTH}" \
      --device "${DEVICE}" \
      --output-path "${out_path}"
    job_idx="$((job_idx + 1))"
  done
done

echo "[DONE] Saved embeddings to ${OUT_DIR}"
