#!/usr/bin/env bash
set -euo pipefail

has_arg() {
  local key="$1"
  shift
  for arg in "$@"; do
    if [[ "$arg" == "$key" || "$arg" == "$key="* ]]; then
      return 0
    fi
  done
  return 1
}

EXTRA_ARGS=()
if ! has_arg --input_pdf_dir "$@"; then
  EXTRA_ARGS+=(--input_pdf_dir "${INPUT_PDF_DIR:-/workspace/input}")
fi
if ! has_arg --output_dir "$@"; then
  EXTRA_ARGS+=(--output_dir "${OUTPUT_DIR:-/workspace/output}")
fi
if ! has_arg --work_dir "$@"; then
  EXTRA_ARGS+=(--work_dir "${WORK_DIR:-/workspace/work}")
fi
if ! has_arg --adapter_checkpoint "$@"; then
  EXTRA_ARGS+=(--adapter_checkpoint "${ADAPTER_CHECKPOINT:-/workspace/adapter/checkpoint-50}")
fi

python3 /app/pdf_to_attrs_qwen_service.py "${EXTRA_ARGS[@]}" "$@"
