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
if ! has_arg --input_dir "$@"; then
  EXTRA_ARGS+=(--input_dir "${INPUT_DIR:-/workspace/input}")
fi
if ! has_arg --output_dir "$@"; then
  EXTRA_ARGS+=(--output_dir "${OUTPUT_DIR:-/workspace/output}")
fi
if ! has_arg --raw_ocr_dir "$@"; then
  EXTRA_ARGS+=(--raw_ocr_dir "${RAW_OCR_DIR:-/workspace/raw_ocr}")
fi

python3 /app/pdf2md_service.py "${EXTRA_ARGS[@]}" "$@"
