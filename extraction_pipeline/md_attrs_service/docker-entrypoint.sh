#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set" >&2
  exit 1
fi

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
if ! has_arg --md_dir "$@"; then
  EXTRA_ARGS+=(--md_dir "${MD_DIR:-/workspace/ocr_md}")
fi

python3 /app/run_attrs_extraction.py "${EXTRA_ARGS[@]}" "$@"
