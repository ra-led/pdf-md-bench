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
if ! has_arg --json_dir "$@"; then
  EXTRA_ARGS+=(--json_dir "${JSON_DIR:-/workspace/input}")
fi
if ! has_arg --output_dir "$@"; then
  EXTRA_ARGS+=(--output_dir "${OUTPUT_DIR:-/workspace/output}")
fi

python3 /app/evaluate_attrs_service.py "${EXTRA_ARGS[@]}" "$@"
