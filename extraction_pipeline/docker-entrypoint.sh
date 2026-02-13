#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set" >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  echo "Usage: docker run ... --pdf /workspace/input/file.pdf --out_json /workspace/output/file.json [extra args]" >&2
  exit 1
fi

PIPELINE_SCRIPT="${PIPELINE_SCRIPT:-pdf_to_characteristics_json.py}"
TARGET="/app/${PIPELINE_SCRIPT}"

if [[ ! -f "${TARGET}" ]]; then
  echo "ERROR: pipeline script not found: ${TARGET}" >&2
  exit 1
fi

python3 "${TARGET}" "$@"
