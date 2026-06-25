#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HDG="${HDG_POSTPROCESS_PATH:-/path/to/HDG_postprocess}"
ENV_NAME="${SOLEDGE_AMCIS_ENV:-soledge-amcis}"

if ! command -v micromamba >/dev/null 2>&1; then
  echo "micromamba required"
  exit 1
fi
if [[ ! -d "$HDG" ]]; then
  echo "HDG_postprocess not found: $HDG"
  exit 1
fi

micromamba env create -f "$ROOT/environment.yml" -n "$ENV_NAME" 2>/dev/null \
  || micromamba env update -f "$ROOT/environment.yml" -n "$ENV_NAME" --yes
micromamba run -n "$ENV_NAME" pip install -U pip setuptools wheel
micromamba run -n "$ENV_NAME" pip install -e "$ROOT[dev]"
micromamba run -n "$ENV_NAME" pip install -e "$HDG"
echo "Done. Run: micromamba activate $ENV_NAME && amcis"
