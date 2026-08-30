#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/opt/anuraset-venv}"

if [[ ! -d /workspace ]]; then
  echo "No existe /workspace; adjunte un Network Volume al crear el Pod." >&2
  exit 1
fi

AVAILABLE_KB="$(df -Pk /workspace | awk 'NR == 2 {print $4}')"
MINIMUM_KB=$((45 * 1024 * 1024))
if (( AVAILABLE_KB < MINIMUM_KB )); then
  echo "Se requieren al menos 45 GB libres en /workspace; se recomiendan 60 GB." >&2
  exit 1
fi

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y curl git libsndfile1 rclone tmux

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="/root/.local/bin:${PATH}"

cd "${PROJECT_ROOT}"
uv python install 3.12
uv sync --frozen --group dev --group tracking

uv run python - <<'PY'
import sys

import torch

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Se esperaba Python 3.12; se obtuvo {sys.version}")
if torch.__version__.split("+")[0] != "2.6.0":
    raise SystemExit(f"Se esperaba PyTorch 2.6.0; se obtuvo {torch.__version__}")
if torch.version.cuda != "12.4":
    raise SystemExit(f"Se esperaba el runtime CUDA 12.4; se obtuvo {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch no detecta CUDA en el Pod")
print(f"CUDA disponible: {torch.cuda.get_device_name(0)}")
print(f"PyTorch {torch.__version__}; runtime CUDA {torch.version.cuda}")
PY

uv run pytest
uv run ruff check .
mkdir -p outputs/runpod/logs /workspace/exports

echo "Entorno Runpod preparado en ${PROJECT_ROOT}."
