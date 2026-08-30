#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export PATH="/root/.local/bin:${PATH}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/opt/anuraset-venv}"
EXPORT_DIR="${ANURASET_EXPORT_DIR:-/workspace/exports}"

cd "${PROJECT_ROOT}"
uv run python -m anuraset_dl.run_experiments --export-dir "${EXPORT_DIR}" "$@"
