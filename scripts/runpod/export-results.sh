#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export PATH="/root/.local/bin:${PATH}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/opt/anuraset-venv}"
EXPORT_DIR="${ANURASET_EXPORT_DIR:-/workspace/exports}"
DRIVE_DESTINATION="${1:-${ANURASET_DRIVE_DESTINATION:-}}"

cd "${PROJECT_ROOT}"
uv run python -m anuraset_dl.package_results --output-dir "${EXPORT_DIR}"

LATEST_ARCHIVE="$(find "${EXPORT_DIR}" -maxdepth 1 -type f -name 'anuraset-results-*.tar.gz' -print | sort | tail -n 1)"
if [[ -z "${LATEST_ARCHIVE}" ]]; then
  echo "No se encontró el paquete recién generado." >&2
  exit 1
fi

if [[ -n "${DRIVE_DESTINATION}" ]]; then
  rclone copy "${LATEST_ARCHIVE}" "${DRIVE_DESTINATION}" --progress
  rclone copy "${LATEST_ARCHIVE}.sha256" "${DRIVE_DESTINATION}" --progress
  echo "Resultados copiados a ${DRIVE_DESTINATION}."
else
  echo "Paquete disponible en ${LATEST_ARCHIVE}."
  echo "Configure ANURASET_DRIVE_DESTINATION o pase un destino rclone para subirlo."
fi
