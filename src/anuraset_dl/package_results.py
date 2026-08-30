"""Empaquetado verificable de los artefactos canónicos del proyecto."""

from __future__ import annotations

import argparse
import io
import json
import platform
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from anuraset_dl.runtime import sha256_file

RESULT_PATHS = (
    "configs",
    "outputs/checkpoints",
    "outputs/filterbanks",
    "outputs/metrics",
    "outputs/mlflow",
    "outputs/runpod",
    "docs/experiments.md",
    "pyproject.toml",
    "uv.lock",
)


def _git_value(root: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _result_files(root: Path) -> list[Path]:
    files: set[Path] = set()
    for relative in RESULT_PATHS:
        path = root / relative
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(item for item in path.rglob("*") if item.is_file())
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def build_manifest(root: str | Path, files: list[Path] | None = None) -> dict[str, Any]:
    """Construye el inventario y las huellas del paquete de resultados."""
    project_root = Path(root).resolve()
    selected = files if files is not None else _result_files(project_root)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git": {
            "commit": _git_value(project_root, "rev-parse", "HEAD"),
            "status": _git_value(project_root, "status", "--short"),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "files": [
            {
                "path": path.relative_to(project_root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in selected
        ],
    }


def package_results(
    root: str | Path = ".", output_dir: str | Path = "outputs/exports"
) -> dict[str, Any]:
    """Crea un ``tar.gz`` autocontenido y su huella SHA-256."""
    project_root = Path(root).resolve()
    files = _result_files(project_root)
    has_checkpoint = any(
        path.is_relative_to(project_root / "outputs/checkpoints") for path in files
    )
    has_metrics = any(path.is_relative_to(project_root / "outputs/metrics") for path in files)
    if not has_checkpoint or not has_metrics:
        raise FileNotFoundError(
            "Se requiere al menos un checkpoint y un reporte de métricas para exportar resultados"
        )

    manifest = build_manifest(project_root, files)
    destination = Path(output_dir)
    if not destination.is_absolute():
        destination = project_root / destination
    destination.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive_path = destination / f"anuraset-results-{timestamp}.tar.gz"
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    with tarfile.open(archive_path, mode="w:gz") as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(project_root).as_posix(), recursive=False)
        info = tarfile.TarInfo("manifest.json")
        info.size = len(manifest_bytes)
        info.mtime = int(datetime.now(UTC).timestamp())
        archive.addfile(info, io.BytesIO(manifest_bytes))

    digest = sha256_file(archive_path)
    checksum_path = archive_path.with_suffix(f"{archive_path.suffix}.sha256")
    checksum_path.write_text(f"{digest}  {archive_path.name}\n", encoding="utf-8")
    return {
        "archive": str(archive_path),
        "checksum": str(checksum_path),
        "sha256": digest,
        "files": len(files),
    }


def main() -> None:
    """Empaqueta resultados existentes desde la línea de comandos."""
    parser = argparse.ArgumentParser(description="Empaqueta resultados verificables")
    parser.add_argument("--root", default=".", help="Raíz del repositorio")
    parser.add_argument("--output-dir", default="outputs/exports", help="Destino del paquete")
    args = parser.parse_args()
    result = package_results(args.root, args.output_dir)
    print(f"Resultados empaquetados: {result['archive']} ({result['sha256']})")


if __name__ == "__main__":
    main()
