"""Artefactos persistentes para representaciones tiempo-frecuencia."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

CACHE_SCHEMA_VERSION = 1


def sha256_file(path: str | Path) -> str:
    """Calcula SHA-256 mediante lectura por bloques."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_enabled(config: dict[str, Any]) -> bool:
    """Indica si la configuración solicita utilizar características persistidas."""
    return bool(config["features"].get("cache", {}).get("enabled", False))


def cache_inputs(config: dict[str, Any]) -> dict[str, Any]:
    """Describe las entradas que determinan el contenido exacto de la caché."""
    features = deepcopy(config["features"])
    features.pop("cache", None)
    bank_sha256 = None
    if features["type"] == "fbrs":
        bank = Path(features.pop("bank_path"))
        if not bank.is_file():
            raise FileNotFoundError(
                f"No existe el banco FBRS {bank}; debe ajustarse antes de crear la caché"
            )
        bank_sha256 = sha256_file(bank)
    payload: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "data": {
            "sample_rate": int(config["data"]["sample_rate"]),
            "clip_seconds": float(config["data"]["clip_seconds"]),
            "metadata_sha256": sha256_file(config["data"]["metadata"]),
            "splits": {
                split: sha256_file(path)
                for split, path in sorted(config["data"]["splits"].items())
            },
        },
        "features": features,
    }
    if bank_sha256 is not None:
        payload["bank_sha256"] = bank_sha256
    return payload


def cache_fingerprint(config: dict[str, Any]) -> str:
    """Calcula la identidad estable de una caché de características."""
    encoded = json.dumps(
        cache_inputs(config), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def cache_directory(config: dict[str, Any]) -> Path:
    """Deriva la ruta inmutable de la caché desde su contenido esperado."""
    settings = config["features"].get("cache", {})
    root = Path(settings.get("root", "outputs/features"))
    return root / f"{config['features']['type']}_{cache_fingerprint(config)}"


def load_cache_manifest(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Carga y valida la identidad global del manifiesto de caché."""
    directory = cache_directory(config)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No existe la caché requerida en {directory}; ejecútese "
            "python -m anuraset_dl.precompute_features --config <config>"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"El manifiesto de caché es ilegible: {manifest_path}") from error
    expected_fingerprint = cache_fingerprint(config)
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("La versión del formato de caché no es compatible")
    if manifest.get("fingerprint") != expected_fingerprint:
        raise ValueError("La caché no corresponde a los datos o a la representación vigente")
    if manifest.get("dtype") != "float32":
        raise ValueError("La implementación vigente requiere una caché float32")
    return directory, manifest


def cached_split_path(
    config: dict[str, Any], split: str, expected_examples: int
) -> tuple[Path, tuple[int, ...]]:
    """Valida la entrada de una partición y devuelve su archivo y forma."""
    directory, manifest = load_cache_manifest(config)
    split_manifest = manifest.get("splits", {}).get(split)
    if not isinstance(split_manifest, dict):
        raise ValueError(f"La caché no contiene la partición {split}")
    shape_value = split_manifest.get("shape")
    if not isinstance(shape_value, list) or not all(
        isinstance(dimension, int) and dimension > 0 for dimension in shape_value
    ):
        raise ValueError(f"La forma persistida de {split} no es válida")
    shape = tuple(shape_value)
    if len(shape) != 4 or shape[0] != expected_examples or shape[1] != 1:
        raise ValueError(f"La caché de {split} no coincide con el manifiesto de datos")
    relative_path = split_manifest.get("path")
    if not isinstance(relative_path, str) or Path(relative_path).name != relative_path:
        raise ValueError(f"La ruta persistida de {split} no es válida")
    path = directory / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Falta el arreglo de caché para {split}: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.float32 or array.shape != shape:
        raise ValueError(f"El arreglo de caché para {split} no coincide con su manifiesto")
    return path, shape
