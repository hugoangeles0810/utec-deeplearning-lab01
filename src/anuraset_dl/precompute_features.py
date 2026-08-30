"""Precomputación reproducible de representaciones tiempo-frecuencia."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from anuraset_dl.data import AnuraDataset, build_transform
from anuraset_dl.feature_cache import (
    CACHE_SCHEMA_VERSION,
    cache_directory,
    cache_fingerprint,
    cache_inputs,
    cached_split_path,
    sha256_file,
)
from anuraset_dl.utils import load_config


def _filenames_sha256(dataset: AnuraDataset) -> str:
    encoded = "\n".join(path.name for path in dataset.paths).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _existing_cache_result(config: dict[str, Any], directory: Path) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    for split in config["data"]["splits"]:
        dataset = AnuraDataset(config, split, use_feature_cache=False)
        path, shape = cached_split_path(config, split, len(dataset))
        splits[split] = {"path": str(path), "shape": list(shape)}
    return {
        "path": str(directory),
        "fingerprint": cache_fingerprint(config),
        "created": False,
        "splits": splits,
    }


def precompute_feature_cache(
    config: dict[str, Any], num_workers: int = 0
) -> dict[str, Any]:
    """Materializa todas las particiones en arreglos NPY float32 inmutables."""
    if num_workers < 0:
        raise ValueError("num_workers no puede ser negativo")
    destination = cache_directory(config)
    if destination.exists():
        return _existing_cache_result(config, destination)

    root = destination.parent
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=root))
    transform = build_transform(config)
    split_manifests: dict[str, Any] = {}
    try:
        for split in config["data"]["splits"]:
            dataset = AnuraDataset(
                config,
                split,
                transform=transform,
                use_feature_cache=False,
            )
            if not len(dataset):
                raise ValueError(f"No se puede cachear una partición vacía: {split}")
            first, _ = dataset[0]
            if not first.dtype.is_floating_point:
                raise ValueError(f"Las características de {split} no son flotantes")
            feature_shape = tuple(int(value) for value in first.shape)
            path = temporary / f"{split}.npy"
            array = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=np.float32,
                shape=(len(dataset), *feature_shape),
            )
            loader = DataLoader(
                dataset,
                batch_size=int(config.get("training", {}).get("batch_size", 32)),
                shuffle=False,
                num_workers=num_workers,
            )
            cursor = 0
            for features, _ in loader:
                if (
                    tuple(features.shape[1:]) != feature_shape
                    or not features.dtype.is_floating_point
                ):
                    raise ValueError(
                        f"La forma o el tipo de características cambió dentro de {split}"
                    )
                next_cursor = cursor + len(features)
                array[cursor:next_cursor] = features.numpy()
                cursor = next_cursor
            if cursor != len(dataset):
                raise ValueError(f"La caché de {split} no conserva todos los ejemplos")
            array.flush()
            del array
            split_manifests[split] = {
                "path": path.name,
                "shape": [len(dataset), *feature_shape],
                "filenames_sha256": _filenames_sha256(dataset),
                "sha256": sha256_file(path),
            }

        manifest = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "fingerprint": cache_fingerprint(config),
            "dtype": "float32",
            "inputs": cache_inputs(config),
            "splits": split_manifests,
        }
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "path": str(destination),
        "fingerprint": manifest["fingerprint"],
        "created": True,
        "splits": split_manifests,
    }


def main() -> None:
    """Precalcula la caché declarada por una configuración YAML."""
    parser = argparse.ArgumentParser(
        description="Precomputación de representaciones tiempo-frecuencia"
    )
    parser.add_argument("--config", required=True, help="Ruta de la configuración YAML")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    result = precompute_feature_cache(load_config(args.config), num_workers=args.num_workers)
    action = "creada" if result["created"] else "reutilizada"
    print(f"Caché {action}: {result['path']} ({result['fingerprint']})")


if __name__ == "__main__":
    main()
