"""Utilidades compartidas de entrenamiento, inferencia y artefactos."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from anuraset_dl.data import AnuraDataset


def set_reproducibility(seed: int) -> None:
    """Inicializa los generadores aleatorios usados por los experimentos."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Resuelve ``auto`` con prioridad CUDA, MPS y CPU."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Se solicitó CUDA, pero no está disponible")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Se solicitó MPS, pero no está disponible")
    return device


def build_loader(
    config: dict[str, Any], split: str, shuffle: bool = False
) -> tuple[DataLoader[tuple[Tensor, Tensor]], tuple[str, ...]]:
    """Construye un DataLoader y devuelve el orden canónico de etiquetas."""
    dataset = AnuraDataset(config, split)
    generator = torch.Generator().manual_seed(int(config["seed"]))
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["training"].get("num_workers", 0)),
        pin_memory=False,
        generator=generator,
    )
    return loader, dataset.labels


def predict_probabilities(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Ejecuta inferencia ordenada y devuelve objetivos y probabilidades."""
    model.eval()
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for inputs, batch_targets in loader:
            logits = model(inputs.to(device))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(batch_targets.numpy())
    if not targets:
        raise ValueError("No se puede inferir sobre una partición vacía")
    return np.concatenate(targets).astype(np.int64), np.concatenate(probabilities)


def _semantic_config(config: dict[str, Any]) -> dict[str, Any]:
    """Excluye opciones operativas que no cambian el experimento definido."""
    semantic = deepcopy(config)
    semantic.pop("tracking", None)
    semantic.get("features", {}).pop("cache", None)
    training = semantic.get("training", {})
    for field in ("device", "num_workers", "checkpoint_dir"):
        training.pop(field, None)
    semantic.get("evaluation", {}).pop("metrics_dir", None)
    return semantic


def config_fingerprint(config: dict[str, Any]) -> str:
    """Calcula una huella estable de la configuración semántica."""
    payload = json.dumps(
        _semantic_config(config), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Calcula SHA-256 mediante lectura por bloques."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_fingerprints(config: dict[str, Any]) -> dict[str, Any]:
    """Identifica por contenido los metadatos y manifiestos del experimento."""
    metadata_path = Path(config["data"]["metadata"])
    split_paths = {
        split: Path(path) for split, path in config["data"]["splits"].items()
    }
    fingerprints: dict[str, Any] = {
        "metadata": {"path": str(metadata_path), "sha256": sha256_file(metadata_path)},
        "splits": {
            split: {"path": str(path), "sha256": sha256_file(path)}
            for split, path in sorted(split_paths.items())
        },
    }
    if config["features"]["type"] == "fbrs":
        bank = Path(config["features"]["bank_path"])
        fingerprints["feature_artifact"] = {
            "path": str(bank),
            "sha256": sha256_file(bank),
        }
    return fingerprints


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    """Guarda un checkpoint mediante reemplazo atómico."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json_dump(payload: dict[str, Any], path: str | Path) -> None:
    """Escribe JSON UTF-8 de forma atómica."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
