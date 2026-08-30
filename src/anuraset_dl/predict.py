"""Inferencia reproducible sobre audios externos sin etiquetas."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from anuraset_dl.data import UnlabeledAudioDataset
from anuraset_dl.models import build_model
from anuraset_dl.runtime import (
    atomic_json_dump,
    config_fingerprint,
    data_fingerprints,
    resolve_device,
    set_reproducibility,
    sha256_file,
)
from anuraset_dl.utils import load_config


def _inventory_sha256(dataset: UnlabeledAudioDataset) -> str:
    """Identifica el inventario externo por nombre y contenido."""
    digest = hashlib.sha256()
    for path in dataset.paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _atomic_csv_dump(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False, lineterminator="\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def predict_unlabeled(
    config: dict[str, Any],
    input_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    evaluation_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Genera probabilidades y decisiones para un directorio sin etiquetas."""
    set_reproducibility(int(config["seed"]))
    device = resolve_device(str(config["training"].get("device", "auto")))
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else (
        Path(config["training"].get("checkpoint_dir", "outputs/checkpoints"))
        / config["experiment"]
        / "best.pt"
    )
    evaluation_path = Path(evaluation_path) if evaluation_path else (
        Path(config["evaluation"].get("metrics_dir", "outputs/metrics"))
        / f"{config['experiment']}.json"
    )
    output_path = Path(output_path) if output_path else (
        Path("outputs/predictions") / f"{config['experiment']}.csv"
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    semantic_hash = config_fingerprint(config)
    fingerprints = data_fingerprints(config)
    labels = tuple(checkpoint.get("labels", []))
    if checkpoint.get("experiment") != config["experiment"]:
        raise ValueError("El checkpoint pertenece a otro experimento")
    if checkpoint.get("config_sha256") != semantic_hash:
        raise ValueError("El checkpoint no corresponde a la configuración efectiva")
    if checkpoint.get("data_fingerprints") != fingerprints:
        raise ValueError("Los datos etiquetados cambiaron desde el entrenamiento")
    if len(labels) != int(config["data"]["num_labels"]):
        raise ValueError("El checkpoint no contiene el orden esperado de etiquetas")

    with evaluation_path.open(encoding="utf-8") as stream:
        evaluation = json.load(stream)
    if evaluation.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError("Los umbrales no corresponden al checkpoint")
    if evaluation.get("config_sha256") != semantic_hash:
        raise ValueError("Los umbrales no corresponden a la configuración")
    threshold_map = evaluation.get("thresholds", {})
    if set(threshold_map) != set(labels):
        raise ValueError("Los umbrales no coinciden con las etiquetas del checkpoint")
    thresholds = np.asarray([threshold_map[label] for label in labels], dtype=np.float64)

    dataset = UnlabeledAudioDataset(config, input_dir)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
        pin_memory=False,
    )
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    filenames: list[str] = []
    batches: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for inputs, batch_filenames in loader:
            batches.append(torch.sigmoid(model(inputs.to(device))).cpu().numpy())
            filenames.extend(batch_filenames)
    probabilities = np.concatenate(batches)
    frame = pd.DataFrame({"filename": filenames})
    for index, label in enumerate(labels):
        frame[f"{label}_probability"] = probabilities[:, index]
        frame[f"{label}_prediction"] = (probabilities[:, index] >= thresholds[index]).astype(int)
    _atomic_csv_dump(frame, output_path)
    result = {
        "output_path": str(output_path),
        "examples": len(frame),
        "labels": list(labels),
        "input_inventory_sha256": _inventory_sha256(dataset),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "thresholds_path": str(evaluation_path),
    }
    metadata_path = output_path.with_suffix(".json")
    result["metadata_path"] = str(metadata_path)
    atomic_json_dump(result, metadata_path)
    return result


def main() -> None:
    """Ejecuta inferencia sobre el directorio externo configurado por el usuario."""
    parser = argparse.ArgumentParser(description="Inferencia sobre audios sin etiquetas")
    parser.add_argument("--config", required=True, help="Ruta de la configuración YAML")
    parser.add_argument("--input-dir", default="dataset/test", help="Directorio con WAV externos")
    parser.add_argument("--checkpoint", help="Checkpoint; por defecto usa best.pt")
    parser.add_argument("--evaluation", help="JSON con umbrales; por defecto usa metrics_dir")
    parser.add_argument("--output", help="CSV de salida")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), help="Sobrescribe el dispositivo"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config = copy.deepcopy(config)
        config["training"]["device"] = args.device
    result = predict_unlabeled(
        config,
        args.input_dir,
        checkpoint_path=args.checkpoint,
        evaluation_path=args.evaluation,
        output_path=args.output,
    )
    print(f"Inferencia completada: {result['examples']:,} audios; {result['output_path']}")


if __name__ == "__main__":
    main()
