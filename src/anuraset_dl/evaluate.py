"""Selección de umbrales en validación y evaluación final sobre test."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any

import torch

from anuraset_dl.metrics import multilabel_metrics, select_f1_thresholds
from anuraset_dl.models import build_model
from anuraset_dl.runtime import (
    atomic_json_dump,
    build_loader,
    config_fingerprint,
    data_fingerprints,
    predict_probabilities,
    resolve_device,
    set_reproducibility,
    sha256_file,
)
from anuraset_dl.tracking import MlflowTracker
from anuraset_dl.utils import load_config


def _tracking_metrics(split: str, metrics: dict[str, Any]) -> dict[str, float]:
    """Aplana métricas macro y por clase para compararlas en MLflow."""
    flattened = {
        f"{split}/macro/{name}": float(value)
        for name, value in metrics["macro"].items()
    }
    for row in metrics["per_class"]:
        label = row["label"]
        for name in ("precision", "recall", "f1", "average_precision", "threshold"):
            flattened[f"{split}/per_class/{label}/{name}"] = float(row[name])
    return flattened


def evaluate_experiment(
    config: dict[str, Any], checkpoint_path: str | Path | None = None
) -> dict[str, Any]:
    """Ajusta umbrales en validación y aplica el protocolo congelado a test."""
    if config["features"]["type"] != "mel" or config["model"]["name"] != "cnn":
        raise NotImplementedError("El pipeline ejecutable actual corresponde a CNN + Mel")
    set_reproducibility(int(config["seed"]))
    device = resolve_device(str(config["training"].get("device", "auto")))
    validation_loader, labels = build_loader(config, "validation")
    test_loader, test_labels = build_loader(config, "test")
    if labels != test_labels:
        raise ValueError("Validation y test no comparten el mismo orden de etiquetas")

    if checkpoint_path is None:
        checkpoint_path = (
            Path(config["training"].get("checkpoint_dir", "outputs/checkpoints"))
            / config["experiment"]
            / "best.pt"
        )
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("experiment") != config["experiment"]:
        raise ValueError("El checkpoint pertenece a otro experimento")
    if checkpoint.get("config_sha256") != config_fingerprint(config):
        raise ValueError("El checkpoint no corresponde a la configuración efectiva")
    current_data_fingerprints = data_fingerprints(config)
    if checkpoint.get("data_fingerprints") != current_data_fingerprints:
        raise ValueError("Los metadatos o manifiestos cambiaron desde el entrenamiento")
    if tuple(checkpoint.get("labels", [])) != labels:
        raise ValueError("El orden de etiquetas del checkpoint no coincide")
    semantic_fingerprint = config_fingerprint(config)
    tracker = MlflowTracker.start(
        config,
        phase="evaluation",
        run_id=checkpoint.get("tracking_run_id"),
    )
    if not tracker.resumed:
        tracker.log_config(config)
        tracker.log_dict(current_data_fingerprints, "inputs/data_fingerprints.json")
    tracker.set_tags({"anuraset.config_sha256": semantic_fingerprint})
    started_at = time.perf_counter()

    try:
        model = build_model(config)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        validation_targets, validation_probabilities = predict_probabilities(
            model, validation_loader, device
        )
        thresholds = select_f1_thresholds(validation_targets, validation_probabilities)
        validation_metrics = multilabel_metrics(
            validation_targets, validation_probabilities, thresholds, labels
        )
        test_targets, test_probabilities = predict_probabilities(model, test_loader, device)
        test_metrics = multilabel_metrics(test_targets, test_probabilities, thresholds, labels)
        checkpoint_sha256 = sha256_file(checkpoint_path)
        duration = time.perf_counter() - started_at

        result = {
            "schema_version": 2,
            "experiment": config["experiment"],
            "seed": int(config["seed"]),
            "device": str(device),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "config_sha256": semantic_fingerprint,
            "data_fingerprints": current_data_fingerprints,
            "threshold_strategy": config["evaluation"]["threshold_strategy"],
            "thresholds": dict(zip(labels, thresholds.tolist())),
            "validation": validation_metrics,
            "test": test_metrics,
            "duration_seconds": duration,
            "tracking_run_id": tracker.run_id,
        }
        output_dir = Path(config["evaluation"].get("metrics_dir", "outputs/metrics"))
        output_path = output_dir / f"{config['experiment']}.json"
        atomic_json_dump(result, output_path)
        tracker.set_tags(
            {
                "anuraset.checkpoint_sha256": checkpoint_sha256,
                "anuraset.evaluation_recorded": "true",
            }
        )
        tracker.log_metrics(
            {
                **_tracking_metrics("validation", validation_metrics),
                **_tracking_metrics("test", test_metrics),
                "evaluation/duration_seconds": duration,
            }
        )
        tracker.log_artifact(output_path, artifact_path="evaluation")
        tracker.end("FINISHED")
        result["metrics_path"] = str(output_path)
        return result
    except BaseException:
        tracker.log_metrics({"evaluation/duration_seconds": time.perf_counter() - started_at})
        tracker.end("FAILED")
        raise


def main() -> None:
    """Evalúa un checkpoint usando validación y test en sus roles definidos."""
    parser = argparse.ArgumentParser(description="Evaluación sobre AnuraSet")
    parser.add_argument("--config", required=True, help="Ruta de la configuración YAML")
    parser.add_argument("--checkpoint", help="Checkpoint; por defecto usa best.pt")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), help="Sobrescribe el dispositivo"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config = copy.deepcopy(config)
        config["training"]["device"] = args.device
    result = evaluate_experiment(config, args.checkpoint)
    macro = result["test"]["macro"]
    print(
        f"Evaluación completada: macro-F1={macro['f1']:.6f}, "
        f"mAP={macro['map']:.6f}. Métricas: {result['metrics_path']}"
    )


if __name__ == "__main__":
    main()
