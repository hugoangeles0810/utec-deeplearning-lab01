"""Entrenamiento reproducible de experimentos multietiqueta."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from anuraset_dl.models import build_model, count_trainable_parameters
from anuraset_dl.runtime import (
    atomic_json_dump,
    atomic_torch_save,
    build_loader,
    config_fingerprint,
    data_fingerprints,
    resolve_device,
    set_reproducibility,
)
from anuraset_dl.tracking import MlflowTracker
from anuraset_dl.utils import load_config


def _run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(inputs)
            total_examples += len(inputs)
    if total_examples == 0:
        raise ValueError("No se puede entrenar o validar con una partición vacía")
    return total_loss / total_examples


def train_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Entrena el experimento y persiste checkpoints ``best`` y ``last``."""
    set_reproducibility(int(config["seed"]))
    device = resolve_device(str(config["training"].get("device", "auto")))
    train_loader, labels = build_loader(config, "train", shuffle=True)
    validation_loader, validation_labels = build_loader(config, "validation")
    if labels != validation_labels:
        raise ValueError("Train y validation no comparten el mismo orden de etiquetas")

    model = build_model(config).to(device)
    criterion = nn.BCEWithLogitsLoss()
    learning_rate = float(config["training"]["learning_rate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    output_dir = Path(config["training"].get("checkpoint_dir", "outputs/checkpoints"))
    experiment_dir = output_dir / config["experiment"]
    best_path = experiment_dir / "best.pt"
    last_path = experiment_dir / "last.pt"
    history_path = experiment_dir / "history.json"
    best_loss = float("inf")
    history: list[dict[str, float | int]] = []
    epochs = int(config["training"]["epochs"])
    input_fingerprints = data_fingerprints(config)
    semantic_fingerprint = config_fingerprint(config)
    tracker = MlflowTracker.start(config, phase="training")
    tracker.log_config(config)
    tracker.log_dict(input_fingerprints, "inputs/data_fingerprints.json")
    tracker.set_tags({"anuraset.config_sha256": semantic_fingerprint})
    started_at = time.perf_counter()

    try:
        for epoch in range(1, epochs + 1):
            epoch_started_at = time.perf_counter()
            train_loss = _run_epoch(model, train_loader, criterion, device, optimizer)
            validation_loss = _run_epoch(
                model, validation_loader, criterion, device, optimizer=None
            )
            epoch_duration = time.perf_counter() - epoch_started_at
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "duration_seconds": epoch_duration,
                }
            )
            checkpoint = {
                "schema_version": 2,
                "experiment": config["experiment"],
                "config_sha256": semantic_fingerprint,
                "data_fingerprints": input_fingerprints,
                "labels": list(labels),
                "model_name": config["model"]["name"],
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "validation_loss": validation_loss,
                "tracking_run_id": tracker.run_id,
            }
            atomic_torch_save(checkpoint, last_path)
            if validation_loss < best_loss:
                best_loss = validation_loss
                atomic_torch_save(checkpoint, best_path)
            total_duration = time.perf_counter() - started_at
            atomic_json_dump(
                {
                    "experiment": config["experiment"],
                    "seed": int(config["seed"]),
                    "device": str(device),
                    "learning_rate": learning_rate,
                    "trainable_parameters": count_trainable_parameters(model),
                    "best_validation_loss": best_loss,
                    "duration_seconds": total_duration,
                    "tracking_run_id": tracker.run_id,
                    "epochs": history,
                },
                history_path,
            )
            tracker.log_metrics(
                {
                    "training/loss": train_loss,
                    "validation/loss": validation_loss,
                    "training/epoch_duration_seconds": epoch_duration,
                },
                step=epoch,
            )
            print(
                f"Época {epoch:03d}/{epochs}: train_loss={train_loss:.6f}, "
                f"validation_loss={validation_loss:.6f}"
            )

        total_duration = time.perf_counter() - started_at
        tracker.log_metrics(
            {
                "training/best_validation_loss": best_loss,
                "training/duration_seconds": total_duration,
            }
        )
        tracker.set_tags({"anuraset.training_recorded": "true"})
        tracker.log_artifact(history_path, artifact_path="training")
        tracker.end("FINISHED")
    except BaseException:
        tracker.log_metrics({"training/duration_seconds": time.perf_counter() - started_at})
        tracker.end("FAILED")
        raise

    return {
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "history": str(history_path),
        "best_validation_loss": best_loss,
        "device": str(device),
        "tracking_run_id": tracker.run_id,
    }


def main() -> None:
    """Entrena un experimento descrito por YAML."""
    parser = argparse.ArgumentParser(description="Entrenamiento sobre AnuraSet")
    parser.add_argument("--config", required=True, help="Ruta de la configuración YAML")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), help="Sobrescribe el dispositivo"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config = copy.deepcopy(config)
        config["training"]["device"] = args.device
    result = train_experiment(config)
    print(f"Entrenamiento completado. Mejor checkpoint: {result['best_checkpoint']}")


if __name__ == "__main__":
    main()
