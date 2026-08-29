"""Entrenamiento reproducible del baseline multietiqueta."""

from __future__ import annotations

import argparse
import copy
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
    if config["features"]["type"] != "mel" or config["model"]["name"] != "cnn":
        raise NotImplementedError("El pipeline ejecutable actual corresponde a CNN + Mel")
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

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, device, optimizer)
        validation_loss = _run_epoch(
            model, validation_loader, criterion, device, optimizer=None
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss}
        )
        checkpoint = {
            "schema_version": 1,
            "experiment": config["experiment"],
            "config_sha256": config_fingerprint(config),
            "data_fingerprints": input_fingerprints,
            "labels": list(labels),
            "model_name": config["model"]["name"],
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "validation_loss": validation_loss,
        }
        atomic_torch_save(checkpoint, last_path)
        if validation_loss < best_loss:
            best_loss = validation_loss
            atomic_torch_save(checkpoint, best_path)
        atomic_json_dump(
            {
                "experiment": config["experiment"],
                "seed": int(config["seed"]),
                "device": str(device),
                "learning_rate": learning_rate,
                "trainable_parameters": count_trainable_parameters(model),
                "best_validation_loss": best_loss,
                "epochs": history,
            },
            history_path,
        )
        print(
            f"Época {epoch:03d}/{epochs}: train_loss={train_loss:.6f}, "
            f"validation_loss={validation_loss:.6f}"
        )

    return {
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "history": str(history_path),
        "best_validation_loss": best_loss,
        "device": str(device),
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
