"""Orquestación del protocolo experimental completo en un host CUDA."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from anuraset_dl.fbrs import load_fbrs_artifact
from anuraset_dl.package_results import package_results
from anuraset_dl.runtime import atomic_json_dump, config_fingerprint, sha256_file
from anuraset_dl.utils import load_config

EXPERIMENT_CONFIGS = {
    "cnn_mel_baseline": "configs/baseline.yaml",
    "cnn_fbrs": "configs/cnn_fbrs.yaml",
    "dlognet_mel": "configs/dlognet_mel.yaml",
    "dlognet_fbrs": "configs/dlognet_fbrs.yaml",
}
DEFAULT_EXPERIMENTS = tuple(EXPERIMENT_CONFIGS)


@dataclass(frozen=True)
class CommandStage:
    """Etapa externa con nombre estable y argumentos sin shell."""

    name: str
    arguments: tuple[str, ...]


CommandRunner = Callable[[CommandStage, Path, Path], None]


def _command(*arguments: str) -> tuple[str, ...]:
    return (sys.executable, "-m", *arguments)


def build_execution_plan(
    experiments: Sequence[str], device: str, num_workers: int
) -> list[CommandStage]:
    """Deriva un plan sin duplicar dependencias compartidas."""
    unknown = set(experiments) - EXPERIMENT_CONFIGS.keys()
    if unknown:
        raise ValueError(f"Experimentos desconocidos: {', '.join(sorted(unknown))}")
    selected = list(dict.fromkeys(experiments))
    stages = [
        CommandStage(
            "provision_train",
            _command("anuraset_dl.provision_data", "--only", "train"),
        ),
        CommandStage(
            "validate_data",
            _command(
                "anuraset_dl.prepare_data",
                "--config",
                EXPERIMENT_CONFIGS["cnn_mel_baseline"],
                "--verify-existing",
            ),
        ),
    ]
    prepared: set[str] = set()
    for experiment in selected:
        config_path = EXPERIMENT_CONFIGS[experiment]
        representation = "fbrs" if experiment.endswith("fbrs") else "mel"
        if representation not in prepared:
            if representation == "fbrs":
                stages.append(
                    CommandStage(
                        "fit_fbrs_bank",
                        _command(
                            "anuraset_dl.fbrs",
                            "--config",
                            EXPERIMENT_CONFIGS["cnn_fbrs"],
                        ),
                    )
                )
            cache_config = (
                EXPERIMENT_CONFIGS["cnn_fbrs"]
                if representation == "fbrs"
                else EXPERIMENT_CONFIGS["cnn_mel_baseline"]
            )
            stages.append(
                CommandStage(
                    f"precompute_{representation}",
                    _command(
                        "anuraset_dl.precompute_features",
                        "--config",
                        cache_config,
                        "--num-workers",
                        str(num_workers),
                    ),
                )
            )
            prepared.add(representation)
        runtime_arguments = (
            "--config",
            config_path,
            "--device",
            device,
            "--num-workers",
            str(num_workers),
        )
        stages.extend(
            (
                CommandStage(
                    f"train_{experiment}",
                    _command("anuraset_dl.train", *runtime_arguments),
                ),
                CommandStage(
                    f"evaluate_{experiment}",
                    _command("anuraset_dl.evaluate", *runtime_arguments),
                ),
            )
        )
    return stages


def _training_artifacts(config: dict[str, Any], root: Path) -> tuple[Path, Path, Path, Path]:
    experiment = config["experiment"]
    checkpoint_root = root / config["training"].get("checkpoint_dir", "outputs/checkpoints")
    metrics_root = root / config["evaluation"].get("metrics_dir", "outputs/metrics")
    experiment_root = checkpoint_root / experiment
    return (
        experiment_root / "best.pt",
        experiment_root / "last.pt",
        experiment_root / "history.json",
        metrics_root / f"{experiment}.json",
    )


def experiment_status(config: dict[str, Any], root: str | Path = ".") -> str:
    """Clasifica artefactos como ausentes, entrenados, completos o inconsistentes."""
    project_root = Path(root)
    best, last, history_path, metrics_path = _training_artifacts(config, project_root)
    training_paths = (best, last, history_path)
    if not any(path.exists() for path in (*training_paths, metrics_path)):
        return "absent"
    if not all(path.is_file() for path in training_paths):
        return "inconsistent"
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
        best_checkpoint = torch.load(best, map_location="cpu", weights_only=True)
        last_checkpoint = torch.load(last, map_location="cpu", weights_only=True)
    except (EOFError, OSError, RuntimeError, ValueError, json.JSONDecodeError):
        return "inconsistent"
    expected_hash = config_fingerprint(config)
    expected_epochs = int(config["training"]["epochs"])
    if (
        history.get("experiment") != config["experiment"]
        or len(history.get("epochs", [])) != expected_epochs
        or best_checkpoint.get("config_sha256") != expected_hash
        or last_checkpoint.get("config_sha256") != expected_hash
        or last_checkpoint.get("epoch") != expected_epochs
    ):
        return "inconsistent"
    if not metrics_path.exists():
        return "trained"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "inconsistent"
    if (
        metrics.get("experiment") != config["experiment"]
        or metrics.get("config_sha256") != expected_hash
        or metrics.get("checkpoint_sha256") != sha256_file(best)
    ):
        return "inconsistent"
    return "complete"


def _run_stage(stage: CommandStage, root: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            stage.arguments,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, stage.arguments)


def _write_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at_utc"] = datetime.now(UTC).isoformat()
    atomic_json_dump(state, path)


def _upload_results(bundle: dict[str, Any], destination: str, root: Path) -> None:
    if not shutil.which("rclone"):
        raise FileNotFoundError("No se encontró rclone para exportar a Google Drive")
    for key in ("archive", "checksum"):
        subprocess.run(
            ["rclone", "copy", bundle[key], destination, "--progress"],
            cwd=root,
            check=True,
        )


def run_pipeline(
    experiments: Sequence[str] = DEFAULT_EXPERIMENTS,
    *,
    root: str | Path = ".",
    device: str = "cuda",
    num_workers: int = 4,
    export_dir: str | Path = "outputs/exports",
    package: bool = True,
    drive_destination: str | None = None,
    command_runner: CommandRunner = _run_stage,
) -> dict[str, Any]:
    """Ejecuta dependencias, experimentos y exportación sin sobrescribir estados parciales."""
    if num_workers < 0:
        raise ValueError("num_workers no puede ser negativo")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("El flujo Runpod requiere una instalación de PyTorch con CUDA activa")
    project_root = Path(root).resolve()
    selected = list(dict.fromkeys(experiments))
    plan = build_execution_plan(selected, device, num_workers)
    state_root = project_root / "outputs/runpod"
    logs_root = state_root / "logs"
    state_path = state_root / "pipeline.json"
    state: dict[str, Any] = {
        "schema_version": 1,
        "experiments": selected,
        "device": device,
        "num_workers": num_workers,
        "status": "running",
        "stages": [],
    }
    _write_state(state_path, state)

    try:
        for stage in plan:
            experiment = next(
                (name for name in selected if stage.name.endswith(name)), None
            )
            if experiment:
                config = load_config(project_root / EXPERIMENT_CONFIGS[experiment])
                status = experiment_status(config, project_root)
                if status == "inconsistent":
                    raise RuntimeError(
                        f"{experiment} contiene artefactos parciales o incompatibles; "
                        "muévalos antes de reiniciar el experimento"
                    )
                if status == "complete":
                    state["stages"].append({"name": stage.name, "status": "skipped_complete"})
                    _write_state(state_path, state)
                    continue
                if stage.name.startswith("train_") and status == "trained":
                    state["stages"].append({"name": stage.name, "status": "skipped_trained"})
                    _write_state(state_path, state)
                    continue

            if stage.name == "fit_fbrs_bank":
                config = load_config(project_root / EXPERIMENT_CONFIGS["cnn_fbrs"])
                bank = project_root / config["features"]["bank_path"]
                if bank.exists():
                    current_directory = Path.cwd()
                    try:
                        os.chdir(project_root)
                        load_fbrs_artifact(config)
                    finally:
                        os.chdir(current_directory)
                    state["stages"].append(
                        {"name": stage.name, "status": "skipped_valid"}
                    )
                    _write_state(state_path, state)
                    continue

            started = time.perf_counter()
            entry = {"name": stage.name, "status": "running"}
            state["stages"].append(entry)
            _write_state(state_path, state)
            command_runner(stage, project_root, logs_root / f"{stage.name}.log")
            entry["status"] = "completed"
            entry["duration_seconds"] = time.perf_counter() - started
            _write_state(state_path, state)

        state["status"] = "completed"
        state["drive_destination"] = drive_destination
        _write_state(state_path, state)
        bundle = package_results(project_root, export_dir) if package else None
        if bundle and drive_destination:
            _upload_results(bundle, drive_destination, project_root)
        state["bundle"] = bundle
        _write_state(state_path, state)
        return state
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        _write_state(state_path, state)
        raise


def main() -> None:
    """Expone el flujo completo y la selección de experimentos."""
    parser = argparse.ArgumentParser(description="Ejecuta experimentos AnuraSet en Runpod")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=tuple(EXPERIMENT_CONFIGS),
        default=list(DEFAULT_EXPERIMENTS),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--root", default=".")
    parser.add_argument("--export-dir", default="outputs/exports")
    parser.add_argument("--no-package", action="store_true")
    parser.add_argument("--drive-destination")
    args = parser.parse_args()
    result = run_pipeline(
        args.experiments,
        root=args.root,
        device=args.device,
        num_workers=args.num_workers,
        export_dir=args.export_dir,
        package=not args.no_package,
        drive_destination=args.drive_destination,
    )
    print(f"Pipeline completado: {len(result['experiments'])} experimento(s)")


if __name__ == "__main__":
    main()
