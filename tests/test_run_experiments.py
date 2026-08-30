from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path

import torch

from anuraset_dl.run_experiments import (
    DEFAULT_EXPERIMENTS,
    build_execution_plan,
    experiment_status,
    run_pipeline,
)
from anuraset_dl.runtime import config_fingerprint, sha256_file
from anuraset_dl.utils import load_config


def test_default_plan_runs_the_complete_factorial_matrix() -> None:
    plan = build_execution_plan(DEFAULT_EXPERIMENTS, "cuda", 4)

    assert [stage.name for stage in plan] == [
        "provision_train",
        "validate_data",
        "precompute_mel",
        "train_cnn_mel_baseline",
        "evaluate_cnn_mel_baseline",
        "fit_fbrs_bank",
        "precompute_fbrs",
        "train_cnn_fbrs",
        "evaluate_cnn_fbrs",
        "train_dlognet_mel",
        "evaluate_dlognet_mel",
        "train_dlognet_fbrs",
        "evaluate_dlognet_fbrs",
    ]
    assert plan[0].arguments[-2:] == ("--only", "train")
    training = next(stage for stage in plan if stage.name == "train_dlognet_mel")
    assert training.arguments[-4:] == ("--device", "cuda", "--num-workers", "4")


def test_individual_fbrs_plan_resolves_shared_dependencies() -> None:
    plan = build_execution_plan(["dlognet_fbrs"], "cuda", 2)

    assert [stage.name for stage in plan] == [
        "provision_train",
        "validate_data",
        "fit_fbrs_bank",
        "precompute_fbrs",
        "train_dlognet_fbrs",
        "evaluate_dlognet_fbrs",
    ]


def _artifact_config(tmp_path: Path) -> dict:
    config = deepcopy(load_config("configs/baseline.yaml"))
    config["experiment"] = "cloud_test"
    config["training"]["epochs"] = 2
    config["training"]["checkpoint_dir"] = "outputs/checkpoints"
    config["evaluation"]["metrics_dir"] = "outputs/metrics"
    return config


def _write_trained_artifacts(tmp_path: Path, config: dict) -> tuple[Path, Path]:
    output = tmp_path / "outputs/checkpoints" / config["experiment"]
    output.mkdir(parents=True)
    fingerprint = config_fingerprint(config)
    best = output / "best.pt"
    last = output / "last.pt"
    epochs = int(config["training"]["epochs"])
    torch.save({"config_sha256": fingerprint, "epoch": 1}, best)
    torch.save({"config_sha256": fingerprint, "epoch": epochs}, last)
    (output / "history.json").write_text(
        json.dumps(
            {
                "experiment": config["experiment"],
                "epochs": [{"epoch": epoch} for epoch in range(1, epochs + 1)],
            }
        ),
        encoding="utf-8",
    )
    return best, last


def test_experiment_status_distinguishes_trained_and_complete(tmp_path: Path) -> None:
    config = _artifact_config(tmp_path)
    assert experiment_status(config, tmp_path) == "absent"
    best, _ = _write_trained_artifacts(tmp_path, config)
    assert experiment_status(config, tmp_path) == "trained"

    metrics = tmp_path / "outputs/metrics/cloud_test.json"
    metrics.parent.mkdir(parents=True)
    metrics.write_text(
        json.dumps(
            {
                "experiment": "cloud_test",
                "config_sha256": config_fingerprint(config),
                "checkpoint_sha256": sha256_file(best),
            }
        ),
        encoding="utf-8",
    )
    assert experiment_status(config, tmp_path) == "complete"


def test_experiment_status_rejects_partial_training(tmp_path: Path) -> None:
    config = _artifact_config(tmp_path)
    checkpoint = tmp_path / "outputs/checkpoints/cloud_test/best.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({}, checkpoint)

    assert experiment_status(config, tmp_path) == "inconsistent"


def test_pipeline_runs_selected_experiment_and_records_state(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    shutil.copy("configs/dlognet_mel.yaml", tmp_path / "configs/dlognet_mel.yaml")
    executed = []

    def runner(stage, root, log_path):  # type: ignore[no-untyped-def]
        executed.append(stage.name)
        assert root == tmp_path
        assert log_path.parent == tmp_path / "outputs/runpod/logs"

    result = run_pipeline(
        ["dlognet_mel"],
        root=tmp_path,
        device="cpu",
        num_workers=0,
        package=False,
        command_runner=runner,
    )

    assert executed == [
        "provision_train",
        "validate_data",
        "precompute_mel",
        "train_dlognet_mel",
        "evaluate_dlognet_mel",
    ]
    assert result["status"] == "completed"
    persisted = json.loads(
        (tmp_path / "outputs/runpod/pipeline.json").read_text(encoding="utf-8")
    )
    assert persisted["status"] == "completed"


def test_pipeline_skips_a_complete_experiment(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    config_path = tmp_path / "configs/dlognet_mel.yaml"
    shutil.copy("configs/dlognet_mel.yaml", config_path)
    config = load_config(config_path)
    best, _ = _write_trained_artifacts(tmp_path, config)
    metrics = tmp_path / "outputs/metrics/dlognet_mel.json"
    metrics.parent.mkdir(parents=True)
    metrics.write_text(
        json.dumps(
            {
                "experiment": "dlognet_mel",
                "config_sha256": config_fingerprint(config),
                "checkpoint_sha256": sha256_file(best),
            }
        ),
        encoding="utf-8",
    )
    executed = []

    def runner(stage, root, log_path):  # type: ignore[no-untyped-def]
        executed.append(stage.name)

    result = run_pipeline(
        ["dlognet_mel"],
        root=tmp_path,
        device="cpu",
        num_workers=0,
        package=False,
        command_runner=runner,
    )

    assert executed == ["provision_train", "validate_data", "precompute_mel"]
    assert result["stages"][-2:] == [
        {"name": "train_dlognet_mel", "status": "skipped_complete"},
        {"name": "evaluate_dlognet_mel", "status": "skipped_complete"},
    ]
