import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from anuraset_dl.evaluate import evaluate_experiment
from anuraset_dl.fbrs import fit_fbrs_bank
from anuraset_dl.precompute_features import precompute_feature_cache
from anuraset_dl.predict import predict_unlabeled
from anuraset_dl.runtime import config_fingerprint, sha256_file
from anuraset_dl.train import train_experiment


def _synthetic_experiment(tmp_path: Path) -> dict:
    dataset_root = tmp_path / "dataset"
    audio_dir = dataset_root / "train"
    audio_dir.mkdir(parents=True)
    rows = []
    split_rows = {"train": [], "validation": []}
    sample_rate = 8000
    samples = 2000
    time = np.arange(samples) / sample_rate
    for split_index, split in enumerate(split_rows):
        for index in range(4):
            label = index % 2
            frequency = 440 if label == 0 else 1200
            filename = f"{split}_{index}.wav"
            waveform = (0.25 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)
            sf.write(audio_dir / filename, waveform, sample_rate, subtype="PCM_16")
            rows.append(
                {"filename": filename, "frog_a": int(label == 0), "frog_b": int(label == 1)}
            )
            split_rows[split].append(
                {"filename": filename, "recording_id": f"recording_{split_index}_{index}"}
            )
    metadata_path = dataset_root / "train.csv"
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    split_paths = {}
    for split, entries in split_rows.items():
        path = tmp_path / f"{split}.csv"
        pd.DataFrame(entries).to_csv(path, index=False)
        split_paths[split] = str(path)

    return {
        "experiment": "synthetic_cnn_mel",
        "seed": 7,
        "data": {
            "root": str(dataset_root),
            "metadata": str(metadata_path),
            "splits": split_paths,
            "sample_rate": sample_rate,
            "clip_seconds": 0.25,
            "num_labels": 2,
            "excluded_labels": [],
            "zero_label_policy": "keep_as_all_negative",
        },
        "features": {
            "type": "mel",
            "pre_emphasis": 0.97,
            "window": "hamming",
            "n_fft": 128,
            "hop_length": 64,
            "center": False,
            "power": 2.0,
            "n_mels": 16,
            "f_min": 0,
            "f_max": 4000,
            "mel_scale": "htk",
            "filter_normalization": "none",
            "log_epsilon": 1e-10,
        },
        "model": {"name": "cnn", "channels": [4, 8], "dropout": 0.0},
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 0.001,
            "loss": "bce_with_logits",
            "device": "cpu",
            "num_workers": 0,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "early_stopping": {
                "enabled": False,
                "monitor": "validation_loss",
                "mode": "min",
                "patience": 5,
                "min_delta": 0.0,
                "warmup_epochs": 0,
            },
        },
        "evaluation": {
            "threshold_strategy": "per_class_validation_f1",
            "zero_positive_class_policy": "error",
            "metrics": ["precision", "recall", "f1", "map"],
            "metrics_dir": str(tmp_path / "metrics"),
        },
    }


def _select_model(config: dict, model_name: str) -> None:
    config["experiment"] = f"synthetic_{model_name}_{config['features']['type']}"
    if model_name == "dlognet":
        config["model"] = {
            "name": "dlognet",
            "channels": [4, 8],
            "kernel_size": 5,
            "initial_angles_degrees": [0, 45, 90, 135],
            "initial_sigma": 1.0,
            "minimum_sigma": 0.3,
            "classifier_hidden": 16,
            "dropout": 0.0,
        }


@pytest.mark.parametrize("model_name", ["cnn", "dlognet"])
def test_training_and_evaluation_complete_end_to_end(
    tmp_path: Path, model_name: str
) -> None:
    config = _synthetic_experiment(tmp_path)
    _select_model(config, model_name)

    training = train_experiment(config)
    evaluation = evaluate_experiment(config, training["best_checkpoint"])

    assert Path(training["best_checkpoint"]).is_file()
    assert Path(training["last_checkpoint"]).is_file()
    assert Path(training["history"]).is_file()
    assert Path(evaluation["metrics_path"]).is_file()
    assert evaluation["checkpoint_sha256"] == sha256_file(training["best_checkpoint"])
    persisted = json.loads(Path(evaluation["metrics_path"]).read_text(encoding="utf-8"))
    assert persisted["checkpoint_sha256"] == evaluation["checkpoint_sha256"]
    assert persisted["data_fingerprints"] == evaluation["data_fingerprints"]
    assert set(evaluation["thresholds"]) == {"frog_a", "frog_b"}
    assert set(evaluation["validation"]) == {"macro", "per_class"}


@pytest.mark.parametrize("model_name", ["cnn", "dlognet"])
def test_fbrs_completes_end_to_end_with_a_frozen_bank(
    tmp_path: Path, model_name: str
) -> None:
    config = _synthetic_experiment(tmp_path)
    config["features"] = {
        "type": "fbrs",
        "pre_emphasis": 0.97,
        "window": "hamming",
        "n_fft": 128,
        "hop_length": 64,
        "center": False,
        "power": 2.0,
        "log_epsilon": 1e-10,
        "wavelet": "db4",
        "level": 4,
        "bank_scope": "corpus",
        "fit_subset": "active_positive_training",
        "energy_normalization": "per_level",
        "band_selection": "target_count",
        "target_bands": 8,
        "filter_shape": "triangular",
        "filter_normalization": "peak",
        "bank_path": str(tmp_path / "filterbanks" / "fbrs.pt"),
    }
    _select_model(config, model_name)
    fitted = fit_fbrs_bank(config)
    config["features"]["cache"] = {
        "enabled": True,
        "root": str(tmp_path / "features"),
        "dtype": "float32",
    }
    cached = precompute_feature_cache(config)

    training = train_experiment(config)
    evaluation = evaluate_experiment(config, training["best_checkpoint"])

    assert fitted["fit_examples"] == 4
    assert cached["created"] is True
    assert Path(training["best_checkpoint"]).is_file()
    assert Path(evaluation["metrics_path"]).is_file()
    assert evaluation["data_fingerprints"]["feature_artifact"]["sha256"] == fitted["sha256"]


def test_runtime_options_do_not_change_semantic_fingerprint(tmp_path: Path) -> None:
    config = _synthetic_experiment(tmp_path)
    changed = copy.deepcopy(config)
    changed["training"]["device"] = "auto"
    changed["training"]["num_workers"] = 3
    changed["training"]["checkpoint_dir"] = str(tmp_path / "other-checkpoints")
    changed["evaluation"]["metrics_dir"] = str(tmp_path / "other-metrics")
    changed["features"]["cache"] = {
        "enabled": True,
        "root": str(tmp_path / "features"),
        "dtype": "float32",
    }
    changed["tracking"] = {
        "enabled": True,
        "experiment_name": "synthetic-tests",
        "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
        "artifact_root": str(tmp_path / "mlflow-artifacts"),
        "log_system_metrics": False,
    }

    assert config_fingerprint(changed) == config_fingerprint(config)

    changed["training"]["learning_rate"] = 0.01
    assert config_fingerprint(changed) != config_fingerprint(config)


def test_early_stopping_persists_a_complete_training_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _synthetic_experiment(tmp_path)
    config["training"]["epochs"] = 10
    config["training"]["early_stopping"] = {
        "enabled": True,
        "monitor": "validation_loss",
        "mode": "min",
        "patience": 2,
        "min_delta": 0.0,
        "warmup_epochs": 3,
    }
    validation_losses = iter((1.0, 0.8, 0.81, 0.82, 0.83))

    def fake_run_epoch(model, loader, criterion, device, optimizer):  # type: ignore[no-untyped-def]
        return 0.5 if optimizer is not None else next(validation_losses)

    monkeypatch.setattr("anuraset_dl.train._run_epoch", fake_run_epoch)

    training = train_experiment(config)
    history = json.loads(Path(training["history"]).read_text(encoding="utf-8"))
    best = torch.load(training["best_checkpoint"], map_location="cpu", weights_only=True)
    last = torch.load(training["last_checkpoint"], map_location="cpu", weights_only=True)

    assert training["stopped_early"] is True
    assert training["completed_epochs"] == 5
    assert training["best_epoch"] == 2
    assert history["stop_reason"] == "early_stopping"
    assert history["training_completed"] is True
    assert history["epochs"][-1]["epochs_without_improvement"] == 2
    assert best["epoch"] == 2
    assert last["epoch"] == 5
    assert last["training_completed"] is True
    assert last["stopped_early"] is True


def test_mlflow_links_training_and_evaluation_in_the_same_run(tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    config = _synthetic_experiment(tmp_path)
    config["tracking"] = {
        "enabled": True,
        "experiment_name": "synthetic-tests",
        "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
        "artifact_root": str(tmp_path / "mlflow-artifacts"),
        "log_system_metrics": False,
    }

    training = train_experiment(config)
    evaluation = evaluate_experiment(config, training["best_checkpoint"])

    assert training["tracking_run_id"]
    assert evaluation["tracking_run_id"] == training["tracking_run_id"]
    client = mlflow.MlflowClient(tracking_uri=config["tracking"]["tracking_uri"])
    run = client.get_run(training["tracking_run_id"])
    assert run.info.status == "FINISHED"
    assert run.data.params["model.name"] == "cnn"
    assert "training/loss" in run.data.metrics
    assert "validation/loss" in run.data.metrics
    assert "validation/macro/f1" in run.data.metrics
    assert run.data.tags["anuraset.training_recorded"] == "true"
    assert run.data.tags["anuraset.evaluation_recorded"] == "true"
    assert [item.path for item in client.list_artifacts(run.info.run_id, "training")] == [
        "training/history.json"
    ]
    assert [item.path for item in client.list_artifacts(run.info.run_id, "evaluation")] == [
        "evaluation/synthetic_cnn_mel.json"
    ]


def test_unlabeled_prediction_uses_validation_thresholds(tmp_path: Path) -> None:
    config = _synthetic_experiment(tmp_path)
    training = train_experiment(config)
    evaluation = evaluate_experiment(config, training["best_checkpoint"])
    input_dir = tmp_path / "external-test"
    input_dir.mkdir()
    sample_rate = int(config["data"]["sample_rate"])
    samples = round(sample_rate * float(config["data"]["clip_seconds"]))
    for filename, frequency in (("b.wav", 1200), ("a.wav", 440)):
        time = np.arange(samples) / sample_rate
        waveform = (0.25 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)
        sf.write(input_dir / filename, waveform, sample_rate, subtype="PCM_16")
    output = tmp_path / "predictions.csv"

    result = predict_unlabeled(
        config,
        input_dir,
        checkpoint_path=training["best_checkpoint"],
        evaluation_path=evaluation["metrics_path"],
        output_path=output,
    )

    predictions = pd.read_csv(output)
    assert result["examples"] == 2
    assert Path(result["metadata_path"]).is_file()
    assert predictions["filename"].tolist() == ["a.wav", "b.wav"]
    assert list(predictions.columns) == [
        "filename",
        "frog_a_probability",
        "frog_a_prediction",
        "frog_b_probability",
        "frog_b_prediction",
    ]


def test_evaluation_rejects_changed_manifests(tmp_path: Path) -> None:
    config = _synthetic_experiment(tmp_path)
    training = train_experiment(config)
    train_manifest = Path(config["data"]["splits"]["train"])
    frame = pd.read_csv(train_manifest)
    frame.iloc[::-1].to_csv(train_manifest, index=False)

    try:
        evaluate_experiment(config, training["best_checkpoint"])
    except ValueError as error:
        assert "manifiestos cambiaron" in str(error)
    else:
        raise AssertionError("Se esperaba rechazar un manifiesto modificado")


def test_evaluation_rejects_changed_metadata(tmp_path: Path) -> None:
    config = _synthetic_experiment(tmp_path)
    training = train_experiment(config)
    metadata_path = Path(config["data"]["metadata"])
    frame = pd.read_csv(metadata_path)
    frame.iloc[::-1].to_csv(metadata_path, index=False)

    try:
        evaluate_experiment(config, training["best_checkpoint"])
    except ValueError as error:
        assert "metadatos" in str(error)
    else:
        raise AssertionError("Se esperaba rechazar metadata modificada")
