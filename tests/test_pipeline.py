import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from anuraset_dl.evaluate import evaluate_experiment
from anuraset_dl.runtime import config_fingerprint, sha256_file
from anuraset_dl.train import train_experiment


def _synthetic_experiment(tmp_path: Path) -> dict:
    dataset_root = tmp_path / "dataset"
    audio_dir = dataset_root / "train"
    audio_dir.mkdir(parents=True)
    rows = []
    split_rows = {"train": [], "validation": [], "test": []}
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
        },
        "evaluation": {
            "threshold_strategy": "per_class_validation_f1",
            "zero_positive_class_policy": "error",
            "metrics": ["precision", "recall", "f1", "map"],
            "metrics_dir": str(tmp_path / "metrics"),
        },
    }


def test_training_and_evaluation_complete_end_to_end(tmp_path: Path) -> None:
    config = _synthetic_experiment(tmp_path)

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
    assert set(evaluation["test"]) == {"macro", "per_class"}


def test_runtime_options_do_not_change_semantic_fingerprint(tmp_path: Path) -> None:
    config = _synthetic_experiment(tmp_path)
    changed = copy.deepcopy(config)
    changed["training"]["device"] = "auto"
    changed["training"]["num_workers"] = 3
    changed["training"]["checkpoint_dir"] = str(tmp_path / "other-checkpoints")
    changed["evaluation"]["metrics_dir"] = str(tmp_path / "other-metrics")

    assert config_fingerprint(changed) == config_fingerprint(config)

    changed["training"]["learning_rate"] = 0.01
    assert config_fingerprint(changed) != config_fingerprint(config)


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
