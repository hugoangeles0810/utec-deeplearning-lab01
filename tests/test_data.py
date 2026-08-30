from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from anuraset_dl.data import AnuraDataset, LogMelSpectrogram, mel_filterbank
from anuraset_dl.precompute_features import precompute_feature_cache


def _config(tmp_path: Path) -> dict:
    return {
        "data": {
            "root": str(tmp_path / "dataset"),
            "metadata": str(tmp_path / "dataset" / "train.csv"),
            "splits": {"train": str(tmp_path / "train.csv")},
            "sample_rate": 8000,
            "clip_seconds": 0.25,
            "num_labels": 2,
            "excluded_labels": [],
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
    }


def test_log_mel_is_deterministic_and_finite(tmp_path: Path) -> None:
    config = _config(tmp_path)
    transform = LogMelSpectrogram(config)
    waveform = torch.zeros(2000)

    first = transform(waveform)
    second = transform(waveform)

    assert first.shape == (16, 30)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()


def test_mel_filterbank_rejects_empty_filters() -> None:
    try:
        mel_filterbank(8000, 16, 128, 0, 4000)
    except ValueError as error:
        assert "vacíos" in str(error)
    else:
        raise AssertionError("Se esperaba detectar filtros vacíos")


def test_dataset_joins_manifest_metadata_and_audio(tmp_path: Path) -> None:
    config = _config(tmp_path)
    audio_dir = tmp_path / "dataset" / "train"
    audio_dir.mkdir(parents=True)
    waveform = np.sin(2 * np.pi * 440 * np.arange(2000) / 8000).astype(np.float32)
    sf.write(audio_dir / "clip.wav", waveform, 8000, subtype="PCM_16")
    pd.DataFrame([{"filename": "clip.wav", "frog_a": 1, "frog_b": 0}]).to_csv(
        config["data"]["metadata"], index=False
    )
    pd.DataFrame([{"filename": "clip.wav", "recording_id": "recording"}]).to_csv(
        config["data"]["splits"]["train"], index=False
    )

    dataset = AnuraDataset(config, "train")
    features, targets = dataset[0]

    assert dataset.labels == ("frog_a", "frog_b")
    assert features.shape == (1, 16, 30)
    assert targets.tolist() == [1.0, 0.0]


def test_dataset_rejects_wrong_label_count(tmp_path: Path) -> None:
    config = _config(tmp_path)
    (tmp_path / "dataset").mkdir()
    pd.DataFrame([{"filename": "clip.wav", "frog_a": 1, "frog_b": 0}]).to_csv(
        config["data"]["metadata"], index=False
    )
    pd.DataFrame([{"filename": "clip.wav"}]).to_csv(
        config["data"]["splits"]["train"], index=False
    )
    invalid = deepcopy(config)
    invalid["data"]["num_labels"] = 3

    try:
        AnuraDataset(invalid, "train")
    except ValueError as error:
        assert "num_labels" in str(error)
    else:
        raise AssertionError("Se esperaba rechazar una taxonomía inconsistente")


def test_feature_cache_preserves_exact_float32_features(tmp_path: Path) -> None:
    config = _config(tmp_path)
    audio_dir = tmp_path / "dataset" / "train"
    audio_dir.mkdir(parents=True)
    waveform = np.sin(2 * np.pi * 440 * np.arange(2000) / 8000).astype(np.float32)
    sf.write(audio_dir / "clip.wav", waveform, 8000, subtype="PCM_16")
    pd.DataFrame([{"filename": "clip.wav", "frog_a": 1, "frog_b": 0}]).to_csv(
        config["data"]["metadata"], index=False
    )
    pd.DataFrame([{"filename": "clip.wav", "recording_id": "recording"}]).to_csv(
        config["data"]["splits"]["train"], index=False
    )
    direct_features, direct_targets = AnuraDataset(config, "train")[0]
    config["features"]["cache"] = {
        "enabled": True,
        "root": str(tmp_path / "features"),
        "dtype": "float32",
    }

    created = precompute_feature_cache(config, num_workers=1)
    cached_features, cached_targets = AnuraDataset(config, "train")[0]
    reused = precompute_feature_cache(config)

    assert created["created"] is True
    assert reused["created"] is False
    assert torch.equal(cached_features, direct_features)
    assert torch.equal(cached_targets, direct_targets)
    assert cached_features.dtype == torch.float32
    assert Path(created["path"], "manifest.json").is_file()


def test_feature_cache_rejects_changed_representation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    audio_dir = tmp_path / "dataset" / "train"
    audio_dir.mkdir(parents=True)
    sf.write(audio_dir / "clip.wav", np.zeros(2000, dtype=np.float32), 8000)
    pd.DataFrame([{"filename": "clip.wav", "frog_a": 1, "frog_b": 0}]).to_csv(
        config["data"]["metadata"], index=False
    )
    pd.DataFrame([{"filename": "clip.wav", "recording_id": "recording"}]).to_csv(
        config["data"]["splits"]["train"], index=False
    )
    config["features"]["cache"] = {
        "enabled": True,
        "root": str(tmp_path / "features"),
        "dtype": "float32",
    }
    precompute_feature_cache(config)
    changed = deepcopy(config)
    changed["features"]["hop_length"] = 32

    try:
        AnuraDataset(changed, "train")
    except FileNotFoundError as error:
        assert "precompute_features" in str(error)
    else:
        raise AssertionError("Se esperaba rechazar una caché de otra representación")
