from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from anuraset_dl.fbrs import (
    FBRSSpectrogram,
    fit_fbrs_bank,
    load_fbrs_artifact,
    select_frequency_partition,
    triangular_filterbank,
    wavelet_packet_energies,
    wavelet_packet_raw_energies,
)


def _config(tmp_path: Path) -> dict:
    return {
        "data": {
            "root": str(tmp_path / "dataset"),
            "metadata": str(tmp_path / "dataset" / "train.csv"),
            "splits": {
                "train": str(tmp_path / "train.csv"),
                "validation": str(tmp_path / "validation.csv"),
                "test": str(tmp_path / "test.csv"),
            },
            "sample_rate": 8000,
            "clip_seconds": 0.25,
            "num_labels": 2,
            "excluded_labels": [],
            "zero_label_policy": "keep_as_all_negative",
        },
        "features": {
            "type": "fbrs",
            "pre_emphasis": 0.97,
            "window": "hamming",
            "n_fft": 128,
            "hop_length": 64,
            "center": False,
            "power": 2.0,
            "log_epsilon": 1e-10,
            "wavelet": "db4",
            "level": 3,
            "bank_scope": "corpus",
            "fit_subset": "active_positive_training",
            "energy_normalization": "per_level",
            "band_selection": "target_count",
            "target_bands": 4,
            "filter_shape": "triangular",
            "filter_normalization": "peak",
            "bank_path": str(tmp_path / "filterbanks" / "fbrs.pt"),
        },
    }


def _write_fit_fixture(tmp_path: Path, config: dict) -> None:
    audio_dir = tmp_path / "dataset" / "train"
    audio_dir.mkdir(parents=True)
    time = np.arange(2000) / 8000
    waveform = (0.25 * np.sin(2 * np.pi * 1000 * time)).astype(np.float32)
    sf.write(audio_dir / "positive.wav", waveform, 8000, subtype="PCM_16")
    pd.DataFrame(
        [
            {"filename": "positive.wav", "frog_a": 1, "frog_b": 0},
            {"filename": "negative_missing.wav", "frog_a": 0, "frog_b": 0},
            {"filename": "validation_missing.wav", "frog_a": 0, "frog_b": 1},
        ]
    ).to_csv(config["data"]["metadata"], index=False)
    pd.DataFrame(
        [
            {"filename": "positive.wav", "recording_id": "positive"},
            {"filename": "negative_missing.wav", "recording_id": "negative"},
        ]
    ).to_csv(config["data"]["splits"]["train"], index=False)
    pd.DataFrame(
        [{"filename": "validation_missing.wav", "recording_id": "validation"}]
    ).to_csv(config["data"]["splits"]["validation"], index=False)
    pd.DataFrame(columns=["filename", "recording_id"]).to_csv(
        config["data"]["splits"]["test"], index=False
    )


def test_wavelet_energies_are_normalized_per_level() -> None:
    time = np.arange(2048) / 8000
    waveform = np.sin(2 * np.pi * 1000 * time)

    energies = wavelet_packet_energies(waveform, "db4", 4)

    for level in range(1, 5):
        total = sum(value for path, value in energies.items() if len(path) == level)
        assert np.isclose(total, 1)


def test_wavelet_packet_conserves_energy() -> None:
    rng = np.random.default_rng(42)
    waveform = rng.normal(size=2048)

    energies = wavelet_packet_raw_energies(waveform, "db4", 4)

    root = energies[""]
    for level in range(1, 5):
        total = sum(value for path, value in energies.items() if len(path) == level)
        assert np.isclose(total, root, rtol=1e-10)


def test_partition_is_complete_ordered_and_respects_siblings() -> None:
    energies = {"": 1.0}
    for depth in range(1, 5):
        for index in range(2**depth):
            path = format(index, f"0{depth}b").replace("0", "a").replace("1", "d")
            energies[path] = (index + 1) / (2**depth)

    intervals = select_frequency_partition(energies, "db4", level=4, target_bands=7)

    assert len(intervals) == 7
    assert intervals[0][1] == 0
    assert intervals[-1][2] == 1
    assert all(left[2] == right[1] for left, right in zip(intervals, intervals[1:]))
    paths = {path for path, _, _ in intervals}
    assert all(not (path + "a" in paths or path + "d" in paths) for path in paths)


def test_triangular_bank_has_peak_normalization_and_no_empty_filters() -> None:
    intervals = [(f"band_{index}", index / 8, (index + 1) / 8) for index in range(8)]

    filters = triangular_filterbank(intervals, sample_rate=8000, n_fft=128)

    assert filters.shape == (8, 65)
    assert torch.all(filters.sum(dim=1) > 0)
    assert torch.allclose(filters.max(dim=1).values, torch.ones(8))


def test_fit_uses_only_active_positive_training_and_roundtrips(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_fit_fixture(tmp_path, config)

    fitted = fit_fbrs_bank(config)
    loaded = load_fbrs_artifact(config)

    assert fitted["fit_examples"] == 1
    assert fitted["fit_subset"] == "active_positive_training"
    assert len(fitted["intervals"]) == 4
    assert torch.equal(fitted["filters"], loaded["filters"])
    waveform = torch.linspace(-1, 1, 2000)
    assert torch.equal(
        FBRSSpectrogram(config, fitted)(waveform),
        FBRSSpectrogram(config, loaded)(waveform),
    )


@pytest.mark.parametrize("frequency", [500, 1000, 2000, 3000])
def test_fbrs_is_finite_deterministic_and_locates_a_tone(
    tmp_path: Path, frequency: int
) -> None:
    config = _config(tmp_path)
    _write_fit_fixture(tmp_path, config)
    artifact = fit_fbrs_bank(config)
    transform = FBRSSpectrogram(config, artifact)
    time = torch.arange(2000) / 8000
    tone = torch.sin(2 * torch.pi * frequency * time)

    first = transform(tone)
    second = transform(tone)
    dominant = int(first.exp().mean(dim=1).argmax())
    interval = artifact["intervals"][dominant]

    assert first.shape == (4, 30)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert interval["lower_hz"] <= frequency <= interval["upper_hz"]
    assert torch.isfinite(transform(torch.zeros(2000))).all()


def test_loading_rejects_a_changed_training_manifest(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_fit_fixture(tmp_path, config)
    fit_fbrs_bank(config)
    manifest = Path(config["data"]["splits"]["train"])
    frame = pd.read_csv(manifest)
    frame.iloc[::-1].to_csv(manifest, index=False)

    try:
        load_fbrs_artifact(config)
    except ValueError as error:
        assert "manifiesto de entrenamiento cambió" in str(error)
    else:
        raise AssertionError("Se esperaba rechazar un manifiesto modificado")


def test_fit_is_reproducible(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_fit_fixture(tmp_path, config)
    first = fit_fbrs_bank(config)
    second_config = deepcopy(config)
    second_config["features"]["bank_path"] = str(tmp_path / "filterbanks" / "second.pt")

    second = fit_fbrs_bank(second_config)

    assert first["intervals"] == second["intervals"]
    assert first["mean_node_energies"] == second["mean_node_energies"]
    assert torch.equal(first["filters"], second["filters"])
