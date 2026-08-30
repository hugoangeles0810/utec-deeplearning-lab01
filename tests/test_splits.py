from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import yaml

from anuraset_dl.splits import (
    PreparationPolicy,
    PreparedData,
    build_manifests,
    build_report,
    optimize_assignments,
    prepare_data,
    recording_id,
    recording_site,
    serialize_artifacts,
    sha256_file,
    validate_existing_artifacts,
    validate_experiment_configs,
    validate_manifests,
    write_artifacts_atomically,
)


def test_recording_id_removes_segment_interval() -> None:
    filename = "INCT20955_20190909_050000_0_3.wav"
    assert recording_id(filename) == "INCT20955_20190909_050000"


def test_recording_id_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        recording_id("audio.wav")


def test_recording_id_preserves_additional_component() -> None:
    filename = "INCT20955_20191123_041500_000_0_3.wav"
    assert recording_id(filename) == "INCT20955_20191123_041500_000"
    assert recording_site(recording_id(filename)) == "20955"


def test_experiment_config_validation_ignores_provisioning_config(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    baseline = {
        "experiment": "baseline",
        "data": {
            "metadata": "dataset/train.csv",
            "splits": {"train": "splits/train.csv", "validation": "splits/validation.csv"},
            "num_labels": 1,
            "excluded_labels": [],
            "zero_label_policy": "keep_as_all_negative",
        },
    }
    baseline_path = config_root / "baseline.yaml"
    baseline_path.write_text(yaml.safe_dump(baseline), encoding="utf-8")
    (config_root / "dataset.yaml").write_text(
        yaml.safe_dump({"schema_version": 1, "source": {"provider": "google_drive"}}),
        encoding="utf-8",
    )

    validate_experiment_configs(baseline_path, baseline)


def test_experiment_config_validation_rejects_missing_data(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    baseline = {"experiment": "baseline", "data": {}}
    baseline_path = config_root / "baseline.yaml"
    baseline_path.write_text(yaml.safe_dump(baseline), encoding="utf-8")
    (config_root / "invalid.yaml").write_text("experiment: invalid\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sección data válida"):
        validate_experiment_configs(baseline_path, baseline)


def _policy(tmp_path: Path, metadata_hash: str = "0" * 64) -> PreparationPolicy:
    return PreparationPolicy(
        seed=42,
        split_names=("train", "validation"),
        proportions=(Fraction("0.8"), Fraction("0.2")),
        minimum_positive_recordings=(8, 2),
        exploratory_labels=("E",),
        exploratory_minimum_positive_recordings=(1, 1),
        expected_metadata_sha256=metadata_hash,
        manifest_paths=(
            tmp_path / "train.csv",
            tmp_path / "validation.csv",
        ),
        report_path=tmp_path / "report.json",
    )


def _prepared_data() -> PreparedData:
    identifiers = [f"INCT4_20200101_{index:06d}" for index in range(12)]
    recordings = pd.DataFrame(
        {
            "site": ["4"] * 12,
            "clip_count": [1] * 12,
            "active_all_negative_clips": [0] * 12,
            "out_of_scope_foreground_clips": [0] * 12,
            "A": [1] * 10 + [0, 0],
            "E": [1, 1, 1] + [0] * 9,
        },
        index=identifiers,
    )
    clips = pd.DataFrame(
        {
            "filename": [f"{identifier}_0_3.wav" for identifier in identifiers],
            "recording_id": identifiers,
            "site": ["4"] * 12,
            "A": recordings["A"].to_numpy(),
            "E": recordings["E"].to_numpy(),
            "active_all_negative": [False] * 12,
            "out_of_scope_foreground": [False] * 12,
        }
    )
    return PreparedData(
        clips=clips,
        recordings=recordings,
        label_columns=("A", "E"),
        active_labels=("A",),
        excluded_labels=("E",),
        exploratory_labels=("E",),
        metadata_sha256="0" * 64,
    )


def test_optimizer_respects_grouping_coverage_and_is_reproducible(tmp_path: Path) -> None:
    data = _prepared_data()
    policy = _policy(tmp_path)

    first = optimize_assignments(data, policy)
    second = optimize_assignments(data, policy)
    manifests = build_manifests(data, first, policy)

    assert first.assignments == second.assignments
    assert first.objectives["exploratory_coverage"] == 2
    assert {len(frame) for frame in manifests.values()} == {2, 10}
    for frame in manifests.values():
        assert list(frame.columns) == ["filename", "recording_id"]


def test_optimizer_rejects_insufficient_main_label_support(tmp_path: Path) -> None:
    data = _prepared_data()
    data.recordings.loc[data.recordings.index[9], "A"] = 0
    with pytest.raises(ValueError, match="cobertura mínima total"):
        optimize_assignments(data, _policy(tmp_path))


def test_manifest_validation_rejects_incorrect_recording_id(tmp_path: Path) -> None:
    data = _prepared_data()
    policy = _policy(tmp_path)
    result = optimize_assignments(data, policy)
    manifests = build_manifests(data, result, policy)
    target = next(frame for frame in manifests.values() if not frame.empty)
    target.loc[0, "recording_id"] = "INCT4_incorrecta"

    with pytest.raises(ValueError, match="recording_id incorrecto"):
        validate_manifests(manifests, data, policy)


def test_existing_artifacts_are_verified_without_regenerating_splits(tmp_path: Path) -> None:
    data = _prepared_data()
    policy = _policy(tmp_path)
    result = optimize_assignments(data, policy)
    manifests = build_manifests(data, result, policy)
    report = build_report(data, manifests, result, policy, "config-hash")
    artifacts = serialize_artifacts(manifests, report, policy)
    write_artifacts_atomically(artifacts)

    verified = validate_existing_artifacts(data, policy)

    assert {name: len(frame) for name, frame in verified.items()} == {
        name: len(frame) for name, frame in manifests.items()
    }


def test_prepare_data_validates_audio_and_binary_labels(tmp_path: Path) -> None:
    audio_dir = tmp_path / "dataset" / "train"
    audio_dir.mkdir(parents=True)
    filenames = ["INCT4_20200101_000000_0_3.wav", "INCT4_20200101_001000_0_3.wav"]
    metadata = pd.DataFrame({"filename": filenames, "A": [1, 0], "E": [0, 1]})
    metadata_path = tmp_path / "dataset" / "train.csv"
    metadata.to_csv(metadata_path, index=False)
    for filename in filenames:
        sf.write(audio_dir / filename, np.zeros(24_000, dtype=np.float32), 8_000, subtype="PCM_16")

    config = {
        "data": {
            "sample_rate": 8_000,
            "clip_seconds": 3,
            "num_labels": 1,
            "excluded_labels": ["E"],
        }
    }
    policy = _policy(tmp_path, sha256_file(metadata_path))
    data = prepare_data(metadata_path, audio_dir, config, policy)
    assert len(data.recordings) == 2

    metadata.loc[0, "A"] = 2
    metadata.to_csv(metadata_path, index=False)
    invalid_policy = _policy(tmp_path, sha256_file(metadata_path))
    with pytest.raises(ValueError, match="binarias"):
        prepare_data(metadata_path, audio_dir, config, invalid_policy)


def test_prepare_data_rejects_missing_audio(tmp_path: Path) -> None:
    audio_dir = tmp_path / "dataset" / "train"
    audio_dir.mkdir(parents=True)
    metadata_path = tmp_path / "dataset" / "train.csv"
    pd.DataFrame(
        {"filename": ["INCT4_20200101_000000_0_3.wav"], "A": [1], "E": [0]}
    ).to_csv(metadata_path, index=False)
    config = {
        "data": {
            "sample_rate": 8_000,
            "clip_seconds": 3,
            "num_labels": 1,
            "excluded_labels": ["E"],
        }
    }
    with pytest.raises(ValueError, match="faltantes=1"):
        prepare_data(
            metadata_path, audio_dir, config, _policy(tmp_path, sha256_file(metadata_path))
        )


def test_prepare_data_rejects_duplicate_filenames(tmp_path: Path) -> None:
    audio_dir = tmp_path / "dataset" / "train"
    audio_dir.mkdir(parents=True)
    metadata_path = tmp_path / "dataset" / "train.csv"
    filename = "INCT4_20200101_000000_0_3.wav"
    pd.DataFrame({"filename": [filename, filename], "A": [1, 0], "E": [0, 1]}).to_csv(
        metadata_path, index=False
    )
    config = {
        "data": {
            "sample_rate": 8_000,
            "clip_seconds": 3,
            "num_labels": 1,
            "excluded_labels": ["E"],
        }
    }
    with pytest.raises(ValueError, match="duplicados"):
        prepare_data(
            metadata_path, audio_dir, config, _policy(tmp_path, sha256_file(metadata_path))
        )


def test_prepare_data_does_not_open_audio_headers(tmp_path: Path) -> None:
    audio_dir = tmp_path / "dataset" / "train"
    audio_dir.mkdir(parents=True)
    metadata_path = tmp_path / "dataset" / "train.csv"
    filename = "INCT4_20200101_000000_0_3.wav"
    pd.DataFrame({"filename": [filename], "A": [1], "E": [0]}).to_csv(
        metadata_path, index=False
    )
    (audio_dir / filename).write_bytes(b"inventario-sin-lectura-de-cabecera")
    config = {
        "data": {
            "sample_rate": 8_000,
            "clip_seconds": 3,
            "num_labels": 1,
            "excluded_labels": ["E"],
        }
    }
    result = prepare_data(
        metadata_path, audio_dir, config, _policy(tmp_path, sha256_file(metadata_path))
    )

    assert len(result.clips) == 1


def test_artifact_writes_are_idempotent_and_protected(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    manifests = {
        split: pd.DataFrame({"filename": [f"{split}.wav"], "recording_id": [split]})
        for split in policy.split_names
    }
    artifacts = serialize_artifacts(manifests, {"schema_version": 1}, policy)
    assert write_artifacts_atomically(artifacts) == "written"
    assert write_artifacts_atomically(artifacts) == "unchanged"

    artifacts[policy.report_path] = b"{}\n"
    with pytest.raises(FileExistsError, match="--force"):
        write_artifacts_atomically(artifacts)
    assert write_artifacts_atomically(artifacts, force=True) == "written"
