from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from urllib.request import Request

import numpy as np
import py7zr
import pytest
import soundfile as sf

import anuraset_dl.provision_data as provisioning
from anuraset_dl.provision_data import (
    ArchiveSpec,
    AudioSpec,
    MetadataSpec,
    ProvisioningConfig,
    download_file,
    extract_archive,
    load_provisioning_config,
    provision,
    validate_archive_members,
)


class _Response(BytesIO):
    def __init__(self, content: bytes, status: int, headers: dict[str, str] | None = None) -> None:
        super().__init__(content)
        self.status = status
        self.headers = headers or {}

    def getcode(self) -> int:
        return self.status


def test_download_file_resumes_partial_download(tmp_path: Path) -> None:
    content = b"0123456789"
    destination = tmp_path / "data.bin"
    partial = tmp_path / "data.bin.part"
    partial.write_bytes(content[:4])

    def opener(request: Request) -> _Response:
        assert request.get_header("Range") == "bytes=4-"
        return _Response(content[4:], 206, {"Content-Range": "bytes 4-9/10"})

    result = download_file(
        "https://example.invalid/data.bin", destination, len(content), opener=opener
    )

    assert result.read_bytes() == content
    assert not partial.exists()


def test_download_file_restarts_when_server_ignores_range(tmp_path: Path) -> None:
    content = b"abcdefghij"
    destination = tmp_path / "data.bin"
    destination.with_name("data.bin.part").write_bytes(b"abcd")

    def opener(request: Request) -> _Response:
        assert request.get_header("Range") == "bytes=4-"
        return _Response(content, 200)

    download_file("https://example.invalid/data.bin", destination, len(content), opener=opener)

    assert destination.read_bytes() == content


@pytest.mark.parametrize("member", ["../escape.wav", "/absolute.wav", "C:\\escape.wav"])
def test_archive_member_validation_rejects_unsafe_paths(member: str) -> None:
    with pytest.raises(ValueError, match="Ruta insegura"):
        validate_archive_members([member])


@pytest.mark.parametrize("wrapped", [False, True])
def test_extract_archive_flattens_supported_layouts(tmp_path: Path, wrapped: bool) -> None:
    source = tmp_path / "source"
    source.mkdir()
    first = source / "first.wav"
    second = source / "second.wav"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    archive_path = tmp_path / "audios.7z"
    with py7zr.SevenZipFile(archive_path, "w") as archive:
        prefix = "train" if wrapped else ""
        archive.write(first, f"{prefix}/first.wav".lstrip("/"))
        archive.write(second, f"{prefix}/second.wav".lstrip("/"))

    ready = extract_archive(archive_path, tmp_path)

    assert (ready / "first.wav").read_bytes() == b"first"
    assert (ready / "second.wav").read_bytes() == b"second"
    assert (ready / ".gitkeep").is_file()


def test_extract_archive_rejects_duplicate_basenames(tmp_path: Path) -> None:
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    archive_path = tmp_path / "duplicates.7z"
    with py7zr.SevenZipFile(archive_path, "w") as archive:
        archive.write(first, "a/clip.wav")
        archive.write(second, "b/clip.wav")

    with pytest.raises(ValueError, match="duplicados"):
        extract_archive(archive_path, tmp_path)


def test_versioned_config_pins_the_audited_cut() -> None:
    config = load_provisioning_config("configs/dataset.yaml")

    assert config.metadata.sha256 == (
        "38df5d408d9bf621cc11f78ed4cd766be63a8406512c1eb4d611d98e4486d276"
    )
    assert config.train.wav_count == 62_191
    assert config.test.wav_count == 31_187


def _write_wav(path: Path) -> None:
    sf.write(path, np.zeros(16, dtype=np.float32), 8_000, subtype="PCM_16")


def _valid_config(tmp_path: Path) -> ProvisioningConfig:
    root = tmp_path / "dataset"
    train = root / "train"
    test = root / "test"
    train.mkdir(parents=True)
    test.mkdir()
    train_name = "INCT4_20200101_000000_0_3.wav"
    test_name = "external.wav"
    _write_wav(train / train_name)
    _write_wav(test / test_name)
    metadata_content = f"filename,A\n{train_name},1\n".encode()
    metadata_path = root / "train.csv"
    metadata_path.write_bytes(metadata_content)
    metadata = MetadataSpec(
        file_id="metadata",
        name="train.csv",
        size=len(metadata_content),
        sha256=hashlib.sha256(metadata_content).hexdigest(),
        rows=1,
    )
    return ProvisioningConfig(
        url_template="https://example.invalid/{file_id}",
        root=root,
        audio=AudioSpec(sample_rate=8_000, channels=1, subtype="PCM_16", frames=16),
        metadata=metadata,
        train=ArchiveSpec(
            file_id="train",
            name="train.7z",
            size=1,
            extracted_bytes=1,
            wav_count=1,
            destination="train",
        ),
        test=ArchiveSpec(
            file_id="test",
            name="test.7z",
            size=1,
            extracted_bytes=1,
            wav_count=1,
            destination="test",
        ),
    )


def test_provision_is_idempotent_for_valid_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _valid_config(tmp_path)

    def unexpected_download(*args: object, **kwargs: object) -> Path:
        raise AssertionError("No debe descargar un dataset ya válido")

    monkeypatch.setattr(provisioning, "_download", unexpected_download)

    assert provision(config) == {
        "metadata": "existente y válido",
        "train": "existente y válido",
        "test": "existente y válido",
    }


def test_provision_protects_invalid_existing_data(tmp_path: Path) -> None:
    config = _valid_config(tmp_path)
    (config.root / "train" / "unexpected.wav").write_bytes(b"invalid")

    with pytest.raises(FileExistsError, match="--force"):
        provision(config, "train")
