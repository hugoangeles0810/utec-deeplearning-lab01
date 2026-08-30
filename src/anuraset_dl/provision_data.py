"""Descarga, extrae y valida el corte versionado de AnuraSet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import py7zr
import soundfile as sf
import yaml

DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
DOWNLOAD_RETRIES = 4
DISK_SAFETY_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class AudioSpec:
    """Formato esperado para todos los clips provisionados."""

    sample_rate: int
    channels: int
    subtype: str
    frames: int


@dataclass(frozen=True)
class RemoteFile:
    """Identidad inmutable de un archivo remoto."""

    file_id: str
    name: str
    size: int


@dataclass(frozen=True)
class MetadataSpec(RemoteFile):
    """Contrato del CSV de anotaciones."""

    sha256: str
    rows: int


@dataclass(frozen=True)
class ArchiveSpec(RemoteFile):
    """Contrato de un archivo de audio comprimido."""

    extracted_bytes: int
    wav_count: int
    destination: str


@dataclass(frozen=True)
class ProvisioningConfig:
    """Configuración completa del aprovisionamiento."""

    url_template: str
    root: Path
    audio: AudioSpec
    metadata: MetadataSpec
    train: ArchiveSpec
    test: ArchiveSpec


def _positive_integer(value: Any, field: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{field} debe ser un entero positivo")
    return result


def load_provisioning_config(path: str | Path) -> ProvisioningConfig:
    """Carga y valida el manifiesto versionado del dataset."""
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("Configuración de aprovisionamiento inválida o incompatible")
    if raw["source"].get("provider") != "google_drive":
        raise ValueError("Solo se admite la fuente google_drive")

    url_template = str(raw["source"]["download_url_template"])
    if "{file_id}" not in url_template:
        raise ValueError("source.download_url_template debe contener {file_id}")

    audio_raw = raw["audio"]
    audio = AudioSpec(
        sample_rate=_positive_integer(audio_raw["sample_rate"], "audio.sample_rate"),
        channels=_positive_integer(audio_raw["channels"], "audio.channels"),
        subtype=str(audio_raw["subtype"]),
        frames=_positive_integer(audio_raw["frames"], "audio.frames"),
    )
    files = raw["files"]
    metadata_raw = files["metadata"]
    metadata = MetadataSpec(
        file_id=str(metadata_raw["file_id"]),
        name=str(metadata_raw["name"]),
        size=_positive_integer(metadata_raw["size"], "files.metadata.size"),
        sha256=str(metadata_raw["sha256"]),
        rows=_positive_integer(metadata_raw["rows"], "files.metadata.rows"),
    )
    if len(metadata.sha256) != 64:
        raise ValueError("files.metadata.sha256 debe ser una huella SHA-256")

    def archive(name: str) -> ArchiveSpec:
        item = files[name]
        return ArchiveSpec(
            file_id=str(item["file_id"]),
            name=str(item["name"]),
            size=_positive_integer(item["size"], f"files.{name}.size"),
            extracted_bytes=_positive_integer(
                item["extracted_bytes"], f"files.{name}.extracted_bytes"
            ),
            wav_count=_positive_integer(item["wav_count"], f"files.{name}.wav_count"),
            destination=str(item["destination"]),
        )

    return ProvisioningConfig(
        url_template=url_template,
        root=Path(raw["destination"]["root"]),
        audio=audio,
        metadata=metadata,
        train=archive("train"),
        test=archive("test"),
    )


def sha256_file(path: Path) -> str:
    """Calcula SHA-256 sin cargar el archivo completo en memoria."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_metadata(path: Path, spec: MetadataSpec) -> set[str]:
    """Valida identidad, esquema mínimo y unicidad del CSV."""
    if not path.is_file():
        raise ValueError(f"No existe el archivo de metadatos: {path}")
    if path.stat().st_size != spec.size:
        raise ValueError(f"Tamaño inesperado para {path}: {path.stat().st_size} != {spec.size}")
    observed_hash = sha256_file(path)
    if observed_hash != spec.sha256:
        raise ValueError(f"Huella SHA-256 inesperada para {path}: {observed_hash}")

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "filename" not in reader.fieldnames:
            raise ValueError("train.csv no contiene la columna filename")
        filenames = [row["filename"] for row in reader]
    if len(filenames) != spec.rows:
        raise ValueError(f"train.csv contiene {len(filenames):,} filas; se esperaban {spec.rows:,}")
    if len(set(filenames)) != len(filenames):
        raise ValueError("train.csv contiene nombres de archivo duplicados")
    invalid_names = (
        not name or Path(name).name != name or not name.lower().endswith(".wav")
        for name in filenames
    )
    if any(invalid_names):
        raise ValueError("train.csv contiene nombres de audio inválidos")
    return set(filenames)


def _wav_inventory(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise ValueError(f"No existe el directorio de audio: {directory}")
    wavs = list(directory.glob("*.wav"))
    inventory = {path.name: path for path in wavs}
    if len(inventory) != len(wavs):
        raise ValueError(f"Existen nombres WAV duplicados en {directory}")
    return inventory


def validate_audio_headers(paths: Iterable[Path], audio: AudioSpec) -> None:
    """Comprueba el contrato técnico de una colección de WAV."""
    failures: list[str] = []
    for path in sorted(paths):
        try:
            info = sf.info(path)
        except (RuntimeError, OSError):
            failures.append(path.name)
        else:
            if (
                info.format != "WAV"
                or info.subtype != audio.subtype
                or info.channels != audio.channels
                or info.samplerate != audio.sample_rate
                or info.frames != audio.frames
            ):
                failures.append(path.name)
        if len(failures) == 10:
            break
    if failures:
        raise ValueError(f"Audios con cabecera incompatible: {', '.join(failures)}")


def validate_train_directory(
    directory: Path, expected_filenames: set[str], spec: ArchiveSpec, audio: AudioSpec
) -> None:
    """Valida la correspondencia exacta entre entrenamiento y metadatos."""
    inventory = _wav_inventory(directory)
    if len(inventory) != spec.wav_count:
        raise ValueError(
            f"{directory} contiene {len(inventory):,} WAV; se esperaban {spec.wav_count:,}"
        )
    missing = expected_filenames - inventory.keys()
    extra = inventory.keys() - expected_filenames
    if missing or extra:
        raise ValueError(
            f"Correspondencia CSV/audio inválida: faltantes={len(missing)}, extras={len(extra)}"
        )
    validate_audio_headers(inventory.values(), audio)


def validate_test_directory(directory: Path, spec: ArchiveSpec, audio: AudioSpec) -> None:
    """Valida inventario y formato del test externo."""
    inventory = _wav_inventory(directory)
    if len(inventory) != spec.wav_count:
        raise ValueError(
            f"{directory} contiene {len(inventory):,} WAV; se esperaban {spec.wav_count:,}"
        )
    validate_audio_headers(inventory.values(), audio)


def _content_range_start(value: str | None) -> int | None:
    if not value or not value.startswith("bytes ") or "/" not in value:
        return None
    interval = value.removeprefix("bytes ").split("/", maxsplit=1)[0]
    try:
        return int(interval.split("-", maxsplit=1)[0])
    except (ValueError, IndexError):
        return None


def _open_request(request: Request) -> BinaryIO:
    return urlopen(request, timeout=60)  # noqa: S310 - URL fijada en configuración versionada


def download_file(
    url: str,
    destination: Path,
    expected_size: int,
    *,
    opener: Callable[[Request], BinaryIO] = _open_request,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Descarga un archivo con reanudación y publicación atómica."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    if destination.exists():
        if destination.stat().st_size == expected_size:
            return destination
        destination.unlink()
    if partial.exists() and partial.stat().st_size > expected_size:
        raise ValueError(f"Descarga parcial mayor que el archivo esperado: {partial}")

    attempts = 0
    while (partial.stat().st_size if partial.exists() else 0) < expected_size:
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "anuraset-dl/0.1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = Request(url, headers=headers)
        try:
            response = opener(request)
            status = getattr(response, "status", response.getcode())
            mode = "ab" if offset and status == 206 else "wb"
            if offset and status == 206:
                observed_start = _content_range_start(response.headers.get("Content-Range"))
                if observed_start != offset:
                    message = (
                        f"El servidor respondió desde el byte {observed_start}; "
                        f"se esperaba {offset}"
                    )
                    raise OSError(message)
            elif offset and status != 200:
                raise OSError(f"Respuesta HTTP {status} incompatible con reanudación")
            with response, partial.open(mode) as stream:
                while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                    stream.write(chunk)
                    observed = stream.tell()
                    if observed > expected_size:
                        raise OSError("La descarga excedió el tamaño esperado")
                    if progress is not None:
                        progress(observed, expected_size)
            attempts = 0
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            attempts += 1
            if attempts > DOWNLOAD_RETRIES:
                raise RuntimeError(f"No se pudo descargar {destination.name}") from error
            time.sleep(2 ** (attempts - 1))

    observed_size = partial.stat().st_size if partial.exists() else 0
    if observed_size != expected_size:
        raise ValueError(
            f"Descarga incompleta para {destination.name}: {observed_size} != {expected_size}"
        )
    os.replace(partial, destination)
    return destination


def validate_archive_members(names: Iterable[str]) -> None:
    """Rechaza rutas capaces de escapar del directorio temporal."""
    for original in names:
        normalized = original.replace("\\", "/")
        posix = PurePosixPath(normalized)
        windows = PureWindowsPath(original)
        if (
            not normalized
            or posix.is_absolute()
            or windows.is_absolute()
            or windows.drive
            or ".." in posix.parts
        ):
            raise ValueError(f"Ruta insegura dentro del archivo 7z: {original}")


def extract_archive(archive_path: Path, root: Path) -> Path:
    """Extrae los WAV en un directorio plano preparado para validación."""
    extraction = Path(tempfile.mkdtemp(prefix=".extract-", dir=root))
    ready = Path(tempfile.mkdtemp(prefix=".ready-", dir=root))
    try:
        with py7zr.SevenZipFile(archive_path, mode="r") as archive:
            members = archive.list()
            validate_archive_members(member.filename for member in members)
            unsafe_types = [
                member.filename
                for member in members
                if member.is_symlink or not (member.is_file or member.is_directory)
            ]
            if unsafe_types:
                raise ValueError(
                    "El archivo 7z contiene miembros no regulares: "
                    + ", ".join(unsafe_types[:10])
                )
            archive.extractall(path=extraction)
        wavs = sorted(extraction.rglob("*.wav"))
        basenames = [path.name for path in wavs]
        if len(set(basenames)) != len(basenames):
            raise ValueError("El archivo 7z contiene nombres WAV duplicados")
        if not wavs:
            raise ValueError("El archivo 7z no contiene audios WAV")
        (ready / ".gitkeep").touch()
        for path in wavs:
            os.replace(path, ready / path.name)
        return ready
    except Exception:
        shutil.rmtree(ready, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(extraction, ignore_errors=True)


def _contains_user_data(path: Path) -> bool:
    return path.exists() and any(item.name != ".gitkeep" for item in path.iterdir())


def install_directory(ready: Path, destination: Path, force: bool) -> None:
    """Publica un directorio validado y revierte si el reemplazo falla."""
    if destination.exists() and _contains_user_data(destination) and not force:
        raise FileExistsError(
            f"{destination} contiene datos inválidos; use --force para reemplazarlos"
        )
    backup = destination.with_name(f".{destination.name}.backup")
    if backup.exists():
        raise FileExistsError(f"Existe un respaldo pendiente que debe revisarse: {backup}")
    moved_existing = False
    try:
        if destination.exists():
            os.replace(destination, backup)
            moved_existing = True
        os.replace(ready, destination)
    except Exception:
        if moved_existing and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    else:
        if moved_existing:
            shutil.rmtree(backup)


def _ensure_disk_space(root: Path, archive: ArchiveSpec, downloaded_bytes: int) -> None:
    required = (
        archive.extracted_bytes
        + max(0, archive.size - downloaded_bytes)
        + DISK_SAFETY_BYTES
    )
    available = shutil.disk_usage(root).free
    if available < required:
        raise OSError(
            f"Espacio insuficiente para {archive.name}: disponibles={available:,}, "
            f"requeridos≈{required:,} bytes"
        )


def _print_progress(name: str) -> Callable[[int, int], None]:
    last_print = 0.0

    def report(current: int, total: int) -> None:
        nonlocal last_print
        now = time.monotonic()
        if now - last_print >= 5 or current == total:
            percentage = current / total * 100
            print(f"\rDescargando {name}: {percentage:6.2f}%", end="", flush=True)
            last_print = now
            if current == total:
                print()

    return report


def _download(config: ProvisioningConfig, file: RemoteFile, downloads: Path) -> Path:
    url = config.url_template.format(file_id=file.file_id)
    return download_file(
        url,
        downloads / file.name,
        file.size,
        progress=_print_progress(file.name),
    )


def _provision_archive(
    config: ProvisioningConfig,
    spec: ArchiveSpec,
    expected_filenames: set[str] | None,
    *,
    force: bool,
    keep_archive: bool,
) -> str:
    destination = config.root / spec.destination
    if expected_filenames is None:
        def validate_destination() -> None:
            validate_test_directory(destination, spec, config.audio)
    else:
        def validate_destination() -> None:
            validate_train_directory(destination, expected_filenames, spec, config.audio)
    try:
        validate_destination()
    except ValueError:
        if _contains_user_data(destination) and not force:
            raise FileExistsError(
                f"{destination} contiene datos incompletos o incompatibles; use --force"
            ) from None
    else:
        archive_path = config.root / ".downloads" / spec.name
        if archive_path.exists() and not keep_archive:
            archive_path.unlink()
        return "existente y válido"

    downloads = config.root / ".downloads"
    completed = downloads / spec.name
    partial = downloads / f"{spec.name}.part"
    downloaded_bytes = completed.stat().st_size if completed.exists() else 0
    if not downloaded_bytes and partial.exists():
        downloaded_bytes = partial.stat().st_size
    _ensure_disk_space(config.root, spec, downloaded_bytes)
    archive_path = _download(config, spec, downloads)
    print(f"Extrayendo {spec.name}...")
    try:
        ready = extract_archive(archive_path, config.root)
    except py7zr.exceptions.ArchiveError:
        archive_path.unlink(missing_ok=True)
        raise
    try:
        if expected_filenames is None:
            validate_test_directory(ready, spec, config.audio)
        else:
            validate_train_directory(ready, expected_filenames, spec, config.audio)
        install_directory(ready, destination, force)
    finally:
        if ready.exists():
            shutil.rmtree(ready)
    if not keep_archive:
        archive_path.unlink()
    return "descargado, extraído y validado"


def provision(
    config: ProvisioningConfig,
    selection: str = "all",
    *,
    force: bool = False,
    keep_archives: bool = False,
) -> dict[str, str]:
    """Aprovisiona uno o ambos subconjuntos de forma idempotente."""
    config.root.mkdir(parents=True, exist_ok=True)
    outcomes: dict[str, str] = {}
    expected_filenames: set[str] | None = None
    if selection in {"all", "train"}:
        metadata_path = config.root / config.metadata.name
        try:
            expected_filenames = validate_metadata(metadata_path, config.metadata)
        except ValueError:
            if metadata_path.exists() and not force:
                raise FileExistsError(
                    f"{metadata_path} no coincide con el corte esperado; use --force"
                ) from None
            downloaded = _download(config, config.metadata, config.root / ".downloads")
            validate_metadata(downloaded, config.metadata)
            os.replace(downloaded, metadata_path)
            expected_filenames = validate_metadata(metadata_path, config.metadata)
            outcomes["metadata"] = "descargado y validado"
        else:
            outcomes["metadata"] = "existente y válido"
        outcomes["train"] = _provision_archive(
            config,
            config.train,
            expected_filenames,
            force=force,
            keep_archive=keep_archives,
        )
    if selection in {"all", "test"}:
        outcomes["test"] = _provision_archive(
            config,
            config.test,
            None,
            force=force,
            keep_archive=keep_archives,
        )
    return outcomes


def main() -> None:
    """Punto de entrada del aprovisionamiento reproducible."""
    parser = argparse.ArgumentParser(description="Aprovisiona el corte versionado de AnuraSet")
    parser.add_argument("--config", default="configs/dataset.yaml", help="Manifiesto del dataset")
    parser.add_argument("--dataset-root", help="Sobrescribe destination.root")
    parser.add_argument("--only", choices=("all", "train", "test"), default="all")
    parser.add_argument("--keep-archives", action="store_true", help="Conserva los archivos .7z")
    parser.add_argument(
        "--force", action="store_true", help="Reemplaza destinos existentes que sean inválidos"
    )
    args = parser.parse_args()

    config = load_provisioning_config(args.config)
    if args.dataset_root:
        config = ProvisioningConfig(
            url_template=config.url_template,
            root=Path(args.dataset_root),
            audio=config.audio,
            metadata=config.metadata,
            train=config.train,
            test=config.test,
        )
    try:
        outcomes = provision(
            config,
            args.only,
            force=args.force,
            keep_archives=args.keep_archives,
        )
    except (
        FileExistsError,
        OSError,
        RuntimeError,
        ValueError,
        py7zr.exceptions.ArchiveError,
        py7zr.exceptions.PasswordRequired,
    ) as error:
        print(f"Error de aprovisionamiento: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    for name, outcome in outcomes.items():
        print(f"{name}: {outcome}")
    print(f"Dataset disponible en {config.root}")


if __name__ == "__main__":
    main()
