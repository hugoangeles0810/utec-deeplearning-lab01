"""Ajuste y aplicación de la representación FBRS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pywt
import soundfile as sf
import torch
from torch import Tensor, nn

from anuraset_dl.data import active_labels
from anuraset_dl.runtime import sha256_file
from anuraset_dl.utils import load_config


def _pre_emphasize(waveform: np.ndarray, coefficient: float) -> np.ndarray:
    if not coefficient:
        return waveform
    emphasized = np.empty_like(waveform)
    emphasized[0] = waveform[0]
    emphasized[1:] = waveform[1:] - coefficient * waveform[:-1]
    return emphasized


def wavelet_packet_raw_energies(
    waveform: np.ndarray, wavelet: str, level: int
) -> dict[str, float]:
    """Calcula energías WPD sin normalizar en orden frecuencial explícito."""
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("La señal para WPD debe ser un vector no vacío")
    if level <= 0:
        raise ValueError("El nivel WPD debe ser positivo")
    packet = pywt.WaveletPacket(
        data=np.asarray(waveform, dtype=np.float64),
        wavelet=wavelet,
        mode="periodization",
        maxlevel=level,
    )
    energies: dict[str, float] = {"": float(np.square(waveform, dtype=np.float64).sum())}
    for depth in range(1, level + 1):
        nodes = packet.get_level(depth, order="freq")
        energies.update(
            {
                node.path: float(np.square(node.data, dtype=np.float64).sum())
                for node in nodes
            }
        )
    return energies


def wavelet_packet_energies(
    waveform: np.ndarray, wavelet: str, level: int
) -> dict[str, float]:
    """Normaliza por nivel las energías WPD usadas para ajustar el banco."""
    raw_energies = wavelet_packet_raw_energies(waveform, wavelet, level)
    energies: dict[str, float] = {"": raw_energies[""]}
    for depth in range(1, level + 1):
        paths = [path for path in raw_energies if len(path) == depth]
        raw = np.asarray([raw_energies[path] for path in paths], dtype=np.float64)
        total = float(raw.sum())
        normalized = raw / total if total > 0 else np.zeros_like(raw)
        energies.update(
            {path: float(value) for path, value in zip(paths, normalized, strict=True)}
        )
    return energies


def _frequency_ranks(wavelet: str, level: int) -> dict[str, int]:
    packet = pywt.WaveletPacket(
        data=np.zeros(2**level, dtype=np.float64),
        wavelet=wavelet,
        mode="periodization",
        maxlevel=level,
    )
    ranks: dict[str, int] = {"": 0}
    for depth in range(1, level + 1):
        ranks.update(
            {node.path: index for index, node in enumerate(packet.get_level(depth, order="freq"))}
        )
    return ranks


def select_frequency_partition(
    energies: dict[str, float], wavelet: str, level: int, target_bands: int
) -> list[tuple[str, float, float]]:
    """Fusiona primero los pares hermanos de menor energía hasta el objetivo."""
    maximum = 2**level
    if not 1 <= target_bands <= maximum:
        raise ValueError(f"target_bands debe pertenecer a [1, {maximum}]")
    ranks = _frequency_ranks(wavelet, level)
    active = {path for path in ranks if len(path) == level}
    while len(active) > target_bands:
        candidates: list[tuple[float, float, int, str]] = []
        for path in active:
            if not path or not path.endswith("a"):
                continue
            parent = path[:-1]
            if parent + "a" in active and parent + "d" in active:
                score = energies[parent + "a"] + energies[parent + "d"]
                lower_frequency = ranks[parent] / (2 ** len(parent)) if parent else 0.0
                candidates.append((score, lower_frequency, len(parent), parent))
        if not candidates:
            raise RuntimeError("La partición activa no contiene un par de hermanos fusionable")
        _, _, _, parent = min(candidates)
        active.remove(parent + "a")
        active.remove(parent + "d")
        active.add(parent)

    intervals = [
        (path, ranks[path] / (2 ** len(path)), (ranks[path] + 1) / (2 ** len(path)))
        for path in active
    ]
    intervals.sort(key=lambda item: item[1])
    validate_partition(intervals)
    return intervals


def validate_partition(intervals: list[tuple[str, float, float]]) -> None:
    """Comprueba cobertura completa, orden y ausencia de huecos o solapamientos."""
    if not intervals:
        raise ValueError("La partición FBRS no puede estar vacía")
    tolerance = 1e-12
    if abs(intervals[0][1]) > tolerance or abs(intervals[-1][2] - 1.0) > tolerance:
        raise ValueError("La partición FBRS no cubre todo el intervalo de Nyquist")
    for (_, lower, upper), (_, next_lower, _) in zip(intervals, intervals[1:], strict=False):
        if lower >= upper or abs(upper - next_lower) > tolerance:
            raise ValueError("La partición FBRS contiene desorden, huecos o solapamientos")
    if intervals[-1][1] >= intervals[-1][2]:
        raise ValueError("La última banda FBRS tiene ancho nulo")


def triangular_filterbank(
    intervals: list[tuple[str, float, float]], sample_rate: int, n_fft: int
) -> Tensor:
    """Proyecta los centros de las bandas a triángulos de altura unitaria."""
    validate_partition(intervals)
    nyquist = sample_rate / 2
    centers = torch.tensor(
        [(lower + upper) * nyquist / 2 for _, lower, upper in intervals],
        dtype=torch.float64,
    )
    frequencies = torch.linspace(0, nyquist, n_fft // 2 + 1, dtype=torch.float64)
    rows: list[Tensor] = []
    for index, center in enumerate(centers):
        lower = centers[index - 1] if index else torch.tensor(0.0, dtype=torch.float64)
        upper = (
            centers[index + 1]
            if index + 1 < len(centers)
            else torch.tensor(nyquist, dtype=torch.float64)
        )
        rising = (frequencies - lower) / (center - lower)
        falling = (upper - frequencies) / (upper - center)
        rows.append(torch.clamp(torch.minimum(rising, falling), min=0.0, max=1.0))
    filters = torch.stack(rows).to(torch.float32)
    if filters.sum(dim=1).eq(0).any():
        raise ValueError("La resolución FFT produce uno o más filtros FBRS vacíos")
    return filters


def _fit_signature(config: dict[str, Any], labels: tuple[str, ...]) -> str:
    features = config["features"]
    payload = {
        "sample_rate": int(config["data"]["sample_rate"]),
        "labels": list(labels),
        "features": {
            key: features[key]
            for key in (
                "pre_emphasis",
                "window",
                "n_fft",
                "hop_length",
                "center",
                "power",
                "log_epsilon",
                "wavelet",
                "level",
                "bank_scope",
                "fit_subset",
                "energy_normalization",
                "band_selection",
                "target_bands",
                "filter_shape",
                "filter_normalization",
            )
        },
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def bank_path(config: dict[str, Any]) -> Path:
    """Devuelve la ubicación configurada del banco congelado."""
    return Path(config["features"]["bank_path"])


def _atomic_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def fit_fbrs_bank(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    """Ajusta un banco global leyendo exclusivamente audios de entrenamiento."""
    features = config["features"]
    if features["type"] != "fbrs":
        raise ValueError("El ajuste FBRS requiere features.type=fbrs")
    destination = bank_path(config)
    if destination.exists() and not force:
        raise FileExistsError(f"El banco ya existe: {destination}; use --force para reemplazarlo")

    metadata_path = Path(config["data"]["metadata"])
    train_path = Path(config["data"]["splits"]["train"])
    metadata = pd.read_csv(metadata_path)
    labels = active_labels(metadata, config["data"]["excluded_labels"])
    if len(labels) != int(config["data"]["num_labels"]):
        raise ValueError("data.num_labels no coincide con la taxonomía activa")
    manifest = pd.read_csv(train_path)
    if "filename" not in manifest or manifest["filename"].duplicated().any():
        raise ValueError("El manifiesto de entrenamiento requiere filenames únicos")
    rows = manifest.loc[:, ["filename"]].merge(
        metadata.loc[:, ["filename", *labels]],
        on="filename",
        how="left",
        validate="one_to_one",
    )
    if rows.loc[:, labels].isna().any().any():
        raise ValueError("El manifiesto de entrenamiento contiene archivos sin metadatos")
    if features["fit_subset"] == "active_positive_training":
        rows = rows.loc[rows.loc[:, labels].sum(axis=1) > 0]
    if rows.empty:
        raise ValueError("El subconjunto de ajuste FBRS está vacío")

    wavelet = str(features["wavelet"])
    level = int(features["level"])
    aggregate: dict[str, float] = {}
    audio_dir = Path(config["data"]["root"]) / "train"
    sample_rate = int(config["data"]["sample_rate"])
    expected_samples = round(sample_rate * float(config["data"]["clip_seconds"]))
    for filename in rows["filename"]:
        waveform, observed_rate = sf.read(
            audio_dir / filename, dtype="float32", always_2d=False
        )
        if observed_rate != sample_rate or waveform.ndim != 1 or len(waveform) != expected_samples:
            raise ValueError(f"Audio incompatible durante el ajuste FBRS: {audio_dir / filename}")
        waveform = _pre_emphasize(waveform, float(features["pre_emphasis"]))
        energies = wavelet_packet_energies(waveform, wavelet, level)
        for path, energy in energies.items():
            aggregate[path] = aggregate.get(path, 0.0) + energy
    aggregate = {path: value / len(rows) for path, value in aggregate.items()}

    intervals = select_frequency_partition(
        aggregate, wavelet, level, int(features["target_bands"])
    )
    filters = triangular_filterbank(
        intervals, sample_rate=sample_rate, n_fft=int(features["n_fft"])
    )
    filenames_sha256 = hashlib.sha256(
        "\n".join(rows["filename"].astype(str)).encode("utf-8")
    ).hexdigest()
    feature_config = {
        key: features[key]
        for key in (
            "pre_emphasis",
            "window",
            "n_fft",
            "hop_length",
            "center",
            "power",
            "log_epsilon",
            "wavelet",
            "level",
            "bank_scope",
            "fit_subset",
            "energy_normalization",
            "band_selection",
            "target_bands",
            "filter_shape",
            "filter_normalization",
        )
    }
    artifact = {
        "schema_version": 1,
        "fit_signature": _fit_signature(config, labels),
        "metadata_sha256": sha256_file(metadata_path),
        "train_manifest_sha256": sha256_file(train_path),
        "fit_subset": features["fit_subset"],
        "fit_examples": int(len(rows)),
        "fit_filenames_sha256": filenames_sha256,
        "sample_rate": sample_rate,
        "wavelet": wavelet,
        "level": level,
        "target_bands": int(features["target_bands"]),
        "feature_config": feature_config,
        "intervals": [
            {
                "path": path,
                "lower_hz": lower * sample_rate / 2,
                "upper_hz": upper * sample_rate / 2,
            }
            for path, lower, upper in intervals
        ],
        "filter_normalization": features["filter_normalization"],
        "filters": filters,
        "mean_node_energies": aggregate,
    }
    _atomic_save(artifact, destination)
    return {**artifact, "path": str(destination), "sha256": sha256_file(destination)}


def load_fbrs_artifact(config: dict[str, Any]) -> dict[str, Any]:
    """Carga el banco y rechaza cambios de configuración o datos de ajuste."""
    path = bank_path(config)
    if not path.is_file():
        raise FileNotFoundError(
            f"No existe el banco FBRS {path}; ejecútese python -m anuraset_dl.fbrs --config ..."
        )
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    metadata = pd.read_csv(config["data"]["metadata"], nrows=0)
    labels = active_labels(metadata, config["data"]["excluded_labels"])
    if artifact.get("fit_signature") != _fit_signature(config, labels):
        raise ValueError("El banco FBRS no corresponde a la configuración efectiva")
    if artifact.get("metadata_sha256") != sha256_file(config["data"]["metadata"]):
        raise ValueError("Los metadatos cambiaron desde el ajuste del banco FBRS")
    if artifact.get("train_manifest_sha256") != sha256_file(config["data"]["splits"]["train"]):
        raise ValueError("El manifiesto de entrenamiento cambió desde el ajuste del banco FBRS")
    filters = artifact.get("filters")
    if not isinstance(filters, Tensor) or filters.ndim != 2:
        raise ValueError("El artefacto FBRS no contiene un banco válido")
    return artifact


class FBRSSpectrogram(nn.Module):
    """Transformación de audio a FBRS mediante un banco global congelado."""

    def __init__(self, config: dict[str, Any], artifact: dict[str, Any] | None = None) -> None:
        super().__init__()
        features = config["features"]
        if features["type"] != "fbrs":
            raise ValueError("FBRSSpectrogram requiere features.type=fbrs")
        artifact = artifact or load_fbrs_artifact(config)
        self.pre_emphasis = float(features["pre_emphasis"])
        self.n_fft = int(features["n_fft"])
        self.hop_length = int(features["hop_length"])
        self.center = bool(features["center"])
        self.power = float(features["power"])
        self.log_epsilon = float(features["log_epsilon"])
        self.register_buffer("window", torch.hamming_window(self.n_fft, periodic=True))
        self.register_buffer("filters", artifact["filters"].to(torch.float32))

    def forward(self, waveform: Tensor) -> Tensor:
        if waveform.ndim != 1:
            raise ValueError("La forma de onda debe ser un tensor unidimensional")
        if self.pre_emphasis:
            waveform = torch.cat(
                (waveform[:1], waveform[1:] - self.pre_emphasis * waveform[:-1])
            )
        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=self.center,
            return_complex=True,
        )
        power = spectrum.abs().pow(self.power)
        return torch.log((self.filters @ power).clamp_min(self.log_epsilon))


def main() -> None:
    """Ajusta y persiste el banco FBRS declarado por una configuración."""
    parser = argparse.ArgumentParser(description="Ajuste del banco FBRS sobre entrenamiento")
    parser.add_argument("--config", required=True, help="Ruta de la configuración YAML")
    parser.add_argument("--force", action="store_true", help="Reemplaza un banco existente")
    args = parser.parse_args()
    result = fit_fbrs_bank(load_config(args.config), force=args.force)
    print(
        f"Banco FBRS ajustado con {result['fit_examples']} ejemplos: "
        f"{result['path']} ({result['sha256']})"
    )


if __name__ == "__main__":
    main()
