"""Lectura de audio, metadatos y representaciones tiempo-frecuencia."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset


def active_labels(metadata: pd.DataFrame, excluded_labels: list[str]) -> tuple[str, ...]:
    """Deriva la taxonomía activa conservando el orden de los metadatos."""
    if "filename" not in metadata:
        raise ValueError("Los metadatos no contienen la columna filename")
    excluded = set(excluded_labels)
    unknown = sorted(excluded - set(metadata.columns))
    if unknown:
        raise ValueError(f"Etiquetas excluidas inexistentes: {', '.join(unknown)}")
    return tuple(
        column
        for column in metadata.columns
        if column != "filename" and column not in excluded
    )


def _validate_metadata(metadata: pd.DataFrame, labels: tuple[str, ...]) -> None:
    if metadata.empty or metadata["filename"].isna().any():
        raise ValueError("Los metadatos están vacíos o contienen filenames nulos")
    if metadata["filename"].duplicated().any():
        raise ValueError("Los metadatos contienen filenames duplicados")
    values = set(np.unique(metadata.loc[:, labels].to_numpy()))
    if not values <= {0, 1}:
        raise ValueError("Las etiquetas activas deben ser binarias")


def _hz_to_mel(frequency: Tensor) -> Tensor:
    return 2595.0 * torch.log10(1.0 + frequency / 700.0)


def _mel_to_hz(mel: Tensor) -> Tensor:
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: float,
) -> Tensor:
    """Construye un banco Mel triangular con escala HTK y sin normalización de área."""
    nyquist = sample_rate / 2
    if not 0 <= f_min < f_max <= nyquist:
        raise ValueError("Se requiere 0 <= f_min < f_max <= Nyquist")
    if n_mels <= 0 or n_fft <= 1:
        raise ValueError("n_mels y n_fft deben ser positivos")

    endpoints = torch.tensor([f_min, f_max], dtype=torch.float64)
    endpoint_mels = _hz_to_mel(endpoints)
    mel_points = torch.linspace(
        endpoint_mels[0], endpoint_mels[1], n_mels + 2, dtype=torch.float64
    )
    hz_points = _mel_to_hz(mel_points)
    frequencies = torch.linspace(0, nyquist, n_fft // 2 + 1, dtype=torch.float64)
    lower = hz_points[:-2, None]
    center = hz_points[1:-1, None]
    upper = hz_points[2:, None]
    rising = (frequencies - lower) / (center - lower)
    falling = (upper - frequencies) / (upper - center)
    filters = torch.clamp(torch.minimum(rising, falling), min=0.0).to(torch.float32)
    if filters.sum(dim=1).eq(0).any():
        raise ValueError("La resolución FFT produce uno o más filtros Mel vacíos")
    return filters


class LogMelSpectrogram(nn.Module):
    """Transformación determinista de audio mono a espectrograma log-Mel."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        features = config["features"]
        if features["type"] != "mel":
            raise ValueError("LogMelSpectrogram requiere features.type=mel")
        if features.get("mel_scale", "htk") != "htk":
            raise ValueError("La implementación actual requiere features.mel_scale=htk")
        if features.get("filter_normalization", "none") != "none":
            raise ValueError("La implementación actual requiere filter_normalization=none")

        self.sample_rate = int(config["data"]["sample_rate"])
        self.pre_emphasis = float(features["pre_emphasis"])
        self.n_fft = int(features["n_fft"])
        self.hop_length = int(features["hop_length"])
        self.center = bool(features["center"])
        self.power = float(features["power"])
        self.log_epsilon = float(features["log_epsilon"])
        if features["window"] != "hamming":
            raise ValueError("La implementación actual requiere features.window=hamming")
        if self.power <= 0 or self.log_epsilon <= 0:
            raise ValueError("power y log_epsilon deben ser positivos")

        self.register_buffer("window", torch.hamming_window(self.n_fft, periodic=True))
        self.register_buffer(
            "filters",
            mel_filterbank(
                self.sample_rate,
                self.n_fft,
                int(features["n_mels"]),
                float(features["f_min"]),
                float(features["f_max"]),
            ),
        )

    def forward(self, waveform: Tensor) -> Tensor:
        """Devuelve ``[n_mels, frames]`` para una forma de onda ``[samples]``."""
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
        mel = self.filters @ power
        return torch.log(mel.clamp_min(self.log_epsilon))


def build_transform(config: dict[str, Any]) -> nn.Module:
    """Construye la representación configurada sin ajustar artefactos aprendidos."""
    feature_type = config["features"]["type"]
    if feature_type == "mel":
        return LogMelSpectrogram(config)
    if feature_type == "fbrs":
        from anuraset_dl.fbrs import FBRSSpectrogram

        return FBRSSpectrogram(config)
    raise ValueError(f"Tipo de representación no reconocido: {feature_type}")


class AnuraDataset(Dataset[tuple[Tensor, Tensor]]):
    """Une un manifiesto con los objetivos y carga los audios correspondientes."""

    def __init__(
        self,
        config: dict[str, Any],
        split: str,
        transform: nn.Module | None = None,
    ) -> None:
        if split not in config["data"]["splits"]:
            raise ValueError(f"Partición no configurada: {split}")
        metadata = pd.read_csv(config["data"]["metadata"])
        self.labels = active_labels(metadata, config["data"]["excluded_labels"])
        if len(self.labels) != int(config["data"]["num_labels"]):
            raise ValueError("data.num_labels no coincide con la taxonomía activa")
        _validate_metadata(metadata, self.labels)

        manifest = pd.read_csv(config["data"]["splits"][split])
        if "filename" not in manifest or manifest["filename"].duplicated().any():
            raise ValueError("El manifiesto requiere filenames únicos")
        unknown = sorted(set(manifest["filename"]) - set(metadata["filename"]))
        if unknown:
            raise ValueError(f"El manifiesto contiene archivos sin metadatos: {unknown[0]}")
        self.rows = manifest.loc[:, ["filename"]].merge(
            metadata.loc[:, ["filename", *self.labels]],
            on="filename",
            how="left",
            validate="one_to_one",
        )
        self.audio_dir = Path(config["data"]["root"]) / "train"
        self.sample_rate = int(config["data"]["sample_rate"])
        self.expected_samples = round(
            self.sample_rate * float(config["data"]["clip_seconds"])
        )
        self.transform = transform or build_transform(config)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        row = self.rows.iloc[index]
        path = self.audio_dir / row["filename"]
        waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        if sample_rate != self.sample_rate or waveform.ndim != 1:
            raise ValueError(f"Audio incompatible: {path}")
        if len(waveform) != self.expected_samples:
            raise ValueError(f"Duración incompatible: {path}")
        features = self.transform(torch.from_numpy(waveform)).unsqueeze(0)
        targets = torch.tensor(row.loc[list(self.labels)].to_numpy(dtype=np.float32))
        return features, targets
