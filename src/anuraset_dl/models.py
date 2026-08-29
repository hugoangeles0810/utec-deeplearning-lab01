"""Modelos CNN de referencia y construcción desde configuración."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn


class BaselineCNN(nn.Module):
    """CNN compacta para representaciones tiempo-frecuencia de tamaño variable."""

    def __init__(self, num_outputs: int, channels: list[int], dropout: float) -> None:
        super().__init__()
        if num_outputs <= 0 or not channels or any(channel <= 0 for channel in channels):
            raise ValueError("La CNN requiere salidas y canales positivos")
        if not 0 <= dropout < 1:
            raise ValueError("dropout debe pertenecer a [0, 1)")
        blocks: list[nn.Module] = []
        input_channels = 1
        for output_channels in channels:
            blocks.extend(
                [
                    nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            input_channels = output_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(input_channels, num_outputs)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.pool(self.features(inputs)))


def build_model(config: dict[str, Any]) -> nn.Module:
    """Construye el modelo indicado sin duplicar el número de clases."""
    model = config["model"]
    if model["name"] != "cnn":
        raise NotImplementedError(f"Modelo todavía no implementado: {model['name']}")
    return BaselineCNN(
        num_outputs=int(config["data"]["num_labels"]),
        channels=[int(value) for value in model.get("channels", [32, 64, 128])],
        dropout=float(model.get("dropout", 0.3)),
    )


def count_trainable_parameters(model: nn.Module) -> int:
    """Cuenta los parámetros que reciben gradientes."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
