"""Modelos CNN de referencia y construcción desde configuración."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from anuraset_dl.dlog import BasicDLoGConvolutionModule


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


class DLoGNet(nn.Module):
    """Red de módulos DLoG adaptada para producir logits multietiqueta."""

    def __init__(
        self,
        num_outputs: int,
        channels: list[int],
        kernel_size: int,
        initial_angles_degrees: list[float],
        initial_sigma: float,
        minimum_sigma: float,
        classifier_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_outputs <= 0 or not channels or any(channel <= 0 for channel in channels):
            raise ValueError("DLoGNet requiere salidas y canales positivos")
        if classifier_hidden <= 0:
            raise ValueError("classifier_hidden debe ser positivo")
        if not 0 <= dropout < 1:
            raise ValueError("dropout debe pertenecer a [0, 1)")

        blocks: list[nn.Module] = []
        input_channels = 1
        for output_channels in channels:
            blocks.append(
                BasicDLoGConvolutionModule(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    initial_angles_degrees=initial_angles_degrees,
                    initial_sigma=initial_sigma,
                    minimum_sigma=minimum_sigma,
                )
            )
            input_channels = output_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_channels, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_outputs),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.pool(self.features(inputs)))


def build_model(config: dict[str, Any]) -> nn.Module:
    """Construye el modelo indicado sin duplicar el número de clases."""
    model = config["model"]
    num_outputs = int(config["data"]["num_labels"])
    if model["name"] == "cnn":
        return BaselineCNN(
            num_outputs=num_outputs,
            channels=[int(value) for value in model.get("channels", [32, 64, 128])],
            dropout=float(model.get("dropout", 0.3)),
        )
    if model["name"] == "dlognet":
        return DLoGNet(
            num_outputs=num_outputs,
            channels=[int(value) for value in model.get("channels", [64, 128, 128, 128, 64])],
            kernel_size=int(model.get("kernel_size", 7)),
            initial_angles_degrees=[
                float(value)
                for value in model.get("initial_angles_degrees", [0, 45, 90, 135])
            ],
            initial_sigma=float(model.get("initial_sigma", 1.0)),
            minimum_sigma=float(model.get("minimum_sigma", 0.3)),
            classifier_hidden=int(model.get("classifier_hidden", 1024)),
            dropout=float(model.get("dropout", 0.3)),
        )
    raise ValueError(f"Modelo no reconocido: {model['name']}")


def count_trainable_parameters(model: nn.Module) -> int:
    """Cuenta los parámetros que reciben gradientes."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
