"""Kernels direccionales Laplacian-of-Gaussian y bloques DLoG."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


def _inverse_softplus(value: float) -> float:
    """Calcula una inicialización estable para un parámetro pasado por ``softplus``."""
    return value + math.log(-math.expm1(-value))


class DirectionalLoG2d(nn.Module):
    """Aplica filtros DLoG *depthwise* con orientación y escala aprendibles.

    Un mismo banco físico se comparte entre canales. Cada canal de entrada produce una
    respuesta por orientación, por lo que la salida contiene
    ``in_channels * num_directions`` canales.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 7,
        initial_angles_degrees: Sequence[float] = (0.0, 45.0, 90.0, 135.0),
        initial_sigma: float = 1.0,
        minimum_sigma: float = 0.3,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels debe ser positivo")
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size debe ser impar y al menos 3")
        if not initial_angles_degrees:
            raise ValueError("Se requiere al menos una orientación DLoG")
        if minimum_sigma <= 0 or initial_sigma <= minimum_sigma:
            raise ValueError("initial_sigma debe ser mayor que minimum_sigma > 0")

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.minimum_sigma = minimum_sigma
        angles = torch.tensor(initial_angles_degrees, dtype=torch.float32)
        self.theta = nn.Parameter(torch.deg2rad(angles))
        sigma_offset = initial_sigma - minimum_sigma
        self.raw_sigma = nn.Parameter(torch.tensor(_inverse_softplus(sigma_offset)))

        radius = kernel_size // 2
        coordinates = torch.arange(-radius, radius + 1, dtype=torch.float32)
        y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        self.register_buffer("x_coordinates", x.clone())
        self.register_buffer("y_coordinates", y.clone())

    @property
    def num_directions(self) -> int:
        """Devuelve la cantidad de orientaciones del banco."""
        return self.theta.numel()

    @property
    def sigma(self) -> Tensor:
        """Expone la escala positiva utilizada para construir los kernels."""
        return functional.softplus(self.raw_sigma) + self.minimum_sigma

    def kernels(self) -> Tensor:
        """Construye los kernels ``[orientaciones, 1, alto, ancho]`` diferenciables."""
        sigma = self.sigma
        sigma_squared = sigma.square()
        radius_squared = self.x_coordinates.square() + self.y_coordinates.square()
        gaussian = torch.exp(-radius_squared / (2.0 * sigma_squared))
        gaussian = gaussian / (2.0 * math.pi * sigma_squared)

        common = gaussian / sigma_squared.square()
        g_xx = (self.x_coordinates.square() - sigma_squared) * common
        g_yy = (self.y_coordinates.square() - sigma_squared) * common
        g_xy = (self.x_coordinates * self.y_coordinates) * common

        theta = self.theta[:, None, None]
        cosine = torch.cos(theta)
        sine = torch.sin(theta)
        kernels = cosine.square() * g_xx + sine.square() * g_yy + 2.0 * sine * cosine * g_xy

        # La discretización y el truncamiento rompen ligeramente la respuesta DC nula teórica.
        kernels = kernels - kernels.mean(dim=(-2, -1), keepdim=True)
        normalizer = kernels.abs().sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        return (kernels / normalizer).unsqueeze(1)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError(
                "DirectionalLoG2d requiere entradas [batch, in_channels, alto, ancho]"
            )
        kernels = self.kernels().repeat(self.in_channels, 1, 1, 1)
        responses = functional.conv2d(
            inputs,
            kernels,
            padding=self.kernel_size // 2,
            groups=self.in_channels,
        )
        batch, _, height, width = responses.shape
        return (
            responses.reshape(batch, self.in_channels, self.num_directions, height, width)
            .permute(0, 2, 1, 3, 4)
            .reshape(batch, self.in_channels * self.num_directions, height, width)
        )


class BasicDLoGConvolutionModule(nn.Module):
    """Fusiona cuatro respuestas DLoG y la identidad antes de reducir resolución."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        initial_angles_degrees: Sequence[float] = (0.0, 45.0, 90.0, 135.0),
        initial_sigma: float = 1.0,
        minimum_sigma: float = 0.3,
    ) -> None:
        super().__init__()
        if out_channels <= 0:
            raise ValueError("out_channels debe ser positivo")
        self.directional = DirectionalLoG2d(
            in_channels=in_channels,
            kernel_size=kernel_size,
            initial_angles_degrees=initial_angles_degrees,
            initial_sigma=initial_sigma,
            minimum_sigma=minimum_sigma,
        )
        fused_channels = in_channels * (self.directional.num_directions + 1)
        self.fusion = nn.Conv2d(fused_channels, out_channels, kernel_size=3, padding=1)
        self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, inputs: Tensor) -> Tensor:
        directional = self.directional(inputs)
        fused = torch.cat((inputs, directional), dim=1)
        return self.pool(self.activation(self.normalization(self.fusion(fused))))
