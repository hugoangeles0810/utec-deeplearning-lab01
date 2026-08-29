import pytest
import torch

from anuraset_dl.dlog import BasicDLoGConvolutionModule, DirectionalLoG2d


def test_dlog_kernels_are_finite_zero_mean_and_directional() -> None:
    layer = DirectionalLoG2d(in_channels=2, kernel_size=7)

    kernels = layer.kernels()

    assert kernels.shape == (4, 1, 7, 7)
    assert torch.isfinite(kernels).all()
    assert torch.allclose(kernels.sum(dim=(-2, -1)), torch.zeros(4, 1), atol=1e-6)
    assert not torch.allclose(kernels[0], kernels[1])
    assert layer.sigma.item() == pytest.approx(1.0)


def test_dlog_orientation_and_scale_receive_gradients() -> None:
    torch.manual_seed(3)
    layer = DirectionalLoG2d(in_channels=1, kernel_size=7)

    output = layer(torch.randn(2, 1, 16, 18))
    output.square().mean().backward()

    assert layer.theta.grad is not None
    assert torch.isfinite(layer.theta.grad).all()
    assert layer.theta.grad.abs().sum() > 0
    assert layer.raw_sigma.grad is not None
    assert torch.isfinite(layer.raw_sigma.grad)
    assert layer.raw_sigma.grad.abs() > 0


def test_dlog_preserves_spatial_shape_and_expands_directions() -> None:
    layer = DirectionalLoG2d(in_channels=3, kernel_size=5)

    output = layer(torch.randn(2, 3, 15, 17))

    assert output.shape == (2, 12, 15, 17)


def test_bdcm_fuses_identity_and_reduces_resolution() -> None:
    block = BasicDLoGConvolutionModule(in_channels=3, out_channels=8, kernel_size=5)

    output = block(torch.randn(2, 3, 16, 20))

    assert output.shape == (2, 8, 8, 10)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"in_channels": 0}, "in_channels"),
        ({"in_channels": 1, "kernel_size": 4}, "kernel_size"),
        ({"in_channels": 1, "initial_sigma": 0.2}, "initial_sigma"),
    ],
)
def test_dlog_rejects_invalid_parameters(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DirectionalLoG2d(**kwargs)
