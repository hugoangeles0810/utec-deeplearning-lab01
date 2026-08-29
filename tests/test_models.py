import torch

from anuraset_dl.models import BaselineCNN, build_model, count_trainable_parameters


def test_baseline_cnn_produces_one_logit_per_label() -> None:
    model = BaselineCNN(num_outputs=31, channels=[8, 16], dropout=0.2)

    output = model(torch.randn(3, 1, 32, 40))

    assert output.shape == (3, 31)
    assert count_trainable_parameters(model) > 0


def test_build_model_derives_outputs_from_data() -> None:
    config = {"data": {"num_labels": 2}, "model": {"name": "cnn", "channels": [4]}}

    output = build_model(config)(torch.randn(2, 1, 8, 8))

    assert output.shape == (2, 2)
