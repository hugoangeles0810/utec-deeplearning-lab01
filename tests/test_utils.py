from copy import deepcopy
from pathlib import Path

import pytest

from anuraset_dl.utils import load_config, num_outputs, validate_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "filename",
    ["baseline.yaml", "cnn_fbrs.yaml", "dlognet_mel.yaml", "dlognet_fbrs.yaml"],
)
def test_project_configs_are_valid(filename: str) -> None:
    config = load_config(ROOT / "configs" / filename)
    assert num_outputs(config) == config["data"]["num_labels"]
    assert num_outputs(config) == 31
    assert config["data"]["excluded_labels"] == [
        "SCIRIZ",
        "SCIALT",
        "ADEDIP",
        "DENELE",
        "RHISCI",
        "AMEPIC",
        "LEPELE",
        "RHIORN",
        "LEPFLA",
        "SCIFUS",
        "SCINAS",
    ]


def test_config_rejects_duplicated_output_count() -> None:
    config = load_config(ROOT / "configs" / "baseline.yaml")
    config["model"]["num_outputs"] = config["data"]["num_labels"]

    with pytest.raises(ValueError, match="duplica data.num_labels"):
        validate_config(config)


def test_fbrs_config_requires_reproducibility_fields() -> None:
    config = deepcopy(load_config(ROOT / "configs" / "dlognet_fbrs.yaml"))
    del config["features"]["energy_normalization"]

    with pytest.raises(ValueError, match="energy_normalization"):
        validate_config(config)


def test_fbrs_config_selects_positives_from_active_taxonomy() -> None:
    config = deepcopy(load_config(ROOT / "configs" / "dlognet_fbrs.yaml"))
    config["features"]["fit_subset"] = "positive_training"

    with pytest.raises(ValueError, match="active_positive_training"):
        validate_config(config)


def test_fbrs_config_allows_all_training_ablation() -> None:
    config = deepcopy(load_config(ROOT / "configs" / "dlognet_fbrs.yaml"))
    config["features"]["fit_subset"] = "all_training"

    validate_config(config)


def test_config_rejects_silently_omitting_classes_without_positives() -> None:
    config = deepcopy(load_config(ROOT / "configs" / "baseline.yaml"))
    config["evaluation"]["zero_positive_class_policy"] = "omit"

    with pytest.raises(ValueError, match="invalidar la evaluación"):
        validate_config(config)
