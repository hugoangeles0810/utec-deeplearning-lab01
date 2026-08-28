"""Utilidades compartidas de configuración y reproducibilidad."""

from pathlib import Path
from typing import Any

import yaml

_REQUIRED_SECTIONS = ("data", "features", "model", "training", "evaluation")
_REQUIRED_SPLITS = ("train", "validation", "test")
_COMMON_FEATURE_FIELDS = (
    "pre_emphasis",
    "window",
    "n_fft",
    "hop_length",
    "center",
    "power",
    "log_epsilon",
)
_FBRS_FIELDS = (
    "wavelet",
    "level",
    "bank_scope",
    "fit_subset",
    "energy_normalization",
    "band_selection",
    "target_bands",
    "filter_shape",
)


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"La sección '{key}' debe ser un objeto YAML")
    return value


def _require_fields(section: dict[str, Any], fields: tuple[str, ...], name: str) -> None:
    missing = [field for field in fields if field not in section]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Faltan campos en la sección '{name}': {joined}")


def validate_config(config: dict[str, Any]) -> None:
    """Valida la estructura común y evita fuentes de verdad duplicadas."""
    if not isinstance(config.get("experiment"), str) or not config["experiment"].strip():
        raise ValueError("La configuración requiere un nombre de experimento")

    for section in _REQUIRED_SECTIONS:
        _require_mapping(config, section)

    data = _require_mapping(config, "data")
    _require_fields(
        data,
        ("root", "metadata", "splits", "num_labels", "excluded_labels", "zero_label_policy"),
        "data",
    )
    num_labels = data["num_labels"]
    if isinstance(num_labels, bool) or not isinstance(num_labels, int) or num_labels <= 0:
        raise ValueError("data.num_labels debe ser un entero positivo")
    excluded_labels = data["excluded_labels"]
    if not isinstance(excluded_labels, list) or not all(
        isinstance(label, str) and label for label in excluded_labels
    ):
        raise ValueError("data.excluded_labels debe ser una lista de etiquetas")
    if len(excluded_labels) != len(set(excluded_labels)):
        raise ValueError("data.excluded_labels no puede contener duplicados")
    if data["zero_label_policy"] != "keep_as_all_negative":
        raise ValueError("Las filas sin positivos deben conservarse como ejemplos negativos")

    splits = data["splits"]
    if not isinstance(splits, dict):
        raise ValueError("data.splits debe ser un objeto YAML")
    _require_fields(splits, _REQUIRED_SPLITS, "data.splits")

    model = _require_mapping(config, "model")
    _require_fields(model, ("name",), "model")
    if "num_outputs" in model:
        raise ValueError(
            "model.num_outputs duplica data.num_labels; la salida debe derivarse del dataset"
        )

    features = _require_mapping(config, "features")
    _require_fields(features, ("type", *_COMMON_FEATURE_FIELDS), "features")
    feature_type = features["type"]
    if feature_type == "mel":
        _require_fields(features, ("n_mels", "f_min", "f_max"), "features")
    elif feature_type == "fbrs":
        _require_fields(features, _FBRS_FIELDS, "features")
    else:
        raise ValueError(f"Tipo de representación no reconocido: {feature_type}")

    training = _require_mapping(config, "training")
    _require_fields(training, ("batch_size", "epochs", "learning_rate", "loss"), "training")
    if training["loss"] != "bce_with_logits":
        raise ValueError("La tarea multietiqueta requiere training.loss=bce_with_logits")

    evaluation = _require_mapping(config, "evaluation")
    _require_fields(evaluation, ("threshold_strategy", "metrics"), "evaluation")


def num_outputs(config: dict[str, Any]) -> int:
    """Obtiene la dimensión de salida desde la única fuente de verdad del dataset."""
    validate_config(config)
    return config["data"]["num_labels"]


def load_config(path: str | Path) -> dict[str, Any]:
    """Carga una configuración YAML y valida su estructura."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"La configuración debe ser un objeto YAML: {config_path}")
    validate_config(config)
    return config
