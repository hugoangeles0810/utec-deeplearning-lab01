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
    "filter_normalization",
    "bank_path",
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
    if features["window"] != "hamming":
        raise ValueError("La implementación requiere features.window=hamming")
    if int(features["n_fft"]) <= 1 or int(features["hop_length"]) <= 0:
        raise ValueError("features.n_fft y features.hop_length deben ser positivos")
    if float(features["power"]) <= 0 or float(features["log_epsilon"]) <= 0:
        raise ValueError("features.power y features.log_epsilon deben ser positivos")
    feature_type = features["type"]
    if feature_type == "mel":
        _require_fields(
            features,
            ("n_mels", "f_min", "f_max", "mel_scale", "filter_normalization"),
            "features",
        )
        if features["mel_scale"] != "htk" or features["filter_normalization"] != "none":
            raise ValueError("El baseline requiere Mel HTK sin normalización del banco")
    elif feature_type == "fbrs":
        _require_fields(features, _FBRS_FIELDS, "features")
        if features["fit_subset"] not in {"active_positive_training", "all_training"}:
            raise ValueError(
                "features.fit_subset debe ser active_positive_training o all_training"
            )
        if features["bank_scope"] != "corpus":
            raise ValueError("La implementación requiere features.bank_scope=corpus")
        if features["energy_normalization"] != "per_level":
            raise ValueError("La implementación requiere energy_normalization=per_level")
        if features["band_selection"] != "target_count":
            raise ValueError("La implementación requiere band_selection=target_count")
        if features["filter_shape"] != "triangular":
            raise ValueError("La implementación requiere filter_shape=triangular")
        if features["filter_normalization"] != "peak":
            raise ValueError("La implementación requiere filter_normalization=peak")
        level = features["level"]
        target_bands = features["target_bands"]
        if isinstance(level, bool) or not isinstance(level, int) or level <= 0:
            raise ValueError("features.level debe ser un entero positivo")
        if (
            isinstance(target_bands, bool)
            or not isinstance(target_bands, int)
            or not 1 <= target_bands <= 2**level
        ):
            raise ValueError("features.target_bands debe pertenecer a [1, 2**level]")
        if not isinstance(features["bank_path"], str) or not features["bank_path"].strip():
            raise ValueError("features.bank_path debe ser una ruta no vacía")
    else:
        raise ValueError(f"Tipo de representación no reconocido: {feature_type}")

    training = _require_mapping(config, "training")
    _require_fields(
        training,
        (
            "batch_size",
            "epochs",
            "learning_rate",
            "loss",
            "device",
            "num_workers",
            "checkpoint_dir",
        ),
        "training",
    )
    if training["loss"] != "bce_with_logits":
        raise ValueError("La tarea multietiqueta requiere training.loss=bce_with_logits")
    if training["device"] not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("training.device debe ser auto, cpu, mps o cuda")
    for field in ("batch_size", "epochs"):
        invalid_type = isinstance(training[field], bool) or not isinstance(
            training[field], int
        )
        if invalid_type or training[field] <= 0:
            raise ValueError(f"training.{field} debe ser un entero positivo")
    if (
        isinstance(training["num_workers"], bool)
        or not isinstance(training["num_workers"], int)
        or training["num_workers"] < 0
    ):
        raise ValueError("training.num_workers debe ser un entero no negativo")
    if float(training["learning_rate"]) <= 0:
        raise ValueError("training.learning_rate debe ser positivo")

    evaluation = _require_mapping(config, "evaluation")
    _require_fields(
        evaluation,
        ("threshold_strategy", "zero_positive_class_policy", "metrics"),
        "evaluation",
    )
    _require_fields(evaluation, ("metrics_dir",), "evaluation")
    if evaluation["threshold_strategy"] != "per_class_validation_f1":
        raise ValueError("Los umbrales deben seleccionarse por F1 en validación")
    if evaluation["zero_positive_class_policy"] != "error":
        raise ValueError(
            "Las clases principales sin positivos deben invalidar la evaluación"
        )

    tracking = config.get("tracking")
    if tracking is not None:
        if not isinstance(tracking, dict):
            raise ValueError("La sección 'tracking' debe ser un objeto YAML")
        _require_fields(
            tracking,
            (
                "enabled",
                "experiment_name",
                "tracking_uri",
                "artifact_root",
                "log_system_metrics",
            ),
            "tracking",
        )
        if not isinstance(tracking["enabled"], bool):
            raise ValueError("tracking.enabled debe ser booleano")
        if not isinstance(tracking["log_system_metrics"], bool):
            raise ValueError("tracking.log_system_metrics debe ser booleano")
        for field in ("experiment_name", "tracking_uri", "artifact_root"):
            if not isinstance(tracking[field], str) or not tracking[field].strip():
                raise ValueError(f"tracking.{field} debe ser un texto no vacío")


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
