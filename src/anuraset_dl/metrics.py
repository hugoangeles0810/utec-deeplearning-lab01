"""Métricas y selección de umbrales para clasificación multietiqueta."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def _validated_arrays(
    targets: np.ndarray, probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if targets.ndim != 2 or targets.shape != probabilities.shape:
        raise ValueError("targets y probabilities deben tener la misma forma bidimensional")
    if not set(np.unique(targets)) <= {0, 1}:
        raise ValueError("targets debe ser binario")
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0) | (probabilities > 1)
    ).any():
        raise ValueError("Las probabilidades deben ser finitas y pertenecer a [0, 1]")
    zero_positive = np.flatnonzero(targets.sum(axis=0) == 0)
    if len(zero_positive):
        raise ValueError(f"Clases sin positivos: {zero_positive.tolist()}")
    return targets, probabilities


def select_f1_thresholds(targets: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Selecciona por clase el umbral que maximiza F1 sobre validación."""
    targets, probabilities = _validated_arrays(targets, probabilities)
    selected = np.empty(targets.shape[1], dtype=np.float64)
    for index in range(targets.shape[1]):
        precision, recall, thresholds = precision_recall_curve(
            targets[:, index], probabilities[:, index]
        )
        if len(thresholds) == 0:
            selected[index] = 0.5
            continue
        denominator = precision[:-1] + recall[:-1]
        f1 = np.divide(
            2 * precision[:-1] * recall[:-1],
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )
        selected[index] = thresholds[int(np.argmax(f1))]
    return selected


def multilabel_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    labels: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Calcula métricas por clase, agregados macro y mean average precision."""
    targets, probabilities = _validated_arrays(targets, probabilities)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    labels = tuple(labels)
    if thresholds.shape != (targets.shape[1],) or len(labels) != targets.shape[1]:
        raise ValueError("Umbrales y etiquetas no coinciden con el número de clases")

    predictions = probabilities >= thresholds[None, :]
    true_positive = np.logical_and(predictions, targets == 1).sum(axis=0)
    false_positive = np.logical_and(predictions, targets == 0).sum(axis=0)
    false_negative = np.logical_and(~predictions, targets == 1).sum(axis=0)
    precision = np.divide(
        true_positive,
        true_positive + false_positive,
        out=np.zeros_like(true_positive, dtype=float),
        where=(true_positive + false_positive) > 0,
    )
    recall = true_positive / (true_positive + false_negative)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    average_precision = np.asarray(
        [
            average_precision_score(targets[:, index], probabilities[:, index])
            for index in range(targets.shape[1])
        ]
    )
    per_class = [
        {
            "label": label,
            "threshold": float(thresholds[index]),
            "support": int(targets[:, index].sum()),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "average_precision": float(average_precision[index]),
        }
        for index, label in enumerate(labels)
    ]
    return {
        "macro": {
            "precision": float(precision.mean()),
            "recall": float(recall.mean()),
            "f1": float(f1.mean()),
            "map": float(average_precision.mean()),
        },
        "per_class": per_class,
    }
