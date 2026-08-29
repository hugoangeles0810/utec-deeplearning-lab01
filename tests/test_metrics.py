import numpy as np
import pytest

from anuraset_dl.metrics import multilabel_metrics, select_f1_thresholds


def test_threshold_selection_and_metrics_are_perfect_for_separable_scores() -> None:
    targets = np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]])
    probabilities = np.asarray([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])

    thresholds = select_f1_thresholds(targets, probabilities)
    result = multilabel_metrics(targets, probabilities, thresholds, ("a", "b"))

    assert result["macro"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "map": 1.0}
    assert [row["support"] for row in result["per_class"]] == [2, 2]


def test_metrics_reject_classes_without_positives() -> None:
    targets = np.asarray([[1, 0], [0, 0]])
    probabilities = np.asarray([[0.8, 0.2], [0.2, 0.3]])

    with pytest.raises(ValueError, match="Clases sin positivos"):
        select_f1_thresholds(targets, probabilities)
