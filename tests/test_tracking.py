import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from anuraset_dl.tracking import MlflowTracker


def test_disabled_tracking_is_a_no_op() -> None:
    tracker = MlflowTracker.start({"tracking": {"enabled": False}}, phase="training")

    tracker.log_metrics({"training/loss": 1.0}, step=1)
    tracker.end("FINISHED")

    assert not tracker.active
    assert tracker.run_id is None


def test_tracking_start_failure_does_not_interrupt_experiment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    broken_mlflow = SimpleNamespace(
        set_tracking_uri=lambda _uri: (_ for _ in ()).throw(RuntimeError("backend caído")),
        end_run=lambda **_kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "mlflow", broken_mlflow)
    config = {
        "experiment": "synthetic",
        "model": {"name": "cnn"},
        "features": {"type": "mel"},
        "tracking": {
            "enabled": True,
            "experiment_name": "tests",
            "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
            "artifact_root": str(tmp_path / "artifacts"),
            "log_system_metrics": False,
        },
    }

    with pytest.warns(RuntimeWarning, match="continuará sin tracking"):
        tracker = MlflowTracker.start(config, phase="training")

    assert not tracker.active


def test_logging_failure_is_reduced_to_a_warning() -> None:
    broken_mlflow = SimpleNamespace(
        log_metrics=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("escritura fallida")
        )
    )
    tracker = MlflowTracker(broken_mlflow, run_id="run-1")

    with pytest.warns(RuntimeWarning, match="experimento continuará"):
        tracker.log_metrics({"training/loss": 1.0}, step=1)

    assert tracker.active
