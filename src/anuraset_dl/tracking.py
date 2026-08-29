"""Integración opcional y tolerante a fallos con MLflow Tracking."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any


def _flatten_params(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Convierte una configuración anidada en parámetros consultables por MLflow."""
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            flattened.update(_flatten_params(item, name))
        elif isinstance(item, (list, tuple)):
            flattened[name] = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        elif item is None:
            flattened[name] = "null"
        else:
            flattened[name] = item
    return flattened


def _prepare_local_storage(settings: dict[str, Any]) -> str | None:
    """Crea directorios locales declarados por la configuración de tracking."""
    tracking_uri = settings.get("tracking_uri")
    if isinstance(tracking_uri, str) and tracking_uri.startswith("sqlite:///"):
        database_path = Path(tracking_uri.removeprefix("sqlite:///"))
        database_path.parent.mkdir(parents=True, exist_ok=True)

    artifact_root = settings.get("artifact_root")
    if not artifact_root:
        return None
    root = Path(artifact_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root.as_uri()


class MlflowTracker:
    """Adaptador pequeño que nunca convierte un fallo de tracking en fallo experimental."""

    def __init__(
        self,
        mlflow: Any | None = None,
        run_id: str | None = None,
        resumed: bool = False,
    ) -> None:
        self._mlflow = mlflow
        self.run_id = run_id
        self.resumed = resumed

    @property
    def active(self) -> bool:
        """Indica si existe un run activo al que se puedan enviar eventos."""
        return self._mlflow is not None and self.run_id is not None

    @classmethod
    def start(
        cls,
        config: dict[str, Any],
        phase: str,
        run_id: str | None = None,
    ) -> MlflowTracker:
        """Inicia o reanuda un run; ante cualquier problema devuelve un no-op."""
        settings = config.get("tracking", {})
        if not settings.get("enabled", False):
            return cls()

        try:
            import mlflow
        except ImportError:
            warnings.warn(
                "MLflow no está instalado; el experimento continuará sin tracking. "
                "Ejecute `uv sync --group dev --group tracking` para habilitarlo.",
                RuntimeWarning,
                stacklevel=2,
            )
            return cls()

        try:
            artifact_location = _prepare_local_storage(settings)
            tracking_uri = settings.get("tracking_uri")
            if tracking_uri:
                mlflow.set_tracking_uri(str(tracking_uri))

            client = mlflow.MlflowClient()
            experiment_name = str(settings["experiment_name"])
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = client.create_experiment(
                    experiment_name,
                    artifact_location=artifact_location,
                )
            else:
                experiment_id = experiment.experiment_id

            active_run = None
            resumed = False
            if run_id:
                try:
                    active_run = mlflow.start_run(
                        run_id=run_id,
                        log_system_metrics=bool(settings.get("log_system_metrics", False)),
                    )
                    resumed = True
                except Exception as error:  # MLflow usa excepciones propias según el backend.
                    warnings.warn(
                        f"No se pudo reanudar el run MLflow {run_id}: {error}. "
                        "Se intentará crear uno nuevo.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            if active_run is None:
                active_run = mlflow.start_run(
                    experiment_id=experiment_id,
                    run_name=str(config["experiment"]),
                    log_system_metrics=bool(settings.get("log_system_metrics", False)),
                )

            tracker = cls(mlflow, active_run.info.run_id, resumed=resumed)
            tracker.set_tags(
                {
                    "anuraset.phase": phase,
                    "anuraset.experiment": config["experiment"],
                    "anuraset.model": config["model"]["name"],
                    "anuraset.features": config["features"]["type"],
                }
            )
            return tracker
        except Exception as error:  # El entrenamiento no depende de la telemetría.
            try:
                mlflow.end_run(status="FAILED")
            except Exception:
                pass
            warnings.warn(
                f"No se pudo iniciar MLflow; el experimento continuará sin tracking: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return cls()

    def _call(self, operation: str, *args: Any, **kwargs: Any) -> None:
        if not self.active:
            return
        try:
            getattr(self._mlflow, operation)(*args, **kwargs)
        except Exception as error:
            warnings.warn(
                f"MLflow no pudo ejecutar {operation}; el experimento continuará: {error}",
                RuntimeWarning,
                stacklevel=2,
            )

    def log_config(self, config: dict[str, Any]) -> None:
        """Registra parámetros consultables y una copia JSON de la configuración efectiva."""
        self._call("log_params", _flatten_params(config))
        self._call("log_dict", config, "inputs/effective_config.json")

    def log_dict(self, payload: dict[str, Any], artifact_file: str) -> None:
        """Registra un diccionario como artefacto JSON pequeño."""
        self._call("log_dict", payload, artifact_file)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Registra métricas escalares, opcionalmente asociadas a una época."""
        self._call("log_metrics", metrics, step=step)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        """Registra un artefacto pequeño existente sin duplicar checkpoints."""
        source = Path(path)
        if source.is_file():
            self._call("log_artifact", str(source), artifact_path=artifact_path)

    def set_tags(self, tags: dict[str, Any]) -> None:
        """Añade metadatos que facilitan filtrar ejecuciones."""
        self._call("set_tags", tags)

    def end(self, status: str) -> None:
        """Cierra el run activo con estado FINISHED o FAILED."""
        if not self.active:
            return
        self._call("end_run", status=status)
        self._mlflow = None
