# UTEC Deep Learning — Laboratorio 01

Transferencia de FBRS y DLoGNet desde clasificación de aves hacia reconocimiento
multietiqueta de anuros sobre grabaciones con formato AnuraSet.

## Estado

El repositorio contiene la auditoría del corpus, las particiones reproducibles por grabación, el
baseline CNN + log-Mel y el pipeline CNN + FBRS. DLoGNet se incorporará progresivamente dentro
de `src/anuraset_dl/`.

## Preparación del entorno

Se requiere Python 3.12 y [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --group dev --group tracking
```

El grupo `tracking` instala MLflow y la captura de métricas del sistema. El pipeline puede
ejecutarse sin ese grupo si `tracking.enabled` se desactiva; si MLflow no está disponible o su
backend falla, el experimento continúa y conserva sus artefactos locales canónicos.

## Comandos principales

```bash
uv run pytest
uv run jupyter lab
uv run --group tracking python -m anuraset_dl.prepare_data --config configs/baseline.yaml
uv run --group tracking python -m anuraset_dl.train --config configs/baseline.yaml
uv run --group tracking python -m anuraset_dl.evaluate --config configs/baseline.yaml
```

Para ajustar el banco FBRS únicamente sobre entrenamiento y ejecutar la comparación con la misma
CNN:

```bash
uv run python -m anuraset_dl.fbrs --config configs/cnn_fbrs.yaml
uv run --group tracking python -m anuraset_dl.train --config configs/cnn_fbrs.yaml
uv run --group tracking python -m anuraset_dl.evaluate --config configs/cnn_fbrs.yaml
```

El ajuste se detiene si el banco ya existe; `--force` permite reemplazarlo de forma explícita.
Entrenamiento y evaluación verifican la huella del banco congelado además de las huellas de los
metadatos y manifiestos.

El comando de preparación valida el CSV y todas las cabeceras de audio antes de reproducir los
manifiestos. El entrenamiento guarda `best.pt`, `last.pt` y el historial bajo
`outputs/checkpoints/cnn_mel_baseline/`. La evaluación selecciona los umbrales por clase sobre
validación, los congela para prueba y escribe el reporte completo en
`outputs/metrics/cnn_mel_baseline.json`.

El dispositivo se selecciona automáticamente con prioridad CUDA, MPS y CPU. Puede forzarse, por
ejemplo, con `--device cpu` tanto en entrenamiento como en evaluación. El cambio de dispositivo
no invalida un checkpoint; la evaluación sí se detiene si detecta cambios en los metadatos o en
alguno de los manifiestos utilizados por el experimento.

Entrenamiento y evaluación comparten un mismo run de MLflow. La interfaz local se abre con:

```bash
uv run --group tracking mlflow server \
  --backend-store-uri sqlite:///outputs/mlflow/mlflow.db \
  --default-artifact-root ./outputs/mlflow/artifacts
```

Después puede consultarse en `http://127.0.0.1:5000`. MLflow registra parámetros, huellas,
pérdidas por época, duración y métricas finales, pero no copia `best.pt` ni `last.pt`; esos
checkpoints permanecen bajo `outputs/checkpoints/`.

## Organización

- `configs/`: configuraciones de experimentos.
- `dataset/`: audios y etiquetas locales, excluidos de Git.
- `splits/`: particiones por grabación, pequeñas y versionables.
- `src/anuraset_dl/`: código fuente reutilizable.
- `notebooks/`: análisis exploratorio y visualización.
- `docs/`: documentación metodológica y técnica; véase [`docs/README.md`](docs/README.md).
- `tests/`: pruebas automáticas.
- `outputs/`: bancos de filtros, checkpoints, métricas y figuras generadas.
