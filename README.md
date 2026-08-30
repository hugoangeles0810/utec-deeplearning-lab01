# UTEC Deep Learning — Laboratorio 01

Transferencia de FBRS y DLoGNet desde clasificación de aves hacia reconocimiento
multietiqueta de anuros sobre grabaciones con formato AnuraSet.

## Estado

El repositorio contiene la auditoría del corpus, las particiones reproducibles por grabación y
los pipelines CNN/DLoGNet con representaciones log-Mel/FBRS. Las dos ejecuciones CNN están
registradas; las ejecuciones completas de DLoGNet permanecen pendientes.

## Preparación del entorno

Se requiere Python 3.12 y [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --group dev --group tracking
```

El grupo `tracking` instala MLflow y la captura de métricas del sistema. El pipeline puede
ejecutarse sin ese grupo si `tracking.enabled` se desactiva; si MLflow no está disponible o su
backend falla, el experimento continúa y conserva sus artefactos locales canónicos.

## Estructura del dataset

Los datos no se distribuyen con el repositorio. Después de obtener AnuraSet, deben organizarse
localmente con esta estructura:

```text
dataset/
├── train.csv
├── train/
│   ├── <filename_1>.wav
│   ├── <filename_2>.wav
│   └── ...
└── test/
    └── <external_filename>.wav
```

`train.csv` debe contener una columna `filename` con nombres únicos y una columna binaria por
etiqueta. Cada valor de `filename` debe identificar exactamente un WAV dentro de `dataset/train/`.
Los audios del protocolo vigente son mono, tienen una frecuencia de muestreo de 22.05 kHz y una
duración exacta de 3 segundos. Los nombres deben conservarse sin cambios, porque los manifiestos
versionados en `splits/` los utilizan para vincular cada clip con su partición.

`dataset/test/` recibe los audios externos sin etiquetas cuando estén disponibles. Esos WAV no
participan en el particionado ni se incorporan a `train.csv`.

El CSV y los WAV permanecen ignorados por Git; los archivos `.gitkeep` conservan únicamente la
estructura vacía. Para validar el dataset local y comprobar que coincide con los manifiestos:

```bash
uv run python -m anuraset_dl.prepare_data --config configs/baseline.yaml
```

La política de etiquetas, particiones y controles se detalla en
[`docs/data-preparation.md`](docs/data-preparation.md), y la huella del corte utilizado por el
proyecto se registra en [`docs/dataset-audit.md`](docs/dataset-audit.md).

## Comandos principales

```bash
uv run pytest
uv run jupyter lab
uv run --group tracking python -m anuraset_dl.prepare_data --config configs/baseline.yaml
uv run --group tracking python -m anuraset_dl.train --config configs/baseline.yaml
uv run --group tracking python -m anuraset_dl.evaluate --config configs/baseline.yaml
uv run python -m anuraset_dl.predict --config configs/baseline.yaml --input-dir dataset/test
```

Para ajustar el banco FBRS únicamente sobre entrenamiento y ejecutar la comparación con la misma
CNN:

```bash
uv run python -m anuraset_dl.fbrs --config configs/cnn_fbrs.yaml
uv run --group tracking python -m anuraset_dl.train --config configs/cnn_fbrs.yaml
uv run --group tracking python -m anuraset_dl.evaluate --config configs/cnn_fbrs.yaml
```

DLoGNet utiliza el mismo protocolo de datos, entrenamiento y evaluación:

```bash
uv run --group tracking python -m anuraset_dl.train --config configs/dlognet_mel.yaml
uv run --group tracking python -m anuraset_dl.evaluate --config configs/dlognet_mel.yaml
uv run --group tracking python -m anuraset_dl.train --config configs/dlognet_fbrs.yaml
uv run --group tracking python -m anuraset_dl.evaluate --config configs/dlognet_fbrs.yaml
```

`dlognet_fbrs` reutiliza el banco congelado declarado en `features.bank_path`; no debe volver a
ajustarse con validación ni con el test externo.

El ajuste se detiene si el banco ya existe; `--force` permite reemplazarlo de forma explícita.
Entrenamiento y evaluación verifican la huella del banco congelado además de las huellas de los
metadatos y manifiestos.

El comando de preparación valida el CSV y todas las cabeceras de audio antes de reproducir los
manifiestos. El entrenamiento guarda `best.pt`, `last.pt` y el historial bajo
`outputs/checkpoints/cnn_mel_baseline/`. La evaluación selecciona los umbrales por clase y calcula
las métricas internas sobre validación en `outputs/metrics/cnn_mel_baseline.json`. La inferencia
aplica esos umbrales al test sin etiquetas y escribe probabilidades y decisiones bajo
`outputs/predictions/`; no calcula métricas de test localmente.

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
