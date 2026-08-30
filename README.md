# UTEC Deep Learning — Laboratorio 01

Transferencia de FBRS y DLoGNet desde clasificación de aves hacia reconocimiento
multietiqueta de anuros sobre grabaciones con formato AnuraSet.

## Estado

El repositorio contiene la auditoría del corpus, las particiones reproducibles por grabación y
los pipelines CNN/DLoGNet con representaciones log-Mel/FBRS. Las cuatro ejecuciones del protocolo
vigente 80/20 permanecen pendientes y no existen resultados experimentales registrados.

## Preparación del entorno

Se requiere Python 3.12 y [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --group dev --group tracking
```

El grupo `tracking` instala MLflow y la captura de métricas del sistema. El pipeline puede
ejecutarse sin ese grupo si `tracking.enabled` se desactiva; si MLflow no está disponible o su
backend falla, el experimento continúa y conserva sus artefactos locales canónicos.

### Ejecución en Runpod

El proyecto incluye un flujo para preparar un Pod con CUDA, aprovisionar únicamente el conjunto
de entrenamiento, ejecutar por defecto los cuatro experimentos y exportar checkpoints, métricas y
MLflow en un paquete verificable:

```bash
scripts/runpod/bootstrap.sh
scripts/runpod/run.sh
```

También pueden seleccionarse experimentos individuales y respaldar automáticamente los resultados
en Google Drive mediante `rclone`. La creación manual del Pod, los requisitos de almacenamiento y
el procedimiento completo se documentan en [`docs/runpod.md`](docs/runpod.md).

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

El corte utilizado por el proyecto puede aprovisionarse directamente desde Google Drive:

```bash
uv run python -m anuraset_dl.provision_data
```

El comando descarga `train.csv`, `train.7z` y `test.7z`, reanuda transferencias interrumpidas,
extrae cada archivo de forma secuencial y valida el resultado antes de publicarlo en `dataset/`.
Es idempotente: si el corpus local ya es válido, no vuelve a descargarlo. Por defecto elimina los
comprimidos después de una extracción correcta para reducir el uso de disco.

Se recomienda disponer de al menos 18 GB libres. `--keep-archives` conserva los comprimidos y
eleva el espacio necesario a unos 23 GB; `--only train` o `--only test` aprovisionan un solo
subconjunto. `--dataset-root RUTA` cambia el destino, y `--force` permite reemplazar contenido
local incompleto o incompatible de forma explícita. La fuente y las huellas esperadas se fijan en
`configs/dataset.yaml` para evitar aceptar silenciosamente otro corte.

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
uv run python -m anuraset_dl.precompute_features --config configs/baseline.yaml
uv run --group tracking python -m anuraset_dl.train --config configs/baseline.yaml
uv run --group tracking python -m anuraset_dl.evaluate --config configs/baseline.yaml
uv run python -m anuraset_dl.predict --config configs/baseline.yaml --input-dir dataset/test
```

Para ajustar el banco FBRS únicamente sobre entrenamiento y ejecutar la comparación con la misma
CNN:

```bash
uv run python -m anuraset_dl.fbrs --config configs/cnn_fbrs.yaml
uv run python -m anuraset_dl.precompute_features --config configs/cnn_fbrs.yaml
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

`dlognet_mel` reutiliza la caché Mel creada con `baseline.yaml` y `dlognet_fbrs` reutiliza la
caché FBRS creada con `cnn_fbrs.yaml`: su identidad depende de los datos y de la representación,
no de la arquitectura. El comando de precomputación es idempotente y reutiliza un artefacto
válido existente.

`dlognet_fbrs` reutiliza el banco congelado declarado en `features.bank_path`; no debe volver a
ajustarse con validación ni con el test externo.

El ajuste se detiene si el banco ya existe; `--force` permite reemplazarlo de forma explícita.
Entrenamiento y evaluación verifican la huella del banco congelado además de las huellas de los
metadatos y manifiestos.

Las representaciones se persisten como arreglos NPY `float32` mapeados en memoria bajo
`outputs/features/`. La caché registra las huellas de metadatos, particiones y parámetros de la
representación; FBRS incorpora además la huella del banco congelado. Si cambia cualquiera de
esas entradas, el pipeline exige crear una caché nueva. Estos artefactos son locales y están
excluidos de Git.

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
- `outputs/features/`: cachés locales de representaciones Mel y FBRS.
