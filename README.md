# UTEC Deep Learning — Laboratorio 01

Transferencia de FBRS y DLoGNet desde clasificación de aves hacia reconocimiento
multietiqueta de anuros sobre grabaciones con formato AnuraSet.

## Estado

El repositorio contiene la auditoría del corpus, las particiones reproducibles por grabación y
un baseline CNN + log-Mel ejecutable de extremo a extremo. FBRS y DLoGNet se incorporarán
progresivamente dentro de `src/anuraset_dl/`.

## Preparación del entorno

Se requiere Python 3.12 y [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --group dev
```

## Comandos principales

```bash
uv run pytest
uv run jupyter lab
uv run python -m anuraset_dl.prepare_data --config configs/baseline.yaml
uv run python -m anuraset_dl.train --config configs/baseline.yaml
uv run python -m anuraset_dl.evaluate --config configs/baseline.yaml
```

El comando de preparación valida el CSV y todas las cabeceras de audio antes de reproducir los
manifiestos. El entrenamiento guarda `best.pt`, `last.pt` y el historial bajo
`outputs/checkpoints/cnn_mel_baseline/`. La evaluación selecciona los umbrales por clase sobre
validación, los congela para prueba y escribe el reporte completo en
`outputs/metrics/cnn_mel_baseline.json`.

El dispositivo se selecciona automáticamente con prioridad CUDA, MPS y CPU. Puede forzarse, por
ejemplo, con `--device cpu` tanto en entrenamiento como en evaluación. El cambio de dispositivo
no invalida un checkpoint; la evaluación sí se detiene si detecta cambios en los metadatos o en
alguno de los manifiestos utilizados por el experimento.

## Organización

- `configs/`: configuraciones de experimentos.
- `dataset/`: audios y etiquetas locales, excluidos de Git.
- `splits/`: particiones por grabación, pequeñas y versionables.
- `src/anuraset_dl/`: código fuente reutilizable.
- `notebooks/`: análisis exploratorio y visualización.
- `docs/`: documentación metodológica y técnica; véase [`docs/README.md`](docs/README.md).
- `tests/`: pruebas automáticas.
- `outputs/`: bancos de filtros, checkpoints, métricas y figuras generadas.
