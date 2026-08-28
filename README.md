# UTEC Deep Learning — Laboratorio 01

Transferencia de FBRS y DLoGNet desde clasificación de aves hacia reconocimiento
multietiqueta de anuros sobre grabaciones con formato AnuraSet.

## Estado

El repositorio contiene el esqueleto reproducible del proyecto. La implementación de FBRS,
DLoGNet, entrenamiento y evaluación se incorporará progresivamente dentro de
`src/anuraset_dl/`.

## Preparación del entorno

Se requiere Python 3.12 y [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --group dev
```

## Comandos principales

```bash
uv run pytest
uv run jupyter lab
uv run python -m anuraset_dl.train --config configs/dlognet_fbrs.yaml
uv run python -m anuraset_dl.evaluate --config configs/dlognet_fbrs.yaml
```

Los dos últimos comandos validan actualmente la configuración y muestran el estado del
pipeline; todavía no ejecutan entrenamiento ni evaluación.

## Organización

- `configs/`: configuraciones de experimentos.
- `dataset/`: audios y etiquetas locales, excluidos de Git.
- `splits/`: particiones por grabación, pequeñas y versionables.
- `src/anuraset_dl/`: código fuente reutilizable.
- `notebooks/`: análisis exploratorio y visualización.
- `docs/`: documentación metodológica y técnica; véase [`docs/README.md`](docs/README.md).
- `tests/`: pruebas automáticas.
- `outputs/`: bancos de filtros, checkpoints, métricas y figuras generadas.
