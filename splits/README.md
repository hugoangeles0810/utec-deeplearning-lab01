# Particiones del dataset

Este directorio contiene los manifiestos versionables de entrenamiento y validación:

```text
splits/
├── train.csv
├── validation.csv
└── report.json
```

Cada manifiesto debe incluir, como mínimo, las columnas `filename` y `recording_id`. El
identificador `recording_id` se obtiene eliminando el sufijo `_inicio_fin.wav` del nombre del
segmento.

Todos los segmentos con el mismo `recording_id` deben pertenecer a una única partición. Los
manifiestos se generan a partir de `dataset/train.csv`; los audios no se copian ni se mueven.

Los manifiestos vigentes se generaron mediante optimización MILP lexicográfica según la política
definida en [`docs/data-preparation.md`](../docs/data-preparation.md). Utilizan semilla `42`,
cubren los 62,191 clips y separan las 1,074 grabaciones de esta forma:

| Partición | Grabaciones | Clips |
|---|---:|---:|
| Entrenamiento | 859 | 49,753 |
| Validación | 215 | 12,438 |

La asignación satisface la cobertura 8/2 de las 31 etiquetas principales y presencia 1/1
de cada etiqueta exploratoria. `report.json` contiene los conteos detallados, las huellas de los
artefactos y los resultados de todas las validaciones.

Los audios externos sin etiquetas de `dataset/test/` no forman parte de estos manifiestos. Se
procesan únicamente mediante el comando de inferencia después de congelar modelo y umbrales.

Los manifiestos no duplican las columnas de etiquetas de `dataset/train.csv`. El archivo
versionable `splits/report.json` debe conservar la huella SHA-256 del archivo de metadatos
utilizado y las verificaciones exigidas por la política de preparación de datos.

Para comprobar la reproducibilidad sin modificar resultados idénticos:

```bash
uv run python -m anuraset_dl.prepare_data --config configs/baseline.yaml
```
