# Particiones del dataset

Este directorio contiene los manifiestos versionables de entrenamiento, validación y prueba:

```text
splits/
├── train.csv
├── validation.csv
├── test.csv
└── report.json
```

Cada manifiesto debe incluir, como mínimo, las columnas `filename` y `recording_id`. El
identificador `recording_id` se obtiene eliminando el sufijo `_inicio_fin.wav` del nombre del
segmento.

Todos los segmentos con el mismo `recording_id` deben pertenecer a una única partición. Los
manifiestos se generan a partir de `dataset/train.csv`; los audios no se copian ni se mueven.

Estos archivos todavía no se incluyen porque la estrategia de estratificación multietiqueta
debe implementarse según la política ya definida en
[`docs/data-preparation.md`](../docs/data-preparation.md). La proporción objetivo es 80/10/10,
con cobertura mínima por etiqueta principal y semilla `42`. Cuando se generen, deben
versionarse para que todos los experimentos utilicen exactamente la misma separación.

Los manifiestos no duplican las columnas de etiquetas de `dataset/train.csv`. El archivo
versionable `splits/report.json` debe conservar la huella SHA-256 del archivo de metadatos
utilizado y las verificaciones exigidas por la política de preparación de datos.
