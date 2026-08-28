# Particiones del dataset

Este directorio contiene los manifiestos versionables de entrenamiento, validación y prueba:

```text
splits/
├── train.csv
├── validation.csv
└── test.csv
```

Cada manifiesto debe incluir, como mínimo, las columnas `filename` y `recording_id`. El
identificador `recording_id` se obtiene eliminando el sufijo `_inicio_fin.wav` del nombre del
segmento.

Todos los segmentos con el mismo `recording_id` deben pertenecer a una única partición. Los
manifiestos se generan a partir de `dataset/train.csv`; los audios no se copian ni se mueven.

Estos archivos todavía no se incluyen porque la estrategia de estratificación multietiqueta
debe definirse antes de crear la partición definitiva. Cuando se generen, deben versionarse
para que todos los experimentos utilicen exactamente la misma separación.
