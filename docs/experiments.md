# Registro de experimentos

Cada experimento debe registrar como mínimo:

- configuración utilizada;
- semilla aleatoria;
- archivos de partición utilizados desde `splits/`;
- versión o alcance del banco FBRS;
- métricas por clase y agregadas;
- ruta de los artefactos generados;
- identificador del run de MLflow cuando el tracking esté habilitado;
- dispositivo y duración del entrenamiento y la evaluación;
- observaciones y anomalías.

| Experimento | Representación | Modelo | Estado | Resultado |
|---|---|---|---|---|
| `cnn_mel_baseline` | Mel | CNN | Ejecutado | F1 macro test 0.5070; mAP test 0.5570 |
| `cnn_fbrs` | FBRS | CNN | Ejecutado | F1 macro test 0.4977; mAP test 0.5303 |
| `dlognet_mel` | Mel | DLoGNet | Pipeline implementado; ejecución pendiente | — |
| `dlognet_fbrs` | FBRS | DLoGNet | Pipeline implementado; ejecución pendiente | — |

Estos cuatro experimentos forman la matriz factorial mínima para separar el efecto de la
representación del efecto de la arquitectura. Todas las celdas deben utilizar las mismas
particiones y el mismo protocolo de entrenamiento y evaluación; solo deben variar los factores
indicados por su fila.

## Estado de DLoGNet

El pipeline implementa kernels DLoG diferenciables con cuatro orientaciones y escala aprendible,
cinco BDCM, conexión de identidad por concatenación y clasificador multietiqueta. Las pruebas
comprueban respuesta DC nula, orientaciones distintas, gradientes finitos para `θ` y `σ`, forma de
los bloques, serialización mediante checkpoint y ejecuciones sintéticas completas tanto con Mel
como con un banco FBRS congelado. Las decisiones exactas de adaptación se registran en
[`methodology.md`](methodology.md#adaptación-de-dlognet).

Las ejecuciones de 50 épocas todavía no se han realizado. Debe ejecutarse primero
`dlognet_mel` para medir el efecto de la arquitectura sin cambiar simultáneamente la
representación y después `dlognet_fbrs`, reutilizando el banco ya ajustado únicamente con
entrenamiento.

## Estado del baseline CNN + Mel

El pipeline ejecutable incluye carga por manifiestos, transformación log-Mel, CNN, entrenamiento
con checkpoints, selección de umbrales en validación y evaluación separada sobre prueba. Su prueba
de integración utiliza un corpus sintético pequeño y no constituye un resultado experimental.

El seguimiento local usa el experimento MLflow `anuraset_dl`. Entrenamiento y evaluación deben
compartir el identificador almacenado en el checkpoint. MLflow complementa el registro documental:
no sustituye este archivo ni convierte una prueba sintética en una ejecución experimental.

La ejecución de 50 épocas sobre el corpus auditado se realizó el 28 de agosto de 2026 en MPS. El
mejor checkpoint corresponde a la época 46, con pérdida de validación `0.080517`. El entrenamiento
duró `4607.06` segundos y la evaluación `13.69` segundos. En validación se obtuvo F1 macro
`0.558183` y mAP `0.554192`; en prueba, F1 macro `0.506981` y mAP `0.557045`. Los artefactos
canónicos se encuentran en `outputs/checkpoints/cnn_mel_baseline/` y
`outputs/metrics/cnn_mel_baseline.json`; el run de MLflow es
`534b0f7702cc4835bcbfbbe084323ee0`. No se registraron anomalías de ejecución.

## Estado de CNN + FBRS

La implementación se validó antes de la ejecución con las 50 pruebas del proyecto y `ruff`. Las
pruebas específicas de FBRS comprueban conservación y normalización de energía, cobertura y orden
de la partición, restricción de hermanos, localización de tonos puros, estabilidad ante silencio,
reproducibilidad, serialización, rechazo de un manifiesto de entrenamiento alterado y una ejecución
sintética de extremo a extremo. Todas las verificaciones terminaron correctamente.

El banco global se ajustó entre el 28 y el 29 de agosto de 2026 con
`configs/cnn_fbrs.yaml`, semilla `42` y exclusivamente los 31,573 clips de entrenamiento con al
menos un positivo en la taxonomía activa (`active_positive_training`). No se utilizaron clips de
validación ni prueba. El artefacto contiene 128 filtros finitos y no vacíos sobre 513 bins, cubre
de 0 a 11,025 Hz y tiene SHA-256
`3497f1e4ddcb491c02fc76bfdf786b91be3905d3039b34bf573770f0334c4584`. Se encuentra en
`outputs/filterbanks/fbrs_db16_l8_b128.pt` y está vinculado por huella al CSV de metadatos y a
`splits/train.csv`.

El entrenamiento completo de 50 épocas se ejecutó en MPS con los manifiestos
`splits/train.csv`, `splits/validation.csv` y `splits/test.csv`. Duró `4958.01` segundos. El mejor
checkpoint fue el de la época 41, con pérdida de validación `0.078872`; la evaluación duró
`13.37` segundos. Los artefactos canónicos se encuentran en
`outputs/checkpoints/cnn_fbrs/` y `outputs/metrics/cnn_fbrs.json`; el run de MLflow es
`4715f505c6a64586bb4a9a649bf4a6bd`.

La comparación agregada contra `cnn_mel_baseline`, con las mismas etiquetas, particiones,
arquitectura y protocolo de umbrales, es:

| Partición | Métrica | Mel | FBRS | Diferencia FBRS − Mel |
|---|---|---:|---:|---:|
| Validación | Precisión macro | 0.549535 | 0.581089 | +0.031555 |
| Validación | Exhaustividad macro | 0.660394 | 0.693056 | +0.032662 |
| Validación | F1 macro | 0.558183 | 0.599570 | +0.041387 |
| Validación | mAP | 0.554192 | 0.581325 | +0.027133 |
| Prueba | Precisión macro | 0.503555 | 0.464745 | −0.038810 |
| Prueba | Exhaustividad macro | 0.683529 | 0.643120 | −0.040409 |
| Prueba | F1 macro | 0.506981 | 0.497746 | −0.009235 |
| Prueba | mAP | 0.557045 | 0.530344 | −0.026701 |

FBRS mejora las cuatro métricas agregadas de validación, pero no transfiere esa mejora a prueba:
allí reduce tanto F1 macro como mAP. Por tanto, esta ejecución no demuestra una mejora de
generalización de FBRS sobre Mel con la CNN. El entrenamiento FBRS fue `350.95` segundos
(`7.62 %`) más lento; la duración de evaluación fue similar.

Las principales limitaciones son que existe una sola ejecución por representación y no pueden
estimarse variabilidad ni significancia; el banco solo se comparó en su variante inicial
`active_positive_training`, sin la ablación con todo entrenamiento; tampoco se han barrido nivel,
número de bandas o forma de filtro. La selección del mejor checkpoint y de los umbrales usa la
misma validación, y la divergencia entre validación y prueba aconseja no ajustar retrospectivamente
estas decisiones a partir del resultado de prueba. El cálculo FBRS se realiza en línea, sin caché
y con `num_workers: 0`, lo que limita la eficiencia. MPS no aportó métricas de GPU a MLflow, aunque
el run, los parámetros, tiempos y resultados sí quedaron registrados.
