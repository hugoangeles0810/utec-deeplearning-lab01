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
| `cnn_fbrs` | FBRS | CNN | Pipeline implementado; ejecución pendiente | — |
| `dlognet_mel` | Mel | DLoGNet | Pendiente | — |
| `dlognet_fbrs` | FBRS | DLoGNet | Pendiente | — |

Estos cuatro experimentos forman la matriz factorial mínima para separar el efecto de la
representación del efecto de la arquitectura. Todas las celdas deben utilizar las mismas
particiones y el mismo protocolo de entrenamiento y evaluación; solo deben variar los factores
indicados por su fila.

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

El pipeline implementa el ajuste reproducible del banco global con ejemplos positivos de
entrenamiento, su serialización y validación, la transformación FBRS y el entrenamiento y la
evaluación con la CNN del baseline. Falta ajustar el banco sobre el corpus completo y ejecutar
las 50 épocas de `configs/cnn_fbrs.yaml`; por tanto, todavía no existen resultados comparables.
