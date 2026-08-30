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
| `cnn_mel_baseline` | Mel | CNN | Ejecución pendiente | — |
| `cnn_fbrs` | FBRS | CNN | Ejecución pendiente | — |
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

Las ejecuciones de 50 épocas todavía no se han realizado con el protocolo vigente. Debe
ejecutarse primero `cnn_mel_baseline`, seguido de `cnn_fbrs`, `dlognet_mel` y `dlognet_fbrs`.
Las variantes Mel comparten su caché; las variantes FBRS reutilizan un banco ajustado únicamente
con entrenamiento y su caché correspondiente.

## Protocolo vigente y estado de los artefactos

El protocolo divide `dataset/train/` en entrenamiento y validación 80/20. El test es un conjunto
externo sin etiquetas y se utiliza únicamente para inferencia. No existen checkpoints, bancos
FBRS, cachés de representaciones, métricas ni runs de MLflow aceptados para este protocolo.

El pipeline ejecutable incluye carga por manifiestos, transformaciones log-Mel y FBRS, modelos
CNN y DLoGNet, entrenamiento con checkpoints, selección de umbrales y evaluación interna sobre
validación. Las pruebas sintéticas verifican estos componentes, pero no constituyen resultados
experimentales.

El seguimiento local utiliza el experimento MLflow `anuraset_dl`. Entrenamiento y evaluación
deben compartir el identificador almacenado en el checkpoint. MLflow complementa el registro
documental: no sustituye este archivo ni convierte una prueba sintética en una ejecución
experimental.
