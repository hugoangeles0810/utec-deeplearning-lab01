# Registro de experimentos

Cada experimento debe registrar como mínimo:

- configuración utilizada;
- semilla aleatoria;
- archivos de partición utilizados desde `splits/`;
- versión o alcance del banco FBRS;
- métricas por clase y agregadas;
- ruta de los artefactos generados;
- observaciones y anomalías.

| Experimento | Representación | Modelo | Estado | Resultado |
|---|---|---|---|---|
| `cnn_mel_baseline` | Mel | CNN | Pipeline implementado; ejecución pendiente | — |
| `cnn_fbrs` | FBRS | CNN | Pendiente | — |
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

La ejecución de 50 épocas sobre el corpus auditado sigue pendiente. Al realizarla deben añadirse
a este registro la fecha, dispositivo, duración, mejor época, pérdidas, métricas, rutas de los
artefactos y cualquier anomalía observada; hasta entonces la celda no se considera ejecutada.
