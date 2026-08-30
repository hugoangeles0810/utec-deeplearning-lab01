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
| `cnn_mel_baseline` | Mel | CNN | Completado | F1 macro 0.5315; mAP 0.5188 |
| `cnn_fbrs` | FBRS | CNN | Completado | F1 macro 0.5199; mAP 0.5229 |
| `dlognet_mel` | Mel | DLoGNet | Completado | F1 macro 0.6564; mAP 0.6601 |
| `dlognet_fbrs` | FBRS | DLoGNet | Completado | F1 macro 0.5603; mAP 0.5659 |

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
[`methodology.md`](methodology.md#adaptación-de-dlognet) y su fundamento técnico se desarrolla en
[`dlognet.md`](dlognet.md).

Las ejecuciones de 50 épocas se realizaron con el protocolo vigente el 30 de agosto de 2026, en
el orden `cnn_mel_baseline`, `cnn_fbrs`, `dlognet_mel` y `dlognet_fbrs`. Las variantes Mel
compartieron su caché; las variantes FBRS reutilizaron un banco ajustado únicamente con
entrenamiento y su caché correspondiente.

## Protocolo vigente y estado de los artefactos

El protocolo divide `dataset/train/` en entrenamiento y validación 80/20. El test es un conjunto
externo sin etiquetas y se utiliza únicamente para inferencia. La ejecución aceptada del 30 de
agosto de 2026 produjo checkpoints, banco FBRS, métricas y runs de MLflow para las cuatro celdas
de la matriz; no se ejecutó inferencia sobre el test externo.

El pipeline ejecutable incluye carga por manifiestos, transformaciones log-Mel y FBRS, modelos
CNN y DLoGNet, entrenamiento con checkpoints, selección de umbrales y evaluación interna sobre
validación. Las pruebas sintéticas verifican estos componentes, pero no constituyen resultados
experimentales.

El seguimiento local utiliza el experimento MLflow `anuraset_dl`. Entrenamiento y evaluación
deben compartir el identificador almacenado en el checkpoint. MLflow complementa el registro
documental: no sustituye este archivo ni convierte una prueba sintética en una ejecución
experimental.

## Ejecución remota

La matriz puede ejecutarse en un host CUDA mediante el flujo documentado en
[`runpod.md`](runpod.md). Runpod es una decisión operacional: el dispositivo, el número de workers,
las rutas de exportación y el mecanismo de respaldo no cambian la definición semántica de los
experimentos. Cada ejecución aceptada debe registrar aquí sus duraciones, métricas, anomalías y
las rutas de sus artefactos canónicos después de verificar el paquete exportado.

## Ejecución aceptada del 30 de agosto de 2026

La matriz completa se ejecutó con semilla `42` sobre una NVIDIA RTX A4500, Python 3.12.14,
PyTorch 2.6.0 y CUDA 12.4. El commit remoto fue
`327ed3aa126cdd8d40b293b40eb7bd1a3e931989` y el árbol de trabajo estaba limpio. Los cuatro
experimentos utilizaron las mismas huellas de metadatos y particiones:

- metadatos: `38df5d408d9bf621cc11f78ed4cd766be63a8406512c1eb4d611d98e4486d276`;
- entrenamiento: `20e458b8e1f732175280c5a8b32e7615d4c075146d19d7a7bc17a921075365fd`;
- validación: `46c633f11e2ae25178194f18424fcf4945b65e9d368a83929228ddde466aba67`.

| Experimento | Mejor época | Entrenamiento | Evaluación | Precisión macro | Recall macro | F1 macro | mAP | Run de MLflow |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `cnn_mel_baseline` | 50 | 23 min 2 s | 14 s | 0.5255 | 0.6366 | 0.5315 | 0.5188 | `98b38ac7241d4a819c329170313b4eb2` |
| `cnn_fbrs` | 50 | 22 min 46 s | 14 s | 0.4916 | 0.6450 | 0.5199 | 0.5229 | `760112c92abb48b7af11217527e21d79` |
| `dlognet_mel` | 4 | 3 h 3 min 46 s | 24 s | 0.6924 | 0.6703 | 0.6564 | 0.6601 | `55eec69fe0814de4ac32e8e08b464f98` |
| `dlognet_fbrs` | 2 | 3 h 3 min 23 s | 24 s | 0.5553 | 0.6804 | 0.5603 | 0.5659 | `fec4c997efce4e679b8404a01884009a` |

El paquete descargado se conserva en
`outputs/exports/anuraset-results-20260830T132819Z.tar.gz` y su copia extraída en
`outputs/exports/anuraset-results-20260830T132819Z/`. La huella SHA-256 del paquete es
`9b16e14f7ee1b2bea7de88ca6a3156828f69d56fda5bff6a3efed5e13b57104e`. Se verificaron el
archivo `.sha256` y las huellas de los 59 archivos del manifiesto interno.

### Observaciones

- `dlognet_mel` fue el mejor resultado de esta ejecución: superó a `cnn_mel_baseline` en 0.1249
  de F1 macro y 0.1413 de mAP.
- FBRS no mejoró F1 macro frente a Mel con ninguna arquitectura. Con la CNN redujo F1 macro en
  0.0116, aunque elevó mAP en 0.0041; con DLoGNet redujo F1 macro en 0.0961 y mAP en 0.0942.
- Las dos CNN alcanzaron su menor pérdida de validación en la época 50. En cambio, las dos
  variantes DLoGNet mostraron sobreajuste temprano: `dlognet_mel` seleccionó la época 4 y
  `dlognet_fbrs` la época 2. Las evaluaciones referencian por huella los respectivos `best.pt`,
  por lo que el deterioro posterior no contaminó las métricas reportadas. Conviene incorporar
  *early stopping* antes de repetir esta matriz.
- Las 31 clases obtuvieron F1 finito y distinto de cero en los cuatro experimentos; no se
  detectaron `NaN`, infinitos, errores CUDA ni artefactos incompletos.
- `validate_data.log` es acumulativo y conserva dos trazas de intentos fallidos anteriores. El
  intento aceptado finaliza en el mismo archivo con la verificación correcta de 49,753 clips de
  entrenamiento y 12,438 de validación; `pipeline.json` marca las 13 etapas como completadas.
