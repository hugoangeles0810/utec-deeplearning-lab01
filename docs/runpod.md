# Ejecución en Runpod

Este documento describe el flujo operacional para ejecutar la matriz experimental completa en un
Pod con CUDA. Runpod aporta el cómputo y el almacenamiento temporal; no modifica el protocolo
metodológico, las particiones ni las configuraciones semánticas del proyecto.

## Recursos recomendados

La infraestructura se crea manualmente desde la consola de Runpod:

- template oficial **Runpod PyTorch 2.4.0**, basado en
  `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`;
- una GPU NVIDIA RTX A5000 de 24 GB; una RTX A4500 de 20 GB o RTX 3090 de 24 GB son
  alternativas compatibles;
- **SSH Terminal Access** habilitado;
- container disk de al menos 20 GB;
- almacenamiento persistente de 60 GB montado en `/workspace`, mediante **Volume Disk** o
  **Network Volume**.

El Python 3.11 y PyTorch 2.4.0 incluidos en el template sirven únicamente para construir la
imagen base y no se reutilizan como entorno del proyecto. `bootstrap.sh` instala Python 3.12 en
`/opt/anuraset-venv` y sincroniza PyTorch 2.6.0 desde el índice oficial CUDA 12.4 fijado en
`pyproject.toml` y `uv.lock`. El host debe exponer un driver NVIDIA compatible con CUDA 12.4;
puede comprobarse con `nvidia-smi` antes de ejecutar el bootstrap.

El flujo es agnóstico al tipo de volumen: solo exige que `/workspace` exista y tenga al menos
45 GB libres. Un Volume Disk conserva los archivos al detener o reiniciar el mismo Pod, pero los
elimina al terminarlo. Un Network Volume sobrevive independientemente del Pod y puede conectarse
a otro Pod compatible. Para una ejecución completa en una sola máquina puede utilizarse Volume
Disk; para cambiar de máquina o conservar los artefactos después de terminarla debe utilizarse
Network Volume o exportarse previamente el paquete de resultados.

El entorno virtual se crea en `/opt/anuraset-venv`, sobre el container disk, porque puede
regenerarse desde `uv.lock` y no necesita ocupar el almacenamiento persistente. Si el container
disk se reinicializa, basta con volver a ejecutar `bootstrap.sh`; el repositorio, dataset, cachés
y resultados permanecen en `/workspace` mientras el volumen correspondiente siga existiendo.

El dataset de entrenamiento ocupa aproximadamente 8.23 GB y cada una de las cachés Mel y FBRS
ocupa unos 8.12 GB. El aprovisionamiento requiere espacio temporal adicional mientras extrae el
archivo `train.7z`. El flujo comprueba un mínimo de 45 GB libres y recomienda 60 GB.

## Clonar y preparar el entorno

Después de crear el Pod, abrir su terminal web o conectarse mediante el comando SSH mostrado en
la pestaña **Connect**. Clonar el repositorio dentro del volumen persistente:

```bash
cd /workspace
git clone URL_DEL_REPOSITORIO utec-deeplearning-lab01
cd /workspace/utec-deeplearning-lab01
```

Si el repositorio es privado, debe utilizarse una deploy key o el mecanismo de autenticación
institucional. No deben almacenarse tokens dentro del repositorio.

Preparar y verificar el entorno:

```bash
scripts/runpod/bootstrap.sh
```

El script instala las dependencias operativas, Python 3.12 y el entorno bloqueado por `uv.lock`.
Después comprueba PyTorch 2.6.0, el runtime CUDA 12.4, la disponibilidad efectiva de la GPU y
ejecuta `pytest` y `ruff`. Es idempotente y puede volver a ejecutarse al crear otro Pod sobre el
mismo Network Volume o al reiniciar un Pod cuyo Volume Disk se conserve.

## Ejecución completa

El comando predeterminado aprovisiona únicamente entrenamiento y ejecuta, en este orden:

1. validación del dataset y de las particiones congeladas, sin regenerarlas;
2. caché Mel;
3. CNN + Mel y su evaluación;
4. ajuste del banco FBRS exclusivamente sobre entrenamiento;
5. caché FBRS;
6. CNN + FBRS y su evaluación;
7. DLoGNet + Mel y su evaluación;
8. DLoGNet + FBRS y su evaluación;
9. paquete verificable de resultados.

Se recomienda iniciar el flujo dentro de `tmux` para que una desconexión SSH no interrumpa el
proceso:

```bash
tmux new -s anuraset
scripts/runpod/run.sh
```

Para salir sin detenerlo se utiliza `Ctrl+B` seguido de `D`. Para volver a la sesión:

```bash
tmux attach -t anuraset
```

Los logs por etapa y el estado atómico del pipeline quedan bajo:

```text
outputs/runpod/
├── pipeline.json
└── logs/
```

## Selección de experimentos

Puede ejecutarse una sola celda de la matriz:

```bash
scripts/runpod/run.sh --experiments dlognet_mel
```

O una selección:

```bash
scripts/runpod/run.sh \
  --experiments cnn_mel_baseline dlognet_mel
```

El orquestador deriva las dependencias. Las variantes Mel comparten su caché y las variantes FBRS
comparten el banco congelado y su caché. `--num-workers` y `--device` son opciones operativas y no
forman parte de la huella semántica. Los workers se aplican tanto a la precomputación paralela de
características como a la carga durante entrenamiento y evaluación:

```bash
scripts/runpod/run.sh --num-workers 4 --device cuda
```

El pipeline no reanuda un entrenamiento interrumpido. Si encuentra artefactos parciales o
incompatibles, se detiene sin sobrescribirlos. Si el entrenamiento terminó y solo falta la
evaluación, reutiliza los checkpoints y ejecuta la evaluación. Un experimento completo se omite.

## Contenido de la exportación

Al terminar, el wrapper guarda en `/workspace/exports`:

```text
anuraset-results-<fecha>.tar.gz
anuraset-results-<fecha>.tar.gz.sha256
```

El paquete incluye:

- `best.pt`, `last.pt` e historiales de los modelos;
- banco FBRS;
- métricas y umbrales;
- base y artefactos pequeños de MLflow;
- configuraciones, estado del pipeline y logs;
- commit Git, estado del repositorio, versiones de Python/PyTorch/CUDA;
- manifiesto con tamaño y SHA-256 de cada archivo.

No incluye el dataset ni `outputs/features/`: contienen datos o artefactos regenerables y las dos
cachés sumarían aproximadamente 16.24 GB.

Para verificar un paquete descargado:

```bash
sha256sum -c anuraset-results-<fecha>.tar.gz.sha256
tar -tzf anuraset-results-<fecha>.tar.gz | less
```

## Respaldo en Google Drive

La primera vez se configura un remoto de Google Drive de forma interactiva:

```bash
rclone config
```

La configuración y el token OAuth permanecen fuera de Git. Después puede indicarse el destino al
ejecutar todo el pipeline:

```bash
scripts/runpod/run.sh \
  --drive-destination gdrive:anuraset-dl/resultados
```

Para volver a empaquetar y subir resultados existentes sin entrenar:

```bash
scripts/runpod/export-results.sh gdrive:anuraset-dl/resultados
```

Se utiliza `rclone copy`, no `sync`, para no eliminar archivos remotos.

## Descarga directa

Con SSH sobre TCP e IP pública, el comando exacto se obtiene en la pestaña **Connect**. Desde el
equipo local puede descargarse el paquete con:

```bash
scp -P PUERTO -i RUTA_CLAVE \
  root@IP_DEL_POD:/workspace/exports/anuraset-results-<fecha>.tar.gz \
  DESTINO_LOCAL
```

Debe descargarse también el `.sha256`. Para transferencias ocasionales puede utilizarse
`runpodctl send` desde el Pod y `runpodctl receive` en el equipo de destino.

## Finalización segura

Antes de eliminar recursos:

1. comprobar que `outputs/runpod/pipeline.json` tiene estado `completed`;
2. verificar la existencia de checkpoints y métricas de los experimentos seleccionados;
3. copiar el paquete y su SHA-256 a Google Drive o al equipo local;
4. verificar la copia descargada;
5. detener el Pod para detener el cobro de GPU;
6. si se utiliza Volume Disk, no terminar el Pod hasta verificar el respaldo, porque esa acción
   elimina el volumen junto con sus datos;
7. si se utiliza Network Volume, terminar el Pod después del respaldo y conservar el volumen
   únicamente mientras queden experimentos pendientes;
8. eliminar el almacenamiento persistente al concluir el proyecto para detener su costo.

## Portabilidad

`anuraset_dl.run_experiments` y `anuraset_dl.package_results` no dependen de la API de Runpod.
También pueden utilizarse en una VM Linux con CUDA de Google Compute Engine, AWS EC2 u otro
proveedor. Solo `scripts/runpod/bootstrap.sh` contiene supuestos específicos del Pod y de su punto
de montaje `/workspace`.
