# Metodología

Este documento registra el protocolo de datos, FBRS, DLoGNet, entrenamiento y
evaluación. Las decisiones pendientes se resolverán antes de ejecutar el experimento final.

## Objetivo y alcance

El proyecto transfiere a un conjunto de grabaciones de anuros las dos contribuciones principales
del artículo *"Towards accurate bird sound recognition through multi-scale texture-aware modeling"*
(Qin y Huang, npj Acoustics 2025):

- **FBRS**, una representación espectrográfica basada en paquetes wavelet y selección de
  subbandas guiada por energía;
- **DLoGNet**, una CNN que sustituye parte de sus núcleos genéricos por filtros
  Laplacian-of-Gaussian con parámetros aprendibles de orientación `θ` y escala `σ`.

El objetivo no es reproducir los resultados del artículo. El dominio, las etiquetas y las
características del audio son diferentes, de modo que las decisiones del artículo se consideran
puntos de partida y deben justificarse nuevamente para este conjunto de datos. `docs/paper-summary.md`
documenta el método original y `docs/fbrs.md` desarrolla la adaptación de FBRS.

## Conjunto de datos y definición de la tarea

`dataset/`, excluido de Git, contiene grabaciones de anuros con formato AnuraSet. La auditoría
del corte local realizada el 27 de agosto de 2026 y verificada el 28 de agosto de 2026 produjo el
siguiente estado; [`dataset-audit.md`](dataset-audit.md) conserva la huella del CSV, el
procedimiento y los hallazgos completos:

| Propiedad | Valor |
|---|---:|
| Clips | 62,191 |
| Grabaciones independientes | 1,074 |
| Sitios | 4 |
| Duración por clip | 3 s |
| Audio | 22.05 kHz, mono, 16 bits |
| Columnas de etiquetas | 42 |
| Especies con positivos observados | 40 |
| Especies del experimento principal | 31 |
| Máximo de especies por clip | 8 |
| Filas sin positivos en las 42 etiquetas originales | 22,504 (36.185 %) |
| Filas sin positivos en las 31 etiquetas principales | 22,731 (36.550 %) |
| Filas con positivos solo en etiquetas excluidas | 227 |

Los segmentos se encuentran en `dataset/train/*.wav`. El nombre termina en
`_<start>_<end>.wav` y su prefijo identifica la grabación; aunque la forma predominante es
`INCT<site>_<date>_<time>_<start>_<end>.wav`, la auditoría encontró una grabación cuyo prefijo
incluye un componente adicional. `dataset/train.csv` contiene `filename` y las columnas multihot.

La tarea es de clasificación multietiqueta. La taxonomía principal contiene 31 especies con
soporte en al menos 10 grabaciones independientes. Las etiquetas con menor soporte se reservan
para experimentos exploratorios, nueva recolección o un protocolo específico de *few-shot
learning*; `SCIFUS` y `SCINAS` no tienen ejemplos positivos. La selección completa y el
tratamiento de clips con etiquetas fuera de alcance se definen en
[`data-preparation.md`](data-preparation.md).

Las filas sin positivos en la taxonomía activa se conservan como ejemplos totalmente negativos
para esa taxonomía. Esto incluye las 22,504 filas sin ningún positivo original y las 227 filas
con positivos únicamente en etiquetas excluidas. No se asume que representen fondo puro y no se
añade una clase de fondo.

Como el conjunto de datos no está versionado, estas cantidades deben volver a auditarse si cambia
`dataset/train.csv`. Una variación no debe corregirse únicamente en este documento: también se
deben revisar las configuraciones, particiones y pruebas que dependan de ella.

## Flujo previsto

1. Auditar etiquetas y grabaciones y aplicar la política de selección de etiquetas.
2. Crear particiones agrupadas por grabación según [`data-preparation.md`](data-preparation.md)
   y guardar sus manifiestos versionables en `splits/`.
3. Ajustar cualquier preprocesamiento usando únicamente entrenamiento.
4. Entrenar la matriz mínima de comparación formada por CNN y DLoGNet con Mel y FBRS.
5. Seleccionar umbrales con validación.
6. Evaluar una sola vez sobre prueba.

## Particiones y prevención de fugas

Ningún segmento de una misma grabación puede aparecer en más de una partición. La proporción
objetivo es 80/10/10 y cada etiqueta principal debe conservar al menos 6 grabaciones positivas
en entrenamiento, 2 en validación y 2 en prueba. La política completa se define en
[`data-preparation.md`](data-preparation.md); los manifiestos deben almacenarse en `splits/` y
seguir el contrato de [`splits/README.md`](../splits/README.md).

Todo preprocesamiento aprendido, incluido el ajuste del banco FBRS, se estima exclusivamente a
partir de entrenamiento. La selección de umbrales utiliza validación. La partición de prueba queda reservada para
la evaluación final.

## Fuente de verdad del número de clases

`data.num_labels` es la fuente de verdad ejecutable para el número de salidas del clasificador y
vale `31` en los experimentos principales.
Los modelos derivan su dimensión de salida desde ese valor; no se duplica en la sección
`model` de las configuraciones. La definición y auditoría de las etiquetas se documentan en
«Conjunto de datos y definición de la tarea».

## Salida del modelo y función de pérdida

El modelo produce logits independientes para cada etiqueta. Durante el entrenamiento se utiliza
`BCEWithLogitsLoss`, que combina internamente sigmoid y binary cross-entropy de forma
numéricamente estable. No se debe aplicar sigmoid a los logits antes de calcular esta pérdida.

Sigmoid se aplica durante inferencia o evaluación para convertir logits en probabilidades. No se
utilizan softmax ni cross-entropy multiclase porque las etiquetas no son mutuamente excluyentes.

## Baseline CNN + Mel

El baseline utiliza preénfasis `0.97`, ventana Hamming periódica, `n_fft = 1024`, salto de 256
muestras, espectro sin centrado y potencia al cuadrado. La proyección consta de 128 filtros
triangulares sobre escala Mel HTK, sin normalización de área, entre 0 Hz y Nyquist. Se aplica el
logaritmo natural después de limitar la energía inferiormente por `1e-10`. Esta transformación no
ajusta estadísticas con validación o prueba y produce tensores con un canal.

La CNN de referencia contiene tres bloques `Conv2d` 3×3, `BatchNorm2d`, ReLU y `MaxPool2d` 2×2,
con 32, 64 y 128 canales. Un `AdaptiveAvgPool2d`, *dropout* de 0.3 y una capa lineal producen los
31 logits. La dimensión de salida se deriva de `data.num_labels`.

El entrenamiento utiliza Adam, la tasa de aprendizaje declarada en configuración, lotes de 32 y
50 épocas. Se conserva como mejor checkpoint el mínimo de `BCEWithLogitsLoss` sobre validación;
también se guarda el estado de la última época y un historial JSON. La semilla inicializa Python,
NumPy y PyTorch. El dispositivo `auto` prioriza CUDA, luego MPS y finalmente CPU. No se aplican
aumentos ni ponderación de clases en este baseline inicial.

Cada checkpoint conserva la huella de la configuración semántica y las huellas SHA-256 de los
metadatos y manifiestos. La configuración semántica excluye el dispositivo, el número de workers y
las rutas de salida porque no alteran la definición del modelo ni del experimento. Esto permite
cambiar de CPU, MPS o CUDA durante evaluación sin aceptar cambios silenciosos en datos, particiones,
representación, arquitectura o hiperparámetros.

## Seguimiento de experimentos con MLflow

MLflow Tracking se utiliza como una capa operacional opcional. Cada entrenamiento crea un run y
guarda su identificador en `best.pt`, `last.pt` y `history.json`; la evaluación reanuda ese mismo
run para incorporar las métricas de validación y prueba. Se registran la configuración efectiva,
las huellas de datos, pérdidas por época, duración, métricas finales y los reportes JSON pequeños.
Los checkpoints no se copian al almacén de artefactos de MLflow para evitar duplicar binarios.

La sección `tracking` de la configuración no forma parte de la huella semántica del experimento:
activar, desactivar o reubicar la telemetría no cambia el modelo ni invalida un checkpoint. Un
fallo de importación, conexión o escritura de MLflow emite una advertencia, pero no interrumpe el
entrenamiento ni la evaluación. Los archivos bajo `outputs/checkpoints/` y `outputs/metrics/`
siguen siendo los artefactos canónicos del proyecto.

## Evaluación

La evaluación debe reportar precisión, exhaustividad y F1 por clase y sus agregaciones macro,
además de precisión media promedio (mAP). La exactitud global no se utiliza como métrica principal porque la
tarea es multietiqueta y contiene una proporción elevada de ejemplos totalmente negativos.

Los umbrales de decisión se seleccionan por clase maximizando F1 sobre validación, según
`evaluation.threshold_strategy` en la configuración. El cálculo de average precision se realiza
sobre probabilidades o puntuaciones continuas y no depende de esos umbrales. Las restricciones de
las particiones garantizan positivos de cada etiqueta principal en validación y prueba. Si una
clase principal no tiene positivos en alguna de esas particiones, la evaluación debe detenerse y
reportar que los manifiestos son inválidos; no debe excluir silenciosamente esa clase de las
métricas macro o de mAP. Esta política se declara mediante
`evaluation.zero_positive_class_policy: error`.

El artefacto de evaluación conserva los umbrales, las métricas de validación y prueba, el orden de
etiquetas, las huellas de los datos y las huellas de configuración y checkpoint. Antes de inferir,
la evaluación exige que los metadatos y todos los manifiestos coincidan por contenido con los
registrados durante entrenamiento. Precisión, exhaustividad, F1 y average precision se reportan
por clase; sus tres primeros agregados son medias macro y mAP es la media de average precision. El
comando de evaluación es la única etapa del baseline que carga ejemplos de prueba.

## Adaptación de FBRS y DLoGNet

La distribución de bandas FBRS debe derivarse del intervalo de Nyquist del audio del proyecto.
Para una descomposición de nivel `L`, la anchura mínima estándar es

$$
\Delta f_{\min} = \frac{f_s/2}{2^L} = \frac{f_s}{2^{L+1}}.
$$

El nivel utilizado por el artículo es un punto de partida, no una decisión transferible sin
validación. Los parámetros vigentes de la etapa inicial y del banco se declaran en
`configs/dlognet_fbrs.yaml`; su justificación, las decisiones pendientes y las alternativas de
ablación se describen en `docs/fbrs.md`.

### Adaptación de DLoGNet

Las representaciones Mel y FBRS producen entradas de un canal con 128 bandas y 255 marcos para
los clips de tres segundos. DLoGNet conserva las cinco etapas del artículo, con canales
`[64, 128, 128, 128, 64]`. Cada etapa aplica un BDCM, `BatchNorm2d`, ReLU y agrupamiento máximo
2×2. Las dimensiones espaciales pasan por `128×255`, `64×127`, `32×63`, `16×31`, `8×15` y
`4×7`; por tanto, las cinco reducciones son válidas sin redimensionar ni rellenar la entrada.
Después se utiliza agrupamiento promedio adaptativo global, una capa oculta de 1024 unidades,
ReLU, *dropout* de 0.3 y una capa lineal de 31 logits.

Cada BDCM contiene cuatro filtros DLoG *depthwise* inicializados en 0°, 45°, 90° y 135°. Un mismo
kernel físico se comparte entre los canales de una rama para conservar la parametrización
interpretable. Las cuatro respuestas y la entrada sin filtrar se concatenan por canales; una
convolución aprendible 3×3 realiza la fusión. Se adopta concatenación porque la descripción del
artículo la denomina conexión de salto y las dimensiones publicadas no permiten sumar directamente
los cuatro grupos direccionales con la entrada.

El kernel DLoG se calcula como la segunda derivada direccional de una Gaussiana discreta 7×7.
Los ángulos `θ` y una escala `σ` por BDCM reciben gradientes. `σ` se inicializa en 1.0 y se
parametriza como `softplus(raw_sigma) + 0.3` para impedir escalas nulas o negativas. Después de
discretizar se resta la media de cada kernel para conservar respuesta DC nula y se normaliza por
su norma L1, lo que evita que un cambio de escala altere la magnitud únicamente por el
truncamiento. El artículo no fija tamaño, normalización ni cota de escala; estos valores son
decisiones reproducibles de la adaptación y deben tratarse como hiperparámetros en una ablación.

La salida son logits independientes y la pérdida continúa siendo `BCEWithLogitsLoss`. No se
aplican el *softmax* ni la entropía cruzada monoclase del artículo. Las dos configuraciones
DLoGNet solo difieren en la representación de entrada y derivan las 31 salidas de
`data.num_labels`.

El resultado de exactitud publicado por el artículo no es un objetivo comparable: corresponde a
otro dominio, otra definición de tarea y otra métrica. Las comparaciones del proyecto deben
realizarse entre modelos entrenados con las mismas particiones y el mismo protocolo de
evaluación. Como mínimo se evalúa la matriz factorial CNN/DLoGNet × Mel/FBRS para no atribuir a la
representación un cambio que también pueda deberse a la arquitectura, o viceversa.

El banco FBRS global se ajusta mediante `python -m anuraset_dl.fbrs` y se persiste en la ruta
`features.bank_path`. La selección comienza con las hojas del nivel configurado y fusiona, de
forma determinista, el par de hermanos activo con menor energía agregada hasta alcanzar
`target_bands`. Los filtros triangulares se construyen sobre los centros de las bandas, se
solapan hasta los centros adyacentes y se normalizan a pico unitario. Entrenamiento y evaluación
cargan el mismo artefacto congelado y rechazan cambios en su firma, los metadatos o el manifiesto
de entrenamiento.
