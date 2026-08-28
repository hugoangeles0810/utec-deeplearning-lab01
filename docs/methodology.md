# Metodología

Este documento registra el protocolo de datos, FBRS, DLoGNet, entrenamiento y
evaluación. Las decisiones pendientes se resolverán antes de ejecutar el experimento final.

## Objetivo y alcance

El proyecto transfiere a un conjunto de grabaciones de anuros las dos contribuciones principales
de *"Towards accurate bird sound recognition through multi-scale texture-aware modeling"*
(Qin y Huang, npj Acoustics 2025):

- **FBRS**, una representación espectrográfica basada en paquetes wavelet y selección de
  subbandas guiada por energía;
- **DLoGNet**, una CNN cuyos kernels se construyen con filtros Laplacian-of-Gaussian y parámetros
  aprendibles de orientación `θ` y escala `σ`.

El objetivo no es reproducir los resultados del paper. El dominio, las etiquetas y las
características del audio son diferentes, de modo que las decisiones del paper se consideran
puntos de partida y deben justificarse nuevamente para este dataset. `docs/paper-summary.md`
documenta el método original y `docs/fbrs-explained.md` desarrolla la adaptación de FBRS.

## Dataset y definición de la tarea

`dataset/`, excluido de Git, contiene grabaciones de anuros con formato AnuraSet. La auditoría
del CSV local realizada el 27 de agosto de 2026 produjo el siguiente estado:

| Propiedad | Valor |
|---|---:|
| Clips | 62,191 |
| Duración por clip | 3 s |
| Audio | 22.05 kHz, mono, 16 bits |
| Columnas de etiquetas | 42 |
| Especies objetivo observadas | 40 |
| Máximo de especies por clip | 8 |
| Filas sin etiquetas positivas | 22,504 (36.185 %) |

Los segmentos se encuentran en `dataset/train/*.wav` y siguen el patrón
`INCT<site>_<date>_<time>_<start>_<end>.wav`. `dataset/train.csv` contiene `filename` y las
columnas multihot.

La tarea es de clasificación multietiqueta. `SCIFUS` y `SCINAS` no tienen ejemplos positivos y
se excluyen de la salida entrenable. Las filas sin etiquetas positivas se conservan como
ejemplos totalmente negativos: no se asume que representen fondo puro y no se añade una clase
de fondo.

Como el dataset no está versionado, estas cantidades deben volver a auditarse si cambia
`dataset/train.csv`. Una variación no debe corregirse únicamente en este documento: también se
deben revisar las configuraciones, particiones y pruebas que dependan de ella.

## Flujo previsto

1. Auditar etiquetas y grabaciones.
2. Crear particiones agrupadas por grabación y guardar sus manifiestos versionables en
   `splits/`.
3. Ajustar cualquier preprocesamiento usando únicamente entrenamiento.
4. Entrenar los modelos de referencia y DLoGNet.
5. Seleccionar umbrales con validación.
6. Evaluar una sola vez sobre test.

## Particiones y prevención de fugas

Ningún segmento de una misma grabación puede aparecer en más de una partición. Los manifiestos
de entrenamiento, validación y prueba deben almacenarse en `splits/` y seguir el contrato
definido en `splits/README.md`.

Todo preprocesamiento aprendido, incluido el ajuste del banco FBRS, se estima exclusivamente a
partir de entrenamiento. La selección de umbrales utiliza validación. Test queda reservado para
la evaluación final.

## Fuente de verdad del número de clases

`data.num_labels` es la fuente de verdad ejecutable para el número de salidas del clasificador.
Los modelos deben derivar su dimensión de salida desde ese valor; no se duplica en la sección
`model` de las configuraciones. La definición y auditoría de las etiquetas se documentan en
«Dataset y definición de la tarea».

## Salida del modelo y función de pérdida

El modelo produce logits independientes para cada etiqueta. Durante el entrenamiento se utiliza
`BCEWithLogitsLoss`, que combina internamente sigmoid y binary cross-entropy de forma
numéricamente estable. No se debe aplicar sigmoid a los logits antes de calcular esta pérdida.

Sigmoid se aplica durante inferencia o evaluación para convertir logits en probabilidades. No se
utilizan softmax ni cross-entropy multiclase porque las etiquetas no son mutuamente excluyentes.

## Evaluación

La evaluación debe reportar precisión, recall y F1 por clase y sus agregaciones macro, además de
mean average precision (mAP). La exactitud global no se utiliza como métrica principal porque la
tarea es multietiqueta y contiene una proporción elevada de ejemplos totalmente negativos.

Los umbrales de decisión se seleccionan por clase maximizando F1 sobre validación, según
`evaluation.threshold_strategy` en la configuración. El cálculo de average precision se realiza
sobre probabilidades o scores continuos y no depende de esos umbrales. Antes del experimento
final debe quedar documentado cómo se tratan, en cada partición, las clases sin positivos al
calcular métricas agregadas.

## Adaptación de FBRS y DLoGNet

La distribución de bandas FBRS debe derivarse del intervalo de Nyquist del audio del proyecto.
Para una descomposición de nivel `L`, la anchura mínima estándar es

$$
\Delta f_{\min} = \frac{f_s/2}{2^L} = \frac{f_s}{2^{L+1}}.
$$

El nivel utilizado por el paper es un punto de partida, no una decisión transferible sin
validación. Los parámetros vigentes del front-end y del banco se declaran en
`configs/dlognet_fbrs.yaml`; su justificación, las decisiones pendientes y las alternativas de
ablación se describen en `docs/fbrs-explained.md`.

La forma de entrada, el apilamiento de pooling y los campos receptivos de DLoGNet deben derivarse
de las dimensiones producidas por el pipeline de este proyecto. La salida y la pérdida siguen la
formulación multietiqueta de este documento, no el clasificador multiclase del paper.

El resultado de exactitud publicado por el paper no es un objetivo comparable: corresponde a
otro dominio, otra definición de tarea y otra métrica. Las comparaciones del proyecto deben
realizarse entre modelos entrenados con las mismas particiones y el mismo protocolo de
evaluación.
