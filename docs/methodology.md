# Metodología

Este documento registra el protocolo de datos, FBRS, DLoGNet, entrenamiento y
evaluación. Las decisiones pendientes se resolverán antes de ejecutar el experimento final.

## Objetivo y alcance

El proyecto transfiere a un conjunto de grabaciones de anuros las dos contribuciones principales
del artículo *"Towards accurate bird sound recognition through multi-scale texture-aware modeling"*
(Qin y Huang, npj Acoustics 2025):

- **FBRS**, una representación espectrográfica basada en paquetes wavelet y selección de
  subbandas guiada por energía;
- **DLoGNet**, una CNN cuyos núcleos se construyen con filtros Laplacian-of-Gaussian y parámetros
  aprendibles de orientación `θ` y escala `σ`.

El objetivo no es reproducir los resultados del artículo. El dominio, las etiquetas y las
características del audio son diferentes, de modo que las decisiones del artículo se consideran
puntos de partida y deben justificarse nuevamente para este conjunto de datos. `docs/paper-summary.md`
documenta el método original y `docs/fbrs.md` desarrolla la adaptación de FBRS.

## Conjunto de datos y definición de la tarea

`dataset/`, excluido de Git, contiene grabaciones de anuros con formato AnuraSet. La auditoría
del CSV local realizada el 27 de agosto de 2026 produjo el siguiente estado:

| Propiedad | Valor |
|---|---:|
| Clips | 62,191 |
| Duración por clip | 3 s |
| Audio | 22.05 kHz, mono, 16 bits |
| Columnas de etiquetas | 42 |
| Especies con positivos observados | 40 |
| Especies del experimento principal | 31 |
| Máximo de especies por clip | 8 |
| Filas sin etiquetas positivas | 22,504 (36.185 %) |

Los segmentos se encuentran en `dataset/train/*.wav` y siguen el patrón
`INCT<site>_<date>_<time>_<start>_<end>.wav`. `dataset/train.csv` contiene `filename` y las
columnas multihot.

La tarea es de clasificación multietiqueta. La taxonomía principal contiene 31 especies con
soporte en al menos 10 grabaciones independientes. Las etiquetas con menor soporte se reservan
para experimentos exploratorios, nueva recolección o un protocolo específico de *few-shot
learning*; `SCIFUS` y `SCINAS` no tienen ejemplos positivos. La selección completa y el
tratamiento de clips con etiquetas fuera de alcance se definen en
[`data-preparation.md`](data-preparation.md).

Las filas sin positivos en la taxonomía activa se conservan como ejemplos totalmente negativos
para esa taxonomía. No se asume que representen fondo puro y no se añade una clase de fondo.

Como el conjunto de datos no está versionado, estas cantidades deben volver a auditarse si cambia
`dataset/train.csv`. Una variación no debe corregirse únicamente en este documento: también se
deben revisar las configuraciones, particiones y pruebas que dependan de ella.

## Flujo previsto

1. Auditar etiquetas y grabaciones y aplicar la política de selección de etiquetas.
2. Crear particiones agrupadas por grabación según [`data-preparation.md`](data-preparation.md)
   y guardar sus manifiestos versionables en `splits/`.
3. Ajustar cualquier preprocesamiento usando únicamente entrenamiento.
4. Entrenar los modelos de referencia y DLoGNet.
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
Los modelos deben derivar su dimensión de salida desde ese valor; no se duplica en la sección
`model` de las configuraciones. La definición y auditoría de las etiquetas se documentan en
«Conjunto de datos y definición de la tarea».

## Salida del modelo y función de pérdida

El modelo produce logits independientes para cada etiqueta. Durante el entrenamiento se utiliza
`BCEWithLogitsLoss`, que combina internamente sigmoid y binary cross-entropy de forma
numéricamente estable. No se debe aplicar sigmoid a los logits antes de calcular esta pérdida.

Sigmoid se aplica durante inferencia o evaluación para convertir logits en probabilidades. No se
utilizan softmax ni cross-entropy multiclase porque las etiquetas no son mutuamente excluyentes.

## Evaluación

La evaluación debe reportar precisión, exhaustividad y F1 por clase y sus agregaciones macro,
además de precisión media promedio (mAP). La exactitud global no se utiliza como métrica principal porque la
tarea es multietiqueta y contiene una proporción elevada de ejemplos totalmente negativos.

Los umbrales de decisión se seleccionan por clase maximizando F1 sobre validación, según
`evaluation.threshold_strategy` en la configuración. El cálculo de average precision se realiza
sobre probabilidades o puntuaciones continuas y no depende de esos umbrales. Antes del experimento
final debe quedar documentado cómo se tratan, en cada partición, las clases sin positivos al
calcular métricas agregadas.

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

La forma de entrada, el apilamiento de agrupamiento y los campos receptivos de DLoGNet deben
derivarse de las dimensiones producidas por el flujo de este proyecto. La salida y la pérdida siguen la
formulación multietiqueta de este documento, no el clasificador multiclase del artículo.

El resultado de exactitud publicado por el artículo no es un objetivo comparable: corresponde a
otro dominio, otra definición de tarea y otra métrica. Las comparaciones del proyecto deben
realizarse entre modelos entrenados con las mismas particiones y el mismo protocolo de
evaluación.
