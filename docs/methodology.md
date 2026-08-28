# Metodología

Este documento registrará el protocolo definitivo de datos, FBRS, DLoGNet, entrenamiento y
evaluación. Las decisiones pendientes se resolverán antes de ejecutar el experimento final.

## Flujo previsto

1. Auditar etiquetas y grabaciones.
2. Crear particiones agrupadas por grabación y guardar sus manifiestos versionables en
   `splits/`.
3. Ajustar cualquier preprocesamiento usando únicamente entrenamiento.
4. Entrenar los modelos de referencia y DLoGNet.
5. Seleccionar umbrales con validación.
6. Evaluar una sola vez sobre test.

## Fuente de verdad del número de clases

`data.num_labels` es la única fuente de verdad para el número de salidas del clasificador.
Los modelos deben derivar su dimensión de salida desde ese valor; no se duplica en la sección
`model` de las configuraciones.

El CSV contiene 42 columnas de etiquetas, pero `SCIFUS` y `SCINAS` no tienen ningún ejemplo
positivo. Se excluyen de la tarea entrenable, que queda definida sobre 40 etiquetas. Las 22,504
filas sin etiquetas positivas se conservan como ejemplos totalmente negativos; no se interpretan
automáticamente como fondo puro ni se añade una salida independiente de fondo.
