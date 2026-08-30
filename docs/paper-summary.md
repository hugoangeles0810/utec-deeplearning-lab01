# Resumen del artículo de referencia

## Referencia

Rui Qin y Jing Huang. *Towards accurate bird sound recognition through multi-scale
texture-aware modeling*. npj Acoustics 1, 22 (2025).
[DOI 10.1038/s44384-025-00025-6](https://doi.org/10.1038/s44384-025-00025-6).

Este documento resume el método original. No define por sí solo las decisiones del proyecto:
la adaptación vigente a AnuraSet se registra en `docs/methodology.md` y
en los documentos técnicos `docs/fbrs.md` y `docs/dlognet.md`.

## Problema y propuesta

El artículo estudia clasificación monoclase de sonidos de aves. Parte de dos limitaciones:

1. las representaciones tiempo-frecuencia convencionales utilizan bancos de filtros fijos y
   no concentran necesariamente la resolución donde se encuentra la energía relevante;
2. los núcleos convolucionales genéricos ofrecen poca interpretabilidad física.

La propuesta combina dos componentes:

| Componente | Función |
|---|---|
| FBRS (*Frequency-Band Recalibrated Spectrogram*) | Construye un banco de filtros adaptativo mediante descomposición por paquetes wavelet y selección de subbandas guiada por energía. |
| DLoGNet (*Directional Laplacian-of-Gaussian Network*) | Sustituye parte de los núcleos genéricos por filtros DLoG cuya orientación `θ` y escala `σ` son aprendibles. |

Los componentes se diseñan conjuntamente: FBRS busca conservar estructura espectral útil y
DLoGNet busca detectar patrones direccionales y multiescala sobre esa representación.

## FBRS en el artículo

El procedimiento descrito por los autores es, a grandes rasgos:

1. aplicar una descomposición por paquetes wavelet de nivel `L`;
2. calcular y normalizar la energía de los nodos;
3. seleccionar o fusionar pares de nodos hermanos según su energía;
4. derivar de esa partición un banco de filtros no uniforme;
5. aplicar el banco al espectro de potencia;
6. comprimir la respuesta con logaritmo y redimensionarla para la red.

La selección conjunta de hermanos se denomina **restricción simbiótica**. Su propósito es
mantener una partición válida del árbol: no se conserva un hijo mientras se descarta de forma
aislada el otro hijo del mismo padre.

El artículo utiliza la wavelet `db16`, explora niveles entre 6 y 9 y adopta `L = 8`. La diferencia
de exactitud reportada en ese barrido es de solo 0.06 puntos porcentuales, por lo que la evidencia
de sensibilidad a `L` es débil.

La salida se expresa como

$$
\operatorname{FBRS}_j = \log\left(\sum_k P(k)H_j(k)\right),
$$

donde `P(k)` es el espectro de potencia y `H_j(k)` es el filtro de la banda `j`.

## DLoGNet en el artículo

El Laplaciano de Gaussiana clásico es isotrópico. DLoG introduce una segunda derivada en la
dirección

$$
\mathbf{n} = (\cos\theta, \sin\theta),
$$

de modo que

$$
\operatorname{DLoG}(x,y,\sigma,\theta)
= \cos^2\theta\,G_{xx}
+ \sin^2\theta\,G_{yy}
+ 2\sin\theta\cos\theta\,G_{xy}.
$$

`θ` representa orientación y `σ` representa escala. Ambos parámetros son diferenciables y se
actualizan junto con los demás parámetros de la red.

Cada módulo BDCM (*Basic DLoG Convolution Module*) contiene cuatro ramas inicializadas cerca de
orientaciones canónicas:

| Orientación aproximada | Patrón esperado en el espectrograma |
|---:|---|
| 0° | Estructuras verticales y transitorios |
| 45° | Modulación de frecuencia ascendente |
| 90° | Bandas horizontales y armónicos estables |
| 135° | Modulación de frecuencia descendente |

Las respuestas direccionales se concatenan, se combinan con una conexión residual y pasan por
una convolución convencional. La red completa apila cinco módulos BDCM y termina en un
clasificador totalmente conectado.

## Configuración experimental original

El experimento del artículo no coincide con el de este repositorio:

| Aspecto | Artículo |
|---|---|
| Dominio | Cantos de aves |
| Tarea | Clasificación monoclase |
| Especies | 8 |
| Audio | Clips de 5 s a 32 kHz |
| Entrada | FBRS redimensionado a 128 × 128 |
| Pérdida | Entropía cruzada |
| Salida | Softmax |
| Entrenamiento | Adam, 50 épocas, lote 32, tasa inicial `1e-4` |

La adaptación a anuros es multietiqueta y, por tanto, utiliza logits independientes,
`BCEWithLogitsLoss` y sigmoid durante inferencia. Esas decisiones se describen en
`docs/methodology.md`.

## Resultados principales

### Ablación de la representación

| Modelo | MFS (Mel) | FBRS | Mejora |
|---|---:|---:|---:|
| CNN de 5 capas | 85.09 % | 87.82 % | +2.73 pp |
| DLoGNet | 87.40 % | 91.18 % | +3.78 pp |

FBRS mejora ambos modelos y la mejora es mayor con DLoGNet. Esto respalda la hipótesis de que
la representación y la arquitectura son complementarias.

### Comparación de modelos con FBRS

| Modelo | Exactitud | F1 |
|---|---:|---:|
| DLoGNet | 91.18 % | 91.16 % |
| Transformer | 91.18 % | 91.16 % |
| MDF-Net | 91.16 % | 91.11 % |
| VGG-16 | 90.52 % | 90.47 % |
| CNN-LSTM | 90.41 % | 90.33 % |
| EfficientNet | 89.82 % | 89.75 % |
| CNN | 87.82 % | 87.79 % |
| LSTM | 87.64 % | 87.60 % |

DLoGNet iguala al Transformer y supera a MDF-Net por un margen mínimo. El resultado no sustenta
una ventaja clara de exactitud frente a los modelos más fuertes; el aporte diferencial es la
interpretabilidad estructural.

## Evidencia de interpretabilidad

Los autores ofrecen cuatro tipos de evidencia:

- valores aprendidos de `θ` y `σ` que conservan significado físico;
- estabilidad de esos parámetros en cinco ejecuciones con semillas distintas;
- mapas de características diferentes para cada orientación;
- visualizaciones Grad-CAM más concentradas en estructuras espectrales coherentes que las de
  CNN y VGG-16.

Los ángulos permanecen cerca de sus inicializaciones canónicas, aunque se ajustan ligeramente.
La escala tiende a aumentar en las primeras cuatro capas, lo que es compatible con campos
receptivos progresivamente más amplios.

## Limitaciones

- La mejora frente a Transformer y MDF-Net es nula o muy pequeña y no se reportan intervalos
  de confianza ni pruebas de significancia.
- No se cuantifica el costo adicional mediante número de parámetros, FLOPs o latencia.
- La evaluación usa ocho especies de un solo conjunto de datos y clips seleccionados por claridad.
- No se documenta con precisión la separación entre entrenamiento, validación y prueba ni si
  evita que clips de una misma grabación aparezcan en particiones distintas.
- No se evalúan escenarios multietiqueta con vocalizaciones solapadas.
- La interpretabilidad visual no fue validada por especialistas del dominio.
- El valor del umbral energético de FBRS, la forma exacta de los filtros y parámetros
  importantes de la etapa inicial no quedan suficientemente especificados.
- El código no está publicado; el artículo indica que puede solicitarse a los autores.

## Lecciones transferibles

La idea más útil no es que DLoGNet sea necesariamente más preciso, sino que un filtro puede
parametrizarse mediante magnitudes interpretables —orientación y escala— y aprenderlas por
descenso por gradiente. La explicación pasa así de ser únicamente *post hoc* a formar parte del
modelo.

La segunda lección es que representación y arquitectura deben evaluarse como factores separados
y también en conjunto. Para este proyecto se requiere comparar, con las mismas particiones y el
mismo protocolo, la matriz factorial CNN/DLoGNet × Mel/FBRS.
