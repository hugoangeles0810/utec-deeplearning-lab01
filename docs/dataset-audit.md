# Auditoría del dataset

Este documento conserva la evidencia empírica que sustenta la política definida en
[`data-preparation.md`](data-preparation.md). No reemplaza esa política: registra el estado del
dataset local auditado, los controles aplicados y la relación entre los hallazgos y las decisiones
metodológicas vigentes.

## Alcance y trazabilidad

La auditoría se realizó sobre `dataset/train.csv` y `dataset/train/*.wav`. El corte fue analizado
el 27 de agosto de 2026 y vuelto a verificar el 28 de agosto de 2026 con los siguientes datos de
trazabilidad:

| Propiedad | Valor |
|---|---|
| Archivo de metadatos | `dataset/train.csv` |
| Tamaño del CSV | 7,358,909 bytes |
| SHA-256 del CSV | `38df5d408d9bf621cc11f78ed4cd766be63a8406512c1eb4d611d98e4486d276` |
| Directorio de audio | `dataset/train/` |
| Clips referenciados | 62,191 |
| Columnas | 43: `filename` y 42 etiquetas |

El dataset no se versiona en Git. Por ello, las cifras de este documento solo son válidas para
el CSV identificado por esa huella. Si la huella cambia, la auditoría debe repetirse antes de
regenerar particiones o ejecutar experimentos.

### Procedimiento

La auditoría puede repetirse y parametrizarse desde
[`notebooks/01_dataset_analysis.ipynb`](../notebooks/01_dataset_analysis.ipynb). El notebook usa
este corte por defecto, no conserva salidas ejecutadas y permite cambiar rutas, columnas,
patrones de identificación y umbrales desde una única celda de configuración.

Los metadatos se cargaron como una tabla completa y se aplicaron los siguientes controles:

1. verificación de nulidad, unicidad de `filename` y dominio binario de las 42 etiquetas;
2. correspondencia bidireccional entre las filas del CSV y los archivos `.wav`;
3. lectura de la cabecera de cada audio para comprobar formato, canales, frecuencia de muestreo y
   cantidad de muestras;
4. obtención de `recording_id` eliminando exclusivamente el sufijo `_inicio_fin.wav`;
5. agregación por máximo para considerar positiva una grabación si alguno de sus clips contiene
   la etiqueta;
6. conteo de etiquetas por clip y separación entre clips sin positivos originales y clips sin
   positivos en la taxonomía principal.

Esta auditoría verifica la estructura y las anotaciones declaradas en el CSV. No constituye una
revisión perceptual de cada audio ni permite concluir que una fila sin positivos corresponda a
silencio o fondo puro.

## Integridad del corpus

| Control | Resultado |
|---|---:|
| Filas con algún valor nulo | 0 |
| Celdas de etiqueta fuera de `{0, 1}` | 0 |
| Nombres de archivo duplicados | 0 |
| Filas del CSV sin audio | 0 |
| Audios sin fila en el CSV | 0 |
| Audios que no pudieron abrirse | 0 |
| Audios WAV, mono, PCM de 16 bits y 22.05 kHz | 62,191 (100 %) |
| Audios con 66,150 muestras, equivalentes a 3 s | 62,191 (100 %) |

La correspondencia uno a uno y la homogeneidad técnica permiten conservar todas las filas sin
una etapa previa de descarte o remuestreo. Estos controles deben repetirse al generar las
particiones porque una futura versión local del dataset puede no conservar las mismas propiedades.

## Grabaciones, segmentos y sitios

Los 62,191 clips proceden de 1,074 grabaciones independientes distribuidas entre cuatro sitios.
La segmentación es fuertemente repetitiva: 1,069 grabaciones tienen 58 clips y las cinco restantes
tienen 26, 28, 42, 42 y 51 clips, respectivamente.

| Sitio | Clips | % de clips | Grabaciones | % de grabaciones |
|---|---:|---:|---:|---:|
| `17` | 13,688 | 22.010 % | 236 | 21.974 % |
| `20955` | 18,111 | 29.122 % | 314 | 29.236 % |
| `4` | 16,240 | 26.113 % | 280 | 26.071 % |
| `41` | 14,152 | 22.756 % | 244 | 22.719 % |
| **Total** | **62,191** | **100 %** | **1,074** | **100 %** |

Tratar los clips como observaciones independientes permitiría que segmentos solapados de una
misma grabación aparecieran en entrenamiento y evaluación. Este hallazgo sustenta que
`recording_id`, y no el clip, sea la unidad indivisible de partición. También justifica aproximar
primero las proporciones por grabaciones y comprobar después el balance por clips y sitios.

### Excepción observada en los nombres

La forma predominante es `INCT<site>_<date>_<time>_<start>_<end>.wav`, pero una grabación utiliza
el identificador `INCT20955_20191123_041500_000` y genera 58 nombres con un componente adicional,
por ejemplo `INCT20955_20191123_041500_000_0_3.wav`.

Por esta razón, `recording_id` se obtiene eliminando solo los dos enteros finales y la extensión;
no se construye tomando una cantidad fija de componentes separados por `_`. La regla agrupa
correctamente los 58 clips excepcionales sin renombrar archivos ni alterar los metadatos.

## Naturaleza multietiqueta

La cantidad de etiquetas positivas originales por clip presenta la siguiente distribución:

| Etiquetas positivas | Clips | % del corpus |
|---:|---:|---:|
| 0 | 22,504 | 36.185 % |
| 1 | 12,886 | 20.720 % |
| 2 | 10,591 | 17.030 % |
| 3 | 8,636 | 13.886 % |
| 4 | 5,032 | 8.091 % |
| 5 | 1,685 | 2.709 % |
| 6 | 664 | 1.068 % |
| 7 | 182 | 0.293 % |
| 8 | 11 | 0.018 % |
| **Total** | **62,191** | **100 %** |

Hay 26,801 clips con dos o más positivos: 43.095 % del corpus y 67.529 % de los clips con al
menos un positivo. La coocurrencia no es marginal; sustenta formular el problema como
clasificación multietiqueta, producir logits independientes, utilizar `BCEWithLogitsLoss` y crear
particiones agrupadas que preserven la cobertura de varias etiquetas simultáneamente.

## Soporte de las etiquetas

El soporte decisivo para curar la taxonomía es la cantidad de grabaciones positivas. El conteo de
clips se conserva para describir la prevalencia, pero no representa observaciones independientes
porque numerosos clips proceden de una misma grabación.

| Etiqueta | Clips positivos | Grabaciones positivas | Categoría vigente |
|---|---:|---:|---|
| `SPHSUR` | 13,258 | 257 | Principal |
| `BOABIS` | 10,888 | 240 | Principal |
| `DENMIN` | 6,070 | 175 | Principal |
| `PHYALB` | 5,374 | 133 | Principal |
| `BOAALB` | 3,704 | 132 | Principal |
| `BOAFAB` | 6,438 | 128 | Principal |
| `PHYCUV` | 4,240 | 128 | Principal |
| `LEPLAT` | 5,244 | 127 | Principal |
| `LEPPOD` | 6,032 | 125 | Principal |
| `PITAZU` | 4,873 | 125 | Principal |
| `DENNAN` | 3,801 | 99 | Principal |
| `SCIPER` | 3,791 | 90 | Principal |
| `SCIFUV` | 2,734 | 67 | Principal |
| `BOALUN` | 2,060 | 48 | Principal |
| `PHYDIS` | 897 | 45 | Principal |
| `BOAALM` | 1,601 | 43 | Principal |
| `PHYSAU` | 1,479 | 39 | Principal |
| `BOARAN` | 1,339 | 37 | Principal |
| `BOALEP` | 846 | 32 | Principal |
| `ELABIC` | 1,214 | 30 | Principal |
| `LEPFUS` | 1,232 | 28 | Principal |
| `BOAPRA` | 480 | 25 | Principal |
| `LEPNOT` | 1,062 | 25 | Principal |
| `LEPLAB` | 1,329 | 24 | Principal |
| `DENCRU` | 602 | 22 | Principal |
| `ELAMAT` | 395 | 20 | Principal |
| `DENNAH` | 467 | 17 | Principal |
| `PHYMAR` | 200 | 13 | Principal |
| `RHIICT` | 310 | 12 | Principal |
| `ADEMAR` | 520 | 10 | Principal |
| `PHYNAT` | 410 | 10 | Principal |
| `ADEDIP` | 390 | 7 | Exploratoria |
| `DENELE` | 149 | 7 | Exploratoria |
| `SCIALT` | 232 | 4 | Exploratoria |
| `SCIRIZ` | 73 | 3 | Exploratoria |
| `AMEPIC` | 68 | 2 | Evidencia insuficiente |
| `LEPELE` | 34 | 2 | Evidencia insuficiente |
| `LEPFLA` | 7 | 2 | Evidencia insuficiente |
| `RHIORN` | 21 | 2 | Evidencia insuficiente |
| `RHISCI` | 11 | 1 | Evidencia insuficiente |
| `SCIFUS` | 0 | 0 | Sin evidencia |
| `SCINAS` | 0 | 0 | Sin evidencia |

La distribución produce cuatro grupos nítidos para el protocolo actual:

- 31 etiquetas tienen al menos 10 grabaciones positivas;
- 4 etiquetas tienen entre 3 y 9 grabaciones positivas;
- 5 etiquetas tienen solamente 1 o 2 grabaciones positivas;
- 2 etiquetas no tienen positivos.

El umbral de 10 grabaciones no afirma que una clase quede bien representada estadísticamente.
Es el mínimo compatible con reservar 8 grabaciones positivas para entrenamiento, 2 para
validación. La coocurrencia puede impedir una asignación concreta incluso cuando
los conteos marginales alcanzan ese mínimo; por eso el generador debe validar las restricciones y
detenerse si no puede cumplirlas.

## Clips sin positivos en la taxonomía activa

| Conjunto de etiquetas considerado | Clips sin positivos | % del corpus |
|---|---:|---:|
| 42 etiquetas originales | 22,504 | 36.185 % |
| 31 etiquetas principales | 22,731 | 36.550 % |

La diferencia corresponde a 227 clips, distribuidos en 14 grabaciones, que contienen algún
positivo original pero ninguno entre las 31 etiquetas principales. Estos clips se registran como
`out_of_scope_foreground`.

Las 22,504 filas sin positivos originales no demuestran ausencia de actividad acústica ni fueron
anotadas como una clase de fondo. Las 227 filas `out_of_scope_foreground` contienen explícitamente
actividad etiquetada fuera de la taxonomía principal. En consecuencia:

- no se elimina ninguna de esas filas;
- no se crea una clase `background` ni `other`;
- para el experimento principal, sus 31 objetivos se representan con ceros;
- `out_of_scope_foreground` se contabiliza por separado en el reporte de particiones;
- el ajuste inicial de FBRS usa positivos de la taxonomía activa para que el 36.550 % de filas
  sin positivos activos no domine la selección energética;
- la exactitud global no se utiliza como métrica principal porque puede verse favorecida por la
  abundancia de objetivos negativos.

## Trazabilidad de hallazgos y decisiones

| Hallazgo auditado | Riesgo o implicación | Decisión sustentada |
|---|---|---|
| 62,191 clips proceden de 1,074 grabaciones y casi todas aportan 58 segmentos | Fuga entre segmentos correlacionados | Partición indivisible por `recording_id` |
| Existe un identificador con el componente adicional `_000` | Una extracción posicional separaría o perdería esa grabación | Eliminar solo el sufijo `_inicio_fin.wav` |
| 26,801 clips contienen dos o más etiquetas | Las clases no son mutuamente excluyentes | Salida multietiqueta, logits independientes y `BCEWithLogitsLoss` |
| 31 etiquetas alcanzan 10 grabaciones positivas | Es posible plantear una cobertura marginal mínima de 8/2 | Taxonomía principal de 31 etiquetas y validación estricta de cobertura |
| 4 etiquetas solo alcanzan entre 3 y 9 grabaciones | No admiten la cobertura 8/2 | Taxonomía exploratoria separada; objetivo secundario de 1 grabación por partición |
| 5 etiquetas tienen 1 o 2 grabaciones y 2 no tienen positivos | No permiten entrenamiento convencional ni evaluación independiente en tres particiones | Exclusión del experimento principal; nueva recolección o protocolo *few-shot* |
| 22,731 clips no tienen positivos entre las 31 etiquetas | Los ceros son frecuentes, pero no equivalen a fondo verificado | Conservar como negativos para la taxonomía activa, sin crear `background` |
| 227 de esos clips contienen positivos excluidos | Hay primer plano conocido fuera del alcance | Conservar anotaciones, no remapear y reportar `out_of_scope_foreground` |
| Los cuatro sitios aportan entre 236 y 314 grabaciones | Una partición puede distorsionar la procedencia aunque cumpla la proporción global | Verificar y optimizar la distribución por sitio después de la cobertura de etiquetas |
| Todos los audios comparten formato, frecuencia y duración | No se requiere normalización estructural para este corte | Configurar 22.05 kHz y 3 s; volver a validar si cambia la huella |
| No hay duplicados ni diferencias entre CSV y audios | El corte actual puede conservarse completo | No descartar filas por integridad, manteniendo esos controles como precondición |

La proporción 80/20 y la semilla 42 son decisiones del protocolo, no resultados derivados de
la auditoría. La evidencia anterior determina las restricciones que esa asignación debe respetar;
los manifiestos y su reporte aportarán la evidencia de que una partición concreta las cumple.
