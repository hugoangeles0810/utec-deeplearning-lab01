# Preparación de datos

Este documento define la política vigente de selección de etiquetas y creación de particiones.
Forma parte del protocolo descrito en [`methodology.md`](methodology.md) y debe aplicarse antes de
ajustar representaciones, entrenar modelos o seleccionar umbrales. Los conteos y controles que
sustentan estas decisiones se registran en [`dataset-audit.md`](dataset-audit.md).

## Aprovisionamiento del corte auditado

El corpus local puede descargarse y estructurarse mediante:

```bash
uv run python -m anuraset_dl.provision_data
```

`configs/dataset.yaml` fija los IDs y tamaños de los archivos remotos, la huella de `train.csv`,
los inventarios esperados y el formato de audio. El aprovisionamiento admite reanudación, comprueba
la extracción del archivo 7z y valida el conteo y los nombres contra el inventario esperado, sin
abrir individualmente las cabeceras de los WAV. No modifica las particiones versionadas. Después
de completarlo debe ejecutarse el comando de preparación descrito en «Reproducibilidad y
manifiestos» para comprobar el contrato metodológico y reproducir los manifiestos vigentes.

## Unidad de partición

La unidad indivisible es `recording_id`, obtenido al eliminar el sufijo `_inicio_fin.wav` del
nombre de cada segmento. Todos los clips provenientes de una misma grabación deben pertenecer
exclusivamente a entrenamiento o validación.

La extracción no debe asumir una cantidad fija de componentes separados por `_`: la auditoría
identificó una grabación cuyo prefijo contiene un componente adicional.

No se permite que un mismo `recording_id` aparezca en más de una partición.

## Selección de etiquetas

El soporte de una etiqueta se mide como el número de grabaciones independientes con al menos un
clip positivo. El número de segmentos no se utiliza como sustituto de observaciones
independientes.

Las etiquetas se clasifican de la siguiente forma:

| Categoría | Grabaciones positivas | Tratamiento |
|---|---:|---|
| Principal | 10 o más | Incluida en el experimento principal |
| Exploratoria | Entre 3 y 9 | Reservada para experimentos separados |
| Evidencia insuficiente | Entre 1 y 2 | Excluida del entrenamiento convencional |
| Sin evidencia | 0 | Excluida |

Esta clasificación es una decisión de curación del corpus realizada antes de congelar las
particiones. No debe modificarse posteriormente en función de resultados externos.

### Taxonomía principal

El experimento principal contiene 31 etiquetas entrenables. Se excluyen las siguientes 11:

| Motivo | Etiquetas |
|---|---|
| Exploratorias | `SCIRIZ`, `SCIALT`, `ADEDIP`, `DENELE` |
| Evidencia insuficiente | `RHISCI`, `AMEPIC`, `LEPELE`, `RHIORN`, `LEPFLA` |
| Sin positivos | `SCIFUS`, `SCINAS` |

`data.num_labels` debe ser `31` y `data.excluded_labels` debe enumerar esas 11 etiquetas en las
configuraciones del experimento principal.

### Taxonomía exploratoria

Un experimento posterior puede utilizar 35 etiquetas incorporando `SCIRIZ`, `SCIALT`, `ADEDIP`
y `DENELE`. Debe reutilizar los mismos manifiestos que el experimento principal y mantener
excluidas las especies con una o dos grabaciones y las etiquetas sin positivos.

Las cinco especies con evidencia insuficiente se reservan para nueva recolección o para un
protocolo específico de *few-shot learning*. No forman parte del entrenamiento convencional.

## Conservación de audios y anotaciones

No se eliminan audios ni filas de `dataset/train.csv` por contener etiquetas excluidas. La
selección se aplica únicamente a las columnas objetivo de cada experimento:

- las anotaciones originales se conservan intactas;
- no se crea una clase `other`;
- las etiquetas excluidas no se convierten en positivos de otra clase;
- un clip sin positivos entre las 31 etiquetas principales se representa como totalmente
  negativo para esa taxonomía.

Los clips que contienen positivos únicamente en etiquetas excluidas se contabilizan como
`out_of_scope_foreground`. No se asume que representen silencio ni fondo puro.

La auditoría vigente contiene 22,504 clips sin positivos en ninguna de las 42 etiquetas y 227
clips `out_of_scope_foreground`. Por tanto, para el experimento principal existen 22,731 clips
sin positivos entre sus 31 etiquetas. Esta última definición es la que debe utilizar cualquier
componente que seleccione ejemplos positivos o negativos respecto de la taxonomía activa. La
desagregación y sus implicaciones se documentan en
[`dataset-audit.md`](dataset-audit.md#clips-sin-positivos-en-la-taxonomía-activa).

## Proporciones de las particiones

La proporción global objetivo es 80 % para entrenamiento y 20 % para validación. Se aproxima
primero por cantidad de grabaciones y después por cantidad de clips. Los audios externos sin
etiquetas de `dataset/test/` no participan en esta asignación.

Cada una de las 31 etiquetas principales debe tener como mínimo:

| Partición | Grabaciones positivas mínimas |
|---|---:|
| Entrenamiento | 8 |
| Validación | 2 |

Para las cuatro etiquetas exploratorias se intenta conservar al menos una grabación positiva en
ambas particiones. Este objetivo es secundario: la cobertura de las etiquetas principales tiene
prioridad sobre la cobertura exploratoria y sobre la proporción global.

Si las restricciones no pueden cumplirse debido a las coetiquetas, el generador debe detenerse
y reportar la incompatibilidad. No debe producir silenciosamente una partición inválida.

## Criterios de asignación

La asignación agrupada y multietiqueta se optimiza en el siguiente orden:

1. separación estricta por `recording_id`;
2. cobertura mínima de etiquetas principales;
3. cobertura de etiquetas exploratorias;
4. proporción global de grabaciones;
5. proporción global de clips;
6. prevalencia de etiquetas principales;
7. distribución por sitio;
8. distribución de clips totalmente negativos;
9. distribución de `out_of_scope_foreground`.

Las etiquetas con una o dos grabaciones no imponen restricciones de asignación. Sus grabaciones
se ubican según las demás características que contengan.

## Reproducibilidad y manifiestos

El generador se ejecuta con:

```bash
uv run python -m anuraset_dl.prepare_data --config configs/baseline.yaml
```

La sección `preparation` de esa configuración declara la huella esperada de los metadatos, las
proporciones, coberturas y etiquetas exploratorias. El generador utiliza la semilla `42` y
produce los siguientes archivos versionables:

```text
splits/train.csv
splits/validation.csv
splits/report.json
```

Cada manifiesto incluye como mínimo `filename` y `recording_id`. Las columnas de etiquetas no se
duplican: se leen desde `dataset/train.csv`. `splits/report.json` es un reporte JSON versionable y
debe registrar una huella SHA-256 del CSV utilizado para vincular los manifiestos con la versión
local de los metadatos.

La asignación se resuelve mediante programación lineal entera mixta con SciPy/HiGHS. La cobertura
principal y la pertenencia exclusiva de cada grabación son restricciones duras. Los criterios de
asignación se optimizan lexicográficamente en el orden declarado anteriormente: cada óptimo se
fija antes de resolver el criterio siguiente. Un objetivo final dependiente de la semilla
desempata de forma determinista las asignaciones equivalentes observadas por el optimizador.

La escritura ocurre de forma atómica después de validar la asignación completa. Una repetición
idéntica no reescribe archivos; para reemplazar artefactos diferentes debe indicarse `--force`.

## Validaciones obligatorias

Antes de aceptar las particiones debe comprobarse que:

- cada fila de los metadatos aparece exactamente una vez;
- no existen nombres de archivo duplicados;
- no faltan audios referenciados ni existen audios sin una fila correspondiente;
- ningún `recording_id` cruza particiones;
- se cumplen las coberturas mínimas de las etiquetas principales y se documenta la cobertura
  alcanzada para las exploratorias;
- las proporciones obtenidas quedan documentadas;
- los mismos datos, configuración y semilla reproducen los mismos manifiestos.

La preparación no abre las cabeceras de los 62,191 WAV. La precómputación posterior lee todos los
audios y se detiene ante frecuencia de muestreo, número de canales o duración incompatibles antes
de iniciar el entrenamiento. Esta separación evita duplicar accesos costosos a archivos pequeños,
especialmente sobre almacenamiento de red.

El reporte debe resumir grabaciones y clips por partición, positivos por etiqueta y partición,
distribución por sitio, clips totalmente negativos para la taxonomía principal,
`out_of_scope_foreground`, etiquetas excluidas, semilla y huella de los metadatos.

## Test externo sin etiquetas

`dataset/test/` queda fuera de los manifiestos etiquetados y puede incorporarse después de
entrenar. Sus WAV se validan y procesan mediante `python -m anuraset_dl.predict`; no se unen con
`dataset/train.csv`, no intervienen en el ajuste del banco FBRS ni producen métricas locales. Los
umbrales aplicados a sus predicciones son los seleccionados exclusivamente sobre validación.
