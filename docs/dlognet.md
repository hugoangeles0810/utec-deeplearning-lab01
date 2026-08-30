# DLoGNet: fundamentos, adaptación y verificación

## Propósito

Este documento explica DLoGNet (*Directional Laplacian-of-Gaussian Network*) y registra cómo se
adapta al conjunto de datos del proyecto. `docs/paper-summary.md` resume el método original,
`docs/fbrs.md` documenta el primer aporte y `docs/methodology.md` es la fuente de verdad del
protocolo experimental.

## Idea central

Un kernel convolucional aprendido libremente puede detectar patrones útiles, pero sus pesos no
tienen necesariamente una interpretación física directa. DLoGNet incorpora una familia analítica
de filtros cuyos parámetros representan dos magnitudes legibles:

| Parámetro | Significado | Efecto esperado |
|---|---|---|
| `θ` | Orientación | Selecciona la dirección de la segunda derivada. |
| `σ` | Escala de la Gaussiana | Controla el tamaño espacial del patrón al que responde el filtro. |

Los kernels DLoG actúan como detectores de curvatura direccional sobre una representación
tiempo-frecuencia. Sus respuestas no sustituyen toda la capacidad aprendible: cada módulo las
concatena con la entrada y una convolución convencional 3×3 aprende a fusionarlas.

La interpretabilidad es estructural, no una garantía de explicación causal. `θ` y `σ` describen
qué filtro se aplica, pero no demuestran por sí solos que una región concreta haya causado una
predicción.

## Operador DLoG

### Gaussiana y Laplaciano

La Gaussiana bidimensional de escala `σ` es

$$
G(x,y;\sigma)=\frac{1}{2\pi\sigma^2}
\exp\left(-\frac{x^2+y^2}{2\sigma^2}\right).
$$

El Laplaciano de Gaussiana clásico suma las segundas derivadas sobre los dos ejes:

$$
\operatorname{LoG}(x,y;\sigma)=G_{xx}+G_{yy}.
$$

Esta operación es isotrópica: una rotación del kernel no cambia su forma. DLoG utiliza, en
cambio, una segunda derivada en una dirección elegida.

### Segunda derivada direccional

Para el vector unitario

$$
\mathbf n=(\cos\theta,\sin\theta),
$$

la segunda derivada direccional de la Gaussiana se obtiene mediante su Hessiano:

$$
\operatorname{DLoG}(x,y;\sigma,\theta)
=\mathbf n^{\mathsf T}H_G\mathbf n
=\cos^2\theta\,G_{xx}
+\sin^2\theta\,G_{yy}
+2\sin\theta\cos\theta\,G_{xy}.
$$

La implementación evalúa analíticamente

$$
G_{xx}=\frac{x^2-\sigma^2}{\sigma^4}G,\qquad
G_{yy}=\frac{y^2-\sigma^2}{\sigma^4}G,\qquad
G_{xy}=\frac{xy}{\sigma^4}G.
$$

`θ` tiene periodicidad `π`: las orientaciones `θ` y `θ + π` producen el mismo operador de
segunda derivada. Por tanto, los ángulos aprendidos deben interpretarse como ejes y no como
vectores con sentido.

### Orientaciones canónicas

El artículo inicializa cuatro ramas. La asociación visual propuesta por los autores es:

| Orientación inicial | Patrón resaltado en el espectrograma |
|---:|---|
| 0° | Estructuras verticales y transitorios. |
| 45° | Modulaciones de frecuencia ascendentes. |
| 90° | Bandas horizontales y componentes tonales sostenidos. |
| 135° | Modulaciones de frecuencia descendentes. |

Estas asociaciones son hipótesis geométricas. La orientación efectiva depende también de la
convención de ejes del tensor, del signo de los gradientes y de la visualización. Debe verificarse
con patrones sintéticos antes de atribuir significado acústico a una rama.

### Discretización

La fórmula continua debe convertirse en un kernel finito. En el proyecto se evalúa sobre una
cuadrícula cuadrada centrada de 7×7. El truncamiento rompe ligeramente la respuesta nula ante una
entrada constante, por lo que a cada kernel se le resta su media. Después se normaliza por norma
L1 para impedir que los cambios de `σ` alteren la magnitud principalmente por el soporte finito:

$$
K' = K-\operatorname{mean}(K),\qquad
\widehat K=\frac{K'}{\lVert K'\rVert_1}.
$$

El relleno simétrico por cantidad, de tres celdas a cada lado para un kernel 7×7, conserva las
dimensiones espaciales de la respuesta. `Conv2d` utiliza relleno con ceros, por lo que la
respuesta en los bordes no tiene la misma interpretación que en el interior.

## Módulo BDCM y arquitectura

### Banco direccional compartido

Cada BDCM (*Basic DLoG Convolution Module*) posee cuatro valores aprendibles de `θ` y un único
valor aprendible de `σ`. El mismo banco físico se aplica de forma *depthwise* a todos los canales
de entrada. Una entrada con `C` canales produce `4C` mapas direccionales:

```python
directional = depthwise_dlog(inputs)  # [B, 4*C, H, W]
```

Compartir los cinco parámetros físicos por etapa mantiene una interpretación común entre
canales y evita aprender un kernel DLoG independiente para cada mapa. La convolución de fusión
posterior sí contiene pesos convencionales distintos por canal.

### Conexión de identidad y fusión

Las cuatro respuestas se concatenan con la entrada sin filtrar. Una convolución 3×3 mezcla los
`5C` canales, seguida de `BatchNorm2d`, ReLU y agrupamiento máximo 2×2:

```python
directional = dlog(inputs)
fused = concatenate([inputs, directional], axis="channels")
outputs = max_pool(relu(batch_norm(conv3x3(fused))))
```

La ruta de identidad conserva información que podría perderse al aplicar filtros de segunda
derivada. No es una suma residual: cambia el número de canales antes de la convolución de fusión.

### Red de cinco etapas

DLoGNet apila cinco BDCM con canales de salida `[64, 128, 128, 128, 64]`. Para las entradas de
AnuraSet de un canal y tamaño `128×255`, las dimensiones son:

| Punto de la red | Forma espacial | Canales |
|---|---:|---:|
| Entrada Mel o FBRS | 128×255 | 1 |
| BDCM 1 | 64×127 | 64 |
| BDCM 2 | 32×63 | 128 |
| BDCM 3 | 16×31 | 128 |
| BDCM 4 | 8×15 | 128 |
| BDCM 5 | 4×7 | 64 |
| Agrupamiento promedio global | 1×1 | 64 |

El clasificador transforma las 64 características en 1024 unidades, aplica ReLU y *dropout* de
0.3 y produce 31 logits independientes. Las dos dimensiones espaciales se reducen por división
entera en cada `MaxPool2d`; no se redimensiona la representación a 128×128 como en el artículo.

## DLoGNet en el artículo

Los autores presentan DLoGNet como una jerarquía de cinco BDCM sobre FBRS. Cada módulo parte de
0°, 45°, 90° y 135°, aprende orientación y escala por descenso por gradiente y fusiona las ramas
con una convolución 3×3. La red original recibe imágenes de 128×128, termina en capas totalmente
conectadas de 1024 y 8 unidades y utiliza *softmax* con entropía cruzada para clasificación
monoclase de ocho especies de aves.

La evidencia presentada incluye:

- una tabla con `θ` y `σ` aprendidos en las cinco etapas;
- estabilidad visual de esos parámetros en cinco entrenamientos con semillas distintas;
- mapas de activación diferentes para las cuatro orientaciones de la primera etapa;
- Grad-CAM más concentrado sobre contornos y armónicos que en CNN y VGG-16;
- separaciones t-SNE más compactas al combinar DLoGNet con FBRS.

En la tabla del artículo, `σ` aumenta desde 1.2955 en la primera etapa hasta 1.4161 en la cuarta y
baja a 1.3762 en la quinta. Los autores lo interpretan como crecimiento progresivo del campo
receptivo seguido de un refinamiento final. Es evidencia descriptiva del experimento con aves;
no establece que la misma trayectoria deba aparecer en AnuraSet.

## Aspectos ambiguos del artículo

La publicación no especifica suficientemente varios detalles necesarios para reproducir
DLoGNet:

- la ecuación (25) muestra `concat(ramas) + X`, una suma dimensionalmente incompatible con la
  tabla de formas, mientras la figura 15 muestra que `X` también entra a la concatenación;
- llama residual a esa conexión aunque la figura representa concatenación y no suma;
- la regla de actualización de la ecuación (21) multiplica la tasa por el propio parámetro, en
  lugar de por los gradientes definidos en la ecuación anterior, lo que parece un error tipográfico;
- no declara tamaño del kernel DLoG, cuadrícula, relleno ni tratamiento de bordes;
- no indica cómo corrige la respuesta DC ni cómo normaliza kernels truncados;
- no define una parametrización que garantice `σ > 0`, ni cotas para evitar escalas degeneradas;
- no precisa si `σ` se comparte entre orientaciones o si los kernels se comparten entre canales;
- no cuantifica parámetros, FLOPs, memoria ni latencia frente a los baselines;
- el código no está publicado y solo se ofrece bajo solicitud.

Además, la afirmación de que los kernels convolucionales convencionales son isotrópicos debe
leerse como una motivación y no como una propiedad general: un kernel libre aprendido puede ser
direccional. La diferencia verificable es que DLoG impone una familia analítica en la que
orientación y escala son parámetros explícitos.

## Adaptación a AnuraSet

### Datos relevantes

| Propiedad | Valor |
|---|---:|
| Frecuencia de muestreo | 22,050 Hz |
| Duración del clip | 3 s |
| Entrada | Mel o FBRS de 128×255 |
| Canales de entrada | 1 |
| Etiquetas del experimento principal | 31 |
| Tarea | Clasificación multietiqueta |

Los contornos espectrotemporales de aves no se transfieren automáticamente a anuros. Las cuatro
orientaciones son una inicialización geométrica razonable, pero su utilidad, estabilidad y
significado deben evaluarse en el dominio del proyecto.

### Decisiones vigentes

`configs/dlognet_mel.yaml` y `configs/dlognet_fbrs.yaml` comparten:

| Decisión | Valor |
|---|---|
| Etapas BDCM | 5 |
| Canales | `[64, 128, 128, 128, 64]` |
| Tamaño del kernel DLoG | 7×7 |
| Orientaciones iniciales | `[0°, 45°, 90°, 135°]` |
| Escala inicial | `1.0` |
| Escala mínima | `0.3` |
| Escalas por BDCM | Una, compartida entre orientaciones y canales |
| Parámetros de orientación por BDCM | Cuatro, compartidos entre canales |
| Ruta de identidad | Concatenación con las respuestas DLoG |
| Fusión | `Conv2d` 3×3 con relleno 1 |
| Orden posterior | `BatchNorm2d`, ReLU, `MaxPool2d` 2×2 |
| Clasificador oculto | 1024 unidades |
| *Dropout* | 0.3 |
| Salida | 31 logits independientes |

La arquitectura tiene 2,314,616 parámetros entrenables en ambas representaciones. De ellos, solo
25 son parámetros físicos DLoG explícitos: cuatro ángulos y una escala por cada una de las cinco
etapas. El resto corresponde principalmente a las convoluciones de fusión, normalizaciones y
clasificador. El baseline CNN configurado tiene 97,119 parámetros, de modo que una comparación
de exactitud no debe interpretarse como una comparación a igual capacidad.

### Escala positiva

Optimizar `σ` directamente permitiría valores nulos o negativos. El proyecto mantiene un
parámetro libre `raw_sigma` y construye la escala efectiva como

$$
\sigma=\operatorname{softplus}(\texttt{raw\_sigma})+0.3.
$$

`raw_sigma` se inicializa mediante la inversa de *softplus* para que la escala efectiva inicial
sea exactamente 1.0. Esta parametrización garantiza positividad, pero no impone una cota superior.

### Concatenación en lugar de suma

La figura 15 y la tabla 6 del artículo son compatibles con concatenar la entrada y las cuatro
ramas: antes de la fusión aparecen `5C` canales. Por ejemplo, la segunda etapa recibe 64 canales
y su convolución de fusión recibe 320. Esta es la interpretación adoptada por el proyecto para
resolver la contradicción con la ecuación (25).

### Adaptación multietiqueta

El clasificador original de ocho salidas con *softmax* se reemplaza por 31 logits independientes.
El entrenamiento utiliza `BCEWithLogitsLoss` y aplica sigmoid solo durante evaluación o inferencia.
La arquitectura deriva su dimensión final de `data.num_labels`; no crea una clase de fondo ni
supone exclusividad entre vocalizaciones simultáneas.

### Dos representaciones, una arquitectura

`dlognet_mel` y `dlognet_fbrs` mantienen idénticos arquitectura, pérdida, optimizador, semilla y
particiones. Solo cambia la representación. FBRS usa el banco global ajustado exclusivamente con
entrenamiento y congelado antes de generar las características. Esta separación permite estimar
el efecto de la representación sin confundirlo con un cambio de modelo.

## Implementación

La lógica definitiva del operador y los BDCM reside en `src/anuraset_dl/dlog.py`; la composición
de la red y su construcción desde configuración residen en `src/anuraset_dl/models.py`:

1. construir la cuadrícula discreta centrada;
2. calcular `G`, `Gxx`, `Gyy` y `Gxy` con la escala positiva;
3. combinar las derivadas según cada `θ`;
4. centrar y normalizar los kernels;
5. repetir el banco por canal y aplicar convolución *depthwise*;
6. concatenar entrada y respuestas direccionales;
7. fusionar con convolución 3×3, normalizar, activar y reducir resolución;
8. repetir cinco etapas y aplicar el clasificador multietiqueta.

El entrenamiento se ejecuta con cualquiera de las dos configuraciones:

```bash
uv run --group tracking python -m anuraset_dl.train --config configs/dlognet_mel.yaml
uv run --group tracking python -m anuraset_dl.train --config configs/dlognet_fbrs.yaml
```

Evaluación carga el mejor checkpoint seleccionado por pérdida de validación. Los checkpoints
incluyen las huellas de configuración, metadatos y manifiestos; la variante FBRS incorpora además
la identidad del banco congelado mediante la representación precalculada.

## Verificación requerida

| Prueba | Invariante esperado |
|---|---|
| Respuesta DC | Cada kernel discreto suma aproximadamente cero. |
| Normalización | Cada kernel tiene norma L1 aproximadamente uno. |
| Direccionalidad | Orientaciones distintas producen kernels y respuestas distintas. |
| Periodicidad | `θ` y `θ + π` producen el mismo kernel dentro de tolerancia numérica. |
| Escala positiva | `σ` permanece estrictamente por encima de `minimum_sigma`. |
| Gradientes | `θ` y `raw_sigma` reciben gradientes finitos y no nulos. |
| Forma *depthwise* | `C` canales y cuatro orientaciones producen `4C` respuestas. |
| Conexión de identidad | La fusión recibe exactamente `5C` canales. |
| Dimensiones | Cinco BDCM aceptan 128×255 y terminan en 4×7 sin relleno adicional. |
| Salida | Cada ejemplo produce exactamente `data.num_labels` logits. |
| Estabilidad numérica | Kernels, activaciones y gradientes no contienen `NaN` ni infinitos. |
| Serialización | Guardar y cargar un checkpoint conserva parámetros y predicciones. |
| Separación factorial | Mel y FBRS construyen la misma arquitectura DLoGNet. |

Las pruebas actuales cubren respuesta DC, finitud, diferencias entre orientaciones, gradientes de
`θ` y `σ`, formas de DLoG y BDCM, validación de parámetros, número de salidas y ejecuciones
sintéticas del pipeline. Las pruebas de periodicidad, norma L1 explícita, patrones orientados
sintéticos y equivalencia exacta después de serialización deben mantenerse como verificaciones
pendientes si todavía no están automatizadas.

## Comparaciones realizadas y ablaciones pendientes

La matriz aceptada del 30 de agosto de 2026 produjo:

| Representación | CNN: F1 macro | DLoGNet: F1 macro | Diferencia de DLoGNet |
|---|---:|---:|---:|
| Mel | 0.5315 | 0.6564 | +0.1249 |
| FBRS | 0.5199 | 0.5603 | +0.0404 |

DLoGNet mejoró F1 macro frente a la CNN en ambas representaciones, pero DLoGNet + FBRS quedó
0.0961 por debajo de DLoGNet + Mel. Las dos variantes DLoGNet mostraron sobreajuste temprano: sus
mejores checkpoints fueron las épocas 4 y 2, respectivamente. Una sola semilla no permite
atribuir estabilidad estadística a `θ`, `σ` ni a las diferencias de métricas.

Permanecen pendientes, con las mismas particiones y el mismo protocolo:

- repetir varias semillas y reportar dispersión de métricas, `θ` y `σ` por etapa;
- comparar orientaciones y escalas aprendibles frente a versiones congeladas;
- evaluar tamaños de kernel y cotas o inicializaciones alternativas de `σ`;
- comparar una escala compartida por BDCM frente a escalas por orientación;
- ablacionar la conexión de identidad y la normalización de los kernels;
- separar el efecto del calendario de entrenamiento mediante *early stopping*;
- reportar parámetros, FLOPs, memoria y latencia junto con la calidad predictiva;
- inspeccionar respuestas a patrones sintéticos y producir Grad-CAM sobre ejemplos de anuros;
- validar si las interpretaciones direccionales son acústicamente coherentes con especialistas.

Los resultados y artefactos deben registrarse en `docs/experiments.md`; cualquier cambio de
arquitectura debe declararse primero en configuración y en `docs/methodology.md`.

## Resumen

DLoGNet incorpora filtros de segunda derivada Gaussiana con orientación y escala aprendibles en
una CNN. Cada etapa aplica cuatro orientaciones compartidas entre canales, concatena esas
respuestas con la entrada y usa una convolución convencional para fusionarlas. Cinco etapas forman
una jerarquía direccional y multiescala antes del clasificador.

Para AnuraSet se conserva la profundidad y el calendario de canales del artículo, pero se adopta
una discretización 7×7 normalizada, `σ` positiva, entradas 128×255 sin redimensionar y una salida
multietiqueta de 31 logits. La primera matriz favoreció DLoGNet + Mel, aunque la estabilidad entre
semillas, la dinámica de sus parámetros interpretables y el costo computacional comparativo
siguen pendientes.
