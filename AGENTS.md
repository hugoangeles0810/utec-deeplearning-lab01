# Guía para agentes

## Propósito

Este archivo contiene únicamente instrucciones operativas para trabajar en el repositorio.
Las decisiones sobre el problema, los datos, la metodología, la arquitectura y los
experimentos pertenecen a `docs/` y no deben duplicarse aquí.

## Lectura obligatoria

Antes de modificar el proyecto:

1. Leer `README.md` para conocer el alcance, la estructura y los comandos principales.
2. Leer `docs/methodology.md`, fuente de verdad del protocolo del proyecto.
3. Consultar la documentación específica de la tarea:
   - `docs/data-preparation.md`: selección de etiquetas, particiones y verificaciones de datos.
   - `docs/paper-summary.md`: descripción del paper de referencia; no define por sí solo las
     decisiones de este proyecto.
   - `docs/fbrs.md`: fundamentos, adaptación y decisiones de FBRS.
   - `docs/experiments.md`: requisitos y estado de los experimentos.
   - `splits/README.md`: contrato de las particiones del dataset.
4. Revisar la configuración aplicable en `configs/` antes de cambiar código o ejecutar un
   experimento.

Si el código, una configuración y la documentación metodológica no coinciden, no resolver la
discrepancia de forma silenciosa. Determinar cuál representa la decisión vigente y actualizar
en la misma tarea todas las fuentes afectadas. Si no puede determinarse sin tomar una nueva
decisión metodológica, solicitar confirmación.

## Organización del trabajo

- Colocar la lógica reutilizable y definitiva en `src/anuraset_dl/`.
- Usar `notebooks/` solo para exploración, análisis y visualización. Extraer a `src/` cualquier
  lógica necesaria para reproducir el pipeline.
- Añadir o actualizar pruebas en `tests/` cuando cambie el comportamiento del código.
- Mantener las configuraciones versionables en `configs/` y las particiones reproducibles en
  `splits/`.
- Registrar decisiones metodológicas en `docs/methodology.md` o en el documento técnico
  correspondiente; registrar la ejecución y los resultados en `docs/experiments.md`.
- Guardar artefactos generados en `outputs/` según su categoría.
- No incorporar audios, checkpoints ni otros binarios grandes a Git. Actualizar `.gitignore`
  cuando aparezca una nueva clase de artefacto local.
- Preservar cambios preexistentes del usuario y evitar modificaciones ajenas a la tarea.
- No crear commits ni hacer `push` salvo petición explícita.

## Idioma y estilo documental

Todo contenido documental nuevo o modificado debe escribirse en español. Esto incluye Markdown,
docstrings y comentarios narrativos, diagramas, tablas y reportes. Pueden conservarse en inglés
los identificadores de código, nombres propios, títulos bibliográficos y términos técnicos cuya
traducción reduzca la precisión. No traducir nombres de clases, columnas, rutas, funciones,
variables ni APIs.

Cuando mejore realmente la claridad, pueden utilizarse diagramas Mermaid y fórmulas LaTeX
incrustadas en Markdown.

## Entorno y verificaciones

El proyecto usa Python 3.12, PyTorch y `uv`.

Preparar el entorno:

```bash
uv sync --group dev
```

Ejecutar las verificaciones generales:

```bash
uv run pytest
uv run ruff check .
```

Durante el desarrollo pueden ejecutarse primero pruebas específicas, pero antes de dar por
terminado un cambio de código deben ejecutarse las verificaciones generales. Si alguna no puede
ejecutarse por una dependencia externa o un artefacto local ausente, reportarlo explícitamente.

## Criterios de finalización

Antes de entregar una tarea:

1. Confirmar que el cambio respeta la documentación metodológica aplicable.
2. Verificar que código, configuraciones, pruebas y documentación no se contradicen.
3. Ejecutar las pruebas y el lint adecuados al alcance.
4. Comprobar que no se añadieron datos o artefactos grandes al control de versiones.
5. Resumir los archivos modificados, las verificaciones ejecutadas y cualquier limitación o
   decisión pendiente.
