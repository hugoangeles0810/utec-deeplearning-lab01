# Documentación

Este directorio reúne la documentación metodológica y técnica del proyecto. Los documentos se
mantienen separados según su función para distinguir las decisiones vigentes, el contexto
bibliográfico y el registro de ejecuciones.

## Orden de lectura

1. [`methodology.md`](methodology.md): fuente de verdad del protocolo, los datos, la tarea, el
   entrenamiento y la evaluación.
2. [`dataset-audit.md`](dataset-audit.md): evidencia del corte auditado y trazabilidad de los
   hallazgos que sustentan la preparación de datos.
3. [`data-preparation.md`](data-preparation.md): política de selección de etiquetas, particiones
   y verificaciones de los manifiestos.
4. [`paper-summary.md`](paper-summary.md): resumen crítico del artículo de referencia y de sus
   diferencias con este proyecto.
5. [`fbrs.md`](fbrs.md): fundamentos de FBRS, adaptación a AnuraSet, decisiones configuradas y
   verificaciones requeridas.
6. [`experiments.md`](experiments.md): requisitos, estado y resultados de los experimentos.

## Fuentes locales

`references/` contiene material bibliográfico disponible solo en el entorno local. El archivo
`references/paper.pdf` corresponde al artículo de referencia y está excluido de Git para evitar
versionar binarios grandes. La referencia bibliográfica y el DOI se conservan en
[`paper-summary.md`](paper-summary.md).

## Responsabilidad de cada documento

| Documento | Debe contener | No debe utilizarse para |
|---|---|---|
| `methodology.md` | Decisiones metodológicas vigentes | Registrar resultados de ejecuciones individuales |
| `dataset-audit.md` | Evidencia y controles del corte local auditado | Definir por sí solo la política vigente |
| `data-preparation.md` | Selección de etiquetas y creación de particiones | Definir el entrenamiento o la evaluación |
| `paper-summary.md` | Método, resultados y limitaciones del artículo | Transferir automáticamente decisiones al proyecto |
| `fbrs.md` | Diseño técnico y adaptación de FBRS | Sustituir el protocolo general del proyecto |
| `experiments.md` | Configuraciones ejecutadas, artefactos y resultados | Introducir decisiones metodológicas sin documentarlas |

Cuando una ejecución requiera cambiar el protocolo, la decisión debe actualizarse primero en
`methodology.md` o en el documento técnico correspondiente y después registrarse en
`experiments.md`.
