# AGENTS.md

## Project

UTEC Deep Learning — Lab 01.

The goal is to **transfer the method of a paper to a different dataset**, not to reproduce
the paper. The method comes from *"Towards accurate bird sound recognition through
multi-scale texture-aware modeling"* (Qin & Huang, npj Acoustics 2025):

- **FBRS** — wavelet-packet spectrogram with energy-guided sub-band selection.
- **DLoGNet** — CNN whose kernels are Laplacian-of-Gaussian filters with learnable
  orientation `θ` and scale `σ`.

Those two ideas are what we carry over. Everything else about the paper's setup —
domain, species, labels, sample rate, clip length — is **different here** and must be
re-derived rather than copied.

## Dataset

`dataset/` (git-ignored, not committed) holds **AnuraSet**-style anuran (frog) recordings:

| | Paper | This project |
|---|---|---|
| Domain | 8 bird species | **40 especies observadas de anuros** en 42 columnas (`SCIFUS` y `SCINAS` no tienen positivos) |
| Task | single-label, softmax + cross-entropy | **multietiqueta**, hasta 8 especies por clip → sigmoid + BCE |
| Audio | 32 kHz | **22.05 kHz**, mono, 16-bit |
| Clips | 5 s | **3 s** |
| Size | ~22.7k clips | **62,191 clips** |

- `dataset/train/*.wav` — 3-second segments, named `INCT<site>_<date>_<time>_<start>_<end>.wav`.
- `dataset/train.csv` — `filename` + 42 columnas multihot; se excluyen `SCIFUS` y `SCINAS`
  del clasificador porque no contienen ejemplos positivos.
- **36% de las filas (22,504) no tienen etiquetas positivas.** Esto no demuestra por sí solo
  que sean fondo puro. Se conservan como ejemplos totalmente negativos y no se crea una clase
  de fondo separada.

## Adaptations this implies

- Classifier head and loss change: **sigmoid + BCE**, not softmax + cross-entropy.
- Metrics change: per-class / macro **precision, recall, F1 and mAP** — plain accuracy is
  meaningless for multi-label with a dominant empty class.
- FBRS band layout must be recomputed over the Nyquist interval:
  `Δf_min = (f_s / 2) / 2^L = f_s / 2^(L+1)` at **22.05 kHz**, so the paper's `L = 8`
  is a starting point, not a settled choice.
- Input size, pooling stack and receptive fields follow from 3-second clips, not 5.
- **The paper's 91.18% is not a comparable target.** Different task, dataset and metric.

## Layout

- `docs/paper-summary.md` — detailed summary of the paper: equations, architecture, results, limitations.
- `docs/fbrs-explained.md` — deep-dive on **Contribution 1 (FBRS)**: intuition, wavelet-packet
  background, the algorithm from Figs. 11–12, the math, plus our adaptation decisions
  (Part II). Read this before implementing the input pipeline.
- `src/anuraset_dl/` — código fuente reutilizable del proyecto.
- `notebooks/` — análisis exploratorios; la lógica definitiva debe vivir en `src/`.
- `configs/` — configuración versionada de los experimentos.
- `splits/` — particiones versionadas; ningún segmento de una misma grabación puede aparecer
  en más de una partición.
- `tests/` — pruebas unitarias de los componentes del pipeline.
- `outputs/` — artefactos generados; checkpoints y resultados grandes no se versionan.

## Conventions

- Python 3.12, PyTorch.
- **Idioma de la documentación:** todo documento nuevo y todo contenido documental generado
  en adelante debe escribirse en **español**. Esto incluye archivos Markdown, explicaciones,
  diagramas, tablas, comentarios narrativos y reportes del proyecto. Se pueden conservar en
  inglés los identificadores de código, nombres propios, títulos bibliográficos y términos
  técnicos cuando traducirlos reduzca la precisión. No se deben traducir nombres de clases,
  columnas, rutas, funciones, variables ni APIs.
- Keep large binaries (audio, checkpoints) out of git; add new ones to `.gitignore`.
- Never commit or push unless asked.
- Consider embebed  Mermaid and LaText in Markdown files for clarity in diagrams and Math.

## Notes for agents

- Prefer reading `docs/paper-summary.md` first — it already contains the paper's equations,
  hyperparameters and result tables.
- When the summary and this file disagree on a number, **this file wins**: the summary
  describes the paper, this file describes our setup.
