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
| Domain | 8 bird species | **42 anuran species** (`SPHSUR`, `BOABIS`, `DENNAH`, …) |
| Task | single-label, softmax + cross-entropy | **multi-label**, up to 7 species per clip → sigmoid + BCE |
| Audio | 32 kHz | **22.05 kHz**, mono, 16-bit |
| Clips | 5 s | **3 s** |
| Size | ~22.7k clips | **62,191 clips** |

- `dataset/train/*.wav` — 3-second segments, named `INCT<site>_<date>_<time>_<start>_<end>.wav`.
- `dataset/train.csv` — `filename` + 42 multi-hot label columns.
- **36% of rows (22,504) carry zero positive labels** — pure background. Decide explicitly
  how to treat them (drop, keep as negatives, or a background class).

## Adaptations this implies

- Classifier head and loss change: **sigmoid + BCE**, not softmax + cross-entropy.
- Metrics change: per-class / macro **precision, recall, F1 and mAP** — plain accuracy is
  meaningless for multi-label with a dominant empty class.
- FBRS band layout must be recomputed: `Δf_min = f_s / 2^L` at **22.05 kHz**, so the
  paper's `L = 8` is a starting point, not a settled choice.
- Input size, pooling stack and receptive fields follow from 3-second clips, not 5.
- **The paper's 91.18% is not a comparable target.** Different task, dataset and metric.

## Layout

- `paper-summary.md` — detailed summary of the paper: equations, architecture, results, limitations.

## Conventions

- Python 3.12, PyTorch.
- Keep large binaries (audio, checkpoints) out of git; add new ones to `.gitignore`.
- Never commit or push unless asked.

## Notes for agents

- Prefer reading `paper-summary.md` first — it already contains the paper's equations,
  hyperparameters and result tables.
- When the summary and this file disagree on a number, **this file wins**: the summary
  describes the paper, this file describes our setup.
