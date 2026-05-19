# BayesianML — DTU 02477

Personal repo for the DTU course 02477 Bayesian Machine Learning.

> **Note for future students:** You will get far more out of this course by solving the problems yourself and building your own utility module from scratch. Use this as a reference, not a shortcut.

## Setup

```bash
uv venv
uv sync
uv pip install -e .
```

## Key file

`src/bayesian/exam_utils.py` — collection of helper functions covering the main exam topics (Laplace approximation, GPs, variational inference, MCMC, etc.), with docstrings. A rendered PDF version is at `exam/exam_utils.py.pdf`.

## Folder structure

```
assignments/        homework notebooks and solutions (aflevering 1–3)
exam/
  old_exams/        past exam PDFs and solutions (2024–2026)
  prep/             preparation notebooks for past exams
  exam_2026.ipynb   my own exam exam
  exam_template.ipynb
  exam_utils_example.py
lectures/           lecture slides (PDF handouts)
                    (see also src/bayesian/ for exercise notebooks)
src/bayesian/
  ex1–ex12_*.ipynb  weekly exercise handouts and solutions
  exam_utils.py     shared utility module
  *.py              per-exercise helper scripts
Notes/              personal notes
```