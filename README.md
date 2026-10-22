# decision-grade-ml-eval
**The leak is the finding.** Two small, fully reproducible synthetic studies on
evaluating ML *for a decision* — target leakage caught by a feature-contract audit,
and the propensity-vs-uplift trap caught by the right causal estimand.

[![CI](https://img.shields.io/badge/CI-flake8%20%2B%20pytest-brightgreen)](../../actions)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![tests](https://img.shields.io/badge/tests-18%20passing-brightgreen)](../../actions)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-pending%20Zenodo-lightgrey)](#how-to-cite)

> A citable technical report + reproducible code. The paper is [`paper/report.md`](paper/report.md);
> the numbers and figures in it regenerate exactly from `scripts/reproduce_all.py`.

---

## The two findings

**Study 1 — target leakage.** One feature (`call_duration`) that is only knowable
*after* the outcome inflates a classifier from an honest **ROC-AUC 0.81** to a
misleading **0.94** (precision@10% 0.52 → 0.85). A one-line **feature contract** —
"is this field known at decision time?" — rejects it before the model ships.

**Study 2 — the propensity-vs-uplift trap.** Ranking a fixed-budget campaign by
*propensity to convert* fills its top decile with **94% "sleeping dogs"** — loyal
customers whose **true treatment effect is negative (−0.09)**: contacting them *loses*
~190 conversions. A two-model **uplift** estimator recovers the average effect
(**0.175** est. vs **0.171** true) and targets the persuadables instead (**+408**).

> Everything is **synthetic on purpose**: it is the only way to know the ground truth
> (the real treatment effect, the exact leak) and prove a method recovers it. No real
> or employer data is used.

---

## Why this exists

Most ML failures in production are **evaluation** failures: the model is scored on a
metric that does not match the decision it drives. This repo isolates two of the most
common and shows a cheap, auditable practice that prevents each — a *feature contract*
for leakage, and the *right estimand* for the wrong-KPI trap. It is deliberately small,
honest, and reproducible: the discipline is the deliverable, not a new algorithm.

## Reproduce it (≈ 15 s, CPU, no network)

```bash
py -3 -m pip install -r requirements-dev.txt     # numpy 2.1.1, scikit-learn 1.5.2, flake8, pytest
py -3 scripts/reproduce_all.py                   # -> results/*.json + paper/figures/*.svg
py -3 -m pytest                                  # 18 deterministic tests
py -3 -m flake8 src tests scripts                # clean
```

Single studies: `py -3 scripts/run_leakage.py`, `py -3 scripts/run_uplift.py`.
(`make all` runs lint + tests + reproduce.)

## What's in the box

```
decision-grade-ml-eval/
├── paper/
│   ├── report.md                 # THE technical report (cite this)
│   └── figures/                  # generated SVGs (ROC; targeting curves)
├── src/dgme/
│   ├── config.py                 # every tunable constant, frozen
│   ├── metrics.py                # AUC, precision@k, mean-effect-in-top-k, Qini
│   ├── leakage.py                # Study 1: data gen + feature contract + models
│   ├── uplift.py                 # Study 2: RCT gen + propensity + T-learner uplift
│   └── figures.py                # dependency-free SVG figures
├── scripts/                      # run_leakage · run_uplift · reproduce_all
├── tests/                        # 18 deterministic unit tests
├── results/                      # committed reference numbers (JSON)
├── CITATION.cff · .zenodo.json   # citation metadata
└── .github/workflows/ci.yml      # flake8 + pytest + reproduce
```

## Headline numbers (seed 20260119)

| Study 1 | Honest | Leaky |
|---|---|---|
| ROC-AUC | 0.8087 | 0.9430 |
| precision@10% | 0.5183 | 0.8508 |

| Study 2 | Propensity top-decile | Uplift top-decile |
|---|---|---|
| True treatment effect | **−0.0949** | **+0.204** |
| Net incremental conversions | **−190** | **+408** |
| True ATE 0.171 · T-learner estimate | **0.175** | |

## Reproducibility & honesty

- **Deterministic:** shared seed, pinned `numpy`/`scikit-learn`; committed `results/`
  reproduce exactly on the reference env (Python 3.13.16; CI runs 3.11). Unit tests use
  tolerant bands so the *conclusions* hold across environments.
- **Synthetic by design**, so ground truth is known; mechanisms are realistic, exact
  magnitudes are not claims about any real dataset.
- **Scope:** logistic regression throughout — the failures are properties of the
  *evaluation*, not the estimator (stronger models inherit the same traps).

## How to cite

Citation metadata is in [`CITATION.cff`](CITATION.cff) (GitHub shows a *"Cite this
repository"* button). To mint a DOI, enable this repo in **Zenodo** and cut a GitHub
release; Zenodo archives it and issues a DOI, after which the badge and
`CITATION.cff` `doi:` field are updated. Suggested citation:

> Yan, H. (2026). *The Leak Is the Finding: Feature-Contract Auditing and the
> Propensity-vs-Uplift Trap in Decision-Grade ML Evaluation* (v1.0.0). Zenodo.
> https://doi.org/10.5281

---

*Author: Henry Yan · github.com/initiative1972 · Synthetic data only; no employer data
used or reproduced. Code MIT; report prose CC BY 4.0.*
