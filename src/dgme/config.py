"""All tunable constants for the two studies, in one place.

The data are synthetic by design: it is the only way to *know* the ground truth
(the true treatment effect, the exact leak) and therefore to show that an
evaluation method recovers it. These constants were tuned once so the headline
effects are clear and stable; they are then frozen.
"""
from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "paper" / "figures"

# A single shared seed makes the whole artefact reproducible.
SEED = 20260119

# ----------------------------------------------------------------------------
# Study 1 — target leakage via a post-outcome feature.
# ----------------------------------------------------------------------------
N_LEAK = 40_000
LEAK_TEST_FRAC = 0.30
# Honest features: a correctly-specified logistic signal tuned to ~0.80 AUC.
LEAK_INTERCEPT = -2.25
LEAK_SIGNAL_SCALE = 0.74
LEAK_BETA = (1.10, -0.90, 0.70, 0.60, -0.50, 0.40, 0.30, -0.25)
# The leaky feature "call_duration": known only AFTER the outcome, so it is
# generated from the label. It pushes AUC to ~0.92 on its own + the signal.
DUR_BASE = 220.0
DUR_PER_LABEL = 185.0
DUR_NOISE = 100.0
DUR_X0_COUPLING = 25.0

# ----------------------------------------------------------------------------
# Study 2 — propensity-vs-uplift on a synthetic randomised trial.
# ----------------------------------------------------------------------------
N_UPLIFT = 20_000
TREATMENT_SHARE = 0.50
TOP_FRAC = 0.10                 # "top decile" targeting budget
# Baseline conversion model (control potential outcome).
UP_INTERCEPT = -1.05
UP_BETA = (0.55, 0.30, -0.20, 0.15)
# A "sleeping dogs" segment: already-loyal, high-value customers (identified by
# a high first covariate) who convert anyway AND are annoyed by contact
# (negative effect). Because the segment is visible in the covariates, a
# propensity-to-convert ranker walks straight into it — that is the trap.
HB_X0_THRESHOLD = 0.8416        # ~top 20% of the first covariate (N(0,1) 80th pct)
HB_BASELINE_BOOST = 1.65        # lifts baseline conversion for the segment
TAU_PERSUADABLE = 0.26          # positive prob. uplift for the rest
TAU_SLEEPING_DOG = -0.12        # negative prob. uplift for the segment
TAU_X0_COUPLING = 0.05          # mild heterogeneity with the first covariate
