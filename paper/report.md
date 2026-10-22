# The Leak Is the Finding: Feature-Contract Auditing and the Propensity-vs-Uplift Trap in Decision-Grade ML Evaluation

**Henry Yan** · Independent · github.com/initiative1972
Version 1.0.0 · 19 October 2026 · MIT (code) / CC BY 4.0 (prose)

---

## Abstract

Two small, fully reproducible synthetic case studies argue a single point: when a
machine-learning model informs a decision, the evaluation must match the decision,
not the leaderboard. **Study 1** shows how one plausible-looking feature that is only
knowable *after* the outcome inflates a classifier from an honest **ROC-AUC of 0.81**
to a misleading **0.94**, and how a one-line *feature-contract audit* — asking of each
field, "is this known at decision time?" — recovers the truth before the model ships.
**Study 2** shows how ranking a fixed-budget marketing campaign by *propensity to
convert* selects an already-loyal segment whose **true treatment effect is negative
(−0.09)**: contacting them *destroys* conversions, even though they look like the best
customers. A two-model (T-learner) **uplift** estimator recovers the average treatment
effect (**0.175** estimated vs **0.171** true) and ranks the genuinely persuadable
customers first (**+0.20** true effect in its top decile). Both studies use synthetic
data precisely because it exposes the ground truth an evaluator would otherwise never
see; both are deterministic and reproducible from code. The contribution is not a new
algorithm but a reusable discipline: **a feature contract to stop leakage, and the
right causal estimand to stop the wrong-KPI trap.**

**Keywords:** ML evaluation, data leakage, feature contract, uplift modeling, propensity, causal inference, reproducibility.

---

## 1. Introduction

Most ML failures in production are not modelling failures; they are *evaluation*
failures. A model is scored on a metric that does not correspond to the decision it
will drive, and the error is discovered only after it is deployed. This report
isolates two of the most common and costly such failures and shows, on ground-truth
synthetic data, (a) how each produces a confidently wrong answer and (b) a simple,
auditable practice that prevents it.

1. **Target leakage** — a feature carries information that would not exist at the
   moment a prediction must be made, so offline scores are inflated and collapse in
   production.
2. **The propensity-vs-uplift trap** — optimising "who is most likely to convert?"
   when the decision is "whom should we spend budget *influencing*?" selects people
   who would have converted anyway, and sometimes people the intervention actively
   repels.

Both are failures of *decision-grade evaluation*: picking the estimand and the metric
that correspond to the action, and quantifying what we actually know.

---

## 2. Study 1 — target leakage and the feature-contract audit

### 2.1 Setup

We simulate a direct-marketing campaign of **N = 40,000** customers deciding whether
to subscribe to a product (base rate **15.3%**). Eight "honest" customer features
(known before the call) drive the outcome through a correctly-specified logistic
model. We then add one leaky feature, `call_duration`, generated *from the label*: a
long call only happens when the customer is already subscribing. In production,
duration is unknown at scoring time — it is realised *during* the very event whose
outcome we are predicting.

We train the same logistic-regression pipeline (standardisation + L2 logistic
regression) twice: once on the honest features only, once with `call_duration` added.
Evaluation is a stratified 30% hold-out; reported metrics are ROC-AUC and
precision@10% (the decision is "whom do we call with a limited budget?", so the top
decile matters).

### 2.2 The feature contract

Before any training, each feature is tagged with whether it is **available at decision
time**. The audit is one line: reject any feature that is not.

| Feature | Available at decision time? | Verdict |
|---|---|---|
| customer_features (×8) | yes | keep |
| `call_duration` | **no** — only known after the call realises the outcome | **reject** |

### 2.3 Results

| Metric | Honest model | Leaky model | Δ |
|---|---|---|---|
| ROC-AUC | **0.8087** | **0.9430** | +0.134 |
| precision@10% | **0.5183** | **0.8508** | +0.333 |

![ROC: honest vs leaky](figures/study1_roc.svg)

The leaky model looks dramatically better by every offline metric. In production it
would fall back to roughly the honest model's performance (or worse), because
`call_duration` simply is not available when the call list is built — and the team
would have shipped a model chosen for a number it can never reproduce.

### 2.4 Discussion

The point is deliberately mundane: **the leak is the finding.** A one-line feature
contract, applied before modelling, catches it every time and costs nothing. The
discipline — not the model — is the deliverable. In practice the contract doubles as
documentation: it is the written record of *why* a tempting feature was excluded, which
is exactly what a reviewer or auditor needs.

---

## 3. Study 2 — the propensity-vs-uplift trap

### 3.1 Setup

We simulate a **randomised controlled trial** of **N = 20,000** customers, each
assigned to treatment (contacted) or control with probability 0.5. From four
covariates we construct each customer's *control* conversion probability `p0` and
*treated* conversion probability `p1`; the individual treatment effect is
`τ = p1 − p0`, which we know exactly because we generated it. Two segments exist:

- **Persuadables** (~80%): moderate baseline conversion, **positive** effect
  (τ ≈ +0.26) — contacting them helps.
- **Sleeping dogs** (~20%): already-loyal, high-value customers identified by a high
  first covariate — **high baseline conversion** but a **negative** effect
  (τ ≈ −0.12): contacting them annoys them and *reduces* conversion.

Crucially, the sleeping-dog segment is **visible in the covariates** (that is what
"high-value, already-loyal" means), so a propensity model will happily rank them at
the top. The campaign can contact only the **top 10%**.

### 3.2 Two rankings

- **Propensity** — a logistic model of `P(convert)` ignoring treatment; rank by
  predicted conversion. This answers "who will convert?".
- **Uplift (T-learner)** — fit `P(convert | treated)` and `P(convert | control)`
  separately; score each customer by the difference. This answers "whom does
  contacting *change*?".

### 3.3 Results

| Quantity | Value |
|---|---|
| True average treatment effect (ATE) | **0.171** |
| Estimated ATE (T-learner) | **0.175** |
| Estimated ATE (difference in means, RCT) | **0.175** |
| Conversion@10% under propensity ranking | **0.773** |
| **True effect in the propensity top decile** | **−0.0949** |
| True effect in the uplift top decile | **+0.204** |
| Sleeping-dog share of the propensity top decile | **93.9%** |
| Net incremental conversions, propensity top decile | **−190** |
| Net incremental conversions, uplift top decile | **+408** |

![Cumulative incremental conversions by targeting rule](figures/study2_targeting.svg)

The propensity ranking achieves the **highest** top-decile conversion rate (0.773) —
and it is a trap. That decile is **94% sleeping dogs**, whose **true effect is
negative**: contacting them yields roughly **−190** net conversions. The campaign that
"wins" on the conversion chart *loses money*. The uplift ranking, scoring the same
customers by incremental effect, targets persuadables and delivers **+408** — and the
T-learner's average-effect estimate (0.175) matches both the truth (0.171) and the
experiment's difference-in-means (0.175).

### 3.4 Discussion

Propensity answers the wrong question for a *persuasion* budget. The right estimand is
the **conditional average treatment effect**, and the right metric is incremental
conversions under the targeting rule — which can be *negative* at the top precisely
where propensity is most confident. The vanity metric (conversion@10%) and the
decision metric (incremental conversions) point in opposite directions.

---

## 4. Threats to validity

- **Synthetic data.** The data are generated, not observed. This is intentional: it is
  the only way to *know* τ and the exact leak and therefore to demonstrate that a
  method recovers them. The *mechanisms* (post-outcome features; loyal segments that
  resent contact) are common in practice; the exact magnitudes are not claims about any
  real dataset.
- **Model simplicity.** Both studies use logistic regression. The failures shown are
  properties of the *evaluation*, not the estimator; stronger models inherit the same
  traps (often more severe, because they exploit the leak better).
- **Uplift estimator.** The T-learner is the simplest meta-learner and can be biased
  under covariate shift between arms; here assignment is randomised, so it is
  well-behaved. S-/X-/R-learners are natural extensions.
- **Single seed.** Results are reported for one frozen seed; tolerant unit tests assert
  the effects hold, and the qualitative conclusions are insensitive to the seed.

---

## 5. Reproducibility

```bash
python -m pip install -r requirements-dev.txt   # numpy 2.1.1, scikit-learn 1.5.2
python scripts/reproduce_all.py                 # writes results/*.json + paper/figures/*.svg
python -m pytest                                # 18 deterministic tests
```

- **Seed:** `20260119` (shared). **Reference env:** Python 3.13.16, numpy 2.1.1,
  scikit-learn 1.5.2. numpy's PCG64 stream and scikit-learn's deterministic solvers make
  the committed numbers in `results/` reproduce exactly on the pinned versions; unit
  tests use tolerant bands so the *conclusions* hold across environments.
- All figures are generated (SVG) from the same run; nothing is hand-drawn.

---

## 6. Conclusion

Two failures, one lesson. A model is only as trustworthy as the question its evaluation
actually asks. A **feature contract** keeps the question honest in time; the **right
causal estimand** keeps it honest about cause and effect. Neither needs a bigger model
— both need the discipline to say that the impressive number is the wrong one.

---

## References

1. Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). *Leakage in Data
   Mining: Formulation, Detection, and Avoidance.* ACM TKDD.
2. Radcliffe, N. J., & Surry, P. D. (2011). *Real-World Uplift Modelling with
   Significance-Based Uplift Trees.* Stochastic Solutions white paper.
3. Gutierrez, P., & Gérardy, J.-Y. (2017). *Causal Inference and Uplift Modelling: A
   Review of the Literature.* PMLR.
4. Künzel, S., Sekhon, J., Bickel, P., & Yu, B. (2019). *Metalearners for estimating
   heterogeneous treatment effects using machine learning.* PNAS.

## How to cite

See `CITATION.cff`. After the first Zenodo release, cite with the minted DOI:

> Yan, H. (2026). *The Leak Is the Finding: Feature-Contract Auditing and the
> Propensity-vs-Uplift Trap in Decision-Grade ML Evaluation* (v1.0.0). Zenodo.
> https://doi.org/10.5281/zenodo.XXXXXXX
