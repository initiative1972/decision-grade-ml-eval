# reproduce_all.py

Reproduce both studies end-to-end: write `results/*.json` and `paper/figures/*.svg`.

Deterministic and offline. Exit code `0` means both studies ran and the headline effects held:

- Leak inflates AUC
- Propensity top-decile has a negative true effect
- Uplift top-decile has a positive true effect

---

## Source Code

```python
#!/usr/bin/env python3
"""
Reproduce both studies end-to-end: write results/*.json and paper/figures/*.svg.
Deterministic and offline. Exit code 0 means both studies ran and the headline
effects held (leak inflates AUC; propensity top-decile has a negative true
effect while uplift's is positive).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")  # quiet a benign scipy/sklearn solver notice

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dgme import config as C
from dgme import figures, leakage, uplift


def main() -> int:
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    leak = leakage.run()
    up = uplift.run()

    (C.RESULTS_DIR / "leakage_results.json").write_text(
        json.dumps(leak.as_dict(), indent=2),
        encoding="utf-8",
    )

    (C.RESULTS_DIR / "uplift_results.json").write_text(
        json.dumps(up.as_dict(), indent=2),
        encoding="utf-8",
    )

    roc = leakage.roc_curves()
    (C.FIGURES_DIR / "study1_roc.svg").write_text(
        figures.roc_svg(
            roc,
            leak.auc_honest,
            leak.auc_leaky,
        ),
        encoding="utf-8",
    )

    curves = uplift.targeting_curves()
    (C.FIGURES_DIR / "study2_targeting.svg").write_text(
        figures.targeting_svg(curves),
        encoding="utf-8",
    )

    print("== Study 1: target leakage ==")
    print(f"  base rate              : {leak.base_rate}")
    print(f"  AUC honest             : {leak.auc_honest}")
    print(f"  AUC leaky              : {leak.auc_leaky}")
    print(f"  AUC inflation          : {leak.auc_inflation}")
    print(f"  precision@10 honest    : {leak.precision_at_10_honest}")
    print(f"  precision@10 leaky     : {leak.precision_at_10_leaky}")
    print(f"  features failing audit : {leak.leaking_features}")

    print("== Study 2: propensity vs uplift ==")
    print(f"  true ATE               : {up.true_ate}")
    print(f"  estimated ATE (T-learn): {up.estimated_ate_tlearner}")
    print(f"  diff-in-means ATE      : {up.diff_in_means_ate}")
    print(f"  conversion@10 propens. : {up.conversion_at_10_by_propensity}")
    print(f"  TRUE effect top10 prop : {up.true_effect_in_top10_propensity}")
    print(f"  TRUE effect top10 upl  : {up.true_effect_in_top10_uplift}")
    print(f"  sleeping-dog share top : {up.sleeping_dog_share_top10_propensity}")

    # Guard the narrative: the studies must actually show what the report claims.
    ok = (
        leak.auc_leaky > leak.auc_honest
        and up.true_effect_in_top10_propensity < 0
        and up.true_effect_in_top10_uplift > 0
    )

    print(f"\nheadline effects hold: {ok}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

---

## What this script does

1. Creates results and figures directories if they do not already exist.
2. Runs a target-leakage study (`leakage.run()`).
3. Runs an uplift-modeling study (`uplift.run()`).
4. Saves results to:
   - `results/leakage_results.json`
   - `results/uplift_results.json`
5. Generates SVG visualizations:
   - `paper/figures/study1_roc.svg`
   - `paper/figures/study2_targeting.svg`
6. Prints summary metrics for both studies.
7. Verifies the expected headline findings:
   - Leaky model AUC exceeds honest model AUC.
   - Top 10% selected by propensity has negative true effect.
   - Top 10% selected by uplift has positive true effect.
8. Returns:
   - `0` if all headline effects hold.
   - `1` otherwise.