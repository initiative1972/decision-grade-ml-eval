"""decision-grade-ml-eval (dgme).

Two reproducible synthetic case studies on evaluating ML for a decision:

  * ``leakage``  -- a post-outcome feature inflates a classifier; a
                    feature-contract audit recovers honest performance.
  * ``uplift``   -- ranking by propensity-to-convert selects a
                    negative-treatment-effect segment; a two-model uplift
                    estimator recovers the incremental effect.

Everything is seeded and deterministic: the committed numbers in ``results/``
and the figures in ``paper/figures/`` regenerate exactly from code.
"""

__version__ = "1.0.0"
