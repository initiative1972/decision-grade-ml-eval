"""Dependency-free SVG figures (deterministic, no matplotlib).

Two figures for the report:
  * ``roc_svg``        -- honest vs leaky ROC curves (Study 1)
  * ``targeting_svg``  -- cumulative-incremental (Qini-style) curves for
                          propensity vs uplift vs random targeting (Study 2)

SVG strings are built from the arrays, so they render identically in CI and in
a browser.
"""
from __future__ import annotations

import numpy as np

W, H = 560, 440
PAD_L, PAD_R, PAD_T, PAD_B = 70, 30, 40, 55
PW = W - PAD_L - PAD_R
PH = H - PAD_T - PAD_B


def _px(x: float, x0: float, x1: float) -> float:
    return PAD_L + (x - x0) / (x1 - x0) * PW if x1 > x0 else PAD_L


def _py(y: float, y0: float, y1: float) -> float:
    return PAD_T + PH - (y - y0) / (y1 - y0) * PH if y1 > y0 else PAD_T + PH


def _polyline(xs, ys, x0, x1, y0, y1, color, width=2.4):
    pts = " ".join(
        f"{_px(x, x0, x1):.1f},{_py(y, y0, y1):.1f}" for x, y in zip(xs, ys)
    )
    return f'<polyline fill="none" stroke="{color}" stroke-width="{width}" points="{pts}" />'


def _frame(title: str, x_label: str, y_label: str) -> str:
    return (
        f'<rect width="{W}" height="{H}" fill="#ffffff" />'
        f'<text x="{W / 2:.0f}" y="24" font-size="15" text-anchor="middle" '
        f'fill="#111">{title}</text>'
        f'<line x1="{PAD_L}" y1="{PAD_T + PH}" x2="{PAD_L + PW}" '
        f'y2="{PAD_T + PH}" stroke="#333" />'
        f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{PAD_T + PH}" stroke="#333" />'
        f'<text x="{PAD_L + PW / 2:.0f}" y="{H - 16}" font-size="12" '
        f'text-anchor="middle" fill="#333">{x_label}</text>'
        f'<text x="18" y="{PAD_T + PH / 2:.0f}" font-size="12" fill="#333" '
        f'transform="rotate(-90 18 {PAD_T + PH / 2:.0f})">{y_label}</text>'
    )


def _legend(items, x, y):
    out = ""
    for i, (label, color) in enumerate(items):
        yy = y + i * 18
        out += (
            f'<rect x="{x}" y="{yy - 10}" width="12" height="12" fill="{color}" />'
            f'<text x="{x + 18}" y="{yy}" font-size="12" fill="#333">{label}</text>'
        )
    return out


def _svg(body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="sans-serif">{body}</svg>'
    )


def roc_svg(curves: dict, auc_honest: float, auc_leaky: float) -> str:
    frame = _frame("Study 1 — ROC: honest vs leaky model",
                   "false positive rate", "true positive rate")
    diag = (f'<line x1="{_px(0, 0, 1):.1f}" y1="{_py(0, 0, 1):.1f}" '
            f'x2="{_px(1, 0, 1):.1f}" y2="{_py(1, 0, 1):.1f}" '
            'stroke="#bbb" stroke-dasharray="4 3" />')
    fpr_l, tpr_l = curves["leaky"]
    fpr_h, tpr_h = curves["honest"]
    leaky = _polyline(fpr_l, tpr_l, 0, 1, 0, 1, "#d9534f")
    honest = _polyline(fpr_h, tpr_h, 0, 1, 0, 1, "#0275d8")
    legend = _legend(
        [(f"leaky  (AUC {auc_leaky:.2f})", "#d9534f"),
         (f"honest (AUC {auc_honest:.2f})", "#0275d8")],
        PAD_L + PW - 180, PAD_T + PH - 40,
    )
    return _svg(frame + diag + leaky + honest + legend)


def targeting_svg(curves: dict) -> str:
    frame = _frame("Study 2 — cumulative incremental conversions",
                   "fraction of customers targeted", "cumulative incremental conversions")
    all_vals = np.concatenate([curves[k][1] for k in curves])
    y0 = float(min(all_vals.min(), 0.0))
    y1 = float(all_vals.max())
    zero = (f'<line x1="{PAD_L}" y1="{_py(0, y0, y1):.1f}" '
            f'x2="{PAD_L + PW}" y2="{_py(0, y0, y1):.1f}" '
            'stroke="#bbb" stroke-dasharray="4 3" />')
    fp, vp = curves["propensity"]
    fu, vu = curves["uplift"]
    fr, vr = curves["random"]
    prop = _polyline(fp, vp, 0, 1, y0, y1, "#d9534f")
    upl = _polyline(fu, vu, 0, 1, y0, y1, "#0275d8")
    rand = _polyline(fr, vr, 0, 1, y0, y1, "#999", width=1.6)
    legend = _legend(
        [("uplift ranking", "#0275d8"),
         ("propensity ranking", "#d9534f"),
         ("random", "#999")],
        PAD_L + 14, PAD_T + 20,
    )
    return _svg(frame + zero + rand + prop + upl + legend)
