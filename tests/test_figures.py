import warnings

from dgme import leakage, uplift
from dgme.figures import roc_svg, targeting_svg

warnings.filterwarnings("ignore")


def test_roc_svg_well_formed():
    curves = leakage.roc_curves()
    svg = roc_svg(curves, 0.81, 0.94)
    assert svg.startswith("<svg")
    assert svg.rstrip().endswith("</svg>")
    assert "honest" in svg and "leaky" in svg


def test_targeting_svg_well_formed():
    curves = uplift.targeting_curves()
    svg = targeting_svg(curves)
    assert svg.startswith("<svg")
    assert svg.rstrip().endswith("</svg>")
    assert "uplift" in svg and "propensity" in svg
