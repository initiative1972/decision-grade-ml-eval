import importlib.util
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reproduce_all_runs_and_writes_artifacts():
    module = _load("reproduce_all")
    assert module.main() == 0
    assert (ROOT / "results" / "leakage_results.json").exists()
    assert (ROOT / "results" / "uplift_results.json").exists()
    assert (ROOT / "paper" / "figures" / "study1_roc.svg").exists()
    assert (ROOT / "paper" / "figures" / "study2_targeting.svg").exists()
