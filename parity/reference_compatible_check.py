"""Check the runner's reference-compatible widest-horizon IBS/IBLL.

Usage: python reference_compatible_check.py <authors' repo> <pyhealth repo>

Compares experiments/run_survival_mdn.py's reference_compatible_widest()
with the authors' own metrics.py, on parity_check.py's six synthetic test
sets (three with the last test time censored, three without). Prints the
largest absolute difference per case.
"""

import contextlib
import importlib.util
import io
import runpy
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
ref_path, pyhealth_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, pyhealth_path)

# parity_check.py runs its own comparison on import; silence that output and
# keep its make_case() and reference() helpers.
sys.argv = ["parity_check.py", ref_path, pyhealth_path]
with contextlib.redirect_stdout(io.StringIO()):
    parity = runpy.run_path(str(here / "parity_check.py"))

spec = importlib.util.spec_from_file_location(
    "runner", here.parent / "experiments" / "run_survival_mdn.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

for last_censored in (False, True):
    for seed in (0, 1, 2):
        times, events, predict = parity["make_case"](400, seed, last_censored)
        theirs = parity["reference"](times, events, predict)
        ours = runner.reference_compatible_widest(times, events, predict)
        gap = max(abs(ours[k] - theirs[k]) for k in ours)
        print(f"last_censored={last_censored} seed={seed} max|diff|={gap:.1e}")
