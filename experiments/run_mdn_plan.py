"""Run NJ's 180 SurvivalMDN fits from the course run plan.

The plan (course-run-plan/run-list.json, from Aritra's Oct 26 handoff) lists
one command per fit with placeholder paths. This script swaps in your real
paths and runs the MDN fits one at a time with the PR's runner
(examples/survival_benchmark/run_survival_mdn.py), in this order:

  1. tuning: the three candidate configs (K=10 softplus, slower learning
     rate, wider network) on all 10 splits of GBSG, then SUPPORT (60 fits)
  2. ablations: K = 1, 2, 3, 5, 20 and K=10 with the exp transform (120 fits)

It is safe to stop and rerun. The runner reuses every completed, verified
fit and never overwrites one. If a fit fails, the script stops and prints
the end of that fit's log; nothing else is touched.

Example (from anywhere, with the survmdn environment active):
    python run_mdn_plan.py --pyhealth ~/PyHealth \\
        --gbsg-root ~/survmdn-data/csv --support-root ~/survmdn-data/support2 \\
        --released-root ~/survmdn-data/Survival-MDN --runs-root ~/runs/course

Add --dry-run to print the commands, --stage tuning to run only the first
block, --dataset gbsg to run one dataset, or --limit N to run the first N
pending fits.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TUNING_ROLES = {"tuning_and_selected_main", "main_candidate_and_ablation"}


def load_fits(plan_dir: Path) -> list[dict]:
    fits = json.loads((plan_dir / "run-list.json").read_text())
    mdn = [f for f in fits if f["model"] == "mdn" and f["owner"] == "NJ"]
    for f in mdn:
        f["stage"] = "tuning" if f["role"] in TUNING_ROLES else "ablations"
    order = {"tuning": 0, "ablations": 1}
    datasets = {"gbsg": 0, "support2": 1}
    return sorted(
        mdn,
        key=lambda f: (
            order[f["stage"]],
            datasets[f["dataset"]],
            f["config"],
            f["split"],
        ),
    )


def localize(command: list[str], paths: dict[str, Path]) -> list[str]:
    """Swap the plan's placeholder paths for real ones (longest first)."""
    out = []
    for arg in command:
        for placeholder in sorted(paths, key=len, reverse=True):
            if arg == placeholder or arg.startswith(placeholder + "/"):
                arg = str(paths[placeholder]) + arg[len(placeholder) :]
                break
        out.append(arg)
    if out[0] == "python":
        out[0] = sys.executable
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--pyhealth",
        type=Path,
        required=True,
        help="PyHealth checkout on feat/survival-mdn (d473380 or later)",
    )
    p.add_argument("--gbsg-root", type=Path, required=True, help="Folder with gbsg.csv")
    p.add_argument(
        "--support-root",
        type=Path,
        required=True,
        help="Folder with support2.csv (first column named sno)",
    )
    p.add_argument(
        "--released-root",
        type=Path,
        required=True,
        help="The authors' Survival-MDN repository folder",
    )
    p.add_argument("--runs-root", type=Path, required=True, help="Where results go")
    p.add_argument("--plan-dir", type=Path, default=HERE / "course-run-plan")
    p.add_argument("--stage", choices=["tuning", "ablations", "all"], default="all")
    p.add_argument("--dataset", choices=["gbsg", "support2"])
    p.add_argument("--limit", type=int, help="Run at most this many fits")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    paths = {
        "/handoff/course-run-plan": a.plan_dir.expanduser().resolve(),
        "/data/Survival-MDN": a.released_root.expanduser().resolve(),
        "/data/gbsg": a.gbsg_root.expanduser().resolve(),
        "/data/support2": a.support_root.expanduser().resolve(),
        "/runs": a.runs_root.expanduser().resolve(),
    }
    checks = [
        paths["/data/gbsg"] / "gbsg.csv",
        paths["/data/support2"] / "support2.csv",
        paths["/data/Survival-MDN"] / "data/gbsg/train_1.npz",
        paths["/data/Survival-MDN"] / "data/support/train_1.npz",
        a.pyhealth.expanduser() / "examples/survival_benchmark/run_survival_mdn.py",
    ]
    missing = [str(c) for c in checks if not c.exists()]
    if missing:
        sys.exit("Missing inputs:\n  " + "\n  ".join(missing))

    fits = [
        f
        for f in load_fits(a.plan_dir)
        if (a.stage == "all" or f["stage"] == a.stage)
        and (a.dataset is None or f["dataset"] == a.dataset)
    ]
    if a.limit is not None:
        fits = fits[: a.limit]
    logs = paths["/runs"] / "_launcher_logs"
    logs.mkdir(parents=True, exist_ok=True)
    print(f"{len(fits)} fits ({a.stage}{', ' + a.dataset if a.dataset else ''})")

    start = time.perf_counter()
    for n, fit in enumerate(fits, 1):
        command = localize(fit["command"], paths)
        if a.dry_run:
            print(" ".join(command))
            continue
        log = logs / (fit["id"].replace("/", "__") + ".log")
        t0 = time.perf_counter()
        with log.open("w") as stream:
            result = subprocess.run(
                command,
                cwd=a.pyhealth.expanduser(),
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
        text = log.read_text()
        if result.returncode != 0:
            print(f"[{n}/{len(fits)}] FAILED {fit['id']} (exit {result.returncode})")
            print("".join(text.splitlines(keepends=True)[-25:]))
            sys.exit(f"Stopped. Full log: {log}")
        reused = "done, skipping" in text or re.search(r"\breus", text, re.I)
        status = "reused (already complete)" if reused else "done"
        line = (
            ""
            if reused
            else next(
                (ln for ln in text.splitlines() if re.match(r"\[\d+/\d+\] ", ln)), ""
            )
        )
        elapsed = time.perf_counter() - start
        eta = elapsed / n * (len(fits) - n)
        print(
            f"[{n}/{len(fits)}] {fit['id']}: {status} "
            f"in {time.perf_counter() - t0:.0f}s"
            f"  {line.split(':', 1)[-1].strip()}  (ETA {eta / 60:.0f} min)"
        )
    if not a.dry_run:
        print(
            f"All {len(fits)} fits finished in "
            f"{(time.perf_counter() - start) / 60:.1f} min. Logs: {logs}"
        )


if __name__ == "__main__":
    main()
