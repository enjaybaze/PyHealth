"""Run the 300-fit course plan (MDN, Cox, DeepSurv) on one machine.

NJ and Aritra agreed on Oct 28 to run every fit in course-run-plan/run-list.json
on NJ's VM, on the CPU, with the same thread settings for every model. This
script does that in four steps:

  1. Preflight (once per output folder, about 5 minutes):
     - installed package versions equal the pinned ones
       (baseline-requirements-pinned.txt plus the core versions below)
     - the benchmark tests pass with no skipped baseline cases
     - Cox, DeepSurv and MDN on split 1 of GBSG and SUPPORT reproduce
       Aritra's PACE pilot within tolerance
  2. Baselines: Aritra's 120 Cox/DeepSurv fits
  3. MDN tuning: 60 fits (K=10 softplus, slower learning rate, wider network)
  4. MDN ablations: 120 fits (K = 1, 2, 3, 5, 20; K=10 with exp)

Every fit runs on the CPU with the same thread count, one at a time. The
runners reuse completed fits, so it is safe to stop and rerun. The script
stops at the first failure and prints that fit's log.

    python run_course_plan.py --pyhealth ~/PyHealth \\
        --gbsg-root ~/survmdn-data/csv --support-root ~/survmdn-data/support2 \\
        --released-root ~/survmdn-data/Survival-MDN --runs-root ~/runs/course

Options: --stage baselines|tuning|ablations, --dataset gbsg|support2,
--limit N, --dry-run, --threads N (default 4, as on PACE, or fewer if the
machine has fewer CPUs), and
--package FILE.tar.gz to bundle the results for Aritra when done.
"""

import argparse
import importlib.metadata as metadata
import json
import os
import platform
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = ("baselines", "tuning", "ablations")
TUNING_ROLES = {"tuning_and_selected_main", "main_candidate_and_ablation"}
CORE_VERSIONS = {
    "numpy": "2.2.6",
    "torch": "2.14.1",
    "scikit-learn": "1.7.2",
    "pandas": "2.3.3",
    "scipy": "1.18.1",
}
# Aritra's PACE CPU pilot (Oct 26 handoff, validation/cpu), split 1 seed 1,
# default settings. The sandbox reproduced GBSG exactly.
REFERENCE = {
    ("gbsg", "cox"): {"c_td": 0.6367, "ibs": 0.1763},
    ("gbsg", "deepsurv"): {"c_td": 0.6717, "ibs": 0.1645},
    ("gbsg", "mdn"): {"val_nll": 0.3557, "c_td": 0.6558},
    ("support2", "cox"): {"c_td": 0.5831, "ibs": 0.1927},
    ("support2", "deepsurv"): {"c_td": 0.5910, "ibs": 0.1905},
    ("support2", "mdn"): {"val_nll": 0.5447, "c_td": 0.6219},
}
TOLERANCE = {"c_td": 0.005, "ibs": 0.005, "val_nll": 0.002}
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


def pinned_versions() -> dict[str, str]:
    pins = {}
    for line in (HERE / "baseline-requirements-pinned.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, version = line.split("==")
            pins[name.strip()] = version.strip()
    return {**CORE_VERSIONS, **pins}


def check_versions() -> list[str]:
    problems = []
    for name, wanted in pinned_versions().items():
        try:
            found = metadata.version(name)
        except metadata.PackageNotFoundError:
            problems.append(f"{name}: not installed (need {wanted})")
            continue
        if found.split("+")[0] != wanted:
            problems.append(f"{name}: {found} installed, pinned {wanted}")
    return problems


def child_env(threads: int) -> dict[str, str]:
    env = dict(os.environ)
    env.update({var: str(threads) for var in THREAD_VARS})
    env["CUDA_VISIBLE_DEVICES"] = ""  # CPU only, for every model
    return env


def run(command, cwd: Path, env, log: Path) -> tuple[int, str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        result = subprocess.run(
            command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT
        )
    return result.returncode, log.read_text()


def load_fits(plan_dir: Path) -> list[dict]:
    fits = json.loads((plan_dir / "run-list.json").read_text())
    for f in fits:
        if f["model"] != "mdn":
            f["stage"] = "baselines"
        else:
            f["stage"] = "tuning" if f["role"] in TUNING_ROLES else "ablations"
    datasets = {"gbsg": 0, "support2": 1}
    return sorted(
        fits,
        key=lambda f: (
            STAGES.index(f["stage"]),
            datasets[f["dataset"]],
            f["model"],
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


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def git_head(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def newest_record(folder: Path, model: str) -> dict:
    pattern = "*/split_1/seed_1/run_*/record.json"
    if model != "mdn":
        pattern = f"{model}/split_1/seed_1/run_*/record.json"
    records = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not records:
        raise FileNotFoundError(f"no {model} record under {folder}")
    return json.loads(records[-1].read_text())


def preflight(a, paths, env, pyhealth: Path, runs: Path) -> dict:
    report = {"versions": check_versions(), "checks": []}
    if report["versions"]:
        return report | {"passed": False}

    code, text = run(
        [sys.executable, "-m", "unittest", "tests.core.test_survival_benchmark"],
        pyhealth,
        env,
        runs / "_preflight" / "benchmark-tests.log",
    )
    skipped = re.search(r"skipped=(\d+)", text)
    tests_ok = code == 0 and not skipped
    report["checks"].append(
        {
            "check": "tests.core.test_survival_benchmark",
            "passed": tests_ok,
            "detail": text.strip().splitlines()[-1] if text.strip() else "",
        }
    )
    if not tests_ok:
        return report | {"passed": False}

    roots = {"gbsg": paths["/data/gbsg"], "support2": paths["/data/support2"]}
    released = {
        "gbsg": paths["/data/Survival-MDN"] / "data/gbsg",
        "support2": paths["/data/Survival-MDN"] / "data/support",
    }
    for dataset in ("gbsg", "support2"):
        common = [
            "--dataset",
            dataset,
            "--root",
            str(roots[dataset]),
            "--released-dir",
            str(released[dataset]),
            "--splits",
            "1",
            "--device",
            "cpu",
        ]
        for runner, out in (
            ("run_baselines", "baselines"),
            ("run_survival_mdn", "mdn"),
        ):
            folder = runs / "_preflight" / f"{dataset}-{out}"
            command = [
                sys.executable,
                "-m",
                f"examples.survival_benchmark.{runner}",
                *common,
                "--out",
                str(folder),
            ]
            code, text = run(command, pyhealth, env, folder.with_suffix(".log"))
            if code != 0:
                report["checks"].append(
                    {
                        "check": f"{dataset} {out}",
                        "passed": False,
                        "detail": text[-2000:],
                    }
                )
                return report | {"passed": False}
            models = ("cox", "deepsurv") if out == "baselines" else ("mdn",)
            for model in models:
                record = newest_record(folder, model)
                got = {"c_td": record["test_metrics"]["c_td"]}
                if model == "mdn":
                    got["val_nll"] = record["val_nll"]
                else:
                    got["ibs"] = record["test_metrics"]["ibs"]
                expected = REFERENCE[(dataset, model)]
                ok = all(abs(got[k] - v) <= TOLERANCE[k] for k, v in expected.items())
                report["checks"].append(
                    {
                        "check": f"{dataset} {model} split 1 vs PACE pilot",
                        "passed": ok,
                        "detail": {
                            k: [round(got[k], 4), v] for k, v in expected.items()
                        },
                    }
                )
    report["passed"] = all(c["passed"] for c in report["checks"])
    return report


def package(runs: Path, target: Path) -> None:
    def skip_cache(info: tarfile.TarInfo):
        return None if "/cache" in f"/{info.name}" else info

    with tarfile.open(target, "w:gz") as tar:
        tar.add(runs, arcname=runs.name, filter=skip_cache)
    print(
        f"Wrote {target} ({target.stat().st_size / 1e6:.1f} MB), cache folders left out"
    )


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
    p.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    p.add_argument("--dataset", choices=["gbsg", "support2"])
    p.add_argument("--limit", type=int, help="Run at most this many fits")
    p.add_argument(
        "--threads",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="CPU threads per fit, the same for every model (default: 4, or "
        "fewer if the machine has fewer CPUs)",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--package",
        type=Path,
        metavar="FILE.tar.gz",
        help="Bundle --runs-root for sharing, then exit",
    )
    a = p.parse_args()

    runs = a.runs_root.expanduser().resolve()
    if a.package:
        package(runs, a.package.expanduser().resolve())
        return

    pyhealth = a.pyhealth.expanduser().resolve()
    paths = {
        "/handoff/course-run-plan": a.plan_dir.expanduser().resolve(),
        "/data/Survival-MDN": a.released_root.expanduser().resolve(),
        "/data/gbsg": a.gbsg_root.expanduser().resolve(),
        "/data/support2": a.support_root.expanduser().resolve(),
        "/runs": runs,
    }
    needed = [
        paths["/data/gbsg"] / "gbsg.csv",
        paths["/data/support2"] / "support2.csv",
        paths["/data/Survival-MDN"] / "data/gbsg/train_1.npz",
        paths["/data/Survival-MDN"] / "data/support/train_1.npz",
        pyhealth / "examples/survival_benchmark/run_baselines.py",
    ]
    missing = [str(n) for n in needed if not n.exists()]
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
    env = child_env(a.threads)
    if a.dry_run:
        print(f"{len(fits)} fits; threads={a.threads}; CPU only")
        for f in fits:
            print(" ".join(localize(f["command"], paths)))
        return

    runs.mkdir(parents=True, exist_ok=True)
    environment = {
        "python": platform.python_version(),
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "threads_per_fit": a.threads,
        "device": "cpu",
        "versions": {
            n: metadata.version(n) for n in pinned_versions() if _installed(n)
        },
        "pyhealth_commit": git_head(pyhealth),
        "team_notes_commit": git_head(HERE),
    }
    preflight_file = runs / "preflight.json"
    previous = json.loads(preflight_file.read_text()) if preflight_file.exists() else {}
    same_setup = (
        previous.get("passed")
        and previous.get("environment", {}).get("versions") == environment["versions"]
        and previous.get("environment", {}).get("threads_per_fit") == a.threads
        and previous.get("environment", {}).get("pyhealth_commit")
        == environment["pyhealth_commit"]
    )
    if not same_setup:
        print(
            "Preflight: versions, benchmark tests, split-1 check against the "
            "PACE pilot (about 5 min)..."
        )
        report = preflight(a, paths, env, pyhealth, runs)
        report["environment"] = environment
        preflight_file.write_text(json.dumps(report, indent=2) + "\n")
        for problem in report["versions"]:
            print("  version:", problem)
        for check in report["checks"]:
            mark = "ok  " if check["passed"] else "FAIL"
            print(f"  {mark} {check['check']}: {check['detail']}")
        if not report["passed"]:
            sys.exit(f"Preflight failed; nothing launched. Details: {preflight_file}")
        print("Preflight passed.")
    (runs / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")

    logs = runs / "_launcher_logs"
    print(
        f"{len(fits)} fits ({a.stage}{', ' + a.dataset if a.dataset else ''}), "
        f"CPU only, {a.threads} threads each"
    )
    start = time.perf_counter()
    for n, fit in enumerate(fits, 1):
        command = localize(fit["command"], paths)
        t0 = time.perf_counter()
        code, text = run(
            command, pyhealth, env, logs / (fit["id"].replace("/", "__") + ".log")
        )
        if code != 0:
            print(f"[{n}/{len(fits)}] FAILED {fit['id']} (exit {code})")
            print("".join(text.splitlines(keepends=True)[-25:]))
            sys.exit(
                "Stopped. Rerun the same command after fixing; finished fits "
                "are reused."
            )
        reused = (
            "done, skipping" in text
            or "Verified existing record" in text
            or re.search(r"\breus", text, re.I)
        )
        summary = (
            ""
            if reused
            else next(
                (
                    ln.split(":", 1)[-1].strip()
                    for ln in reversed(text.splitlines())
                    if "C^td" in ln
                ),
                "",
            )
        )
        eta = (time.perf_counter() - start) / n * (len(fits) - n)
        print(
            f"[{n}/{len(fits)}] {fit['id']}: "
            f"{'reused (already complete)' if reused else 'done'} "
            f"in {time.perf_counter() - t0:.0f}s  {summary}  (ETA {eta / 60:.0f} min)"
        )
    print(
        f"All {len(fits)} fits finished in "
        f"{(time.perf_counter() - start) / 60:.1f} min. Logs: {logs}"
    )


def _installed(name: str) -> bool:
    try:
        metadata.version(name)
        return True
    except metadata.PackageNotFoundError:
        return False


if __name__ == "__main__":
    main()
