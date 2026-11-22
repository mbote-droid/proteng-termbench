#!/usr/bin/env python3
"""Validation campaign: false-reject rate of the references, false-accept rate of wrong methods.

Fresh 128-bit seeds are drawn for every run and recorded, so the result is reproducible
from its own file. Each submission is run as a subprocess exactly as the grader runs it
(single thread, own directory) and graded with the task's checks.

    uv run python validation/campaign.py --out validation/results --n 20
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance

import protengbench.tasks  # noqa: F401
from protengbench.checks.stats import wilson_interval
from protengbench.core import registry
from protengbench.core.scoring import Status
from protengbench.core.seeds import new_seed, to_hex

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = {
    "t01-model-triage": "triage.py",
    "t02-stability-design": "design_variant.py",
    "t03-md-switch-kinetics": "switch_analysis.py",
    "t04-spr-kinetics": "spr_fit.py",
    "t05-qsar-scaffold-hop": "qsar_model.py",
    "t06-inhibitor-mechanism": "inhibitor_fit.py",
}
BASELINES = {
    "t02-stability-design": ("additive_top5", ROOT / "validation/baselines/naive_t02.py"),
    "t05-qsar-scaffold-hop": (
        "fingerprint_rf_random_cv",
        ROOT / "validation/baselines/naive_t05.py",
    ),
}
N_SCALE = {
    "t04-spr-kinetics": 0.4,
    "t06-inhibitor-mechanism": 0.25,
}  # these references take about a minute


def run_once(task_id: str, script: Path, seed: int) -> dict[str, Any]:
    task = registry.get(task_id)
    case = task.make_case(seed)
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        (base / "in").mkdir()
        (base / "out").mkdir()
        task.write_inputs(case, base / "in")
        t0 = time.monotonic()
        proc = subprocess.run(
            [
                sys.executable,
                str(script),
                "--indir",
                str(base / "in"),
                "--outdir",
                str(base / "out"),
            ],
            capture_output=True,
            text=True,
            timeout=task.timeout_s * 2,
            check=False,
            env={
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "PATH": "/usr/bin:/bin",
                "VIRTUAL_ENV": sys.prefix,
                "PYTHONHASHSEED": "0",
            },
        )
        elapsed = time.monotonic() - t0
        if proc.returncode != 0:
            return {
                "seed": to_hex(seed),
                "run_error": proc.stderr[-300:],
                "seconds": elapsed,
                "failed": ["RUN"],
            }
        results = task.evaluate_outputs(base / "out", case)
    return {
        "seed": to_hex(seed),
        "seconds": round(elapsed, 1),
        "failed": [r.code for r in results if r.status is Status.FAIL],
        "measured": {r.code: [r.measured, r.threshold] for r in results if r.measured is not None},
    }


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    passed = sum(1 for r in rows if not r["failed"])
    codes: dict[str, int] = {}
    for r in rows:
        for c in r["failed"]:
            codes[c] = codes.get(c, 0) + 1
    return {
        "n": n,
        "passed": passed,
        "pass_rate_wilson95": [round(v, 3) for v in wilson_interval(passed, n)],
        "failures_by_check": dict(sorted(codes.items())),
        "max_seconds": max(r["seconds"] for r in rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--tasks", nargs="*", default=list(REFERENCES))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for task_id in args.tasks:
        n = max(3, int(args.n * N_SCALE.get(task_id, 1.0)))
        seeds = [new_seed() for _ in range(n)]
        methods: dict[str, Path] = {
            "reference": ROOT / "tasks" / task_id / "solution" / REFERENCES[task_id]
        }
        if task_id in BASELINES:
            name, path = BASELINES[task_id]
            methods[name] = path
        record: dict[str, Any] = {
            "task": task_id,
            "version": registry.get(task_id).version,
            "provenance": provenance.collect(),
            "methods": {},
        }
        for name, script in methods.items():
            rows = [run_once(task_id, script, s) for s in seeds]
            record["methods"][name] = {"summary": summarise(rows), "runs": rows}
            s = record["methods"][name]["summary"]
            print(
                f"{task_id:26s} {name:26s} pass {s['passed']}/{s['n']} fails {s['failures_by_check']}",
                flush=True,
            )
        (args.out / f"{task_id}.json").write_text(
            json.dumps(record, indent=1, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
