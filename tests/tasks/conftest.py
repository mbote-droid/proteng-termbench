"""Shared helpers: run a task's reference solution once per seed and grade variations of its output."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import protengbench.tasks  # noqa: F401  (registers the tasks)
from protengbench.checks.base import Status
from protengbench.core import registry
from protengbench.core.seeds import from_hex

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = {
    "t01-model-triage": "triage.py",
    "t02-stability-design": "design_variant.py",
    "t03-md-switch-kinetics": "switch_analysis.py",
    "t04-spr-kinetics": "spr_fit.py",
    "t05-qsar-scaffold-hop": "qsar_model.py",
    "t06-inhibitor-mechanism": "inhibitor_fit.py",
}
_CACHE: dict[tuple[str, int], tuple[Any, Path]] = {}
_TMP = Path(tempfile.mkdtemp(prefix="proteng-tests-"))


def dev_seeds(task_id: str) -> list[int]:
    path = ROOT / "tasks" / task_id / "tests" / "dev_seeds.txt"
    return [from_hex(line) for line in path.read_text().split() if line.strip()]


def reference(task_id: str, seed: int) -> tuple[Any, Path]:
    """(case, output directory) of the reference solution for one seed; cached."""
    key = (task_id, seed)
    if key not in _CACHE:
        task = registry.get(task_id)
        case = task.make_case(seed)
        base = _TMP / f"{task_id}-{seed:032x}"
        (base / "in").mkdir(parents=True)
        (base / "out").mkdir()
        task.write_inputs(case, base / "in")
        script = ROOT / "tasks" / task_id / "solution" / SCRIPTS[task_id]
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--indir",
                str(base / "in"),
                "--outdir",
                str(base / "out"),
            ],
            check=True,
            capture_output=True,
        )
        _CACHE[key] = (case, base / "out")
    return _CACHE[key]


def find_seed(task_id: str, predicate: Callable[[Any], bool]) -> int:
    """First dev seed whose case satisfies `predicate` (some negative controls need a kind of case)."""
    task = registry.get(task_id)
    for seed in dev_seeds(task_id):
        if predicate(task.make_case(seed)):
            return seed
    pytest.skip("no dev seed has the case kind this control needs")
    raise AssertionError  # unreachable


def grade(task_id: str, case: Any, outdir: Path) -> dict[str, Status]:
    results = registry.get(task_id).evaluate_outputs(outdir, case)
    return {r.code: r.status for r in results}


def failed(task_id: str, case: Any, outdir: Path) -> list[str]:
    return [code for code, st in grade(task_id, case, outdir).items() if st is Status.FAIL]


def variant(src: Path, edit: Callable[[Path], None]) -> Path:
    """Copy an output directory and apply `edit` to the copy."""
    dst = Path(tempfile.mkdtemp(dir=_TMP))
    shutil.rmtree(dst)
    shutil.copytree(src, dst)
    edit(dst)
    return dst


def edit_json(name: str, fn: Callable[[dict[str, Any]], None]) -> Callable[[Path], None]:
    def apply(d: Path) -> None:
        raw = json.loads((d / name).read_text())
        fn(raw)
        (d / name).write_text(json.dumps(raw))

    return apply
