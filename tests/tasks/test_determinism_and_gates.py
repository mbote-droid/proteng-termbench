"""FR-2 (same seed, same data and truth), gates, and one-line shortcuts that must fail."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from conftest import SCRIPTS, dev_seeds, failed, grade

from protengbench.checks.base import Status
from protengbench.core import registry

TASKS = sorted(SCRIPTS)


def _digest(directory: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()


@pytest.mark.parametrize("task_id", TASKS)
def test_same_seed_same_inputs(task_id: str, tmp_path: Path) -> None:
    task = registry.get(task_id)
    seed = dev_seeds(task_id)[0]
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    task.write_inputs(task.make_case(seed), a)
    registry.get(task_id).write_inputs(registry.get(task_id).make_case(seed), b)
    assert _digest(a) == _digest(b)


@pytest.mark.parametrize("task_id", TASKS)
def test_different_seeds_different_inputs(task_id: str, tmp_path: Path) -> None:
    task = registry.get(task_id)
    s1, s2 = dev_seeds(task_id)[:2]
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    task.write_inputs(task.make_case(s1), a)
    task.write_inputs(task.make_case(s2), b)
    assert _digest(a) != _digest(b)


@pytest.mark.parametrize("task_id", TASKS)
def test_empty_output_fails_at_the_gate_and_skips_the_rest(task_id: str, tmp_path: Path) -> None:
    case = registry.get(task_id).make_case(dev_seeds(task_id)[0])
    result = grade(task_id, case, tmp_path)
    assert result["FILES"] is Status.FAIL
    assert all(st is Status.SKIPPED for code, st in result.items() if code not in ("FILES",))


@pytest.mark.parametrize("task_id", TASKS)
def test_garbage_files_fail_the_schema_gate(task_id: str, tmp_path: Path) -> None:
    names = {
        "t01-model-triage": ["triage.json"],
        "t02-stability-design": ["design.json"],
        "t03-md-switch-kinetics": ["analysis.json", "states.csv"],
        "t04-spr-kinetics": ["fit.json"],
        "t05-qsar-scaffold-hop": ["predictions.csv", "report.json"],
        "t06-inhibitor-mechanism": ["inhibition.json"],
    }[task_id]
    for n in names:
        (tmp_path / n).write_text("[]" if n.endswith(".json") else "x,y\n")
    case = registry.get(task_id).make_case(dev_seeds(task_id)[0])
    codes = failed(task_id, case, tmp_path)
    assert codes[0] == "SCHEMA", codes
