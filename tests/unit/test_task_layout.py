"""FR-1: the suite holds six tasks and each ships everything a Harbor-style task needs."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import protengbench.tasks  # noqa: F401
from protengbench.core import registry
from protengbench.core.seeds import SEED_HEX_LEN

ROOT = Path(__file__).resolve().parents[2]
TASKS = sorted(registry.task_ids())


def test_six_tasks_cover_the_domains() -> None:
    assert TASKS == [
        "t01-model-triage",
        "t02-stability-design",
        "t03-md-switch-kinetics",
        "t04-spr-kinetics",
        "t05-qsar-scaffold-hop",
        "t06-inhibitor-mechanism",
    ]


@pytest.mark.parametrize("task_id", TASKS)
def test_task_folder_is_complete(task_id: str) -> None:
    d = ROOT / "tasks" / task_id
    for rel in (
        "instruction.md",
        "task.toml",
        "solution/solve.sh",
        "tests/test.sh",
        "tests/Dockerfile",
        "tests/dev_seeds.txt",
        "environment/Dockerfile",
    ):
        assert (d / rel).is_file(), f"{task_id}: missing {rel}"
    assert (d / "environment" / "data").is_dir() and any((d / "environment" / "data").iterdir())
    assert any((d / "solution").glob("*.py")), "no reference solution"
    assert "harbor-canary" in (d / "instruction.md").read_text()


@pytest.mark.parametrize("task_id", TASKS)
def test_task_toml_is_consistent(task_id: str) -> None:
    meta = tomllib.loads((ROOT / "tasks" / task_id / "task.toml").read_text())
    assert meta["task"]["name"] == f"proteng-termbench/{task_id}"
    assert meta["agent"]["network_mode"] == "no-network"
    assert meta["verifier"]["network_mode"] == "no-network"
    assert meta["verifier"]["environment_mode"] == "separate"
    assert meta["environment"]["cpus"] == 1
    script = meta["artifacts"][0].rsplit("/", 1)[1]
    assert (ROOT / "tasks" / task_id / "solution" / script).is_file()
    assert script in (ROOT / "tasks" / task_id / "tests" / "test.sh").read_text()
    assert task_id in (ROOT / "tasks" / task_id / "tests" / "test.sh").read_text()


@pytest.mark.parametrize("task_id", TASKS)
def test_dev_seeds_are_128_bit_and_distinct(task_id: str) -> None:
    seeds = (ROOT / "tasks" / task_id / "tests" / "dev_seeds.txt").read_text().split()
    assert len(seeds) >= 4 and len(set(seeds)) == len(seeds)
    assert all(len(s) == SEED_HEX_LEN for s in seeds)


@pytest.mark.parametrize("task_id", TASKS)
def test_example_data_contains_no_truth(task_id: str) -> None:
    """The image shipped to the agent must not carry answer keys or seeds."""
    for path in (ROOT / "tasks" / task_id / "environment" / "data").iterdir():
        name = path.name.lower()
        assert not any(w in name for w in ("truth", "answer", "seed", "solution")), name
