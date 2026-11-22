"""FR-3: the expert reference solution passes every check on the public dev seeds."""

from __future__ import annotations

import pytest
from conftest import SCRIPTS, dev_seeds, failed, reference

pytestmark = pytest.mark.slow
N_SEEDS = {"t06-inhibitor-mechanism": 1, "t04-spr-kinetics": 2}


@pytest.mark.parametrize("task_id", sorted(SCRIPTS))
def test_reference_passes_dev_seeds(task_id: str) -> None:
    for seed in dev_seeds(task_id)[: N_SEEDS.get(task_id, 3)]:
        case, out = reference(task_id, seed)
        assert failed(task_id, case, out) == [], f"{task_id} seed {seed:032x}"


@pytest.mark.parametrize(
    "seed_hex", ["9709e2724c4e0c661f828d7812a0a2ee", "e7f2295e9c7dd6b285ed63b5b307355b"]
)
def test_t01_regression_hinge_adjacent_residues_are_not_penalised(seed_hex: str) -> None:
    """Found by the validation campaign: an NMP rotation whose first residues lie on the
    hinge axis moves them by less than the noise, so the reference may start the moved
    segment up to three residues early. The boundary tolerance now accounts for that."""
    from conftest import from_hex

    case, out = reference("t01-model-triage", from_hex(seed_hex))
    assert case.slack is not None and case.slack[0] >= 1
    assert failed("t01-model-triage", case, out) == []
