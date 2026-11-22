"""t03: read a trajectory: fluctuations, a conformational switch, and its kinetics.

The trajectory is a coarse-grained (C-alpha network) simulation of ubiquitin
with one segment that switches between two states (a displacement of several
Angstrom) as a two-state Markov chain, and every frame is stored in the
laboratory frame (random orientation and position), as if it came straight out
of a simulation of a tumbling molecule. The agent must align the frames,
measure fluctuations, find the switching residues, label the frames and
estimate the populations and the two rate constants.

How each output is judged, and why:

* RMSF per residue (root-mean-square fluctuation about the mean structure) is
  compared with the exact value from the internal coordinates the generator
  knows. Superposing on all C-alpha atoms is the classic mistake: the mobile
  segment drags the fit, so the rigid core appears to fluctuate and the mobile
  region is underestimated. Iterating the fit on the low-fluctuation core
  removes it.
* The switching residues are those whose mean position differs by at least
  1.5 Angstrom between the two states; the reported range must match within
  one residue at each end.
* Frame labels must agree with the hidden Markov state sequence for at
  least 97 percent of frames (label 1 is the less populated state).
* The population of the minor state and the two rate constants are compared
  with the realised sequence: the rate from state X is the number of
  transitions out of X divided by the time spent in X, which is the maximum
  likelihood estimate for the realised chain. Tolerance is 30 percent.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from protengbench.checks.base import CheckContext, CheckGroup, Outcome, check
from protengbench.checks.io import FloatArray, SchemaError
from protengbench.core.registry import register
from protengbench.core.seeds import streams
from protengbench.core.task import TaskPlugin
from protengbench.models import elastic, structure

N_FRAMES = 4000
DT_NS = 0.1
SWITCH_MIN_DISPLACEMENT = 1.5  # Angstrom between state means
RMSF_REL_TOL = 0.10
RMSF_MEAN_REL_TOL = 0.04
BOUNDARY_TOL = 1
STATE_AGREEMENT = 0.97
POPULATION_TOL = 0.03
RATE_REL_TOL = 0.30


@dataclass(frozen=True)
class SwitchCase:
    resseq: list[int]
    resname: list[str]
    xyz: FloatArray  # laboratory-frame coordinates, (frames, residues, 3), float32
    rmsf_true: FloatArray
    segment: tuple[int, int]  # residue numbers (inclusive) switching by >= 1.5 A
    state: FloatArray  # 0 = major, 1 = minor, per frame
    pop_minor: float
    rate_major_to_minor: float
    rate_minor_to_major: float


def _two_state_chain(rng: np.random.Generator, k_mm: float, k_mM: float) -> FloatArray:
    p_leave_major = 1.0 - math.exp(-k_mm * DT_NS)
    p_leave_minor = 1.0 - math.exp(-k_mM * DT_NS)
    state = np.zeros(N_FRAMES)
    s = 1 if rng.random() < k_mm / (k_mm + k_mM) else 0
    u = rng.random(N_FRAMES)
    for f in range(N_FRAMES):
        state[f] = s
        if u[f] < (p_leave_minor if s else p_leave_major):
            s = 1 - s
    return state


def _rates(state: FloatArray) -> tuple[float, float, float]:
    """Maximum-likelihood rates (1/ns) and minor population of a realised two-state chain."""
    major_to_minor = int(np.sum((state[:-1] == 0) & (state[1:] == 1)))
    minor_to_major = int(np.sum((state[:-1] == 1) & (state[1:] == 0)))
    t_major = float(np.sum(state[:-1] == 0)) * DT_NS
    t_minor = float(np.sum(state[:-1] == 1)) * DT_NS
    if t_major == 0.0 or t_minor == 0.0:  # a state never visited: no rate to estimate
        return 0.0, 0.0, float(state.mean())
    return major_to_minor / t_major, minor_to_major / t_minor, float(state.mean())


@lru_cache(maxsize=16)
def make_case(seed: int) -> SwitchCase:
    r_par, r_net, r_chain, r_rot = streams(seed, 4)
    ca, resseq, resname = structure.ca_trace(structure.read_pdb(structure.bundled("1UBI.pdb")))
    n = ca.shape[0]
    net = elastic.NetworkDynamics(
        ca, target_rmsf=r_par.uniform(0.6, 1.0), slowest_ns=r_par.uniform(1.0, 3.0)
    )
    fluct = net.sample(N_FRAMES, DT_NS, r_net)

    length = int(r_par.integers(9, 14))
    start = int(r_par.integers(8, 66 - length))  # index of first switching residue
    amp = r_par.uniform(4.0, 6.0)
    direction = r_par.normal(size=3)
    direction /= np.linalg.norm(direction)
    profile = np.zeros(n)
    profile[start : start + length] = amp * np.sin(np.pi * (np.arange(length) + 1) / (length + 1))
    delta = profile[:, None] * direction[None, :]

    while True:
        state = _two_state_chain(r_chain, r_par.uniform(0.01, 0.04), r_par.uniform(0.04, 0.12))
        k_mm, k_mM, pop = _rates(state)
        n_up = int(np.sum((state[:-1] == 0) & (state[1:] == 1)))
        n_dn = int(np.sum((state[:-1] == 1) & (state[1:] == 0)))
        if min(n_up, n_dn) >= 6 and 0.08 <= pop <= 0.5:
            break

    internal = ca[None, :, :] + fluct + state[:, None, None] * delta[None, :, :]
    rmsf_true = np.sqrt(((internal - internal.mean(axis=0)) ** 2).sum(axis=-1).mean(axis=0))
    moving = np.nonzero(profile >= SWITCH_MIN_DISPLACEMENT)[0]
    segment = (resseq[int(moving[0])], resseq[int(moving[-1])])

    rots = Rotation.random(N_FRAMES, random_state=int(r_rot.integers(0, 2**31 - 1)))
    shifts = r_rot.normal(0.0, 20.0, size=(N_FRAMES, 1, 3))
    lab = np.stack([rots[f].apply(internal[f]) for f in range(N_FRAMES)]) + shifts
    return SwitchCase(
        resseq, resname, lab.astype(np.float32), rmsf_true, segment, state, pop, k_mm, k_mM
    )


def write_inputs(case: SwitchCase, input_dir: Path) -> None:
    ref = case.xyz[0].astype(float)
    (input_dir / "topology.pdb").write_text(structure.format_ca_pdb(ref, case.resseq, case.resname))
    np.savez(input_dir / "trajectory.npz", xyz=case.xyz, time_ns=np.arange(N_FRAMES) * DT_NS)
    meta = {
        "frames": N_FRAMES,
        "frame_interval_ns": DT_NS,
        "coordinates": "C-alpha, Angstrom, laboratory frame (not superposed)",
        "residue_numbers": case.resseq,
        "state_label_convention": "1 = the less populated state, 0 = the more populated",
        "switching_definition": "residues whose mean position differs by >= 1.5 Angstrom between the two states",
    }
    (input_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


# --- checks ------------------------------------------------------------------------


def _case(ctx: CheckContext) -> SwitchCase:
    c: SwitchCase = ctx.truth
    return c


@check("FILES", gate=True)
def files_present(ctx: CheckContext) -> Outcome:
    missing = [n for n in ("analysis.json", "states.csv") if not (ctx.output_dir / n).is_file()]
    return Outcome(not missing, f"missing: {', '.join(missing)}")


@check("SCHEMA", gate=True)
def schema(ctx: CheckContext) -> Outcome:
    case = _case(ctx)
    raw = json.loads((ctx.output_dir / "analysis.json").read_text())
    if not isinstance(raw, dict):
        raise SchemaError("analysis.json: top level must be an object")
    rmsf = raw.get("rmsf_A")
    if not isinstance(rmsf, list) or len(rmsf) != len(case.resseq):
        raise SchemaError(f"rmsf_A must be a list of {len(case.resseq)} numbers")
    arr = np.array([float(x) for x in rmsf])
    if not (np.all(np.isfinite(arr)) and np.all(arr > 0)):
        raise SchemaError("rmsf_A must be finite and positive")
    seg = raw.get("switching_residues")
    if not (isinstance(seg, list) and len(seg) == 2 and all(isinstance(v, int) for v in seg)):
        raise SchemaError("switching_residues must be [first, last] residue numbers (integers)")
    nums: dict[str, float] = {}
    for key in ("population_minor", "rate_major_to_minor_per_ns", "rate_minor_to_major_per_ns"):
        v = raw.get(key)
        if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v) or v <= 0:
            raise SchemaError(f"{key} must be a positive number")
        nums[key] = float(v)
    lines = (ctx.output_dir / "states.csv").read_text().strip().splitlines()
    if lines[0].strip() != "frame,state" or len(lines) != N_FRAMES + 1:
        raise SchemaError(f"states.csv must have header 'frame,state' and {N_FRAMES} rows")
    labels = np.empty(N_FRAMES)
    for i, line in enumerate(lines[1:]):
        f, s = line.split(",")
        if int(f) != i or s.strip() not in ("0", "1"):
            raise SchemaError(f"states.csv row {i + 2}: expected frame {i} and state 0 or 1")
        labels[i] = int(s)
    ctx.outputs.update(rmsf=arr, seg=seg, nums=nums, labels=labels)
    return Outcome(True)


@check("RMSF")
def rmsf(ctx: CheckContext) -> Outcome:
    rel = np.abs(ctx.outputs["rmsf"] / _case(ctx).rmsf_true - 1.0)
    worst, mean = float(rel.max()), float(rel.mean())
    ok = worst <= RMSF_REL_TOL and mean <= RMSF_MEAN_REL_TOL
    return Outcome(
        ok,
        f"RMSF off by up to {worst:.0%} (mean {mean:.1%}); is the fit on the rigid core?",
        worst,
        RMSF_REL_TOL,
    )


@check("SWITCH_RESIDUES")
def switching(ctx: CheckContext) -> Outcome:
    seg, want = ctx.outputs["seg"], _case(ctx).segment
    err = max(abs(seg[0] - want[0]), abs(seg[1] - want[1]))
    return Outcome(
        err <= BOUNDARY_TOL, f"switching range {seg}, expected {list(want)}", err, BOUNDARY_TOL
    )


@check("STATES")
def states(ctx: CheckContext) -> Outcome:
    agree = float(np.mean(ctx.outputs["labels"] == _case(ctx).state))
    return Outcome(
        agree >= STATE_AGREEMENT,
        f"only {agree:.1%} of frames carry the right state",
        agree,
        STATE_AGREEMENT,
    )


@check("POPULATION")
def population(ctx: CheckContext) -> Outcome:
    err = abs(ctx.outputs["nums"]["population_minor"] - _case(ctx).pop_minor)
    return Outcome(err <= POPULATION_TOL, "minor-state population is wrong", err, POPULATION_TOL)


@check("RATES")
def rates(ctx: CheckContext) -> Outcome:
    c, nums = _case(ctx), ctx.outputs["nums"]
    e1 = abs(nums["rate_major_to_minor_per_ns"] / c.rate_major_to_minor - 1.0)
    e2 = abs(nums["rate_minor_to_major_per_ns"] / c.rate_minor_to_major - 1.0)
    return Outcome(
        max(e1, e2) <= RATE_REL_TOL,
        f"rates off by {e1:.0%} and {e2:.0%} (units are 1/ns, frames are {DT_NS} ns apart)",
        max(e1, e2),
        RATE_REL_TOL,
    )


@register
class SwitchTask(TaskPlugin):
    task_id = "t03-md-switch-kinetics"
    version = "1.0.0"
    timeout_s = 240.0

    def make_case(self, seed: int) -> SwitchCase:
        return make_case(seed)

    def write_inputs(self, case: SwitchCase, input_dir: Path) -> None:
        write_inputs(case, input_dir)

    def argv(self, lang: str = "python") -> tuple[str, ...]:
        return ("{submission}", "--indir", "{input_dir}", "--outdir", "{output_dir}")

    def checks(self) -> CheckGroup:
        return CheckGroup(
            "t03",
            [
                CheckGroup("load", [files_present, schema], gate=True),
                CheckGroup("fluctuations", [rmsf]),
                CheckGroup("switch", [switching, states, population, rates]),
            ],
        )
