"""t02: design a stabilised variant of ubiquitin from noisy mutational scanning data.

What the agent gets (a stability scan, as from a deep mutational scan or a
folding-stability assay panel):

* `scan_single.csv`: measured ddG for all 76 x 19 single substitutions
  (kcal/mol, measurement noise 0.12, as from pooled replicates);
* `scan_double.csv`: measured ddG of the double mutants for every pair of
  contacting residues among the 60 most stabilising singles, plus the designed
  complementary pairs (noise 0.15);
* `contacts.csv`: residue pairs whose C-alpha atoms are within 8 Angstrom;
* `constraints.json`: at most 5 substitutions at distinct positions, and a list
  of forbidden positions (the functional surface and the C-terminal tail, plus
  four positions that are among the most stabilising ones);
* `ubiquitin.pdb`: the structure (PDB 1UBI, public domain).

It must choose substitutions to maximise stability and say how sure it is.

The hidden landscape (src/protengbench/models/stability.py) is not additive:
two stabilising substitutions in contact partly cancel, and twelve designed pairs
are synergistic. The data contain what is needed to find both effects.

Checks (a measurement-driven design is judged on what it does, not on how):

* CONSTRAINTS: at most 5, distinct positions, none forbidden, wild type matches.
* STABILISED: the true ddG of the set must reach at least 90 percent of the
  exhaustive optimum (reference solution: 88-100 percent, 49 of 50 validation seeds above 90;
  additive top-five strategy: 28-98 percent, median 67, so a lucky additive
  pick that really is near-optimal is a legitimate pass, about 1 in 30).
* CALIBRATION: the stated sigma must lie in [0.2, 1.0] kcal/mol and the stated
  ddG must be within 2.5 sigma of the true value. A large sigma cannot buy a
  pass (it is capped), and an additive model that ignores antagonism
  over-predicts by more than it admits.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from protengbench.checks.base import CheckContext, CheckGroup, Outcome, check
from protengbench.checks.io import SchemaError
from protengbench.core.registry import register
from protengbench.core.seeds import streams
from protengbench.core.task import TaskPlugin
from protengbench.models import stability as st
from protengbench.models import structure

FIXED_FORBIDDEN = (1, 8, 44, 48, 68, 70, 71, 72, 73, 74, 75, 76)
N_RANDOM_FORBIDDEN = 4
SINGLE_NOISE = 0.12
DOUBLE_NOISE = 0.15
TOP_SINGLES_FOR_DOUBLES = 60
ACCEPT_FRACTION = 0.90
SIGMA_MIN, SIGMA_MAX, SIGMA_K = 0.2, 1.0, 2.5
MUTATION = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d{1,2})([ACDEFGHIKLMNPQRSTVWY])$")


@dataclass(frozen=True)
class StabilityCase:
    land: st.Landscape
    forbidden: tuple[int, ...]
    best_ddg: float
    best_set: tuple[tuple[int, str], ...]
    single_rows: tuple[tuple[int, str, str, float], ...]
    double_rows: tuple[tuple[int, str, str, int, str, str, float], ...]
    contact_rows: tuple[tuple[int, int, float], ...]


@lru_cache(maxsize=16)
def make_case(seed: int) -> StabilityCase:
    r_cav, r_eps, r_syn, r_ns, r_nd, r_forb = streams(seed, 6)
    ca, _resseq, resname = structure.ca_trace(structure.read_pdb(structure.bundled("1UBI.pdb")))
    seq = "".join(st.THREE[r] for r in resname)
    land = st.make_landscape(ca, seq, r_cav, r_eps, r_syn)
    n = len(seq)

    # forbidden: the functional surface, plus a few of the most stabilising positions
    best_by_pos = land.single.min(axis=1)
    order = [int(i) + 1 for i in np.argsort(best_by_pos) if (int(i) + 1) not in FIXED_FORBIDDEN]
    extra = [int(x) for x in r_forb.choice(order[:12], size=N_RANDOM_FORBIDDEN, replace=False)]
    forbidden = tuple(sorted(set(FIXED_FORBIDDEN) | set(extra)))
    allowed = set(range(1, n + 1)) - set(forbidden)
    best_ddg, best = st.best_set(land, allowed)

    # measured scan
    single_rows = []
    measured = np.zeros_like(land.single)
    for i in range(n):
        for k, a in enumerate(st.AA):
            if a == seq[i]:
                continue
            measured[i, k] = land.single[i, k] + r_ns.normal(0.0, SINGLE_NOISE)
            single_rows.append((i + 1, seq[i], a, float(measured[i, k])))

    top = sorted(
        (measured[i, k], i + 1, st.AA[k])
        for i in range(n)
        for k in range(len(st.AA))
        if st.AA[k] != seq[i]
    )[:TOP_SINGLES_FOR_DOUBLES]
    pairs: set[tuple[int, str, int, str]] = set()
    for x in range(len(top)):
        for y in range(x + 1, len(top)):
            (_, i, a), (_, j, b) = top[x], top[y]
            if i != j and land.dist[i - 1, j - 1] < st.CONTACT_A:
                pairs.add((i, a, j, b) if i < j else (j, b, i, a))
    pairs |= set(land.synergy)  # the designed complementary pairs are always measured
    double_rows = []
    for i, a, j, b in sorted(pairs):
        value = land.ddg([(i, a), (j, b)]) + r_nd.normal(0.0, DOUBLE_NOISE)
        double_rows.append((i, seq[i - 1], a, j, seq[j - 1], b, float(value)))

    contact_rows = tuple(
        (i + 1, j + 1, float(land.dist[i, j]))
        for i in range(n)
        for j in range(i + 1, n)
        if land.dist[i, j] < st.CONTACT_A
    )
    return StabilityCase(
        land, forbidden, best_ddg, tuple(best), tuple(single_rows), tuple(double_rows), contact_rows
    )


def write_inputs(case: StabilityCase, input_dir: Path) -> None:
    (input_dir / "ubiquitin.pdb").write_text(structure.bundled("1UBI.pdb").read_text())
    lines = ["position,wt,mut,ddG_kcal_per_mol"]
    lines += [f"{i},{w},{m},{v:.3f}" for i, w, m, v in case.single_rows]
    (input_dir / "scan_single.csv").write_text("\n".join(lines) + "\n")
    lines = ["pos1,wt1,mut1,pos2,wt2,mut2,ddG_kcal_per_mol"]
    lines += [f"{i},{w},{a},{j},{x},{b},{v:.3f}" for i, w, a, j, x, b, v in case.double_rows]
    (input_dir / "scan_double.csv").write_text("\n".join(lines) + "\n")
    lines = ["pos1,pos2,ca_distance_A"]
    lines += [f"{i},{j},{d:.2f}" for i, j, d in case.contact_rows]
    (input_dir / "contacts.csv").write_text("\n".join(lines) + "\n")
    (input_dir / "constraints.json").write_text(
        json.dumps(
            {
                "max_substitutions": st.MAX_SUBSTITUTIONS,
                "forbidden_positions": list(case.forbidden),
                "sequence": case.land.sequence,
                "sign": "ddG < 0 means more stable than wild type (kcal/mol)",
            },
            indent=2,
        )
        + "\n"
    )


# --- checks ------------------------------------------------------------------------


def _case(ctx: CheckContext) -> StabilityCase:
    c: StabilityCase = ctx.truth
    return c


@check("FILES", gate=True)
def files_present(ctx: CheckContext) -> Outcome:
    return Outcome((ctx.output_dir / "design.json").is_file(), "missing design.json")


@check("SCHEMA", gate=True)
def schema(ctx: CheckContext) -> Outcome:
    raw = json.loads((ctx.output_dir / "design.json").read_text())
    if not isinstance(raw, dict):
        raise SchemaError("design.json: top level must be an object")
    subs = raw.get("substitutions")
    if not isinstance(subs, list) or not subs or not all(isinstance(s, str) for s in subs):
        raise SchemaError("substitutions must be a non-empty list of strings like 'K27L'")
    parsed: list[tuple[int, str, str]] = []
    seq = _case(ctx).land.sequence
    for s in subs:
        m = MUTATION.match(s.strip())
        if m is None:
            raise SchemaError(f"'{s}' is not wild type + position + mutant, e.g. K27L")
        wt, pos, mut = m.group(1), int(m.group(2)), m.group(3)
        if not 1 <= pos <= len(seq) or seq[pos - 1] != wt:
            raise SchemaError(
                f"'{s}': position {pos} is {seq[pos - 1] if 1 <= pos <= len(seq) else '?'}, not {wt}"
            )
        if mut == wt:
            raise SchemaError(f"'{s}' is not a substitution")
        parsed.append((pos, wt, mut))
    for key in ("predicted_ddG_kcal_per_mol", "sigma_kcal_per_mol"):
        v = raw.get(key)
        if isinstance(v, bool) or not isinstance(v, int | float) or not np.isfinite(v):
            raise SchemaError(f"{key} must be a finite number")
    ctx.outputs.update(
        subs=[(p, m) for p, _w, m in parsed],
        pred=float(raw["predicted_ddG_kcal_per_mol"]),
        sigma=float(raw["sigma_kcal_per_mol"]),
    )
    return Outcome(True)


@check("CONSTRAINTS")
def constraints(ctx: CheckContext) -> Outcome:
    subs = ctx.outputs["subs"]
    positions = [p for p, _ in subs]
    if len(subs) > st.MAX_SUBSTITUTIONS:
        return Outcome(False, f"{len(subs)} substitutions, at most {st.MAX_SUBSTITUTIONS} allowed")
    if len(set(positions)) != len(positions):
        return Outcome(False, "two substitutions at the same position")
    bad = sorted(set(positions) & set(_case(ctx).forbidden))
    return Outcome(not bad, f"forbidden positions used: {bad}")


@check("STABILISED")
def stabilised(ctx: CheckContext) -> Outcome:
    case = _case(ctx)
    true = case.land.ddg(ctx.outputs["subs"])
    needed = ACCEPT_FRACTION * case.best_ddg  # both negative
    return Outcome(
        true <= needed,
        f"true ddG {true:.2f} kcal/mol; this set reaches {true / case.best_ddg:.0%} of "
        f"the best achievable ({case.best_ddg:.2f}), {ACCEPT_FRACTION:.0%} is required",
        true / case.best_ddg,
        ACCEPT_FRACTION,
    )


@check("CALIBRATION")
def calibration(ctx: CheckContext) -> Outcome:
    sigma, pred = ctx.outputs["sigma"], ctx.outputs["pred"]
    true = _case(ctx).land.ddg(ctx.outputs["subs"])
    if not SIGMA_MIN <= sigma <= SIGMA_MAX:
        return Outcome(False, f"sigma {sigma:.2f} outside [{SIGMA_MIN}, {SIGMA_MAX}] kcal/mol")
    err = abs(pred - true)
    return Outcome(
        err <= SIGMA_K * sigma,
        f"predicted {pred:.2f} but the true ddG is {true:.2f}: off by {err:.2f}, "
        f"more than {SIGMA_K} x the stated sigma {sigma:.2f}",
        err / sigma,
        SIGMA_K,
    )


@register
class StabilityTask(TaskPlugin):
    task_id = "t02-stability-design"
    version = "1.0.0"
    timeout_s = 240.0

    def make_case(self, seed: int) -> StabilityCase:
        return make_case(seed)

    def write_inputs(self, case: StabilityCase, input_dir: Path) -> None:
        write_inputs(case, input_dir)

    def argv(self, lang: str = "python") -> tuple[str, ...]:
        return ("{submission}", "--indir", "{input_dir}", "--outdir", "{output_dir}")

    def checks(self) -> CheckGroup:
        return CheckGroup(
            "t02",
            [
                CheckGroup("load", [files_present, schema], gate=True),
                CheckGroup("design", [constraints, stabilised, calibration]),
            ],
        )
