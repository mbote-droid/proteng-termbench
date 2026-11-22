"""t01: triage a predicted protein structure against an experimental reference.

What the agent gets: `model.pdb`, a full-atom predicted model, and
`reference.pdb`, an experimental structure of the same protein (E. coli
adenylate kinase, PDB 1AKE, public domain). The two files disagree about
everything except the coordinates that matter:

* the model carries a three-residue N-terminal tag that the reference lacks,
  and its numbering starts at 1;
* the reference is renumbered from an arbitrary start and is missing an
  unresolved loop, so after the gap no constant offset maps one file onto the
  other (matching residues must be found from the sequences);
* the model's confidence column (pLDDT, stored as B-factor) is high
  everywhere except on the tag, so it says nothing about where the model is
  wrong.

Three kinds of model are generated, by construction:

  domain_rotation  one domain (NMP or LID) is rotated 12-45 degrees about the
                   axis through the C-alpha atoms of its two hinge residues;
  local_error      one 9-12 residue segment is displaced by 4-7 Angstrom;
  accurate         no systematic error; only coordinate noise.

Every model also has independent coordinate noise (0.3-0.7 Angstrom rms).

How each output is judged, and why:

* n_matched, the global C-alpha RMSD and the TM-score are recomputed by the
  grader from the planted pairing. A naive index alignment pairs the wrong
  residues, so all three fail at once.
* The verdict, moved segment, hinge residues, rotation angle and per-part RMSD
  are the diagnosis. Reporting only the global RMSD cannot give them: one
  superposition of everything spreads a domain rotation over the whole
  structure (the global RMSD looks like "a bit off everywhere").
* A constant verdict cannot pass: rotations are half of the cases, local errors
  and accurate models a quarter each, so any fixed answer fails on a block of
  cases with overwhelming probability.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from protengbench.checks.base import CheckContext, CheckGroup, Outcome, check
from protengbench.checks.io import FloatArray, SchemaError
from protengbench.core.registry import register
from protengbench.core.seeds import streams
from protengbench.core.task import TaskPlugin
from protengbench.models import structure

KINDS = ("domain_rotation", "local_error", "accurate")
DOMAINS = {"NMP": (30, 59), "LID": (122, 159)}
TAG = (("GLY", 1), ("SER", 2), ("HIS", 3))

RMSD_REL_TOL = 0.03
RMSD_ABS_FLOOR = 0.03
TM_TOL = 0.01
BOUNDARY_TOL_ROTATION = 2
BOUNDARY_TOL_LOCAL = 2
ANGLE_TOL_DEG = 3.0
UNDETECTABLE_SHIFT_A = 0.6  # planted displacement below which a residue cannot be told apart


@dataclass(frozen=True)
class TriageCase:
    kind: str
    model_pdb: str
    reference_pdb: str
    n_matched: int
    rmsd_global: float
    tm_score: float
    moved: tuple[int, int] | None  # model residue numbers (inclusive)
    hinge: tuple[int, int] | None  # model residue numbers
    angle_deg: float | None
    rmsd_moved: float | None
    rmsd_rest: float | None
    slack: tuple[int, int] | None  # undetectable boundary residues at the (start, end) edges
    pairs: tuple[tuple[int, int], ...]  # (model residue number, reference residue number)
    model_ca: FloatArray  # matched C-alpha coordinates, in `pairs` order
    ref_ca: FloatArray


def tm_score(distances: FloatArray, length: int) -> float:
    """TM-score for per-residue distances d_i after a superposition, normalised by the
    length L of the reference: (1/L) sum 1 / (1 + (d_i / d0)^2), d0 = 1.24 (L - 15)^(1/3) - 1.8."""
    d0 = 1.24 * (length - 15) ** (1.0 / 3.0) - 1.8
    return float(np.sum(1.0 / (1.0 + (distances / d0) ** 2)) / length)


def superposed_distances(mobile: FloatArray, target: FloatArray) -> FloatArray:
    r, t = structure.kabsch(mobile, target)
    return np.sqrt(((mobile @ r.T + t - target) ** 2).sum(axis=1))


def _rodrigues(
    points: FloatArray, origin: FloatArray, axis: FloatArray, angle: float
) -> FloatArray:
    k = axis / np.linalg.norm(axis)
    v = points - origin
    c, s = math.cos(angle), math.sin(angle)
    return origin + v * c + np.cross(k, v) * s + np.outer(v @ k, k) * (1.0 - c)


@lru_cache(maxsize=32)
def make_case(seed: int) -> TriageCase:
    r_kind, r_geo, r_noise, r_ref = streams(seed, 4)
    residues = structure.residues(structure.read_pdb(structure.bundled("1AKE.pdb")))
    n_res = len(residues)  # 214, numbered 1..214
    ca = np.array([[a.xyz for a in res if a.name == "CA"][0] for res in residues])

    # one of four draws is `accurate`, two are rotations, one is a local error
    draw = int(r_kind.integers(0, 4))
    kind = ("domain_rotation", "domain_rotation", "local_error", "accurate")[draw]

    new_xyz = [np.array([a.xyz for a in res], dtype=float) for res in residues]
    moved_orig: tuple[int, int] | None = None
    angle_deg: float | None = None
    if kind == "domain_rotation":
        name = ("NMP", "LID")[int(r_geo.integers(0, 2))]
        s, e = DOMAINS[name]
        moved_orig = (s, e)
        angle = math.radians(r_geo.uniform(12.0, 45.0)) * (1 if r_geo.random() < 0.5 else -1)
        angle_deg = abs(math.degrees(angle))
        origin = ca[s - 2]  # residue s-1 (0-based index s-2)
        axis = ca[e] - origin  # residue e+1 (0-based index e)
        for k in range(s - 1, e):
            new_xyz[k] = _rodrigues(new_xyz[k], origin, axis, angle)
    elif kind == "local_error":
        length = int(r_geo.integers(9, 13))
        pool = list(range(62, 109)) + list(range(165, 196))
        start = int(r_geo.choice(pool))
        moved_orig = (start, start + length - 1)
        amp = r_geo.uniform(4.0, 7.0)
        direction = r_geo.normal(size=3)
        direction /= np.linalg.norm(direction)
        for i in range(length):
            shift = amp * math.sin(math.pi * (i + 1) / (length + 1)) * direction
            new_xyz[start - 1 + i] = new_xyz[start - 1 + i] + shift

    rms = r_noise.uniform(0.3, 0.7)
    sigma = rms / math.sqrt(3.0)
    model_rows: list[tuple[str, str, str, int, FloatArray, float]] = []
    # tag: a straight line of backbone atoms leading into residue 1, low confidence
    first_n = new_xyz[0][0]
    for i, (resname, num) in enumerate(TAG):
        base = first_n + np.array([-3.8 * (len(TAG) - i), 0.0, 0.0])
        for j, an in enumerate(("N", "CA", "C", "O")):
            model_rows.append((an, resname, "A", num, base + np.array([0.9 * j, 0.5, 0.0]), 38.0))
    plddt = np.clip(r_noise.normal(92.0, 2.5, n_res), 80.0, 98.0)
    for k, res in enumerate(residues):
        for a, xyz in zip(res, new_xyz[k], strict=True):
            noisy = xyz + r_noise.normal(0.0, sigma, 3)
            model_rows.append((a.name, a.resname, "A", a.resseq + len(TAG), noisy, float(plddt[k])))

    # reference: trim a few N-terminal residues, drop an unresolved loop, renumber from a new start
    trim = int(r_ref.integers(1, 4))
    loop_len = int(r_ref.integers(5, 9))
    avoid = {*range(58, 62), *range(120, 124), *range(158, 162)}  # keep hinges intact
    if moved_orig is not None:
        avoid |= set(range(moved_orig[0] - 3, moved_orig[1] + 4))
    seq = [res[0].resname for res in residues]  # residue k is seq[k - 1]

    def unambiguous(c: int) -> bool:
        """The gap's position is unique: shifting it by one residue would pair different
        residue types, so a sequence alignment cannot place it elsewhere."""
        return seq[c - 2] != seq[c + loop_len - 2] and seq[c - 1] != seq[c + loop_len - 1]

    pool = [
        c
        for c in list(range(70, 118)) + list(range(165, 200))
        if not set(range(c, c + loop_len)) & avoid and unambiguous(c)
    ]
    loop_start = int(r_ref.choice(pool))
    dropped = set(range(loop_start, loop_start + loop_len))
    kept = [k for k in range(1, n_res + 1) if k > trim and k not in dropped]
    start_number = int(r_ref.integers(11, 61))
    ref_rows: list[tuple[str, str, str, int, FloatArray, float]] = []
    pairs = []
    for idx, k in enumerate(kept):
        ref_num = start_number + idx  # contiguous after the gap: no constant offset
        pairs.append((k + len(TAG), ref_num))
        for a in residues[k - 1]:
            ref_rows.append((a.name, a.resname, "A", ref_num, np.array(a.xyz), 20.0))

    model_pdb = structure.format_pdb(model_rows)
    reference_pdb = structure.format_pdb(ref_rows)

    # truth, from the exact pairing (C-alpha atoms only)
    model_ca = {r[3]: r[4] for r in model_rows if r[0] == "CA"}
    ref_ca = {r[3]: r[4] for r in ref_rows if r[0] == "CA"}
    mm = np.array([model_ca[m] for m, _ in pairs])
    rr = np.array([ref_ca[r] for _, r in pairs])
    n = len(pairs)
    d_all = superposed_distances(mm, rr)
    rmsd_global = float(np.sqrt((d_all**2).mean()))
    tm = tm_score(d_all, n)

    moved = hinge = None
    slack: tuple[int, int] | None = None
    rmsd_moved = rmsd_rest = None
    if moved_orig is not None:
        # Residues next to the hinge sit almost on the rotation axis, so the planted
        # fault displaces them by less than the coordinate noise: whether they are
        # called moved or not cannot be told from the data. Count them at each edge.
        if kind == "domain_rotation":
            # displacement that the planted rotation would give each residue, whether or
            # not it is part of the moved domain
            shift_ca = np.linalg.norm(_rodrigues(ca, origin, axis, angle) - ca, axis=1)
        else:
            shift_ca = np.full(n_res, np.inf)
            for i in range(moved_orig[1] - moved_orig[0] + 1):
                shift_ca[moved_orig[0] - 1 + i] = np.linalg.norm(
                    amp * math.sin(math.pi * (i + 1) / (length + 1)) * direction
                )
        first0, last0 = moved_orig[0] - 1, moved_orig[1] - 1  # 0-based indices
        lo = hi = 0
        k = first0 - 1
        while k >= 0 and shift_ca[k] < UNDETECTABLE_SHIFT_A:
            lo, k = lo + 1, k - 1
        k = first0
        while k < last0 and shift_ca[k] < UNDETECTABLE_SHIFT_A:
            lo, k = lo + 1, k + 1
        k = last0 + 1
        while k < n_res and shift_ca[k] < UNDETECTABLE_SHIFT_A:
            hi, k = hi + 1, k + 1
        k = last0
        while k > first0 and shift_ca[k] < UNDETECTABLE_SHIFT_A:
            hi, k = hi + 1, k - 1
        slack = (lo, hi)
        moved = (moved_orig[0] + len(TAG), moved_orig[1] + len(TAG))
        hinge = (moved[0] - 1, moved[1] + 1)
        in_moved = np.array([moved[0] <= m <= moved[1] for m, _ in pairs])
        rmsd_moved = float(np.sqrt((superposed_distances(mm[in_moved], rr[in_moved]) ** 2).mean()))
        rmsd_rest = float(np.sqrt((superposed_distances(mm[~in_moved], rr[~in_moved]) ** 2).mean()))
    return TriageCase(
        kind,
        model_pdb,
        reference_pdb,
        n,
        rmsd_global,
        tm,
        moved,
        hinge,
        angle_deg if kind == "domain_rotation" else None,
        rmsd_moved,
        rmsd_rest,
        slack,
        tuple(pairs),
        mm,
        rr,
    )


def write_inputs(case: TriageCase, input_dir: Path) -> None:
    (input_dir / "model.pdb").write_text(case.model_pdb)
    (input_dir / "reference.pdb").write_text(case.reference_pdb)


# --- checks ------------------------------------------------------------------------


def _case(ctx: CheckContext) -> TriageCase:
    c: TriageCase = ctx.truth
    return c


def _number(raw: dict[str, object], key: str, positive: bool = True) -> float:
    v = raw.get(key)
    if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
        raise SchemaError(f"{key} must be a finite number, got {v!r}")
    if positive and v <= 0:
        raise SchemaError(f"{key} must be positive")
    return float(v)


def _int_pair(raw: dict[str, object], key: str) -> tuple[int, int]:
    v = raw.get(key)
    ok = (
        isinstance(v, list)
        and len(v) == 2
        and all(isinstance(x, int) and not isinstance(x, bool) for x in v)
    )
    if not ok:
        raise SchemaError(f"{key} must be [first, last] integer residue numbers or null")
    return int(v[0]), int(v[1])  # type: ignore[index]


@check("FILES", gate=True)
def files_present(ctx: CheckContext) -> Outcome:
    return Outcome((ctx.output_dir / "triage.json").is_file(), "missing triage.json")


@check("SCHEMA", gate=True)
def schema(ctx: CheckContext) -> Outcome:
    raw = json.loads((ctx.output_dir / "triage.json").read_text())
    if not isinstance(raw, dict):
        raise SchemaError("triage.json: top level must be an object")
    out: dict[str, object] = {"n_matched": int(_number(raw, "n_matched"))}
    if raw["n_matched"] != out["n_matched"]:
        raise SchemaError("n_matched must be an integer")
    out["rmsd"] = _number(raw, "rmsd_global_A")
    tm = _number(raw, "tm_score")
    if tm > 1.0:
        raise SchemaError("tm_score must be in (0, 1]")
    out["tm"] = tm
    verdict = raw.get("verdict")
    if verdict not in KINDS:
        raise SchemaError(f"verdict must be one of {KINDS}")
    out["verdict"] = verdict
    for key in ("moved_segment", "hinge_residues"):
        out[key] = None if raw.get(key) is None else _int_pair(raw, key)
    angle = raw.get("rotation_angle_deg")
    out["angle"] = None if angle is None else _number(raw, "rotation_angle_deg", positive=False)
    parts = raw.get("rmsd_parts_A")
    if parts is None:
        out["parts"] = None
    else:
        if not isinstance(parts, dict):
            raise SchemaError("rmsd_parts_A must be an object with moved and rest, or null")
        out["parts"] = (_number(parts, "moved"), _number(parts, "rest"))
    if verdict == "accurate":
        if any(out[k] is not None for k in ("moved_segment", "hinge_residues", "angle", "parts")):
            raise SchemaError(
                "for verdict 'accurate' moved_segment, hinge_residues, "
                "rotation_angle_deg and rmsd_parts_A must be null"
            )
    else:
        if out["moved_segment"] is None or out["hinge_residues"] is None or out["parts"] is None:
            raise SchemaError(
                "moved_segment, hinge_residues and rmsd_parts_A are required "
                "unless the verdict is 'accurate'"
            )
        if verdict == "domain_rotation" and out["angle"] is None:
            raise SchemaError("rotation_angle_deg is required for verdict 'domain_rotation'")
        if verdict == "local_error" and out["angle"] is not None:
            raise SchemaError("rotation_angle_deg must be null for verdict 'local_error'")
    ctx.outputs.update(out)
    return Outcome(True)


@check("MATCHED")
def matched(ctx: CheckContext) -> Outcome:
    got, want = ctx.outputs["n_matched"], _case(ctx).n_matched
    return Outcome(
        got == want,
        f"matched {got} residues, expected {want} "
        "(match by sequence; numbering differs between the files)",
        abs(got - want),
        0,
    )


@check("RMSD_GLOBAL")
def rmsd_global(ctx: CheckContext) -> Outcome:
    want = _case(ctx).rmsd_global
    err = abs(ctx.outputs["rmsd"] - want)
    tol = max(RMSD_ABS_FLOOR, RMSD_REL_TOL * want)
    return Outcome(
        err <= tol,
        f"global C-alpha RMSD {ctx.outputs['rmsd']:.3f}, expected about {want:.3f}",
        err,
        tol,
    )


@check("TM_SCORE")
def tm(ctx: CheckContext) -> Outcome:
    want = _case(ctx).tm_score
    err = abs(ctx.outputs["tm"] - want)
    return Outcome(
        err <= TM_TOL, f"TM-score {ctx.outputs['tm']:.3f}, expected about {want:.3f}", err, TM_TOL
    )


@check("VERDICT")
def verdict(ctx: CheckContext) -> Outcome:
    got, want = ctx.outputs["verdict"], _case(ctx).kind
    return Outcome(got == want, f"verdict {got!r}, but the model's error is {want!r}")


@check("SEGMENT")
def segment(ctx: CheckContext) -> Outcome:
    case = _case(ctx)
    got = ctx.outputs["moved_segment"]
    if case.moved is None:
        return Outcome(got is None, "there is no moved segment in this model")
    if got is None:
        return Outcome(False, "a moved segment exists but none was reported")
    tol = BOUNDARY_TOL_ROTATION if case.kind == "domain_rotation" else BOUNDARY_TOL_LOCAL
    slack = case.slack or (0, 0)
    err = max(abs(got[0] - case.moved[0]) - slack[0], abs(got[1] - case.moved[1]) - slack[1], 0)
    return Outcome(
        err <= tol,
        f"moved segment {list(got)}, expected {list(case.moved)} (model numbering)",
        err,
        tol,
    )


@check("HINGE")
def hinge(ctx: CheckContext) -> Outcome:
    case = _case(ctx)
    got = ctx.outputs["hinge_residues"]
    if case.hinge is None:
        return Outcome(got is None, "there is no hinge in this model")
    if got is None:
        return Outcome(False, "hinge residues missing")
    slack = case.slack or (0, 0)
    err = max(abs(got[0] - case.hinge[0]) - slack[0], abs(got[1] - case.hinge[1]) - slack[1], 0)
    return Outcome(
        err <= BOUNDARY_TOL_ROTATION,
        f"hinge {list(got)}, expected {list(case.hinge)}",
        err,
        BOUNDARY_TOL_ROTATION,
    )


@check("ANGLE")
def angle(ctx: CheckContext) -> Outcome:
    case = _case(ctx)
    got = ctx.outputs["angle"]
    if case.angle_deg is None:
        return Outcome(got is None, "this model has no domain rotation")
    if got is None:
        return Outcome(False, "rotation angle missing")
    err = abs(abs(got) - case.angle_deg)
    return Outcome(
        err <= ANGLE_TOL_DEG,
        f"rotation {got:.1f} degrees, expected about {case.angle_deg:.1f}",
        err,
        ANGLE_TOL_DEG,
    )


PART_RMSD_REL_TOL = 0.03
PART_RMSD_ABS_FLOOR = 0.02


def _part_rmsd(case: TriageCase, first: int, last: int) -> tuple[float, float] | None:
    """RMSD of the residues inside [first, last] and of the rest, each part superposed
    on its own, from the planted pairing."""
    numbers = np.array([m for m, _ in case.pairs])
    inside = (numbers >= first) & (numbers <= last)
    if inside.sum() < 4 or (~inside).sum() < 4:
        return None
    parts = []
    for sel in (inside, ~inside):
        d = superposed_distances(case.model_ca[sel], case.ref_ca[sel])
        parts.append(float(np.sqrt((d**2).mean())))
    return parts[0], parts[1]


@check("PART_RMSD")
def parts(ctx: CheckContext) -> Outcome:
    """The reported part RMSDs must be what the reported segment gives when each part is
    superposed on its own. Reporting the same residues after one global superposition
    (the usual shortcut) gives much larger numbers and fails."""
    case = _case(ctx)
    got, seg = ctx.outputs["parts"], ctx.outputs["moved_segment"]
    if case.moved is None:
        return Outcome(got is None, "no parts expected")
    if got is None or seg is None:
        return Outcome(False, "part RMSDs missing")
    own = _part_rmsd(case, seg[0], seg[1])
    if own is None:
        return Outcome(False, "the reported segment leaves fewer than 4 residues in a part")
    worst = 0.0
    for g, w in zip(got, own, strict=True):
        tol = max(PART_RMSD_ABS_FLOOR, PART_RMSD_REL_TOL * w)
        worst = max(worst, abs(g - w) / tol)
    return Outcome(
        worst <= 1.0,
        f"part RMSDs {got[0]:.2f}/{got[1]:.2f}, but your own segment "
        f"gives {own[0]:.2f}/{own[1]:.2f} when each part is superposed on its own",
        worst,
        1.0,
    )


@register
class TriageTask(TaskPlugin):
    task_id = "t01-model-triage"
    version = "1.0.0"
    timeout_s = 180.0

    def make_case(self, seed: int) -> TriageCase:
        return make_case(seed)

    def write_inputs(self, case: TriageCase, input_dir: Path) -> None:
        write_inputs(case, input_dir)

    def argv(self, lang: str = "python") -> tuple[str, ...]:
        return ("{submission}", "--indir", "{input_dir}", "--outdir", "{output_dir}")

    def checks(self) -> CheckGroup:
        return CheckGroup(
            "t01",
            [
                CheckGroup("load", [files_present, schema], gate=True),
                CheckGroup("alignment", [matched, rmsd_global, tm]),
                CheckGroup("diagnosis", [verdict, segment, hinge, angle, parts]),
            ],
        )
