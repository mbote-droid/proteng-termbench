"""t06: characterise an enzyme inhibitor from progress curves.

The agent gets a 4 x 6 plate of fluorescence progress curves (four substrate
concentrations, an inhibitor-free control and five inhibitor concentrations,
product formed over 600 s) from an enzyme with a reversible inhibitor, the
enzyme concentration, the calibration (background, RFU per uM product), and
must report the mechanism, Km, kcat, Ki (and alpha), the initial-rate IC50 at
10 uM substrate, and the Ki obtained from that IC50 by the equation that fits
the mechanism.

Data are simulated from full mass action (see models/enzyme.py), so nothing in
them assumes the analysis. Two features make naive analysis fail:

* Substrate depletion: curves bend, because enough substrate is consumed that
  rate falls during the run. Reading a rate from the whole curve (or an
  endpoint) biases Km upwards and can turn competitive inhibition into
  something else. Initial rates from the first minutes, or a fit of the
  integrated curves, are not affected.
* Tight binding: in some seeds the competitive inhibitor's Ki is about as
  small as the enzyme concentration. Then the free inhibitor is much less than
  the total, the textbook equations fail, and the IC50 exceeds the apparent Ki
  by about E0/2 (Morrison).

How each output is judged, and why:

* The mechanism must match. Mixed inhibition is only generated with
  alpha <= 0.25 or >= 4 and noncompetitive with alpha = 1, so the data can
  separate them.
* Km and kcat within the tolerances below; Ki and alpha within factors that
  grow when Ki is weakly determined; IC50 within 15 percent of the exact
  mass-action value (operational definition in the instruction).
* Ki from IC50 must be the correct conversion of the agent's own IC50, Km,
  mechanism and alpha, to 10 percent: Ki = IC50 / (1 + S/Km) for competitive,
  Ki' = IC50 / (1 + Km/S) for uncompetitive, and the mixed expression
  otherwise; for tight binding the E0/2 is removed first.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from protengbench.checks.base import CheckContext, CheckGroup, Outcome, check
from protengbench.checks.io import FloatArray, SchemaError
from protengbench.core.registry import register
from protengbench.core.seeds import streams
from protengbench.core.task import TaskPlugin
from protengbench.models import enzyme

S_REF_UM = 10.0
T_END_S = 600.0
DT_S = 10.0
NOISE_RFU = 40.0
BACKGROUND_RFU = 500.0
FULL_SCALE_RFU = 20000.0
S_FACTORS = (0.25, 0.7, 2.0, 6.0)  # substrate as multiples of Km
I_FACTORS = (0.0, 0.25, 0.6, 1.5, 4.0, 10.0)  # inhibitor as multiples of the apparent Ki

KM_TOL = 0.20  # natural-log tolerances
KCAT_TOL = 0.20
KI_TOL = 0.40
KI_TOL_TIGHT = 0.55
ALPHA_TOL = math.log(2.5)
IC50_TOL = 0.15
KI_FROM_IC50_TOL = 0.10
REPORT_KEYS = ("Km_uM", "kcat_per_s", "Ki_uM", "IC50_uM_at_10uM_S", "Ki_from_IC50_uM")


@dataclass(frozen=True)
class InhibitorCase:
    mechanism: str
    km: float
    kcat: float
    ki: float  # Ki; for uncompetitive this is Ki' (inhibitor binding to ES)
    alpha: float
    e0: float
    tight: bool
    gain: float
    wells: tuple[tuple[float, float], ...]  # (S0, I0) per well
    t: FloatArray
    signal: FloatArray  # (n_wells, n_t)
    ic50_true: float


def _initial_rate(
    case_args: tuple[str, float, float, float, float, float], s0: float, i0: float
) -> float:
    mech, km, kcat, ki, alpha, e0 = case_args
    t = np.array([0.0, 5.0, 15.0])
    p = enzyme.simulate(mech, km, kcat, ki, alpha, e0, s0, i0, t)
    return float((p[2] - p[1]) / 10.0)


def true_ic50(mech: str, km: float, kcat: float, ki: float, alpha: float, e0: float) -> float:
    """Total inhibitor giving half the uninhibited initial rate at S = 10 uM (mass action)."""
    args = (mech, km, kcat, ki, alpha, e0)
    v0 = _initial_rate(args, S_REF_UM, 0.0)
    hi = max(50.0 * ki, 50.0 * e0)
    return float(brentq(lambda i: _initial_rate(args, S_REF_UM, i) - 0.5 * v0, 1e-6, hi, xtol=1e-9))


@lru_cache(maxsize=64)
def make_case(seed: int) -> InhibitorCase:
    r_par, r_noise = streams(seed, 2)
    mech = enzyme.MECHANISMS[int(r_par.integers(0, 4))]
    km = 10 ** r_par.uniform(0.7, 1.7)  # 5 - 50 uM
    kcat = 10 ** r_par.uniform(0.0, 1.0)  # 1 - 10 /s
    vmax = r_par.uniform(0.5, 1.2) * km / T_END_S  # uM/s: depletes 25-60 % at S0 = Km
    e0 = vmax / kcat
    tight = mech == "competitive" and bool(r_par.random() < 0.5)
    if tight:
        ki = e0 * r_par.uniform(0.3, 1.5)
    else:
        ki = max(10 ** r_par.uniform(-0.3, 1.2), 20.0 * e0)
    if mech == "mixed":
        alpha = float(
            10 ** r_par.choice([-0.75, -0.6, 0.65, 0.9]) * 10 ** r_par.uniform(-0.05, 0.05)
        )
    elif mech == "noncompetitive":
        alpha = 1.0
    else:
        alpha = 1.0
    t = np.arange(0.0, T_END_S + 1e-9, DT_S)
    gain = FULL_SCALE_RFU / (1.2 * km)
    # inhibitor series scaled to the apparent inhibition constant at Km
    k_app = ki * (2.0 if mech == "competitive" else 1.0) + (e0 / 2 if tight else 0.0)
    wells = tuple((km * sf, k_app * f) for sf in S_FACTORS for f in I_FACTORS)
    signal = np.empty((len(wells), t.size))
    for w, (s0, i0) in enumerate(wells):
        p = enzyme.simulate(mech, km, kcat, ki, alpha, e0, s0, i0, t)
        signal[w] = BACKGROUND_RFU + gain * p + r_noise.normal(0.0, NOISE_RFU, t.size)
    return InhibitorCase(
        mech,
        km,
        kcat,
        ki,
        alpha,
        e0,
        tight,
        gain,
        wells,
        t,
        signal,
        true_ic50(mech, km, kcat, ki, alpha, e0),
    )


def write_inputs(case: InhibitorCase, input_dir: Path) -> None:
    rows = []
    for w, (s0, i0) in enumerate(case.wells):
        for t, f in zip(case.t, case.signal[w], strict=True):
            rows.append(f"{w},{s0:.6g},{i0:.6g},{t:.0f},{f:.3f}")
    (input_dir / "plate.csv").write_text(
        "well,substrate_uM,inhibitor_uM,time_s,signal_RFU\n" + "\n".join(rows) + "\n"
    )
    assay = {
        "enzyme_nM": round(case.e0 * 1000.0, 4),
        "background_RFU": BACKGROUND_RFU,
        "gain_RFU_per_uM_product": round(case.gain, 4),
        "ic50_substrate_uM": S_REF_UM,
        "mechanisms": list(enzyme.MECHANISMS),
        "note": "inhibitor_uM and substrate_uM are total concentrations at t = 0",
    }
    (input_dir / "assay.json").write_text(json.dumps(assay, indent=2) + "\n")


def correct_ki_from_ic50(mech: str, ic50: float, km: float, alpha: float, e0: float) -> float:
    """Ki (Ki' for uncompetitive) from the IC50 at S_REF with the equation for `mech`."""
    free = max(ic50 - e0 / 2.0, 1e-12)  # tight-binding correction (Morrison); negligible otherwise
    if mech == "competitive":
        return free / (1.0 + S_REF_UM / km)
    if mech == "uncompetitive":
        return free / (1.0 + km / S_REF_UM)
    # mixed / noncompetitive: IC50 = (Km + S) / (Km/Ki + S/(alpha Ki))  =>  Ki = IC50 (Km/ (Km+S) + S / (alpha (Km+S)))
    return free * (km + S_REF_UM / alpha) / (km + S_REF_UM)


# --- checks ------------------------------------------------------------------------


def _case(ctx: CheckContext) -> InhibitorCase:
    c: InhibitorCase = ctx.truth
    return c


@check("FILES", gate=True)
def files_present(ctx: CheckContext) -> Outcome:
    return Outcome((ctx.output_dir / "inhibition.json").is_file(), "missing inhibition.json")


@check("SCHEMA", gate=True)
def schema(ctx: CheckContext) -> Outcome:
    raw = json.loads((ctx.output_dir / "inhibition.json").read_text())
    if not isinstance(raw, dict):
        raise SchemaError("inhibition.json: top level must be an object")
    mech = raw.get("mechanism")
    if mech not in enzyme.MECHANISMS:
        raise SchemaError(f"mechanism must be one of {enzyme.MECHANISMS}")
    vals: dict[str, float] = {}
    for key in REPORT_KEYS:
        v = raw.get(key)
        if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v) or v <= 0:
            raise SchemaError(f"{key} must be a positive number, got {v!r}")
        vals[key] = float(v)
    alpha = raw.get("alpha")
    if mech in ("mixed", "noncompetitive"):
        if isinstance(alpha, bool) or not isinstance(alpha, int | float) or not alpha > 0:
            raise SchemaError("alpha must be a positive number for mixed/noncompetitive")
        vals["alpha"] = float(alpha)
    ctx.outputs["mech"] = mech
    ctx.outputs["vals"] = vals
    return Outcome(True)


@check("MECHANISM")
def mechanism(ctx: CheckContext) -> Outcome:
    got, want = ctx.outputs["mech"], _case(ctx).mechanism
    return Outcome(got == want, f"mechanism {got!r}, the inhibitor is {want!r}")


@check("KM_KCAT")
def km_kcat(ctx: CheckContext) -> Outcome:
    c, v = _case(ctx), ctx.outputs["vals"]
    e_km = abs(math.log(v["Km_uM"] / c.km))
    e_kc = abs(math.log(v["kcat_per_s"] / c.kcat))
    ok = e_km <= KM_TOL and e_kc <= KCAT_TOL
    return Outcome(
        ok,
        f"Km off by {e_km:.2f}, kcat off by {e_kc:.2f} (ln units)",
        max(e_km / KM_TOL, e_kc / KCAT_TOL),
        1.0,
    )


@check("KI")
def ki(ctx: CheckContext) -> Outcome:
    c, v = _case(ctx), ctx.outputs["vals"]
    if ctx.outputs["mech"] != c.mechanism:
        return Outcome(True)  # judged by MECHANISM
    err = abs(math.log(v["Ki_uM"] / c.ki))
    tol = KI_TOL_TIGHT if c.tight else KI_TOL
    return Outcome(err <= tol, f"Ki off by a factor {math.exp(err):.2f}", err, tol)


@check("ALPHA")
def alpha_check(ctx: CheckContext) -> Outcome:
    c, v = _case(ctx), ctx.outputs["vals"]
    if c.mechanism != "mixed" or ctx.outputs["mech"] != "mixed":
        return Outcome(True)
    err = abs(math.log(v["alpha"] / c.alpha))
    return Outcome(err <= ALPHA_TOL, f"alpha off by a factor {math.exp(err):.2f}", err, ALPHA_TOL)


@check("IC50")
def ic50_check(ctx: CheckContext) -> Outcome:
    c, v = _case(ctx), ctx.outputs["vals"]
    err = abs(v["IC50_uM_at_10uM_S"] / c.ic50_true - 1.0)
    return Outcome(
        err <= IC50_TOL,
        f"IC50 {v['IC50_uM_at_10uM_S']:.3g} uM, exact {c.ic50_true:.3g} uM",
        err,
        IC50_TOL,
    )


@check("KI_FROM_IC50")
def ki_from_ic50(ctx: CheckContext) -> Outcome:
    c, v = _case(ctx), ctx.outputs["vals"]
    mech = ctx.outputs["mech"]
    expected = correct_ki_from_ic50(
        mech, v["IC50_uM_at_10uM_S"], v["Km_uM"], v.get("alpha", 1.0), c.e0
    )
    err = abs(v["Ki_from_IC50_uM"] / expected - 1.0)
    return Outcome(
        err <= KI_FROM_IC50_TOL,
        f"Ki_from_IC50 {v['Ki_from_IC50_uM']:.3g}, the equation for {mech} gives {expected:.3g}",
        err,
        KI_FROM_IC50_TOL,
    )


@register
class InhibitorTask(TaskPlugin):
    task_id = "t06-inhibitor-mechanism"
    version = "1.0.0"
    timeout_s = 300.0

    def make_case(self, seed: int) -> InhibitorCase:
        return make_case(seed)

    def write_inputs(self, case: InhibitorCase, input_dir: Path) -> None:
        write_inputs(case, input_dir)

    def argv(self, lang: str = "python") -> tuple[str, ...]:
        return ("{submission}", "--indir", "{input_dir}", "--outdir", "{output_dir}")

    def checks(self) -> CheckGroup:
        return CheckGroup(
            "t06",
            [
                CheckGroup("load", [files_present, schema], gate=True),
                CheckGroup("mechanism", [mechanism]),
                CheckGroup("constants", [km_kcat, ki, alpha_check, ic50_check, ki_from_ic50]),
            ],
        )
