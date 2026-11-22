"""t04: fit surface plasmon resonance sensorgrams, choose the binding model, report kinetics.

What the agent gets: reference-subtracted sensorgrams for a blank and five
analyte concentrations (nanomolar), each cycle with a baseline, a 120 s
injection and a 300 s dissociation. The data carry a bulk refractive-index
jump during injection (proportional to concentration), a slow common drift,
a small random baseline offset per cycle and white noise. The surface is one of

  * one_to_one       Langmuir binding,
  * mass_transport   Langmuir binding limited by diffusion to the surface (kt),
  * two_site         two independent kinds of ligand with different koff.

How each output is judged, and why:

* The model label must match how the data were made. The three cases were
  built so their residual structure is visible (section Design in docs).
* One-to-one and mass-transport parameters must agree with the simulation
  truth within the larger of a floor and four times the Fisher standard
  error that the design allows at the true parameters. A fixed tolerance
  would punish correct fits of weakly determined parameters (the same lesson
  as the t02 pharmacokinetics task of the sister benchmark).
* Reported log standard errors of kon and koff must lie within a factor of
  two of that Fisher standard error: over-confident and absurdly wide
  intervals both fail.
* Two-site fits are judged on the two dissociation rates and the binding
  capacity fractions, which are identifiable when the rates differ by 20x.
* The reported KD must equal koff / kon of the reported parameters.
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
from protengbench.models import spr

MODELS = ("one_to_one", "mass_transport", "two_site")
NOISE_RU = 0.25
CONC_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)  # analyte concentration as multiples of KD (one site)
TOL_FLOOR_LOG = 0.12  # a 13 % error is always accepted
TOL_SIGMAS = 4.0
SE_FACTOR = 2.0
KD_CONSISTENCY = 0.01
SITE_KOFF_TOL_LOG = 0.35
SITE_FRACTION_TOL = 0.10

PARAM_KEYS = {
    "one_to_one": ("kon_per_M_per_s", "koff_per_s", "Rmax_RU", "KD_M"),
    "mass_transport": ("kon_per_M_per_s", "koff_per_s", "Rmax_RU", "KD_M", "kt_RU_per_M_per_s"),
    "two_site": (
        "kon1_per_M_per_s",
        "koff1_per_s",
        "Rmax1_RU",
        "kon2_per_M_per_s",
        "koff2_per_s",
        "Rmax2_RU",
    ),
}


@dataclass(frozen=True)
class SprCase:
    model: str
    sites: tuple[tuple[float, float, float], ...]  # (kon, koff, rmax) per site
    kt: float  # only meaningful for mass_transport
    conc_nm: FloatArray  # per cycle, first is the blank (0)
    bulk_ru_per_nm: float
    drift_ru_per_s: float
    offsets: FloatArray
    time_s: FloatArray  # within a cycle: -BASELINE .. ASSOC + DISSOC
    response: FloatArray  # shape (n_cycles, n_time)
    fisher_se_log: dict[str, float]


def _cycle_truth(
    model: str,
    sites: tuple[tuple[float, float, float], ...],
    kt: float,
    conc_m: float,
    t: FloatArray,
) -> FloatArray:
    if conc_m == 0.0:
        return np.zeros_like(t)
    if model == "one_to_one":
        return spr.langmuir_cycle(t, conc_m, *sites[0])
    if model == "mass_transport":
        return spr.mass_transport_cycle(t, conc_m, *sites[0], kt)
    return spr.two_site_cycle(t, conc_m, sites[0], sites[1])


def nuisance_design(conc_nm: FloatArray, t: FloatArray) -> FloatArray:
    """Linear nuisance terms of the data model, stacked over all cycles (rows) :
    one baseline offset per cycle, one drift slope shared by all cycles, and a
    bulk jump proportional to concentration during the injection."""
    n_c, n_t = conc_nm.size, t.size
    inj = ((t >= 0.0) & (t <= spr.ASSOC_S)).astype(float)
    offsets = np.zeros((n_c * n_t, n_c))
    for i in range(n_c):
        offsets[i * n_t : (i + 1) * n_t, i] = 1.0
    drift = np.tile(t + spr.BASELINE_S, n_c)
    bulk = np.concatenate([inj * c for c in conc_nm])
    return np.column_stack([offsets, drift, bulk])


def _fisher_se_log(
    model: str,
    sites: tuple[tuple[float, float, float], ...],
    kt: float,
    conc_nm: FloatArray,
    t: FloatArray,
) -> dict[str, float]:
    """Standard error of ln kon, ln koff, ... from the Fisher information at the truth.

    The linear nuisance terms (offsets, drift, bulk) are estimated too: the
    kinetic columns of the Jacobian are projected off the nuisance columns
    (variable projection), which is the information left for the kinetics.
    Returns {} for two_site (not used)."""
    if model == "two_site":
        return {}
    mt = model == "mass_transport"
    names = ["kon", "koff", "rmax"] + (["kt"] if mt else [])
    base = np.array([*sites[0]] + ([kt] if mt else []))

    def curve(logp: FloatArray) -> FloatArray:
        p = np.exp(logp)
        s = ((float(p[0]), float(p[1]), float(p[2])),)
        k = float(p[3]) if mt else kt
        parts = [np.zeros_like(t)] + [
            _cycle_truth(model, s, k, float(c) * 1e-9, t) for c in conc_nm[1:]
        ]
        return np.concatenate(parts)

    h = 1e-4
    lp = np.log(base)
    cols = []
    for i in range(lp.size):
        d = np.zeros_like(lp)
        d[i] = h
        cols.append((curve(lp + d) - curve(lp - d)) / (2 * h))
    j = np.column_stack(cols)
    nuis = nuisance_design(conc_nm, t)
    j = j - nuis @ np.linalg.lstsq(nuis, j, rcond=None)[0]
    cov = np.linalg.pinv(j.T @ j / NOISE_RU**2)
    se = np.sqrt(np.abs(np.diag(cov)))
    return {n: float(se[i]) for i, n in enumerate(names)}


@lru_cache(maxsize=64)
def make_case(seed: int) -> SprCase:
    r_par, r_noise, r_nuis, r_model = streams(seed, 4)
    # a seed fixes the surface; the three models are used in rotation by the
    # first draw so that any block of seeds contains all of them
    model = MODELS[int(r_model.integers(0, 3))]
    t = np.arange(-spr.BASELINE_S, spr.ASSOC_S + spr.DISSOC_S + 1e-9, spr.SAMPLE_DT_S)

    kon = 10 ** r_par.uniform(4.9, 6.0)
    # the two-site surface needs its fast site slow enough to resolve at 1 Hz
    koff = 10 ** r_par.uniform(-2.6, -2.0 if model == "two_site" else -1.5)
    rmax = r_par.uniform(60.0, 160.0)
    kt = 0.0
    sites: tuple[tuple[float, float, float], ...]
    if model == "one_to_one":
        sites = ((kon, koff, rmax),)
    elif model == "mass_transport":
        # kt chosen so the correction kon*Rmax/kt is between 1.5 and 4 at the surface
        kt = kon * rmax / r_par.uniform(1.5, 4.0)
        sites = ((kon, koff, rmax),)
    else:
        f = r_par.uniform(0.35, 0.65)
        ratio = 10 ** r_par.uniform(1.3, 1.65)  # koff ratio between 20x and 45x
        sites = (
            (kon, koff, rmax * f),
            (kon * 10 ** r_par.uniform(-0.2, 0.2), koff * ratio, rmax * (1 - f)),
        )
    kd_ref = sites[0][1] / sites[0][0]
    conc_nm = np.concatenate([[0.0], np.array(CONC_FACTORS) * kd_ref * 1e9])
    conc_nm = np.round(conc_nm, 4)

    bulk = r_nuis.uniform(1.5, 6.0) / conc_nm[-1]  # RU per nM: a 1.5 to 6 RU jump at the top conc
    drift = r_nuis.uniform(-0.004, 0.004)  # RU per s, common to all cycles
    offsets = r_nuis.normal(0.0, 0.4, conc_nm.size)
    inj = (t >= 0.0) & (t <= spr.ASSOC_S)
    resp = np.empty((conc_nm.size, t.size))
    for i, c in enumerate(conc_nm):
        clean = _cycle_truth(model, sites, kt, c * 1e-9, t)
        resp[i] = (
            clean
            + inj * bulk * c
            + drift * (t + spr.BASELINE_S)
            + offsets[i]
            + r_noise.normal(0.0, NOISE_RU, t.size)
        )
    fisher = _fisher_se_log(model, sites, kt, conc_nm, t)
    return SprCase(model, sites, kt, conc_nm, bulk, drift, offsets, t, resp, fisher)


def write_inputs(case: SprCase, input_dir: Path) -> None:
    rows = []
    for i, c in enumerate(case.conc_nm):
        for t, r in zip(case.time_s, case.response[i], strict=True):
            rows.append(f"{i},{t:.1f},{c:.6g},{r:.5f}")
    (input_dir / "sensorgrams.csv").write_text(
        "cycle,time_s,concentration_nM,response_RU\n" + "\n".join(rows) + "\n"
    )
    meta = {
        "baseline_s": spr.BASELINE_S,
        "association_s": spr.ASSOC_S,
        "dissociation_s": spr.DISSOC_S,
        "sampling_interval_s": spr.SAMPLE_DT_S,
        "cycle_0": "blank buffer injection (concentration 0)",
        "time_zero": "start of analyte injection in every cycle",
        "allowed_models": list(MODELS),
    }
    (input_dir / "experiment.json").write_text(json.dumps(meta, indent=2) + "\n")


# --- checks ------------------------------------------------------------------------


def _case(ctx: CheckContext) -> SprCase:
    c: SprCase = ctx.truth
    return c


@check("FILES", gate=True)
def files_present(ctx: CheckContext) -> Outcome:
    ok = (ctx.output_dir / "fit.json").is_file()
    return Outcome(ok, "missing fit.json")


@check("SCHEMA", gate=True)
def schema(ctx: CheckContext) -> Outcome:
    raw = json.loads((ctx.output_dir / "fit.json").read_text())
    if not isinstance(raw, dict):
        raise SchemaError("fit.json: top level must be an object")
    model = raw.get("model")
    if model not in MODELS:
        raise SchemaError(f"fit.json: model must be one of {MODELS}")
    params = raw.get("parameters")
    if not isinstance(params, dict):
        raise SchemaError("fit.json: 'parameters' must be an object")
    out: dict[str, float] = {}
    for key in PARAM_KEYS[model]:
        v = params.get(key)
        if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v) or v <= 0:
            raise SchemaError(f"fit.json: parameters.{key} must be a positive number, got {v!r}")
        out[key] = float(v)
    se: dict[str, float] = {}
    if model != "two_site":
        raw_se = raw.get("se_log")
        if not isinstance(raw_se, dict):
            raise SchemaError("fit.json: 'se_log' must be an object for one_to_one/mass_transport")
        for key in ("kon", "koff"):
            v = raw_se.get(key)
            if isinstance(v, bool) or not isinstance(v, int | float) or not v > 0:
                raise SchemaError(f"fit.json: se_log.{key} must be a positive number")
            se[key] = float(v)
    ctx.outputs["model"] = model
    ctx.outputs["params"] = out
    ctx.outputs["se"] = se
    return Outcome(True)


@check("MODEL")
def model_label(ctx: CheckContext) -> Outcome:
    got, want = ctx.outputs["model"], _case(ctx).model
    return Outcome(got == want, f"model {got!r}, but the data were generated by {want!r}")


@check("KD_CONSISTENT")
def kd_consistent(ctx: CheckContext) -> Outcome:
    if ctx.outputs["model"] == "two_site":
        return Outcome(True)
    p = ctx.outputs["params"]
    err = abs(p["KD_M"] / (p["koff_per_s"] / p["kon_per_M_per_s"]) - 1.0)
    return Outcome(err <= KD_CONSISTENCY, "KD_M is not koff / kon", err, KD_CONSISTENCY)


def _log_tol(case: SprCase, name: str) -> float:
    se = case.fisher_se_log.get(name, 0.0)
    return max(TOL_FLOOR_LOG, TOL_SIGMAS * se)


@check("KINETICS")
def kinetics(ctx: CheckContext) -> Outcome:
    case = _case(ctx)
    if case.model == "two_site" or ctx.outputs["model"] != case.model:
        return Outcome(True)  # judged by MODEL / SITES; a wrong label already fails
    p = ctx.outputs["params"]
    kon0, koff0, rmax0 = case.sites[0]
    errs = {
        "kon": (abs(math.log(p["kon_per_M_per_s"] / kon0)), _log_tol(case, "kon")),
        "koff": (abs(math.log(p["koff_per_s"] / koff0)), _log_tol(case, "koff")),
        "rmax": (abs(math.log(p["Rmax_RU"] / rmax0)), _log_tol(case, "rmax")),
    }
    if case.model == "mass_transport":
        errs["kt"] = (abs(math.log(p["kt_RU_per_M_per_s"] / case.kt)), _log_tol(case, "kt"))
    for name, (e, tol) in errs.items():
        if e > tol:
            return Outcome(False, f"{name} off by {e:.2f} in ln units (limit {tol:.2f})", e, tol)
    worst = max(e / tol for e, tol in errs.values())
    return Outcome(True, "", worst, 1.0)


@check("KD")
def kd_accuracy(ctx: CheckContext) -> Outcome:
    case = _case(ctx)
    if case.model == "two_site" or ctx.outputs["model"] != case.model:
        return Outcome(True)
    kd0 = case.sites[0][1] / case.sites[0][0]
    err = abs(math.log(ctx.outputs["params"]["KD_M"] / kd0))
    tol = max(
        TOL_FLOOR_LOG,
        TOL_SIGMAS * math.hypot(case.fisher_se_log["kon"], case.fisher_se_log["koff"]),
    )
    return Outcome(err <= tol, f"KD off by a factor {math.exp(err):.2f}", err, tol)


@check("UNCERTAINTY")
def uncertainty(ctx: CheckContext) -> Outcome:
    case = _case(ctx)
    if case.model == "two_site" or ctx.outputs["model"] != case.model:
        return Outcome(True)
    se = ctx.outputs["se"]
    worst = 0.0
    for name in ("kon", "koff"):
        ratio = se[name] / case.fisher_se_log[name]
        worst = max(worst, ratio, 1.0 / ratio)
    msg = "reported standard errors are not within a factor of 2 of the data's information"
    return Outcome(worst <= SE_FACTOR, msg, worst, SE_FACTOR)


def _slow_fraction(sites_sorted: list[tuple[float, float, float]], conc_m: float) -> float:
    """Share of the end-of-injection signal (top concentration) carried by the slow site.
    Rmax alone is weakly determined for a fast site; this amplitude is what the
    data constrain."""
    amps = []
    for kon, koff, rmax in sites_sorted:
        k_obs = kon * conc_m + koff
        amps.append(rmax * kon * conc_m / k_obs * (1.0 - math.exp(-k_obs * spr.ASSOC_S)))
    return amps[0] / (amps[0] + amps[1])


@check("SITES")
def sites(ctx: CheckContext) -> Outcome:
    case = _case(ctx)
    if case.model != "two_site" or ctx.outputs["model"] != "two_site":
        return Outcome(True)
    p = ctx.outputs["params"]
    got = sorted(
        [
            (p["kon1_per_M_per_s"], p["koff1_per_s"], p["Rmax1_RU"]),
            (p["kon2_per_M_per_s"], p["koff2_per_s"], p["Rmax2_RU"]),
        ],
        key=lambda x: x[1],
    )
    want = sorted(case.sites, key=lambda x: x[1])
    for g, w in zip(got, want, strict=True):
        e = abs(math.log(g[1] / w[1]))
        if e > SITE_KOFF_TOL_LOG:
            return Outcome(
                False, f"a dissociation rate is off by {e:.2f} in ln units", e, SITE_KOFF_TOL_LOG
            )
    conc_m = float(case.conc_nm.max()) * 1e-9
    frac_err = abs(_slow_fraction(got, conc_m) - _slow_fraction(want, conc_m))
    return Outcome(
        frac_err <= SITE_FRACTION_TOL,
        "share of the signal carried by the slow site is wrong",
        frac_err,
        SITE_FRACTION_TOL,
    )


@register
class SprTask(TaskPlugin):
    task_id = "t04-spr-kinetics"
    version = "1.0.0"
    timeout_s = 240.0

    def make_case(self, seed: int) -> SprCase:
        return make_case(seed)

    def write_inputs(self, case: SprCase, input_dir: Path) -> None:
        write_inputs(case, input_dir)

    def argv(self, lang: str = "python") -> tuple[str, ...]:
        return ("{submission}", "--indir", "{input_dir}", "--outdir", "{output_dir}")

    def checks(self) -> CheckGroup:
        return CheckGroup(
            "t04",
            [
                CheckGroup("load", [files_present, schema], gate=True),
                CheckGroup("model", [model_label]),
                CheckGroup(
                    "kinetics",
                    [kd_consistent, kinetics, kd_accuracy, uncertainty, sites],
                ),
            ],
        )
