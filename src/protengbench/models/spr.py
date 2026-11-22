"""Surface plasmon resonance (SPR) binding models.

Units throughout: time s, concentration M, response RU (resonance units),
kon M^-1 s^-1, koff s^-1, KD = koff / kon in M, Rmax RU, kt RU M^-1 s^-1.

One-to-one Langmuir binding of analyte (concentration C) to immobilised
ligand with capacity Rmax:

    dR/dt = kon C (Rmax - R) - koff R

During injection C is constant and the solution is exact,

    R(t) = Req (1 - exp(-(kon C + koff) t)),   Req = Rmax kon C / (kon C + koff)

and after injection C = 0 and R(t) = R0 exp(-koff t).

Mass transport limitation. When the surface binds analyte faster than
diffusion through the unstirred layer can replace it, the concentration at the
surface Cs falls below the bulk C. With the standard two-compartment model and a
quasi-steady layer, kt (C - Cs) = dR/dt, which gives

    dR/dt = [kon C (Rmax - R) - koff R] / (1 + kon (Rmax - R) / kt)

so the apparent association is slowed by the factor 1 + kon (Rmax - R) / kt and
kon is underestimated by a model that ignores it.

Heterogeneous surface (two independent sites): R = R1 + R2, each following the
one-to-one equation with its own kon, koff, Rmax.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

FloatArray = NDArray[np.float64]

BASELINE_S = 20.0
ASSOC_S = 120.0
DISSOC_S = 300.0
CYCLE_S = BASELINE_S + ASSOC_S + DISSOC_S
SAMPLE_DT_S = 1.0


@dataclass(frozen=True)
class OneToOne:
    kon: float
    koff: float
    rmax: float

    @property
    def kd(self) -> float:
        return self.koff / self.kon


def langmuir_cycle(
    t: FloatArray, conc_m: float, kon: float, koff: float, rmax: float
) -> FloatArray:
    """Exact one-to-one response; `t` is time since the start of the injection (s)."""
    t = np.asarray(t, dtype=float)
    k_obs = kon * conc_m + koff
    r_eq = rmax * kon * conc_m / k_obs
    r_end = r_eq * (1.0 - np.exp(-k_obs * ASSOC_S))
    inj = r_eq * (1.0 - np.exp(-k_obs * np.clip(t, 0.0, ASSOC_S)))
    dis = r_end * np.exp(-koff * np.clip(t - ASSOC_S, 0.0, None))
    out: FloatArray = np.where(t < 0.0, 0.0, np.where(t <= ASSOC_S, inj, dis))
    return out


def mass_transport_cycle(
    t: FloatArray, conc_m: float, kon: float, koff: float, rmax: float, kt: float
) -> FloatArray:
    """Response with the mass-transport-limited rate law (integrated numerically)."""
    t = np.asarray(t, dtype=float)

    def rhs(_: float, r: FloatArray, c: float) -> list[float]:
        free = rmax - r[0]
        return [(kon * c * free - koff * r[0]) / (1.0 + kon * free / kt)]

    opts = {"method": "LSODA", "rtol": 1e-9, "atol": 1e-10}
    out = np.zeros_like(t)
    inj = (t >= 0.0) & (t <= ASSOC_S)
    sol = solve_ivp(rhs, (0.0, ASSOC_S), [0.0], t_eval=t[inj], args=(conc_m,), **opts)
    out[inj] = sol.y[0]
    r_end = float(solve_ivp(rhs, (0.0, ASSOC_S), [0.0], args=(conc_m,), **opts).y[0, -1])
    dis = t > ASSOC_S
    if dis.any():
        sol = solve_ivp(rhs, (ASSOC_S, t[dis][-1]), [r_end], t_eval=t[dis], args=(0.0,), **opts)
        out[dis] = sol.y[0]
    return out


def two_site_cycle(
    t: FloatArray, conc_m: float, k1: tuple[float, float, float], k2: tuple[float, float, float]
) -> FloatArray:
    return langmuir_cycle(t, conc_m, *k1) + langmuir_cycle(t, conc_m, *k2)
