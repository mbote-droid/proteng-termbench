"""Enzyme kinetics with a reversible inhibitor, simulated from mass action.

Species (all in uM except where noted): E, S, ES, I, EI, ESI, P.

    E + S  <-> ES   (kon_s, koff_s)          ES -> E + P   (kcat)
    E + I  <-> EI   (kon_i, koff_i = Ki  kon_i)
    ES + I <-> ESI  (kon_i, koff_i' = alpha Ki kon_i)

Mechanisms are different settings of the two inhibitor-binding branches:

    competitive      inhibitor binds E only                (no ESI branch)
    uncompetitive    inhibitor binds ES only               (no EI branch), Ki' = alpha Ki
    mixed            both, alpha != 1 (alpha > 1 favours E, alpha < 1 favours ES)
    noncompetitive   both, alpha = 1

Mass action is exact: nothing assumes rapid equilibrium or a free inhibitor
concentration equal to the total, so substrate depletion, slow binding and
tight binding (enzyme concentration comparable to Ki) are all in the data.

Rapid-equilibrium (quasi-steady) rate law, valid when I >> E0 and binding is fast:

    v = Vmax S / (Km (1 + I/Ki) + S (1 + I/(alpha Ki))),   Vmax = kcat E0

and the concentration of inhibitor giving half the uninhibited rate at substrate S:

    IC50 = (Km + S) / (Km / Ki + S / (alpha Ki)).

Classical conversions follow: competitive Ki = IC50 / (1 + S/Km); uncompetitive
Ki' = IC50 / (1 + Km/S); noncompetitive Ki = IC50.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

FloatArray = NDArray[np.float64]

MECHANISMS = ("competitive", "uncompetitive", "mixed", "noncompetitive")
KON_S = 10.0  # uM^-1 s^-1 (1e7 M^-1 s^-1)
KON_I = 100.0  # uM^-1 s^-1 (1e8 M^-1 s^-1): fast, so equilibrium is reached in seconds


def simulate(
    mechanism: str,
    km: float,
    kcat: float,
    ki: float,
    alpha: float,
    e0: float,
    s0: float,
    i0: float,
    t: FloatArray,
) -> FloatArray:
    """Product concentration (uM) at times `t` (s) for one well. Km, Ki in uM, e0 in uM."""
    koff_s = km * KON_S - kcat
    if koff_s <= 0:
        raise ValueError("Km too small for kcat with kon_s fixed")
    bind_e = mechanism in ("competitive", "mixed", "noncompetitive")
    bind_es = mechanism in ("uncompetitive", "mixed", "noncompetitive")
    koff_e = ki * KON_I
    koff_es = alpha * ki * KON_I if mechanism != "uncompetitive" else ki * KON_I

    def rhs(_: float, y: FloatArray) -> list[float]:
        e, s, es, i, ei, esi, _p = y
        v_bind_s = KON_S * e * s - koff_s * es
        v_bind_ei = KON_I * e * i - koff_e * ei if bind_e else 0.0
        v_bind_esi = KON_I * es * i - koff_es * esi if bind_es else 0.0
        v_cat = kcat * es
        return [
            -v_bind_s - v_bind_ei + v_cat,  # E
            -v_bind_s,  # S
            v_bind_s - v_bind_esi - v_cat,  # ES
            -v_bind_ei - v_bind_esi,  # I
            v_bind_ei,  # EI
            v_bind_esi,  # ESI
            v_cat,  # P
        ]

    sol = solve_ivp(
        rhs,
        (0.0, float(t[-1])),
        [e0, s0, 0.0, i0, 0.0, 0.0, 0.0],
        t_eval=t,
        method="LSODA",
        rtol=1e-8,
        atol=1e-12,
    )
    out: FloatArray = sol.y[6]
    return out


def ic50(mechanism: str, km: float, ki: float, alpha: float, s: float) -> float:
    """Inhibitor concentration giving half the uninhibited initial rate at substrate s."""
    if mechanism == "competitive":
        return ki * (1.0 + s / km)
    if mechanism == "uncompetitive":
        return ki * (1.0 + km / s)  # here ki is Ki'
    return (km + s) / (km / ki + s / (alpha * ki))
