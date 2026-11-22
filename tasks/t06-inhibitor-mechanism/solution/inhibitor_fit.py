"""Reference solution for t06: global fit of integrated progress curves.

Model. Each well follows dS/dt = -v, P = S0 - S, with the quasi-steady rate law
for an enzyme with a reversible inhibitor. The free inhibitor is not assumed to
equal the total: it solves the conservation law If + Ib(If) = I0 (Morrison),
which keeps the fit valid when Ki is comparable to the enzyme concentration.

    Ef = E0 / (1 + a + kE + kES),  a = S/Km, kE = If/Ki, kES = a If/(alpha Ki)
    v  = kcat Ef a,                Ib  = Ef (kE + kES)

All wells are integrated together with fixed-step RK4 at the data spacing.
Four mechanisms are fitted (competitive, uncompetitive, mixed, noncompetitive)
and compared with AICc; mixed must beat the best three-parameter model by 6.
The IC50 at the stated substrate concentration is found by solving v(I0) =
v(0) / 2 for the fitted model, and Ki is recovered from it with the equation for
the chosen mechanism.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, least_squares


def free_inhibitor(s, i0, e0, km, ki, alpha, bind_e, bind_es):
    lo = np.zeros_like(s)
    hi = np.maximum(i0, 1e-12) * np.ones_like(s)
    a = s / km
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        k_e = mid / ki if bind_e else 0.0
        k_es = a * mid / (alpha * ki) if bind_es else 0.0
        ib = e0 * (k_e + k_es) / (1 + a + k_e + k_es)
        g = mid + ib - i0
        hi = np.where(g > 0, mid, hi)
        lo = np.where(g > 0, lo, mid)
    return 0.5 * (lo + hi)


def rate(s, i0, e0, km, kcat, ki, alpha, bind_e, bind_es):
    i_free = free_inhibitor(s, i0, e0, km, ki, alpha, bind_e, bind_es)
    a = s / km
    k_e = i_free / ki if bind_e else 0.0
    k_es = a * i_free / (alpha * ki) if bind_es else 0.0
    return kcat * e0 * a / (1 + a + k_e + k_es)


def progress(mech, par, s0, i0, e0, t):
    km, kcat, ki = par[0], par[1], par[2]
    alpha = par[3] if mech == "mixed" else 1.0
    bind_e = mech in ("competitive", "mixed", "noncompetitive")
    bind_es = mech in ("uncompetitive", "mixed", "noncompetitive")
    s = s0.copy()
    out = np.empty((s0.size, t.size))
    out[:, 0] = 0.0
    for k in range(1, t.size):
        h = t[k] - t[k - 1]

        def f(x):
            return -rate(np.maximum(x, 1e-9), i0, e0, km, kcat, ki, alpha, bind_e, bind_es)

        k1 = f(s)
        k2 = f(s + 0.5 * h * k1)
        k3 = f(s + 0.5 * h * k2)
        k4 = f(s + h * k3)
        s = s + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        out[:, k] = s0 - s
    return out


def aicc(rss, n, k):
    return n * math.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    ind = Path(a.indir)
    assay = json.loads((ind / "assay.json").read_text())
    rows = np.genfromtxt(ind / "plate.csv", delimiter=",", names=True)
    wells = sorted(set(rows["well"].astype(int)))
    t = rows["time_s"][rows["well"] == 0]
    s0 = np.array([rows["substrate_uM"][rows["well"] == w][0] for w in wells])
    i0 = np.array([rows["inhibitor_uM"][rows["well"] == w][0] for w in wells])
    sig = np.array([rows["signal_RFU"][rows["well"] == w] for w in wells])
    e0 = assay["enzyme_nM"] / 1000.0
    p_obs = (sig - assay["background_RFU"]) / assay["gain_RFU_per_uM_product"]
    # the background is known from the blank wells, so no well is re-zeroed on its own
    # (zeroing on one noisy reading adds a correlated error that favours extra parameters)
    s_ref = assay["ic50_substrate_uM"]

    # starting values from the uninhibited wells' initial rates
    free_w = i0 == 0
    v_init = (p_obs[free_w, 3] - p_obs[free_w, 0]) / (t[3] - t[0])
    vmax0 = 1.5 * float(v_init.max())
    km0 = float(np.median(s0))
    kcat0 = vmax0 / e0
    pos = i0[i0 > 0]
    ki_scale = float(np.median(pos))

    n = p_obs.size
    fits = {}

    def run(mech, x0):
        def resid(lp):
            return (progress(mech, np.exp(lp), s0, i0, e0, t) - p_obs).ravel()

        k = len(x0)
        lb = np.log([0.1, 0.01, 1e-5] + ([1e-3] if k == 4 else []))
        ub = np.log([1e3, 1e3, 1e3] + ([1e4] if k == 4 else []))
        x0 = np.clip(np.log(x0), lb + 1e-6, ub - 1e-6)
        try:
            return least_squares(resid, x0, bounds=(lb, ub), method="trf", x_scale=1.0)
        except Exception:
            return None

    # mixed first (it contains the other three as limits), from several starts
    best = None
    for kis in (0.1, 1.0):
        for al in (0.2, 5.0):
            r = run("mixed", [km0, kcat0, ki_scale * kis, al])
            if r is not None and (best is None or r.cost < best.cost):
                best = r
    fits["mixed"] = best
    km1, kc1, ki1, al1 = np.exp(best.x)
    # the other mechanisms start from the mixed solution (and from the data-based guess)
    starts = {
        "competitive": [[km1, kc1, ki1], [km0, kcat0, ki_scale]],
        "uncompetitive": [[km1, kc1, ki1 * al1], [km0, kcat0, ki_scale]],
        "noncompetitive": [[km1, kc1, ki1], [km0, kcat0, ki_scale]],
    }
    for mech, xs in starts.items():
        cand = [run(mech, x) for x in xs]
        cand = [c for c in cand if c is not None]
        fits[mech] = min(cand, key=lambda r: r.cost)
    kpar = {"competitive": 3, "uncompetitive": 3, "mixed": 4, "noncompetitive": 3}
    score = {m: aicc(2 * f.cost, n, kpar[m]) for m, f in fits.items()}
    three = min(("competitive", "uncompetitive", "noncompetitive"), key=lambda m: score[m])
    mech = "mixed" if score["mixed"] < score[three] - 6 else three
    par = np.exp(fits[mech].x)
    km, kcat, ki = par[0], par[1], par[2]
    alpha = par[3] if mech == "mixed" else 1.0

    bind_e = mech in ("competitive", "mixed", "noncompetitive")
    bind_es = mech in ("uncompetitive", "mixed", "noncompetitive")

    def v(i):
        return float(rate(np.array([s_ref]), np.array([i]), e0, km, kcat, ki, alpha, bind_e, bind_es)[0])

    v0 = v(0.0)
    hi = 100 * max(ki, e0)
    ic50 = brentq(lambda i: v(i) - 0.5 * v0, 1e-9, hi * 1e3, xtol=1e-12)
    free = max(ic50 - e0 / 2, 1e-12)
    if mech == "competitive":
        ki_conv = free / (1 + s_ref / km)
    elif mech == "uncompetitive":
        ki_conv = free / (1 + km / s_ref)
    else:
        ki_conv = free * (km + s_ref / alpha) / (km + s_ref)

    out = {
        "mechanism": mech, "Km_uM": km, "kcat_per_s": kcat, "Ki_uM": ki,
        "IC50_uM_at_10uM_S": ic50, "Ki_from_IC50_uM": ki_conv, "aicc": score,
    }
    if mech in ("mixed", "noncompetitive"):
        out["alpha"] = alpha
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "inhibition.json").write_text(json.dumps(out, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
