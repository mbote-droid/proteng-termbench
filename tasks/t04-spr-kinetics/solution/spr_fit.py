"""Reference solution for t04: global fit of SPR sensorgrams with model selection.

Method: for each candidate model (one-to-one, mass transport, two site) the
nonlinear kinetic parameters are fitted by least squares in log space; the
linear nuisance terms (per-cycle baseline offset, one drift slope shared by all
cycles, a bulk jump proportional to concentration during injection) are
eliminated at every step by linear least squares (variable projection). Models
are compared with AICc, a more complex model being accepted only if it lowers
AICc by more than 10. Standard errors come from the projected Jacobian.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


def langmuir(t, c, kon, koff, rmax, assoc):
    k = kon * c + koff
    req = rmax * kon * c / k
    r_end = req * (1 - np.exp(-k * assoc))
    inj = req * (1 - np.exp(-k * np.clip(t, 0, assoc)))
    return np.where(t < 0, 0.0, np.where(t <= assoc, inj, r_end * np.exp(-koff * np.clip(t - assoc, 0, None))))


def mass_transport(t, c, kon, koff, rmax, kt, assoc):
    def rhs(_, r, cc):
        free = rmax - r[0]
        return [(kon * cc * free - koff * r[0]) / (1 + kon * free / kt)]

    out = np.zeros_like(t)
    inj = (t >= 0) & (t <= assoc)
    opts = dict(method="LSODA", rtol=1e-8, atol=1e-9)
    sol = solve_ivp(rhs, (0, assoc), [0.0], t_eval=t[inj], args=(c,), **opts)
    out[inj] = sol.y[0]
    r_end = solve_ivp(rhs, (0, assoc), [0.0], args=(c,), **opts).y[0, -1]
    dis = t > assoc
    sol = solve_ivp(rhs, (assoc, t[dis][-1]), [r_end], t_eval=t[dis], args=(0.0,), **opts)
    out[dis] = sol.y[0]
    return out


class Data:
    def __init__(self, indir):
        meta = json.loads((indir / "experiment.json").read_text())
        self.assoc = meta["association_s"]
        rows = np.genfromtxt(indir / "sensorgrams.csv", delimiter=",", names=True)
        self.cycles = sorted(set(rows["cycle"].astype(int)))
        self.t = rows["time_s"][rows["cycle"] == 0]
        self.conc_nm = np.array([rows["concentration_nM"][rows["cycle"] == i][0] for i in self.cycles])
        self.y = np.concatenate([rows["response_RU"][rows["cycle"] == i] for i in self.cycles])
        n_c, n_t = len(self.cycles), self.t.size
        inj = ((self.t >= 0) & (self.t <= self.assoc)).astype(float)
        offsets = np.zeros((n_c * n_t, n_c))
        for i in range(n_c):
            offsets[i * n_t:(i + 1) * n_t, i] = 1
        drift = np.tile(self.t - self.t.min(), n_c)
        bulk = np.concatenate([inj * c for c in self.conc_nm])
        self.nuis = np.column_stack([offsets, drift, bulk])
        self.q = np.linalg.pinv(self.nuis)

    def curves(self, model, p):
        parts = []
        for c_nm in self.conc_nm:
            c = c_nm * 1e-9
            if c == 0:
                parts.append(np.zeros_like(self.t))
            elif model == "one_to_one":
                parts.append(langmuir(self.t, c, p[0], p[1], p[2], self.assoc))
            elif model == "mass_transport":
                parts.append(mass_transport(self.t, c, p[0], p[1], p[2], p[3], self.assoc))
            else:
                parts.append(langmuir(self.t, c, p[0], p[1], p[2], self.assoc)
                             + langmuir(self.t, c, p[3], p[4], p[5], self.assoc))
        return np.concatenate(parts)

    def residual(self, model, logp):
        r = self.y - self.curves(model, np.exp(logp))
        return r - self.nuis @ (self.q @ r)


def fit(data, model, starts):
    best = None
    for s in starts:
        try:
            r = least_squares(lambda lp: data.residual(model, lp), np.log(s), method="trf", x_scale=1.0)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best


def aicc(rss, n, k):
    return n * math.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    d = Data(Path(a.indir))
    n = d.y.size

    # starting points from the data: kon from the association rate at the top concentration
    kon0, koff0 = 3e5, 5e-3
    rmax0 = float(np.max(d.y)) * 1.5
    fits = {}
    starts = [[kon0 * m, koff0 * k, rmax0] for m in (0.3, 1, 3) for k in (0.3, 1, 3)]
    fits["one_to_one"] = fit(d, "one_to_one", starts)
    starts_mt = [[s[0] * 2, s[1], s[2], s[0] * s[2] / q] for s in starts[::2] for q in (1.5, 4)]
    fits["mass_transport"] = fit(d, "mass_transport", starts_mt)
    two = []
    for ratio in (20, 60):
        for f in (0.4, 0.6):
            for m in (0.5, 2):
                two.append([kon0 * m, koff0 / 4, rmax0 * f, kon0 * m, koff0 / 4 * ratio, rmax0 * (1 - f)])
    fits["two_site"] = fit(d, "two_site", two)

    k_par = {"one_to_one": 3, "mass_transport": 4, "two_site": 6}
    score = {m: aicc(2 * f.cost, n, k_par[m] + d.nuis.shape[1]) for m, f in fits.items()}
    model = "one_to_one"
    for cand in ("mass_transport", "two_site"):
        if score[cand] < score[model] - 10:
            model = cand
    f = fits[model]
    p = np.exp(f.x)

    # standard errors of ln parameters from the projected Jacobian (sigma from the residuals)
    sigma2 = 2 * f.cost / (n - d.nuis.shape[1] - k_par[model])
    cov = np.linalg.pinv(f.jac.T @ f.jac) * sigma2
    se = np.sqrt(np.abs(np.diag(cov)))

    out = {"model": model}
    if model == "two_site":
        order = np.argsort([p[1], p[4]])
        sites = [p[0:3], p[3:6]]
        s1, s2 = sites[order[0]], sites[order[1]]
        out["parameters"] = {
            "kon1_per_M_per_s": s1[0], "koff1_per_s": s1[1], "Rmax1_RU": s1[2],
            "kon2_per_M_per_s": s2[0], "koff2_per_s": s2[1], "Rmax2_RU": s2[2],
        }
    else:
        params = {"kon_per_M_per_s": p[0], "koff_per_s": p[1], "Rmax_RU": p[2], "KD_M": p[1] / p[0]}
        if model == "mass_transport":
            params["kt_RU_per_M_per_s"] = p[3]
        out["parameters"] = params
        out["se_log"] = {"kon": se[0], "koff": se[1]}
    out["aicc"] = score
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "fit.json").write_text(json.dumps(out, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
