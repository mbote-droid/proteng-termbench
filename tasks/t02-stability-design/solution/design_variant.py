"""Reference solution for t02: design a stabilised variant from scan data.

Method
1. The single scan gives each substitution's own effect. Stabilising effects
   are not additive: two stabilising substitutions in contact partly cover the
   same interactions. The double scan measures the pair effect directly, so
   the pair coupling is (double - single_i - single_j).
2. Fit the antagonism rule on the measured pairs: for two stabilising
   substitutions in contact, coupling = k f(d) min(|s_i|, |s_j|), with
   f = 1 below 6 Angstrom and 0.6 otherwise. k is estimated with a median,
   which ignores the few synergistic pairs.
3. Pairs whose coupling is clearly more negative than the rule (by more than
   0.4 kcal/mol, about two standard deviations of the measurement noise) are
   synergistic, and their measured coupling is used as measured. (Shrinking
   it toward the rule was tried and lost about 3 percent of the achievable
   stability on average: genuine synergy is several times the noise.)
4. Search sets of up to five substitutions at distinct, allowed positions
   exhaustively over the most stabilising singles plus the partners of every
   synergistic pair, scoring each set with singles + couplings.
5. Report the set's predicted ddG and an honest sigma: the measurement noise
   of the parts, plus an allowance for choosing the best of many sets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np

CONTACT, CLOSE = 8.0, 6.0
POOL = 38
SYNERGY_CUT = -0.40
SHRINK = 1.0


def load(ind: Path):
    cons = json.loads((ind / "constraints.json").read_text())
    single: dict[tuple[int, str], float] = {}
    wt: dict[int, str] = {}
    with (ind / "scan_single.csv").open() as fh:
        for row in csv.DictReader(fh):
            single[(int(row["position"]), row["mut"])] = float(row["ddG_kcal_per_mol"])
            wt[int(row["position"])] = row["wt"]
    dist: dict[tuple[int, int], float] = {}
    with (ind / "contacts.csv").open() as fh:
        for row in csv.DictReader(fh):
            i, j = int(row["pos1"]), int(row["pos2"])
            dist[(i, j)] = dist[(j, i)] = float(row["ca_distance_A"])
    doubles = []
    with (ind / "scan_double.csv").open() as fh:
        for row in csv.DictReader(fh):
            doubles.append((int(row["pos1"]), row["mut1"], int(row["pos2"]), row["mut2"],
                            float(row["ddG_kcal_per_mol"])))
    return cons, single, wt, dist, doubles


def fit_couplings(single, dist, doubles):
    """Antagonism rule (k estimated by the median) plus shrunk synergy for outlying pairs."""

    def rule_unit(i, x, j, y):
        d = dist.get((i, j))
        si, sj = single[(i, x)], single[(j, y)]
        if d is None or si >= 0 or sj >= 0:
            return 0.0
        return (1.0 if d < CLOSE else 0.6) * min(-si, -sj)

    rows = []
    for i, x, j, y, value in doubles:
        rows.append((i, x, j, y, value - single[(i, x)] - single[(j, y)], rule_unit(i, x, j, y)))
    ratios = [obs / unit for *_, obs, unit in rows if unit > 0.35]
    k = float(np.median(ratios)) if ratios else 0.0
    coupling: dict[tuple[int, str, int, str], float] = {}
    synergistic = []
    for i, x, j, y, obs, unit in rows:
        pred = k * unit
        resid = obs - pred
        c = pred + (SHRINK * resid if resid < SYNERGY_CUT else 0.0)
        coupling[(i, x, j, y)] = coupling[(j, y, i, x)] = c
        if resid < SYNERGY_CUT:
            synergistic.append(((i, x), (j, y)))

    def pair(p, q):
        if p[0] == q[0]:
            return 1e3
        known = coupling.get((p[0], p[1], q[0], q[1]))
        return known if known is not None else k * rule_unit(p[0], p[1], q[0], q[1])

    return k, synergistic, pair


def search(single, forbidden, max_subs, synergistic, pair):
    cands = sorted((key for key in single if key[0] not in forbidden), key=lambda t: single[t])
    pool = cands[:POOL]
    for p, q in synergistic:
        for item in (p, q):
            if item[0] not in forbidden and item not in pool:
                pool.append(item)
    m = len(pool)
    s = np.array([single[p] for p in pool])
    c = np.zeros((m, m))
    for u in range(m):
        for v in range(u + 1, m):
            c[u, v] = c[v, u] = pair(pool[u], pool[v])
    best, best_idx = math.inf, None
    for size in range(1, max_subs + 1):
        combos = np.fromiter((v for t in combinations(range(m), size) for v in t), dtype=np.int32,
                             count=math.comb(m, size) * size).reshape(-1, size)
        total = s[combos].sum(axis=1)
        for u in range(size):
            for v in range(u + 1, size):
                total = total + c[combos[:, u], combos[:, v]]
        arg = int(np.argmin(total))
        if total[arg] < best:
            best, best_idx = float(total[arg]), combos[arg]
    return best, [pool[int(q)] for q in best_idx], pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    ind, outd = Path(a.indir), Path(a.outdir)
    outd.mkdir(parents=True, exist_ok=True)
    cons, single, wt, dist, doubles = load(ind)
    forbidden, max_subs = set(cons["forbidden_positions"]), int(cons["max_substitutions"])
    _k, synergistic, pair = fit_couplings(single, dist, doubles)
    best, chosen, _pool = search(single, forbidden, max_subs, synergistic, pair)
    pairs_in_set = sum(1 for p, q in combinations(chosen, 2) if (p[0], q[0]) in dist)
    sigma = math.sqrt(len(chosen) * 0.12**2 + pairs_in_set * 0.20**2) + 0.20
    (outd / "design.json").write_text(json.dumps({
        "substitutions": [f"{wt[p]}{p}{x}" for p, x in chosen],
        "predicted_ddG_kcal_per_mol": best,
        "sigma_kcal_per_mol": min(max(sigma, 0.25), 1.0),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
