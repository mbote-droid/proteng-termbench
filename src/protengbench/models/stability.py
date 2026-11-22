"""A hidden folding-stability landscape for point mutants of a small protein.

Sign convention: ddG = G_folded(mutant) - G_folded(wild type) for unfolding free
energy change, reported here as the change in folding free energy, in kcal/mol.
Negative ddG means the mutant is more stable than wild type.

Single substitution of residue i from `wt` to `a`, with burial b_i in [0, 1]
(from C-alpha neighbour counts within 10 Angstrom):

    ddG_i(a) = -0.75 b_i^2 (h_a - h_wt)                          hydrophobic burial
               + 0.6 b_i (|v_a - v*_i| - |v_wt - v*_i|) / 60     packing: volume mismatch
               + 1.4 [a = Pro] + 0.5 [a = Gly]                   backbone strain
               + eps_ia                                          idiosyncratic, N(0, 0.25)

h is the Kyte-Doolittle hydropathy / 4.5, v the residue volume in cubic
Angstrom, and v*_i = v_wt (1 + delta_i), delta_i ~ N(0, 0.12): the cavity next to
position i is a little larger or smaller than the wild-type side chain, which
is what makes some larger side chains stabilising.

Several substitutions. Effects add, except for residue pairs whose C-alpha atoms
are within 8 Angstrom (the contact list):

* Antagonism. If both substitutions are stabilising, they partly cover the same
  interactions, and the pair costs back a fraction of the smaller effect:

      c_ij = +0.8 f(d_ij) min(|ddG_i|, |ddG_j|),   f = 1 for d < 6 A, 0.6 otherwise.

* Synergy. A few pairs (listed in the double-mutant scan) are designed to
  complement each other and gain an extra c_ij = -U(1.0, 1.8).

    ddG(set) = sum_i ddG_i + sum_{i<j} c_ij.

So choosing the five best single mutants (an additive model) over-predicts the
gain when they touch each other, and misses the synergistic pairs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

AA = "ACDEFGHIKLMNPQRSTVWY"
HYDROPATHY = dict(
    zip(
        AA,
        np.array(
            [
                1.8,
                2.5,
                -3.5,
                -3.5,
                2.8,
                -0.4,
                -3.2,
                4.5,
                -3.9,
                3.8,
                1.9,
                -3.5,
                -1.6,
                -3.5,
                -4.5,
                -0.8,
                -0.7,
                4.2,
                -0.9,
                -1.3,
            ]
        )
        / 4.5,
        strict=True,
    )
)
VOLUME = dict(
    zip(
        AA,
        [
            88.6,
            108.5,
            111.1,
            138.4,
            189.9,
            60.1,
            153.2,
            166.7,
            168.6,
            166.7,
            162.9,
            114.1,
            112.7,
            143.8,
            173.4,
            89.0,
            116.1,
            140.0,
            227.8,
            193.6,
        ],
        strict=True,
    )
)
THREE = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}

CONTACT_A = 8.0
CLOSE_A = 6.0
ANTAGONISM = 0.8
MAX_SUBSTITUTIONS = 5


def burial(ca: FloatArray) -> FloatArray:
    d = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    count = (d < 10.0).sum(axis=1) - 1.0
    out: FloatArray = (count - count.min()) / (count.max() - count.min())
    return out


@dataclass(frozen=True)
class Landscape:
    sequence: str  # one-letter wild-type sequence, positions 1..n
    single: FloatArray  # (n, 20) true ddG, column order AA; wild type column is 0
    dist: FloatArray  # (n, n) C-alpha distances
    synergy: dict[tuple[int, str, int, str], float]  # ((i, a), (j, b)) with i < j, 1-based

    def pair_term(self, i: int, a: str, j: int, b: str) -> float:
        """Interaction between substitution i->a and j->b (1-based positions, i != j)."""
        if i > j:
            i, a, j, b = j, b, i, a
        d = self.dist[i - 1, j - 1]
        out = self.synergy.get((i, a, j, b), 0.0)
        if d < CONTACT_A:
            si, sj = self.single[i - 1, AA.index(a)], self.single[j - 1, AA.index(b)]
            if si < 0 and sj < 0:
                f = 1.0 if d < CLOSE_A else 0.6
                out += ANTAGONISM * f * min(-si, -sj)
        return out

    def ddg(self, subs: list[tuple[int, str]]) -> float:
        total = sum(float(self.single[i - 1, AA.index(a)]) for i, a in subs)
        for (i, a), (j, b) in combinations(subs, 2):
            total += self.pair_term(i, a, j, b)
        return total


def make_landscape(
    ca: FloatArray,
    sequence: str,
    rng_cavity: np.random.Generator,
    rng_eps: np.random.Generator,
    rng_syn: np.random.Generator,
) -> Landscape:
    n = len(sequence)
    b = burial(ca)
    delta = rng_cavity.normal(0.0, 0.12, n)
    eps = rng_eps.normal(0.0, 0.25, (n, len(AA)))
    single = np.zeros((n, len(AA)))
    for i, wt in enumerate(sequence):
        v_opt = VOLUME[wt] * (1.0 + delta[i])
        for k, a in enumerate(AA):
            if a == wt:
                continue
            single[i, k] = (
                -0.75 * b[i] ** 2 * (HYDROPATHY[a] - HYDROPATHY[wt])
                + 0.6 * b[i] * (abs(VOLUME[a] - v_opt) - abs(VOLUME[wt] - v_opt)) / 60.0
                + (1.4 if a == "P" else 0.0)
                + (0.5 if a == "G" else 0.0)
                + eps[i, k]
            )
    dist = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)

    # synergistic pairs: contacts between two clearly stabilising substitutions
    stabilising = [
        (i + 1, AA[k])
        for i in range(n)
        for k in range(len(AA))
        if single[i, k] < -0.35 and AA[k] != sequence[i]
    ]
    cand = [
        (p, q)
        for p, q in combinations(stabilising, 2)
        if p[0] != q[0] and dist[p[0] - 1, q[0] - 1] < CONTACT_A
    ]
    synergy: dict[tuple[int, str, int, str], float] = {}
    if cand:
        picks = rng_syn.choice(len(cand), size=min(12, len(cand)), replace=False)
        for idx in picks:
            (i, a), (j, bb) = cand[int(idx)]
            if i > j:
                i, a, j, bb = j, bb, i, a
            synergy[(i, a, j, bb)] = -float(rng_syn.uniform(1.0, 1.8))
    return Landscape(sequence, single, dist, synergy)


def best_set(
    land: Landscape, allowed: set[int], pool_size: int = 40, max_subs: int = MAX_SUBSTITUTIONS
) -> tuple[float, list[tuple[int, str]]]:
    """Exhaustive optimum over the pool: the `pool_size` most stabilising allowed singles plus
    the allowed partners of every synergistic pair; sets of 1..max_subs substitutions at
    distinct positions."""
    cands = [
        (i + 1, AA[k])
        for i in range(len(land.sequence))
        if (i + 1) in allowed
        for k in range(len(AA))
        if AA[k] != land.sequence[i]
    ]
    cands.sort(key=lambda t: land.single[t[0] - 1, AA.index(t[1])])
    pool = cands[:pool_size]
    for i, a, j, b in land.synergy:
        for item in ((i, a), (j, b)):
            if item[0] in allowed and item not in pool:
                pool.append(item)
    k = len(pool)
    pos = np.array([p for p, _ in pool])
    s = np.array([land.single[p - 1, AA.index(a)] for p, a in pool])
    c = np.zeros((k, k))
    for x in range(k):
        for y in range(k):
            if pool[x][0] == pool[y][0]:
                c[x, y] = 1e3
            elif x < y:
                c[x, y] = c[y, x] = land.pair_term(*pool[x], *pool[y])
    best: tuple[float, list[tuple[int, str]]] = (math.inf, [])
    for m in range(1, max_subs + 1):
        combos = np.fromiter(
            (v for t in combinations(range(k), m) for v in t),
            dtype=np.int32,
            count=math.comb(k, m) * m,
        ).reshape(-1, m)
        total = s[combos].sum(axis=1)
        for u in range(m):
            for v in range(u + 1, m):
                total = total + c[combos[:, u], combos[:, v]]
        arg = int(np.argmin(total))
        if total[arg] < best[0]:
            best = (float(total[arg]), [pool[int(q)] for q in combos[arg]])
    del pos
    return best
