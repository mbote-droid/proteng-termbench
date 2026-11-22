"""Reference solution for t01: align by sequence, superpose, then find what moved.

Method
1. Read the C-alpha atoms of both files and match residues by sequence with a
   global alignment with affine gap penalties. Residue numbers are not used: the
   reference is renumbered and has a gap, and the model has an N-terminal tag.
2. Superpose all matched C-alpha atoms (Kabsch) for the global RMSD and the
   TM-score (d0 = 1.24 (L - 15)^(1/3) - 1.8, L = reference length).
3. Look for one contiguous segment that is a better rigid body of its own:
   for every segment [s, e] superpose it separately and superpose the rest
   separately, and keep the segment with the smallest total squared error.
   If the best two-body split does not beat one body by a clear margin, the
   model is accurate.
4. A moved segment of at least 20 residues is a domain rotation (its rotation
   angle is the angle between the two separate superpositions); a shorter
   one is a local error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from Bio.Align import PairwiseAligner

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def read_ca(path: Path):
    nums, seq, xyz = [], [], []
    for line in path.read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            nums.append(int(line[22:26]))
            seq.append(THREE_TO_ONE.get(line[17:20].strip(), "X"))
            xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(nums), "".join(seq), np.array(xyz)


def align_sequences(a: str, b: str):
    """Global alignment with affine gap penalties; returns index pairs (i in a, j in b).

    Affine gaps (a cost to open a gap, a smaller cost to extend it) prefer one long
    gap over several short ones, which is how an unresolved loop looks."""
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    best = aligner.align(a, b)[0]
    pairs = []
    for (a0, a1), (b0, _b1) in zip(*best.aligned, strict=True):
        pairs += [(a0 + k, b0 + k) for k in range(a1 - a0)]
    return pairs


def kabsch(x: np.ndarray, y: np.ndarray):
    xc, yc = x.mean(0), y.mean(0)
    u, _, vt = np.linalg.svd((x - xc).T @ (y - yc))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, yc - r @ xc


def sq_dist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r, t = kabsch(x, y)
    return ((x @ r.T + t - y) ** 2).sum(1)


def rotation_angle_deg(r: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    ind, outd = Path(a.indir), Path(a.outdir)
    outd.mkdir(parents=True, exist_ok=True)

    m_nums, m_seq, m_xyz = read_ca(ind / "model.pdb")
    _r_nums, r_seq, r_xyz = read_ca(ind / "reference.pdb")
    pairs = align_sequences(m_seq, r_seq)
    mi = np.array([p[0] for p in pairs])
    ri = np.array([p[1] for p in pairs])
    x, y, nums = m_xyz[mi], r_xyz[ri], m_nums[mi]
    n = len(pairs)

    d2 = sq_dist(x, y)
    rmsd = float(np.sqrt(d2.mean()))
    d0 = 1.24 * (len(r_seq) - 15) ** (1.0 / 3.0) - 1.8
    tm = float(np.sum(1.0 / (1.0 + d2 / d0**2)) / len(r_seq))

    # best contiguous rigid segment
    one_body = float(d2.sum())
    best = (one_body, None)
    min_len = 8
    for s in range(0, n - min_len):
        for e in range(s + min_len, n + 1):
            if e - s > n // 2:
                break
            inside = np.zeros(n, dtype=bool)
            inside[s:e] = True
            two = sq_dist(x[inside], y[inside]).sum() + sq_dist(x[~inside], y[~inside]).sum()
            if two < best[0]:
                best = (two, (s, e))
    out: dict[str, object] = {"n_matched": n, "rmsd_global_A": rmsd, "tm_score": tm}
    # How much does the split explain, in units of the residual noise? Measured on
    # accurate models this statistic is at most 10; any real error scores above 150.
    stat = (one_body - best[0]) / (best[0] / n) if best[1] else 0.0
    if stat < 50.0:
        out.update(verdict="accurate", moved_segment=None, hinge_residues=None,
                   rotation_angle_deg=None, rmsd_parts_A=None)
    else:
        s, e = best[1]
        inside = np.zeros(n, dtype=bool)
        inside[s:e] = True
        r_in, _ = kabsch(x[inside], y[inside])
        r_out, _ = kabsch(x[~inside], y[~inside])
        angle = rotation_angle_deg(r_in @ r_out.T)
        first, last = int(nums[s]), int(nums[e - 1])
        # a domain is tens of residues; a displaced stretch of about ten is a local error
        # (a rigid fit of such a stretch also shows a spurious rotation, so length decides)
        rotation = (e - s) >= 20
        out.update(
            verdict="domain_rotation" if rotation else "local_error",
            moved_segment=[first, last],
            hinge_residues=[first - 1, last + 1],
            rotation_angle_deg=angle if rotation else None,
            rmsd_parts_A={"moved": float(np.sqrt(sq_dist(x[inside], y[inside]).mean())),
                          "rest": float(np.sqrt(sq_dist(x[~inside], y[~inside]).mean()))},
        )
    (outd / "triage.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
