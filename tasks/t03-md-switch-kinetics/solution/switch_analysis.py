"""Reference solution for t03: align on the rigid core, find the switch, label frames, get rates.

Method
1. Superpose every frame on a reference with the Kabsch algorithm, but only on
   the residues that barely move. Fitting on all residues lets the mobile
   segment drag the fit, so the core is chosen iteratively: fit, measure the
   fluctuation of each residue about the mean, keep the quietest 60 percent,
   refit, repeat until the core stops changing.
2. The switch is the dominant collective motion: project the aligned frames on
   the first principal component and split the two clusters.
3. A two-state hidden Markov model (Gaussian emissions, Viterbi decoding)
   removes single-frame noise before transitions are counted.
4. Rates are the maximum-likelihood estimates for a two-state Markov chain:
   transitions out of a state divided by the time spent in it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def kabsch(mobile: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None):
    """Rotation r and translation so that mobile @ r.T + t best matches target."""
    w = np.ones(len(mobile)) if weights is None else weights
    w = w / w.sum()
    mc, tc = (mobile * w[:, None]).sum(0), (target * w[:, None]).sum(0)
    h = ((mobile - mc) * w[:, None]).T @ (target - tc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, tc - r @ mc


def align(frames: np.ndarray, ref: np.ndarray, core: np.ndarray) -> np.ndarray:
    out = np.empty_like(frames)
    for i, f in enumerate(frames):
        r, t = kabsch(f[core], ref[core])
        out[i] = f @ r.T + t
    return out


def core_alignment(frames: np.ndarray, keep: float = 0.6, rounds: int = 12) -> np.ndarray:
    n = frames.shape[1]
    core = np.ones(n, dtype=bool)
    ref = frames[0]
    aligned = frames
    for _ in range(rounds):
        aligned = align(frames, ref, core)
        ref = aligned.mean(axis=0)
        rmsf = np.sqrt(((aligned - ref) ** 2).sum(-1).mean(0))
        new = rmsf <= np.quantile(rmsf, keep)
        if (new == core).all():
            break
        core = new
    return align(frames, ref, core)


def viterbi_two_state(x: np.ndarray, mu: np.ndarray, sd: float, stay: float) -> np.ndarray:
    log_stay, log_move = np.log(stay), np.log(1.0 - stay)
    emit = -0.5 * ((x[:, None] - mu[None, :]) / sd) ** 2
    score = emit[0].copy()
    back = np.zeros((len(x), 2), dtype=np.int8)
    for t in range(1, len(x)):
        from_same = score + log_stay
        from_other = score[::-1] + log_move
        pick_other = from_other > from_same
        back[t] = np.where(pick_other, 1 - np.arange(2), np.arange(2))
        score = np.where(pick_other, from_other, from_same) + emit[t]
    path = np.empty(len(x), dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(len(x) - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def slowest_coordinate(aligned: np.ndarray, lag: int, dims: int = 12) -> np.ndarray:
    """First time-lagged independent component of the aligned coordinates."""
    x = (aligned - aligned.mean(axis=0)).reshape(len(aligned), -1)
    u, sv, _vt = np.linalg.svd(x, full_matrices=False)
    z = u[:, :dims] * sv[:dims]  # principal components, uncorrelated
    z = z / z.std(axis=0)  # whitened: C(0) = identity
    c_lag = z[:-lag].T @ z[lag:] / (len(z) - lag)
    c_lag = 0.5 * (c_lag + c_lag.T)
    w, v = np.linalg.eigh(c_lag)
    return z @ v[:, int(np.argmax(w))]


def two_means_1d(x: np.ndarray, rounds: int = 50) -> np.ndarray:
    lo, hi = np.percentile(x, [5, 95])
    mu = np.array([lo, hi])
    for _ in range(rounds):
        lab = np.abs(x[:, None] - mu[None, :]).argmin(1)
        mu = np.array([x[lab == k].mean() for k in range(2)])
    return mu


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    ind, outd = Path(a.indir), Path(a.outdir)
    outd.mkdir(parents=True, exist_ok=True)

    meta = json.loads((ind / "meta.json").read_text())
    dt = float(meta["frame_interval_ns"])
    resnums = [int(r) for r in meta["residue_numbers"]]
    frames = np.load(ind / "trajectory.npz")["xyz"].astype(float)

    aligned = core_alignment(frames)
    mean = aligned.mean(axis=0)
    rmsf = np.sqrt(((aligned - mean) ** 2).sum(-1).mean(0))

    # The switch is the slowest process in the trajectory, not the largest motion:
    # a floppy tail can have more variance than the switch but relaxes in a few
    # frames. Time-lagged independent component analysis (tICA) finds the
    # coordinate with the longest autocorrelation time.
    pc1 = slowest_coordinate(aligned, lag=max(int(round(1.0 / dt)), 1))
    mu = two_means_1d(pc1)
    sd = max(float(np.std(pc1[np.abs(pc1 - mu[0]) < np.abs(pc1 - mu[1])])), 1e-6)
    path = viterbi_two_state(pc1, mu, sd, stay=0.98)
    # label 1 is the less populated state
    if path.mean() > 0.5:
        path = 1 - path
    state = path.astype(int)

    # the switching segment: one contiguous run of residues whose state means
    # differ by >= 1.5 Angstrom, grown from the largest displacement. (Noise in the
    # flexible tails can exceed the threshold by chance; a run cannot.)
    disp = np.linalg.norm(aligned[state == 1].mean(0) - aligned[state == 0].mean(0), axis=1)
    top = int(disp.argmax())
    lo = hi = top
    while lo > 0 and disp[lo - 1] >= 1.5:
        lo -= 1
    while hi < len(disp) - 1 and disp[hi + 1] >= 1.5:
        hi += 1
    seg = [resnums[lo], resnums[hi]]

    # Final fluctuations. Rigid-body motion is removed relative to the structure of
    # the frame's own state, using every residue: that is the frame in which the
    # fluctuations of the underlying dynamics have no net translation or rotation.
    # Each state structure is kept in one common frame by superposing it on the
    # other on the residues outside the switch.
    outside = np.ones(frames.shape[1], dtype=bool)
    outside[max(lo - 1, 0): hi + 2] = False
    everything = np.ones(frames.shape[1], dtype=bool)
    work = aligned
    for _ in range(8):
        refs = [work[state == k].mean(axis=0) for k in (0, 1)]
        r, t = kabsch(refs[1][outside], refs[0][outside])
        refs[1] = refs[1] @ r.T + t
        work = np.empty_like(frames)
        for i, f in enumerate(frames):
            rot, shift = kabsch(f[everything], refs[state[i]][everything])
            work[i] = f @ rot.T + shift
    rmsf = np.sqrt(((work - work.mean(axis=0)) ** 2).sum(-1).mean(0))

    up = int(np.sum((state[:-1] == 0) & (state[1:] == 1)))
    down = int(np.sum((state[:-1] == 1) & (state[1:] == 0)))
    t_major = float(np.sum(state[:-1] == 0)) * dt
    t_minor = float(np.sum(state[:-1] == 1)) * dt

    (outd / "analysis.json").write_text(json.dumps({
        "rmsf_A": [float(v) for v in rmsf],
        "switching_residues": seg,
        "population_minor": float(state.mean()),
        "rate_major_to_minor_per_ns": up / t_major,
        "rate_minor_to_major_per_ns": down / t_minor,
    }, indent=2) + "\n")
    (outd / "states.csv").write_text(
        "frame,state\n" + "\n".join(f"{i},{int(s)}" for i, s in enumerate(state)) + "\n")


if __name__ == "__main__":
    main()
