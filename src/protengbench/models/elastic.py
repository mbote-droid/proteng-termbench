"""Anisotropic network model (ANM) of a protein and its Langevin dynamics.

A protein is a set of C-alpha beads joined by identical springs (constant gamma)
between every pair closer than a cutoff. The potential energy is quadratic,

    U = 1/2 x^T H x,    H_ij block = -gamma (r r^T) / |r|^2  for each contact,

(x are displacements from the reference structure; the diagonal blocks make the
rows sum to zero). H has six zero eigenvalues (rigid-body motion) and 3N - 6
positive ones lambda_k with eigenvectors v_k. At temperature T the equilibrium
fluctuation covariance is

    <x x^T> = kT sum_k v_k v_k^T / lambda_k        (zero modes excluded),

so mode k has variance kT / lambda_k. Overdamped Langevin dynamics decouples
into independent Ornstein-Uhlenbeck processes, one per mode, with relaxation
rate r_k = lambda_k / zeta (zeta is the friction), and the exact update over a
step dt is

    a_k <- a_k exp(-r_k dt) + sqrt( kT/lambda_k (1 - exp(-2 r_k dt)) ) xi,   xi ~ N(0, 1).

Because the update is exact for any dt, frames far apart in time are sampled
correctly without integration error.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def hessian(coords: FloatArray, cutoff: float = 13.0, gamma: float = 1.0) -> FloatArray:
    n = coords.shape[0]
    h = np.zeros((3 * n, 3 * n))
    for i in range(n):
        d = coords[i + 1 :] - coords[i]
        r2 = np.einsum("ij,ij->i", d, d)
        for k in np.nonzero(r2 < cutoff**2)[0]:
            j = i + 1 + int(k)
            block = -gamma * np.outer(d[k], d[k]) / r2[k]
            h[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] += block
            h[3 * j : 3 * j + 3, 3 * i : 3 * i + 3] += block
            h[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] -= block
            h[3 * j : 3 * j + 3, 3 * j : 3 * j + 3] -= block
    return h


class NetworkDynamics:
    """Sampler of correlated frames from the network's Ornstein-Uhlenbeck dynamics."""

    def __init__(
        self, coords: FloatArray, target_rmsf: float, slowest_ns: float, cutoff: float = 13.0
    ) -> None:
        self.n = coords.shape[0]
        w, v = np.linalg.eigh(hessian(coords, cutoff))
        keep = np.arange(6, 3 * self.n)  # drop the six rigid-body zero modes
        w, v = w[keep], v[:, keep]
        if w[0] < 1e-6:
            raise ValueError("network is not connected: extra zero modes")
        var = 1.0 / w  # kT = 1 for now
        msf = (v**2 * var).sum(axis=1).reshape(self.n, 3).sum(axis=1)  # per residue
        scale = target_rmsf**2 / msf.mean()
        self.modes = v
        self.sd = np.sqrt(var * scale)  # equilibrium std of each mode amplitude (Angstrom)
        self.rate = (w / w[0]) / slowest_ns  # relaxation rate r_k (1/ns)

    def sample(self, n_frames: int, dt_ns: float, rng: np.random.Generator) -> FloatArray:
        decay = np.exp(-self.rate * dt_ns)
        kick = self.sd * np.sqrt(1.0 - decay**2)
        a = rng.normal(0.0, self.sd)
        out = np.empty((n_frames, self.n, 3))
        for f in range(n_frames):
            out[f] = (self.modes @ a).reshape(self.n, 3)
            a = a * decay + kick * rng.normal(size=a.size)
        return out
