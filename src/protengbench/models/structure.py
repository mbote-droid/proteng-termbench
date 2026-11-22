"""Structure file helpers: a minimal PDB reader and the Kabsch superposition.

Only ATOM records are read. Coordinates are in Angstrom. The reader returns the
atoms in file order with residue numbers, names and chain, which is all the
tasks need and avoids a heavy dependency in the grader.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Atom:
    serial: int
    name: str
    resname: str
    chain: str
    resseq: int
    xyz: tuple[float, float, float]


def read_pdb(path: Path) -> list[Atom]:
    atoms = []
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        atoms.append(
            Atom(
                int(line[6:11]),
                line[12:16].strip(),
                line[17:20].strip(),
                line[21],
                int(line[22:26]),
                (float(line[30:38]), float(line[38:46]), float(line[46:54])),
            )
        )
    return atoms


def bundled(name: str) -> Path:
    """Path of a structure shipped inside the package (public-domain PDB entries)."""
    return Path(str(resources.files("protengbench") / "data" / name))


def ca_trace(atoms: list[Atom], chain: str = "A") -> tuple[FloatArray, list[int], list[str]]:
    sel = [a for a in atoms if a.name == "CA" and a.chain == chain]
    xyz = np.array([a.xyz for a in sel], dtype=float)
    return xyz, [a.resseq for a in sel], [a.resname for a in sel]


def format_ca_pdb(
    xyz: FloatArray,
    resseq: list[int],
    resname: list[str],
    chain: str = "A",
    bfactor: FloatArray | None = None,
) -> str:
    lines = []
    for i, (p, n, r) in enumerate(zip(xyz, resseq, resname, strict=True), start=1):
        b = 50.0 if bfactor is None else float(bfactor[i - 1])
        lines.append(
            f"ATOM  {i:5d}  CA  {r:>3s} {chain}{n:4d}    "
            f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}{1.0:6.2f}{b:6.2f}           C"
        )
    lines.append("END")
    return "\n".join(lines) + "\n"


def kabsch(
    mobile: FloatArray, target: FloatArray, weights: FloatArray | None = None
) -> tuple[FloatArray, FloatArray]:
    """Rotation R and translation t minimising sum w |R x + t - y|^2.

    With centroids removed, the optimal rotation is R = V diag(1, 1, det(V U^T)) U^T
    where H = X^T W Y = U S V^T. Returns (R, t) so that aligned = mobile @ R.T + t."""
    w = np.ones(mobile.shape[0]) if weights is None else weights
    w = w / w.sum()
    cm = (mobile * w[:, None]).sum(axis=0)
    ct = (target * w[:, None]).sum(axis=0)
    h = ((mobile - cm) * w[:, None]).T @ (target - ct)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, ct - r @ cm


def rmsd(a: FloatArray, b: FloatArray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


def residues(atoms: list[Atom]) -> list[list[Atom]]:
    """Group atoms (in file order) into residues by chain and residue number."""
    groups: list[list[Atom]] = []
    last: tuple[str, int] | None = None
    for a in atoms:
        key = (a.chain, a.resseq)
        if key != last:
            groups.append([])
            last = key
        groups[-1].append(a)
    return groups


def format_pdb(rows: list[tuple[str, str, str, int, FloatArray, float]]) -> str:
    """Write ATOM records. Each row is (atom name, residue name, chain, residue number,
    xyz, B-factor). Atoms are numbered in the order given."""
    lines = []
    for i, (name, resname, chain, resseq, xyz, bfac) in enumerate(rows, start=1):
        field = f" {name:<3s}" if len(name) < 4 else name
        element = name.lstrip("0123456789")[0]
        lines.append(
            f"ATOM  {i:5d} {field} {resname:>3s} {chain}{resseq:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{1.0:6.2f}{bfac:6.2f}          {element:>2s}"
        )
    lines.append("END")
    return "\n".join(lines) + "\n"
