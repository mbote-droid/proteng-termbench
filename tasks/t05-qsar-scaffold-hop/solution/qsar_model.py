#!/usr/bin/env python3
"""Reference solution for t05: a substituent-level QSAR model validated by core.

Method, and why:

1. Standardise: keep the largest fragment (drops salts), canonicalise, and
   collapse repeat measurements of one compound to their mean. Without this
   the number of compounds is overstated and replicates leak between folds.
2. Describe each compound as chemistry that transfers between cores: find the
   fused ring system (the core), cut the two bonds to it, and describe each
   substituent by lipophilicity, size (and its square, for an optimum) and
   hydrogen-bond donors. The two positions are told apart by the core
   atom they sit on (the carbon between the two ring nitrogens is position 1).
   Whole-molecule fingerprints would tie the model to the training cores.
3. Fit a ridge regression on those descriptors.
4. Estimate performance for a new core by leaving one training core out at a
   time. A random split would put compounds of every core on both sides and
   promise far more than a new core delivers.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")
CHOOSER = rdMolStandardize.LargestFragmentChooser()


def standardise(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"cannot parse {smiles!r}")
    return Chem.MolFromSmiles(Chem.MolToSmiles(CHOOSER.choose(mol)))


def _ring_systems(mol: Chem.Mol) -> list[tuple[set[int], int]]:
    systems: list[tuple[set[int], int]] = []
    for ring in mol.GetRingInfo().AtomRings():
        atoms = set(ring)
        merged = [s for s in systems if s[0] & atoms]
        for s in merged:
            systems.remove(s)
        systems.append((atoms.union(*[s[0] for s in merged]), 1 + sum(s[1] for s in merged)))
    return systems


def _side(mol: Chem.Mol, start: int, core: set[int]) -> set[int]:
    seen, stack = {start}, [start]
    while stack:
        a = stack.pop()
        for n in mol.GetAtomWithIdx(a).GetNeighbors():
            j = n.GetIdx()
            if j not in core and j not in seen:
                seen.add(j)
                stack.append(j)
    return seen


def describe(mol: Chem.Mol) -> tuple[str, np.ndarray]:
    """(core key, [logp, mr, mr^2, hbd] for position 1 then position 2)."""
    core = max(_ring_systems(mol), key=lambda s: s[1])[0]
    mol_h = Chem.AddHs(mol)
    contribs = rdMolDescriptors._CalcCrippenContribs(mol_h)
    out: dict[int, list[float]] = {}
    for ai in core:
        for nb in mol.GetAtomWithIdx(ai).GetNeighbors():
            if nb.GetIdx() in core:
                continue
            side = _side(mol, nb.GetIdx(), core)
            idx = set(side)
            for a in side:  # include implicit hydrogens of the fragment
                idx.update(n.GetIdx() for n in mol_h.GetAtomWithIdx(a).GetNeighbors() if n.GetAtomicNum() == 1)
            idx.update(n.GetIdx() for n in mol_h.GetAtomWithIdx(ai).GetNeighbors()
                       if n.GetAtomicNum() == 1)  # none for substituted ring carbon
            logp = sum(contribs[i][0] for i in idx) + 0.123  # H cap (Crippen H on C)
            mr = sum(contribs[i][1] for i in idx) + 1.057
            hbd = sum(1 for a in side if mol.GetAtomWithIdx(a).GetAtomicNum() in (7, 8)
                      and mol.GetAtomWithIdx(a).GetTotalNumHs() > 0)
            n_ring_n = sum(1 for n in mol.GetAtomWithIdx(ai).GetNeighbors()
                           if n.GetAtomicNum() == 7 and n.GetIdx() in core)
            pos = 1 if n_ring_n == 2 else 2
            out[pos] = [logp, mr, mr * mr, float(hbd)]
    key = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(core))
    return key, np.array(out[1] + out[2])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    with (a.indir / "train.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    merged: dict[str, list[float]] = {}
    for r in rows:
        merged.setdefault(Chem.MolToSmiles(standardise(r["smiles"])), []).append(float(r["pIC50"]))
    smiles = list(merged)
    y = np.array([np.mean(merged[s]) for s in smiles])
    feats = [describe(Chem.MolFromSmiles(s)) for s in smiles]
    groups = np.array([f[0] for f in feats])
    x = np.vstack([f[1] for f in feats])

    def fit(xtr: np.ndarray, ytr: np.ndarray) -> tuple[StandardScaler, Ridge]:
        sc = StandardScaler().fit(xtr)
        return sc, Ridge(alpha=3.0).fit(sc.transform(xtr), ytr)

    # Validation that mimics the deployment: hold out a whole core AND a few
    # substituents per position (the test set contains substituents never seen
    # in training), and score on a half-and-half mix of compounds with and
    # without the held-out substituents. Repeated with different holdouts.
    sub1 = [tuple(np.round(r[:4], 3)) for r in x]
    sub2 = [tuple(np.round(r[4:], 3)) for r in x]
    u1, u2 = sorted(set(sub1)), sorted(set(sub2))
    rng = np.random.default_rng(0)
    scores = []
    for g in sorted(set(groups)):
        for _ in range(10):
            h1 = {u1[i] for i in rng.choice(len(u1), size=min(4, len(u1) // 3), replace=False)}
            h2 = {u2[i] for i in rng.choice(len(u2), size=min(4, len(u2) // 3), replace=False)}
            novel = np.array([a in h1 or b in h2 for a, b in zip(sub1, sub2, strict=True)])
            te_old = (groups == g) & ~novel
            te_new = (groups == g) & novel
            if te_new.sum() < 5 or te_old.sum() < 5:
                continue
            m = min(te_old.sum(), te_new.sum())
            te = np.concatenate([np.flatnonzero(te_old)[:m], np.flatnonzero(te_new)[:m]])
            tr = (groups != g) & ~novel
            sc, model = fit(x[tr], y[tr])
            scores.append(float(spearmanr(model.predict(sc.transform(x[te])), y[te]).statistic))
    claimed = float(np.mean(scores))

    sc, model = fit(x, y)
    with (a.indir / "test.csv").open() as fh:
        test = list(csv.DictReader(fh))
    tx = np.vstack([describe(standardise(r["smiles"]))[1] for r in test])
    pred = model.predict(sc.transform(tx))
    with (a.outdir / "predictions.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "pic50_pred"])
        for r, p in zip(test, pred, strict=True):
            w.writerow([r["id"], f"{p:.4f}"])
    top = [test[i]["id"] for i in np.argsort(-pred)[:20]]
    (a.outdir / "report.json").write_text(json.dumps({
        "n_unique_training_compounds": len(smiles),
        "expected_spearman_new_core": round(claimed, 3),
        "top20": top,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
