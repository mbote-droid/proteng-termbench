#!/usr/bin/env python3
"""Wrong method for t05: Morgan-fingerprint random forest, validated with a random split.

Typical of published QSAR: no salt stripping or duplicate merging (the count is of distinct
strings), whole-molecule bit patterns that cannot describe substituents never seen in training,
and a random-split score reported as the expectation for a new series.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict

RDLogger.DisableLog("rdApp.*")
ap = argparse.ArgumentParser()
ap.add_argument("--indir", type=Path, required=True)
ap.add_argument("--outdir", type=Path, required=True)
a = ap.parse_args()
a.outdir.mkdir(parents=True, exist_ok=True)
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fp(smiles: str) -> np.ndarray:
    arr = np.zeros(2048)
    DataStructs.ConvertToNumpyArray(gen.GetFingerprint(Chem.MolFromSmiles(smiles)), arr)
    return arr


train = list(csv.DictReader((a.indir / "train.csv").open()))
test = list(csv.DictReader((a.indir / "test.csv").open()))
x = np.array([fp(r["smiles"]) for r in train])
y = np.array([float(r["pIC50"]) for r in train])
cv = cross_val_predict(
    RandomForestRegressor(100, random_state=0, n_jobs=1),
    x,
    y,
    cv=KFold(5, shuffle=True, random_state=0),
)
claimed = float(spearmanr(cv, y).statistic)
model = RandomForestRegressor(200, random_state=0, n_jobs=1).fit(x, y)
pred = model.predict(np.array([fp(r["smiles"]) for r in test]))
with (a.outdir / "predictions.csv").open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["id", "pic50_pred"])
    w.writerows([[r["id"], f"{p:.4f}"] for r, p in zip(test, pred, strict=True)])
(a.outdir / "report.json").write_text(
    json.dumps(
        {
            "n_unique_training_compounds": len({r["smiles"] for r in train}),
            "expected_spearman_new_core": claimed,
            "top20": [test[i]["id"] for i in np.argsort(-pred)[:20]],
        }
    )
)
