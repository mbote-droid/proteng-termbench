#!/usr/bin/env python3
"""Wrong method for t02: take the five most stabilising measured singles and add their effects.

Ignores that stabilising substitutions in contact partly cancel, so the design is worse than
the best available and the stated ddG overshoots what the variant really achieves.
"""

import argparse
import csv
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--indir", type=Path, required=True)
ap.add_argument("--outdir", type=Path, required=True)
a = ap.parse_args()
a.outdir.mkdir(parents=True, exist_ok=True)
cons = json.loads((a.indir / "constraints.json").read_text())
rows = [
    (int(r["position"]), r["wt"], r["mut"], float(r["ddG_kcal_per_mol"]))
    for r in csv.DictReader((a.indir / "scan_single.csv").open())
    if int(r["position"]) not in cons["forbidden_positions"]
]
rows.sort(key=lambda r: r[3])
chosen, used, total = [], set(), 0.0
for pos, wt, mut, v in rows:
    if pos in used:
        continue
    chosen.append(f"{wt}{pos}{mut}")
    used.add(pos)
    total += v
    if len(chosen) == cons["max_substitutions"]:
        break
(a.outdir / "design.json").write_text(
    json.dumps(
        {"substitutions": chosen, "predicted_ddG_kcal_per_mol": total, "sigma_kcal_per_mol": 0.5}
    )
)
