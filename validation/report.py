#!/usr/bin/env python3
"""Refresh the results table in the validation report from the recorded campaign files.

Only the text between the two marker comments in the report is replaced; the
rest of the report (thresholds, flaws found, discussion) is hand-written and
left as it is.

    uv run python validation/report.py validation/results docs/VALIDATION_REPORT.md
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from protengbench.checks.stats import wilson_interval

START = "<!-- results:start -->"
END = "<!-- results:end -->"
HEADER = (
    "| task | method | passed | 95% Wilson interval for the pass rate | checks that failed |\n"
    "|---|---|---|---|---|"
)
NAMES = {
    "additive_top5": "additive top-five baseline",
    "fingerprint_rf_random_cv": "fingerprint random forest, random-split claim",
}


def table(results_dir: Path) -> str:
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text())
        methods = sorted(data["methods"], key=lambda m: (m != "reference", m))
        for name in methods:
            runs = data["methods"][name]["runs"]
            passed = sum(1 for r in runs if not r["failed"])
            lo, hi = wilson_interval(passed, len(runs))
            codes = Counter(c for r in runs for c in set(r["failed"]))
            fails = ", ".join(f"{c} {n}" for c, n in codes.most_common()) or "none"
            rows.append(
                f"| {data['task']} | {NAMES.get(name, name)} | {passed}/{len(runs)} "
                f"| {lo:.3f} to {hi:.3f} | {fails} |"
            )
    return HEADER + "\n" + "\n".join(rows)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    results_dir, report = Path(argv[1]), Path(argv[2])
    text = report.read_text()
    if START not in text or END not in text:
        print(f"{report} has no {START} / {END} markers", file=sys.stderr)
        return 1
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    report.write_text(f"{head}{START}\n{table(results_dir)}\n{END}{tail}")
    print(f"updated the results table in {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
