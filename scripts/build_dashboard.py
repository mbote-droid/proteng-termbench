"""Build docs/dashboard.md from validation/results/*.json.

Every figure on the page is read from the recorded campaign files, nothing is
typed in by hand. Run it before `mkdocs build`:

    python scripts/build_dashboard.py
"""

from __future__ import annotations

import html
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "validation" / "results"
OUT = ROOT / "docs" / "dashboard.md"

STAGE = {
    "t01-model-triage": "folding",
    "t02-stability-design": "design",
    "t03-md-switch-kinetics": "dynamics",
    "t04-spr-kinetics": "kinetics",
    "t05-qsar-scaffold-hop": "drug discovery",
    "t06-inhibitor-mechanism": "pharmacology",
}


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def interval_svg(lo: float, hi: float, p: float) -> str:
    w = 100
    x0, x1, xp = lo * w, hi * w, p * w
    return (
        f'<svg width="{w + 4}" height="14" role="img" aria-label="{lo:.2f} to {hi:.2f}">'
        f'<line x1="2" y1="7" x2="{w + 2}" y2="7" stroke="#ccc"/>'
        f'<line x1="{x0 + 2:.1f}" y1="7" x2="{x1 + 2:.1f}" y2="7" '
        'stroke="#3b6ea5" stroke-width="4"/>'
        f'<circle cx="{xp + 2:.1f}" cy="7" r="3.5" fill="#1d3f6e"/></svg>'
    )


def load() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))]


def main() -> None:
    rows, total_runs, kinds = [], 0, Counter()
    for d in load():
        for name, m in d["methods"].items():
            runs = m["runs"]
            k = sum(1 for r in runs if not r["failed"])
            n = len(runs)
            codes = Counter(c for r in runs for c in set(r["failed"]))
            lo, hi = wilson(k, n)
            is_ref = name == "reference"
            kinds["reference" if is_ref else "wrong method"] += 1
            total_runs += n
            rows.append(
                (d["task"], name, is_ref, k, n, lo, hi, codes, max(r["seconds"] for r in runs))
            )
    tasks = len({r[0] for r in rows})
    refs = [r for r in rows if r[2]]
    wrongs = [r for r in rows if not r[2]]
    ref_pass = sum(r[3] for r in refs)
    ref_n = sum(r[4] for r in refs)
    wr_pass = sum(r[3] for r in wrongs)
    wr_n = sum(r[4] for r in wrongs)

    def card(big: str, small: str) -> str:
        return f'<div class="card"><div class="big">{big}</div><div>{small}</div></div>'

    cards = "".join(
        [
            card(str(tasks), "tasks"),
            card(f"{ref_pass}/{ref_n}", "reference runs passed"),
            card(f"{wr_pass}/{wr_n}", "wrong-method runs passed"),
            card(str(total_runs), "graded runs, fresh seeds"),
        ]
    )
    body = []
    for task, name, is_ref, k, n, lo, hi, codes, secs in rows:
        fails = ", ".join(f"{c} {v}" for c, v in codes.most_common()) or "none"
        body.append(
            "<tr>"
            f"<td><code>{html.escape(task)}</code><br><small>{STAGE.get(task, '')}</small></td>"
            f"<td>{'reference' if is_ref else html.escape(name)}</td>"
            f"<td>{k}/{n}</td><td>{interval_svg(lo, hi, k / n)}<br>"
            f"<small>{lo:.3f} to {hi:.3f}</small></td>"
            f"<td>{html.escape(fails)}</td><td>{secs:.1f}</td></tr>"
        )
    prov = load()[0]["provenance"]["packages"]
    pk = ", ".join(f"{k} {v}" for k, v in sorted(prov.items()))
    page = f"""# Validation dashboard

Built from `validation/results/*.json` by `scripts/build_dashboard.py`; every
number below comes from those recorded campaign files. A good grader passes the
reference solution and fails realistic wrong methods, and says which check
caught them.

<style>
.cards {{display:flex;gap:1rem;flex-wrap:wrap;margin:1rem 0}}
.card {{border:1px solid var(--md-default-fg-color--lightest);border-radius:6px;
padding:.6rem 1rem;min-width:9rem}}
.card .big {{font-size:1.8rem;font-weight:700}}
.dash td, .dash th {{vertical-align:top}}
</style>

<div class="cards">{cards}</div>

<div class="dash">
<table>
<thead><tr><th>task</th><th>method</th><th>passed</th>
<th>pass rate, 95% Wilson interval</th><th>checks that failed (count)</th>
<th>slowest run (s)</th></tr></thead>
<tbody>
{"".join(body)}
</tbody></table>
</div>

The slow tasks (t04, t06) ran on fewer seeds, so their intervals are wide:
read them as "no false rejects seen", not as a tight bound. Wrong methods exist
for t02 and t05 only; the other tasks rely on the negative controls in
`tests/tasks/test_negative_controls.py`. Package versions for these runs:
{pk}.

## Agent results

No agent runs are published yet. The harness in `agent_eval/` can run Ollama,
Gemini or Anthropic models on any task; this section will list real runs with
their transcripts once they exist.
"""
    OUT.write_text(page)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(rows)} rows")


if __name__ == "__main__":
    main()
