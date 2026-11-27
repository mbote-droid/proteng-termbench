"""validation/report.py rewrites only the results table and leaves the prose alone."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("report", ROOT / "validation" / "report.py")
assert spec and spec.loader
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def _results(tmp: Path) -> Path:
    d = tmp / "results"
    d.mkdir()
    runs = [{"failed": []}, {"failed": ["RANK"]}, {"failed": ["RANK", "DEDUP"]}]
    (d / "t99-demo.json").write_text(
        json.dumps(
            {
                "task": "t99-demo",
                "methods": {"reference": {"runs": runs[:1]}, "naive": {"runs": runs[1:]}},
            }
        )
    )
    return d


def test_only_the_table_changes(tmp_path: Path) -> None:
    doc = tmp_path / "r.md"
    doc.write_text(f"intro\n\n{report.START}\nold\n{report.END}\n\nhand-written tail\n")
    assert report.main(["report.py", str(_results(tmp_path)), str(doc)]) == 0
    out = doc.read_text()
    assert out.startswith("intro\n\n") and out.endswith("\n\nhand-written tail\n")
    assert "| t99-demo | reference | 1/1 |" in out
    assert "| t99-demo | naive | 0/2 | 0.000 to 0.658 | RANK 2, DEDUP 1 |" in out
    assert "old" not in out


def test_missing_markers_is_an_error(tmp_path: Path) -> None:
    doc = tmp_path / "r.md"
    doc.write_text("no markers here\n")
    assert report.main(["report.py", str(_results(tmp_path)), str(doc)]) == 1
    assert doc.read_text() == "no markers here\n"


def test_wrong_arguments_exit_2() -> None:
    with pytest.raises(SystemExit):
        raise SystemExit(report.main(["report.py"]))
