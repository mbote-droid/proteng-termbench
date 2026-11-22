# Validation report

Produced by `uv run python validation/campaign.py --out validation/results --n 30`.
Every run uses a fresh 128-bit seed, recorded with its metrics, timing and the
package versions in `validation/results/<task>.json`. Each submission runs as a
subprocess, single-threaded, exactly as the grader runs it.

## Results

| task | method | passed | 95% Wilson interval for the pass rate | checks that failed |
|---|---|---|---|---|
| t01-model-triage | reference | 30/30 | 0.886 to 1.000 | none |
| t02-stability-design | reference | 30/30 | 0.886 to 1.000 | none |
| t02-stability-design | additive top-five baseline | 0/30 | 0.000 to 0.114 | STABILISED 30, CALIBRATION 14 |
| t03-md-switch-kinetics | reference | 30/30 | 0.886 to 1.000 | none |
| t04-spr-kinetics | reference | 12/12 | 0.758 to 1.000 | none |
| t05-qsar-scaffold-hop | reference | 30/30 | 0.886 to 1.000 | none |
| t05-qsar-scaffold-hop | fingerprint random forest, random-split claim | 0/30 | 0.000 to 0.114 | DEDUP 30, HONESTY 29, RANK 3, SELECTION 2 |
| t06-inhibitor-mechanism | reference | 7/7 | 0.646 to 1.000 | none |

The slow tasks (t04, t06) were run on fewer seeds because the reference fits
take 15 to 75 seconds; their intervals are correspondingly wide and should be
read as "no false rejects seen", not as a tight bound.

Reference run time: at most 2.5 s (t01), 1.6 s (t02), 3.3 s (t03), 16.5 s
(t04), 1.9 s (t05) and 73.9 s (t06) per case, against a limit of 5 minutes.

## Where the thresholds sit

* **t02.** Over 80 seeds across three runs the reference reaches 88 to 100
  percent of the best achievable stability (93 to 100 in the final run); the
  additive baseline reaches 28 to 98 percent, median about 67 (39 to 88 in the
  final run, where none passed). The cutoff is 90 percent. An additive pick
  that lands near the optimum by luck is a legitimate pass (3 in 30 at an
  earlier 85 percent cutoff, none at 90); the calibration check is the second
  guard.
* **t05.** Claimed minus realised Spearman correlation: the reference ranges
  from -0.35 to +0.12; the random-split baseline from +0.20 to +0.76. The
  cutoff for over-claiming is +0.22, so one baseline run was inside it on
  honesty alone; the duplicate count (DEDUP) failed in all 30 baseline runs,
  so the baseline never passed.
* **t06 and t04.** Tolerances follow the Fisher standard error the design
  allows (t04) and the identifiability of the quantity (t06, `KI_TOL_TIGHT`
  for tight binding).

## Flaws the campaign found, and what changed

1. **t03 generator crash.** On a seed where a state is never visited, the rate
   estimate divided by zero. It now returns a rejected case and the generator
   draws again (`tests/unit/test_t03_rates.py`).
2. **t02 threshold.** A first threshold of 82 percent let the additive
   baseline pass too often; a second of 85 percent still let 3 in 30 through.
   Set to 90 percent after the reference was improved (shrinkage of the
   measured synergy removed: genuine synergy is several times the noise).
3. **t01 false rejects.** The reference failed 2 of 30 fresh seeds on the
   moved-segment and hinge checks. Cause: for a small NMP rotation the first
   residues beside the hinge move less than the coordinate noise, so no method
   can place the boundary there. The boundary tolerance now excludes edge
   residues whose planted displacement is below 0.6 Angstrom. After the change
   the reference had 0 failures in 80 further rotation cases, and the control
   with a hinge nine residues off still fails
   (`tests/tasks/test_references.py::test_t01_regression_...`).

## What has not been measured

* Only one reference method per task exists. The thresholds were therefore
  set against the reference and against the wrong methods, not against a
  second valid method; a different valid approach could fall near a threshold.
* The campaign ran on the development interpreter (Python 3.13). The sandbox
  configuration (grader as root, submission as an unprivileged user, an
  interpreter built only from the hashed agent requirements) was checked by
  grading every reference on its dev seeds; the container images themselves
  have not been built or run here.
* No external agent has yet been evaluated; `agent_eval/harness.py` is
  tested up to the grading step only.
* The grader's own correctness on adversarial submissions (reading seeds,
  tampering with files) was not red-teamed beyond the isolation tests in
  `tests/unit/test_cli_isolation.py`.
