# Requirements

Stakeholders: (1) reviewers at AI labs and data vendors who accept or reject
tasks; (2) agent developers who need a verifiable reward; (3) the author, who
must be able to defend every number.

Problem statement. Agentic evaluation of computational protein engineering and
drug discovery needs tasks that are hard for the right reasons, gradable by
code, and runnable offline. Existing public suites mostly ask questions or run
long pipelines. A task is only useful as a training or evaluation signal if its
grader has been validated: it must pass correct work done by any valid method,
reject realistic mistakes for the right reason, and resist shortcuts.

## Functional requirements

| ID | Requirement | Verified by |
|---|---|---|
| FR-1 | The suite holds six tasks covering structure assessment, design, dynamics, binding kinetics, drug-discovery modelling and inhibitor pharmacology | `tests/unit/test_task_layout.py` |
| FR-2 | Every task is generated from a 128-bit hidden seed; the same seed always yields the same data, truth and verdict | `tests/tasks/test_determinism_and_gates.py` |
| FR-3 | Every task ships an expert reference solution that passes on all hidden seeds | `tests/tasks/test_references.py`, validation campaign |
| FR-4 | Every task has at least four realistic wrong solutions, each failing with a named check code | `tests/tasks/test_negative_controls.py`, validation campaign |
| FR-5 | Grading has three layers: format gates, accuracy against planted truth, and method-validity fingerprints | `docs/DESIGN.md`, per-task check lists |
| FR-6 | Reward is binary (all hidden cases pass all checks); diagnostics report which check failed | `core/scoring.py` |
| FR-7 | Tasks with a plausible naive method (t02, t05) ship it as a baseline that must fail | `validation/baselines/`, validation campaign |
| FR-8 | A command-line interface lists tasks, makes inputs, grades submissions | `tooling/cli.py` |

## Non-functional requirements

| ID | Requirement | Verified by |
|---|---|---|
| NFR-1 | Offline: no task, reference or grader touches the network | network-off CI job, `docker run --network none` |
| NFR-2 | One CPU core, 4 GB memory, at most 5 minutes per case for a reference solution | per-task `timeout_s`, campaign timings |
| NFR-3 | Deterministic: single-threaded maths, key-sorted rounded JSON, byte-identical repeated verdicts | determinism tests, CI double grading |
| NFR-4 | Adversarial submissions cannot read seeds, the grader or truth | sandbox user, separate verifier container, red team |
| NFR-5 | Every scientific input is either a public-domain PDB deposit with recorded provenance or generated from stated equations | `docs/DATA.md` |
| NFR-6 | Dependencies are pinned; images are built from hash-pinned requirements | `uv.lock`, `docker/` |
| NFR-7 | Static quality: lint, strict typing for the framework, tests | `make check` |

## Acceptance criteria for the grader of each task

1. The reference solution passes on the campaign seeds; the Wilson interval of its pass rate is reported.
2. Every planted wrong solution fails, with the expected code first among the failures.
3. Each accuracy threshold sits inside the measured gap between valid methods and wrong methods (see `validation/campaign.py` and `docs/VALIDATION_REPORT.md`).
4. Every check code is exercised by at least one negative control that fails it.
5. A one-line shortcut (return nothing, return everything, copy the input) fails.
