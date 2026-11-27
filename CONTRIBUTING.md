# Contributing

Bug reports, grader false rejects and new task ideas are all welcome.

## Set up

```bash
uv sync --frozen --group agent
make check      # lint, strict types, tests, reference solutions on dev seeds
```

The `agent` group matters: the reference solutions for t01 and t05 import
biopython and scikit-learn, and `make test` runs them.

## Before opening a pull request

- `make check` passes.
- A grader change comes with a negative control in
  `tests/tasks/test_negative_controls.py` (a plausible wrong method that must
  fail the named check) and, if a threshold moved, a fresh `make validate`
  run. Say which numbers changed and why.
- Generators stay deterministic for a given seed.
- Keep commits small, and say in the message why the change was needed.

## Reporting a task or grader problem

Open an issue with the task id, the seed (or the 128-bit hex string from the
diagnostics) and the check that fired. A reference solution that fails a seed,
or a wrong method that passes one, is the most useful report.

## Adding a task

Read `docs/DESIGN.md` and `docs/REQUIREMENTS.md` first. A new task needs a
generator with planted truth, an expert reference, a grader with the three
layers (format, accuracy, method fingerprints), negative controls, and a
validation run on fresh seeds.
