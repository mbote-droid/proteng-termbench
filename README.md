# ProtEng-TermBench

[![ci](https://github.com/mbote-droid/proteng-termbench/actions/workflows/ci.yml/badge.svg)](https://github.com/mbote-droid/proteng-termbench/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
[![open in codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/mbote-droid/proteng-termbench)

A benchmark for AI agents doing protein engineering and early drug discovery
in a terminal, with graders that check the method as well as the answer.

Six terminal tasks that follow one protein-engineering and early drug-discovery
programme from a predicted structure to a characterised inhibitor. Each task
is generated from a hidden seed, solved by an expert reference, and graded by
code that checks the answer against planted truth and checks that the method
was sound. The graders have been validated: the references pass, realistic
wrong methods fail, and each failure names the check that caught it.

| task | stage | what the agent must decide | the trap |
|---|---|---|---|
| `t01-model-triage` | folding | is this predicted model wrong, or are its domains right and only their arrangement wrong? | numbering offset and a missing loop; a global RMSD hides a domain rotation |
| `t02-stability-design` | design | the combination of up to five substitutions that stabilises a protein most | stabilising neighbours are not additive; synergy hides in the double-mutant data |
| `t03-md-switch-kinetics` | dynamics | which segment switches, how populated each state is, how fast it switches | frames are rotated; superposing on all atoms smears the mobile region |
| `t04-spr-kinetics` | kinetics | kon, koff, KD with honest error bars, and which binding model | mass transport, drift, bulk shift, two kinds of ligand |
| `t05-qsar-scaffold-hop` | drug discovery | rank compounds from chemical series never seen in training, and say how good the model really is | duplicates and salts; a random split overstates performance on a new series |
| `t06-inhibitor-mechanism` | pharmacology | mechanism of inhibition and Ki | substrate depletion; tight binding; IC50 to Ki by the wrong equation |

Reward is binary: a submission passes a task only if it passes every check on
every hidden case. Diagnostics list which checks failed.

## Why the grading can be trusted

* **Planted truth.** Every datum comes from a model whose parameters the
  generator knows, so the correct answer is known exactly, not estimated by
  another program.
* **Three layers.** A format gate, then accuracy against the truth, then
  method-validity fingerprints that only sound reasoning produces (a hinge
  located within two residues, a claimed performance that matches the realised
  performance).
* **Tolerances from data.** Thresholds sit in the measured gap between the
  reference and wrong methods. `validation/campaign.py` records every metric
  on fresh 128-bit seeds; results are in `validation/results/` and summarised
  in `docs/VALIDATION_REPORT.md`.
* **Negative controls.** `tests/tasks/test_negative_controls.py` edits the
  reference output into what a plausible wrong method would produce and
  asserts that the named check fails. Every scored check has at least one.
* **Naive baselines.** t02 (additive top five) and t05 (fingerprint random
  forest with a random-split claim) ship a plausible wrong method that must
  fail.
* **Isolation.** The grader runs as root in a separate environment; the
  submission runs as an unprivileged user with an interpreter that holds only
  the agent packages. It cannot read seeds, graders or truth.
* **Determinism.** Single-threaded maths, fixed seeds, rounded JSON; the same
  seed always gives the same verdict.

## Quick start

```bash
uv sync --frozen --group agent
uv run proteng list
uv run proteng seeds new 3 --out /tmp/seeds.txt
uv run proteng grade t05-qsar-scaffold-hop \
    --submission tasks/t05-qsar-scaffold-hop/solution/qsar_model.py \
    --seeds tasks/t05-qsar-scaffold-hop/tests/dev_seeds.txt \
    --runner local --trusted
make check          # lint, strict types, tests, reference solutions on dev seeds
```

`--trusted` is only for your own reference code when running as root; for an
untrusted submission use `--sandbox-user` (see `docs/DESIGN.md`).

## Task layout

```
tasks/<id>/
  instruction.md      what the agent reads
  task.toml           limits (1 CPU, 4 GB, no network)
  environment/        agent image recipe and one public example case
  solution/           expert reference (hidden from the agent)
  tests/              verifier entry point and public dev seeds
src/protengbench/     generators, graders, shared models, CLI
validation/           campaign, baselines, recorded results
agent_eval/           run a real model on the tasks and grade it
docs/                 requirements, design, data, validation report
```

Tasks are written for [Harbor](https://github.com/laude-institute/harbor)
(`harbor run -p tasks/<id>`): the agent and verifier run in separate images
with the network off. `docker/` holds the hash-pinned base images.

## Evaluating an agent

The harness drives a model through one `bash` tool in a sandboxed workspace and
grades what it leaves behind with the real grader. It works with a local
Ollama model (no key), Gemini's free tier, or Anthropic:

```bash
# local: ollama serve && ollama pull qwen2.5-coder:7b
uv run python agent_eval/harness.py --provider ollama --model qwen2.5-coder:7b \
    --task t05-qsar-scaffold-hop --attempts 3

# Gemini free tier; the key is read from the environment, never written to disk
export GEMINI_API_KEY=...
uv run python agent_eval/harness.py --provider gemini --model gemini-2.5-flash \
    --task t05-qsar-scaffold-hop --attempts 3

uv run python agent_eval/harness.py --summarise
```

Small local models may not call tools reliably, and free API tiers are rate
limited (the harness retries with backoff). The harness is for fast
development; it does not disable the network. Use Harbor for reported scores.

## Scope and limits

The data are synthetic apart from two public PDB structures (`docs/DATA.md`).
Passing a task shows that a method avoids the named trap and is
quantitatively right on that generator; it does not show that the method would
rank real compounds or predict real stability. Six tasks cover one pipeline,
not the field. The container images and Harbor runs are defined (`docker/`,
`.github/workflows/images.yml`) but were not built where this was developed (no
container runtime); the sandbox configuration was exercised directly instead,
and CI builds and runs the images.

## Citing

See `CITATION.cff`. MIT licence.
