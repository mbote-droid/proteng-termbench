# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Fixed
- CI installs the `agent` dependency group, so the t01 and t05 reference
  solutions run in the check, reproducibility and validation jobs.
- The test helper prints a crashed reference solution's stderr instead of
  failing later with `FileExistsError`.

### Added
- Contributing guide, security policy, code of conduct, issue and pull request
  templates, dev container.

## 0.2.0

- Six tasks (t01 model triage to t06 inhibitor mechanism) with generators,
  expert references and layered graders.
- Validation campaign, negative controls and naive baselines
  (`docs/VALIDATION_REPORT.md`).
- Agent evaluation harness for Ollama, Gemini and Anthropic models.
