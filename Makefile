# Every workflow is one command. All targets run inside the locked uv environment.
SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c
.DEFAULT_GOAL := help

UV ?= uv
RUN := $(UV) run --frozen
VERSION := $(shell $(RUN) python -c "import protengbench; print(protengbench.__version__)")
REGISTRY ?= ghcr.io/mbote-droid

.PHONY: help setup lint typecheck test test-fast check build images clean validate

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

setup:  ## create the locked environment (uv.lock)
	$(UV) sync --frozen --group agent

lint:  ## ruff lint + format check, shellcheck
	$(RUN) ruff check src tests tasks validation agent_eval
	$(RUN) ruff format --check src tests
	$(RUN) shellcheck tasks/*/solution/solve.sh tasks/*/tests/test.sh scripts/*.sh

typecheck:  ## mypy --strict on the package
	$(RUN) mypy

test-fast:  ## unit tests and negative controls only
	$(RUN) pytest -m "not slow"

test:  ## full suite, including reference solutions on dev seeds
	$(RUN) pytest

check: lint typecheck test  ## everything CI runs

build:  ## wheel for the grader image
	$(UV) build --wheel --out-dir dist

images: build  ## build agent and grader base images
	docker build -f docker/agent-base.Dockerfile -t $(REGISTRY)/protengbench-agent-base:$(VERSION) .
	docker build -f docker/grader-base.Dockerfile -t $(REGISTRY)/protengbench-grader-base:$(VERSION) .

N ?= 20

validate:  ## validation campaign: reference false-reject rates and wrong-method controls -> validation/results
	$(RUN) python validation/campaign.py --out validation/results --n $(N)
	$(RUN) python validation/report.py validation/results docs/VALIDATION_REPORT.md

clean:
	rm -rf dist results .pytest_cache .mypy_cache .ruff_cache .hypothesis
