PY ?= python
export PYTHONNOUSERSITE=1

.PHONY: test check lint fmt install

install:
	$(PY) -m pip install -e '.[dev]'

test:
	$(PY) -m pytest -m "not gpu and not openrouter and not slow"

lint:
	$(PY) -m ruff check swarm tests

fmt:
	$(PY) -m ruff format swarm tests

# check = lint + fast tests (WP9 adds `swarm doctor` here)
check: lint test
