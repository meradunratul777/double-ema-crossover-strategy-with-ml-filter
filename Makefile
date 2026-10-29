PY ?= .venv/bin/python

.PHONY: setup test audit features gate clean
setup:
	python3 -m venv .venv && ./.venv/bin/pip install -q -r requirements.txt && ./.venv/bin/pip install -q -e .

test:
	$(PY) -m pytest tests/ -q

audit:
	$(PY) -m dema.cli audit --smoke-test

gate: test audit            ## CI-equivalent: tests + leakage gate

features:
	$(PY) -m dema.cli features --smoke-test

clean:
	rm -rf artifacts/features artifacts/checkpoints .pytest_cache
