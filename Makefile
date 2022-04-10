# ===================================================================
# Development. Start Work
# ===================================================================

#* Poetry
.PHONY: poetry-download
poetry-download:
	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | $(PYTHON) -

.PHONY: poetry-remove
poetry-remove:
	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | $(PYTHON) - --uninstall

.PHONY: dev-init
dev-init: poetry-install tools-install

.PHONY: poetry-install
poetry-install:
	poetry install --no-interaction

.PHONY: tools-install
tools-install:
	poetry run pre-commit install
	poetry run pre-commit install --hook-type commit-msg
	poetry run nbdime config-git --enable

# ===================================================================
# Testing
# ===================================================================

PYTEST_USE_COLOR ?= yes
PYTEST_OPTS ?= -v --durations=10 --color=${PYTEST_USE_COLOR}

.PHONY: tests
tests:
	poetry run pytest ${PYTEST_OPTS} tests
