.ONESHELL:
#* Variables
SHELL := bash
PYTHON := python
PYTHONPATH := `pwd`
CONDA := conda
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

#* Docker variables
IMAGE := cfpi
VERSION := latest

#* Poetry
.PHONY: poetry-download
poetry-download:
	curl -sSL https://install.python-poetry.org | $(PYTHON) -

.PHONY: poetry-remove
poetry-remove:
	curl -sSL https://install.python-poetry.org | $(PYTHON) - --uninstall

#* Installation
.PHONY: install
install:
	! type -P poetry &> /dev/null && curl -sSL https://install.python-poetry.org | python3 -
	! type -P $(CONDA) &> /dev/null && { echo "Please install conda (https://docs.conda.io/en/latest/miniconda.html)"; exit 1; }

	# install cfpi conda environment
	$(CONDA) create -n cfpi python=3.8 -y
	$(CONDA_ACTIVATE) cfpi

	@if [ -z "$(NO_DATA)" ]; then\
		echo "Downloading data...";\
		git submodule update --init --recursive ./checkpoints;\
	fi

	type python
	
	pip3 install torch torchvision torchaudio
	pip install swig numpy==1.24.4

	# install mujoco-py dependencies https://github.com/openai/mujoco-py/issues/627
	conda install -y -c conda-forge mesa-libgl-cos7-x86_64
	conda install -y -c conda-forge glfw
	conda install -y -c conda-forge mesalib
	conda install -y -c menpo glfw3	
	export CPATH=$(CONDA_PREFIX)/include
	cp /usr/lib64/libGL.so.1  $(CONDA_PREFIX)/lib/
	ln -s $(CONDA_PREFIX)/lib/libGL.so.1 $(CONDA_PREFIX)/lib/libGL.so
	export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
	pip install patchelf "cython<3"	
	pip install cffi lockfile glfw imageio
	pip install mujoco-py==1.50.1.68
	
	poetry lock -n && poetry export --without-hashes > requirements.txt
	poetry install -n
	# -poetry run mypy --install-types --non-interactive ./

.PHONY: pre-commit-install
pre-commit-install:
	poetry run pre-commit install

#* Formatters
.PHONY: codestyle
codestyle:
	poetry run pyupgrade --exit-zero-even-if-changed --py38-plus **/*.py
	poetry run isort --settings-path pyproject.toml ./
	poetry run black --config pyproject.toml ./

.PHONY: formatting
formatting: codestyle

#* Linting
.PHONY: test
test:
	PYTHONPATH=$(PYTHONPATH) poetry run pytest -c pyproject.toml --cov-report=html --cov=cfpi tests/

.PHONY: check-codestyle
check-codestyle:
	poetry run isort --diff --check-only --settings-path pyproject.toml ./
	poetry run black --diff --check --config pyproject.toml ./
	poetry run darglint --verbosity 2 cfpi tests

.PHONY: mypy
mypy:
	poetry run mypy --config-file pyproject.toml ./

.PHONY: check-safety
check-safety:
	poetry check
	poetry run safety check --full-report
	poetry run bandit -ll --recursive cfpi tests

.PHONY: lint
lint: test check-codestyle mypy check-safety

.PHONY: update-dev-deps
update-dev-deps:
	poetry add -D bandit@latest darglint@latest "isort[colors]@latest" mypy@latest pre-commit@latest pydocstyle@latest pylint@latest pytest@latest pyupgrade@latest safety@latest coverage@latest coverage-badge@latest pytest-html@latest pytest-cov@latest
	poetry add -D --allow-prereleases black@latest

#* Docker
# Example: make docker-build VERSION=latest
# Example: make docker-build IMAGE=some_name VERSION=0.1.0
.PHONY: docker-build
docker-build:
	@echo Building docker $(IMAGE):$(VERSION) ...
	docker build \
		-t $(IMAGE):$(VERSION) . \
		-f ./docker/Dockerfile --no-cache

# Example: make docker-remove VERSION=latest
# Example: make docker-remove IMAGE=some_name VERSION=0.1.0
.PHONY: docker-remove
docker-remove:
	@echo Removing docker $(IMAGE):$(VERSION) ...
	docker rmi -f $(IMAGE):$(VERSION)

#* Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

.PHONY: dsstore-remove
dsstore-remove:
	find . | grep -E ".DS_Store" | xargs rm -rf

.PHONY: mypycache-remove
mypycache-remove:
	find . | grep -E ".mypy_cache" | xargs rm -rf

.PHONY: ipynbcheckpoints-remove
ipynbcheckpoints-remove:
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf

.PHONY: pytestcache-remove
pytestcache-remove:
	find . | grep -E ".pytest_cache" | xargs rm -rf

.PHONY: build-remove
build-remove:
	rm -rf build/

.PHONY: cleanup
cleanup: pycache-remove dsstore-remove mypycache-remove ipynbcheckpoints-remove pytestcache-remove
