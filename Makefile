.DEFAULT_GOAL := help
.PHONY: all tpls coverage deps format dev docker test test-dev help

all: ## Builds the package
	python setup.py build

install: tpls ## Builds & installs TINerator
	python setup.py install

tpls: ## Builds TPLs (Exodus, JIGSAW, and PyLaGriT)
	./tpls/build-tpls.sh -A -M

coverage: ## Run tests with code coverage
	coverage erase
	coverage run --include=tinerator/* -m pytest -ra
	coverage report -m

clean: ## Clean build artifacts
	python setup.py clean --all || true
	rm -rf build/
	rm -rf coverage-debug .coverage coverage.xml
	rm -rf doc/_build
	rm -rf .eggs/ *.egg-info/ .cache/ __pycache__/ *.pyc */*.pyc */*/*.pyc

deps: ## Install developer dependencies
	python -m pip install --upgrade pip
	python -m pip install -e '.[all]'

format: ## Format the code with Black
	python -m black --verbose --config pyproject.toml .

dev: ## Temporary build for active development
	python -m pip install --editable .

docs: ## Builds HTML documentation
	cd docs/ && make html

docker: ## Build Docker container
	docker build -t ees16/tinerator:latest -f Dockerfile ./

test: ## Run tests
	python -m pytest -ra

test-dev: dev test ## Builds & tests TINerator within active development

help: ## Show help message
	@IFS=$$'\n' ; \
	help_lines=(`fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##/:/'`); \
	printf "%s\n\n" "Usage: make [task]"; \
	printf "%-20s %s\n" "task" "help" ; \
	printf "%-20s %s\n" "------" "----" ; \
	for help_line in $${help_lines[@]}; do \
		IFS=$$':' ; \
		help_split=($$help_line) ; \
		help_command=`echo $${help_split[0]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		help_info=`echo $${help_split[2]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		printf '\033[36m'; \
		printf "%-20s %s" $$help_command ; \
		printf '\033[0m'; \
		printf "%s\n" $$help_info; \
	done