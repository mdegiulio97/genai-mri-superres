# Shortcuts for the project's uv commands (see README, Development section).
# Each target wraps the actual uv command already documented in the README.

.DEFAULT_GOAL := help

.PHONY: help setup data train test lint app all

help: ## Show this list of targets
	@echo "Available targets:"
	@echo "  setup  - install dependencies (uv sync --extra dev)"
	@echo "  data   - download the BloodMNIST dataset (scripts/download_data.py)"
	@echo "  train  - train the ESPCN model (scripts/train.py)"
	@echo "  test   - run the offline pytest suite (uv run pytest -q)"
	@echo "  lint   - run ruff (uv run ruff check .)"
	@echo "  app    - start the Streamlit demo (app/main.py)"
	@echo "  all    - setup + lint + test"

setup: ## Install runtime and development dependencies
	uv sync --extra dev

data: ## Download the BloodMNIST dataset (idempotent)
	uv run python scripts/download_data.py

train: ## Train the ESPCN model -> results/model.pt
	uv run python scripts/train.py

test: ## Run the offline test suite
	uv run pytest -q

lint: ## Run the ruff linter on the project
	uv run ruff check .

app: ## Start the interactive Streamlit demo
	uv run streamlit run app/main.py

all: setup lint test ## Setup, lint and test in sequence
