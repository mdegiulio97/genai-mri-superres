# Scorciatoie per i comandi uv del progetto (vedi README, sezione Sviluppo).
# Ogni target avvolge il comando uv reale gia documentato nel README.

.DEFAULT_GOAL := help

.PHONY: help setup data train test lint app all

help: ## Mostra questo elenco di target
	@echo "Target disponibili:"
	@echo "  setup  - installa le dipendenze (uv sync --extra dev)"
	@echo "  data   - scarica il dataset BloodMNIST (scripts/download_data.py)"
	@echo "  train  - addestra il modello ESPCN (scripts/train.py)"
	@echo "  test   - esegue la suite pytest offline (uv run pytest -q)"
	@echo "  lint   - esegue ruff (uv run ruff check .)"
	@echo "  app    - avvia la demo Streamlit (app/main.py)"
	@echo "  all    - setup + lint + test"

setup: ## Installa le dipendenze di runtime e di sviluppo
	uv sync --extra dev

data: ## Scarica il dataset BloodMNIST (idempotente)
	uv run python scripts/download_data.py

train: ## Addestra il modello ESPCN -> results/model.pt
	uv run python scripts/train.py

test: ## Esegue la suite di test offline
	uv run pytest -q

lint: ## Esegue il linter ruff sul progetto
	uv run ruff check .

app: ## Avvia la demo interattiva Streamlit
	uv run streamlit run app/main.py

all: setup lint test ## Setup, lint e test in sequenza
