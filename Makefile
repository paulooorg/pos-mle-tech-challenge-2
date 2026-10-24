RAW_FILE := data/raw/online_shoppers_intention.csv

.DEFAULT_GOAL := help
.PHONY: help install check repro data-v1 data-v2 up train

help: ## Lista os alvos disponíveis
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-9s %s\n", $$1, $$2}'

install: ## Instala as dependências (uv sync)
	uv sync

check: ## Lint, formatação e testes (rodar antes de commitar)
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest

repro: ## Executa o pipeline DVC (apenas o que mudou)
	uv run dvc repro

data-v1: ## Libera o dado v1 (Feb a Oct) e registra no DVC
	uv run python scripts/release_data.py v1
	uv run dvc add $(RAW_FILE)

data-v2: ## Libera o dado v2 (completo) e registra no DVC
	uv run python scripts/release_data.py v2
	uv run dvc add $(RAW_FILE)

up: ## Sobe a infra local (RustFS e MLflow), cria os buckets e envia o dado
	docker compose up -d --wait rustfs init mlflow
	uv run dvc push

train: ## Roda o trainer no container (reconstrói a imagem antes)
	docker compose run --build --rm trainer
