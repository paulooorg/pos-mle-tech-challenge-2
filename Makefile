RAW_FILE := data/raw/online_shoppers_intention.csv

.DEFAULT_GOAL := help
MODEL ?= champion

.PHONY: help install check repro data-v1 data-v2 up train deploy-inference clean

help: ## Lista os alvos disponíveis
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-17s %s\n", $$1, $$2}'

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
	docker compose up -d --build --wait rustfs init mlflow
	uv run dvc push

train: ## Roda o trainer no container: treina, registra no MLflow e aplica o portão
	docker compose run --build --rm trainer

deploy-inference: ## Sobe/atualiza a inferência: make deploy-inference MODEL=champion|<versão>
	@docker compose ps --status running --services | grep -qx mlflow || { echo "O MLflow não está rodando: rode make up e make train antes."; exit 1; }
	MODEL_VERSION=$(MODEL) docker compose up -d --build --force-recreate --no-deps --wait inference
	@echo "Inferência no ar com MODEL_VERSION=$(MODEL): http://localhost:8000/docs"

clean: ## Reseta o estado local (como numa máquina nova) para repetir docs/reproducao.md
	@echo "Isto apaga: volumes do Docker (bucket e MLflow do compose), data/raw/*.csv,"
	@echo "data/processed/*.csv, mlflow.db, mlruns/, cache e credenciais locais do DVC."
	@echo "Arquivos versionados no git (dvc.lock, metrics.json, ponteiro .dvc) não são tocados."
	@read -p "Continuar? [s/N] " answer && [ "$$answer" = "s" ]
	docker compose down -v || echo "Docker indisponível: os volumes não foram removidos"
	rm -rf mlflow.db mlruns mlartifacts .dvc/cache .dvc/tmp .dvc/config.local
	rm -f $(RAW_FILE) data/processed/train.csv data/processed/test.csv
	@echo "Pronto. Para repetir o guia: configure as credenciais do DVC (passo 2) e rode make data-v1."
	@echo "Se aparecerem alterações abaixo, descarte com: git restore dvc.lock metrics.json $(RAW_FILE).dvc"
	@git status --short -- dvc.lock metrics.json $(RAW_FILE).dvc
