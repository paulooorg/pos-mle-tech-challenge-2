# pos-mle-tech-challenge-2

FIAP Pós MLE — **Tech Challenge Fase 2**: pipeline de engenharia de ML para prever a **propensão de compra** de usuários de e-commerce a partir do comportamento de navegação (dataset *Online Shoppers Purchasing Intention*).

O foco é a engenharia: código limpo, ambiente reproduzível, dados versionados com DVC, experimentos e Model Registry no MLflow e execução via Docker.

> Documentação completa (DVC, MLflow, Docker e fluxo de retreino) será adicionada ao final do projeto.

## Pré-requisitos

- [uv](https://docs.astral.sh/uv/) (gerenciador de dependências)
- Python 3.12 (o uv instala automaticamente se necessário)

## Instalação

```bash
uv sync                  # instala dependências de produção e de desenvolvimento
cp .env.example .env     # ajuste as variáveis se necessário
```

Somente produção (como na imagem Docker): `uv sync --no-dev`.

## Desenvolvimento

```bash
uv run pytest                      # testes
uv run ruff check .                # lint
uv run ruff format --check .       # formatação
uv run jupyter lab --notebook-dir=notebooks    # EDA e modelagem
```

## Estrutura

```
src/pos_mle_tech_challenge_2/   # código do projeto (config, data, features, training, pipelines)
tests/                          # testes automatizados (pytest)
notebooks/                      # EDA e escolha do modelo (fora do pipeline)
data/source/                    # dataset completo (semente, commitado)
docs/                           # enunciado do desafio
specs/                          # planejamento por etapa
```

## Configuração

Variáveis de ambiente ficam no `.env` (modelo em `.env.example`). Hiperparâmetros, seed e split ficam no `params.yaml`.
