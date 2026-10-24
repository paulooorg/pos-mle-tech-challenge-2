# Imagem do projeto (trainer e servidor MLflow). Não carrega dados: eles chegam via `dvc pull`.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app /mlflow-data \
    && chown app:app /app /mlflow-data

USER app
WORKDIR /app

# Dependências primeiro: a camada fica em cache enquanto pyproject.toml e uv.lock não mudam
COPY --chown=app:app pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Projeto: código, pipeline DVC, ponteiros de dados e o .git (usado pelo DVC)
COPY --chown=app:app src ./src
COPY --chown=app:app dvc.yaml dvc.lock metrics.json params.yaml .dvcignore ./
COPY --chown=app:app .dvc ./.dvc
COPY --chown=app:app data ./data
COPY --chown=app:app .git ./.git
COPY --chown=app:app scripts ./scripts
RUN uv sync --frozen --no-dev

CMD ["/app/scripts/trainer_entrypoint.sh"]
