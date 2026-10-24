"""API de inferência (FastAPI): `/health`, `/model` e `/predict`.

O modelo é escolhido pela variável de ambiente `MODEL_VERSION` (`champion` ou o número
de uma versão). Trocar de modelo é um deploy: recriar o container com outro valor.
"""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from fastapi import Body, Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel

from pos_mle_tech_challenge_2 import __version__
from pos_mle_tech_challenge_2.config import configure_logging, load_settings
from pos_mle_tech_challenge_2.serving.examples import (
    EXAMPLE_SESSIONS,
    EXAMPLE_SUMMARIES,
)
from pos_mle_tech_challenge_2.serving.model_loader import LoadedModel, ModelLoader
from pos_mle_tech_challenge_2.serving.predictor import Predictor
from pos_mle_tech_challenge_2.serving.schemas import (
    ModelInfo,
    PredictRequest,
    PredictResponse,
)

logger = logging.getLogger(__name__)

LoadModel = Callable[[], LoadedModel]
NOT_READY = {503: {"description": "Modelo não carregado (veja `detail`)"}}


class HealthResponse(BaseModel):
    """Resposta de `/health`: o serviço está pronto e qual versão atende."""

    status: Literal["ok"]
    model_version: int


def predict_examples() -> dict[str, dict[str, Any]]:
    """Exemplos nomeados do Swagger: um lote com as três sessões e cada uma sozinha."""
    all_sessions = list(EXAMPLE_SESSIONS.values())
    examples: dict[str, dict[str, Any]] = {
        "lote_com_as_tres_sessoes": {
            "summary": "Lote com as três sessões abaixo",
            "value": {"sessions": all_sessions},
        }
    }
    for name, session in EXAMPLE_SESSIONS.items():
        examples[name] = {
            "summary": EXAMPLE_SUMMARIES[name],
            "value": {"sessions": [session]},
        }
    return examples


def get_predictor(request: Request) -> Predictor:
    """Dependência: o preditor pronto, ou 503 com o motivo se o modelo não carregou."""
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail=request.app.state.load_error)
    return predictor


def try_load(load_model: LoadModel) -> tuple[Predictor | None, str | None]:
    """Carrega o modelo sem derrubar o serviço: em caso de falha, guarda o motivo."""
    try:
        return Predictor(load_model()), None
    except Exception as error:
        logger.exception("Falha ao carregar o modelo")
        return None, f"{type(error).__name__}: {error}"


def register_routes(app: FastAPI) -> None:
    """Registra os endpoints."""

    @app.get("/health", response_model=HealthResponse, responses=NOT_READY)
    def health(
        predictor: Annotated[Predictor, Depends(get_predictor)],
    ) -> HealthResponse:
        """Pronto para atender? Informa a versão do modelo carregado."""
        return HealthResponse(status="ok", model_version=predictor.info.version)

    @app.get("/model", response_model=ModelInfo, responses=NOT_READY)
    def model_info(
        predictor: Annotated[Predictor, Depends(get_predictor)],
    ) -> ModelInfo:
        """Qual modelo está carregado, de onde veio e com qual threshold."""
        return predictor.info

    @app.post("/predict", response_model=PredictResponse, responses=NOT_READY)
    def predict(
        request: Annotated[PredictRequest, Body(openapi_examples=predict_examples())],
        predictor: Annotated[Predictor, Depends(get_predictor)],
    ) -> PredictResponse:
        """Probabilidade de compra e classe de cada sessão do lote, na mesma ordem."""
        return predictor.predict(request.sessions)


def create_app(load_model: LoadModel) -> FastAPI:
    """Cria a aplicação. O modelo é carregado na subida pela função informada."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.predictor, app.state.load_error = try_load(load_model)
        yield

    app = FastAPI(
        title="Propensão de compra: inferência",
        description=(
            "Classifica sessões de navegação de e-commerce com o modelo registrado "
            "no MLflow Model Registry, escolhido pela variável `MODEL_VERSION`."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    register_routes(app)
    return app


def create_app_from_env() -> FastAPI:
    """Monta a aplicação a partir do ambiente (usada pelo uvicorn com `--factory`)."""
    settings = load_settings()
    configure_logging(settings.log_level)
    loader = ModelLoader(
        settings.mlflow_tracking_uri,
        settings.mlflow_registered_model_name,
        settings.mlflow_champion_alias,
    )
    return create_app(lambda: loader.load(settings.model_version))
