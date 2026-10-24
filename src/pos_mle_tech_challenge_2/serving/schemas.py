"""Contratos da API de inferência: entrada, saída e informações do modelo."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_SESSIONS_PER_REQUEST = 1000

NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
Rate = Annotated[float, Field(ge=0, le=1)]
CategoryCode = Annotated[int, Field(ge=1)]

MonthName = Literal[
    "Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]
VisitorTypeName = Literal["Returning_Visitor", "New_Visitor", "Other"]


class SessionFeatures(BaseModel):
    """Uma sessão de navegação: as colunas do dataset, sem o alvo (`Revenue`)."""

    model_config = ConfigDict(extra="forbid")

    Administrative: NonNegativeInt = Field(
        description="Páginas administrativas visitadas"
    )
    Administrative_Duration: NonNegativeFloat = Field(description="Segundos nelas")
    Informational: NonNegativeInt = Field(description="Páginas informativas visitadas")
    Informational_Duration: NonNegativeFloat = Field(description="Segundos nelas")
    ProductRelated: NonNegativeInt = Field(description="Páginas de produto visitadas")
    ProductRelated_Duration: NonNegativeFloat = Field(description="Segundos nelas")
    BounceRates: Rate = Field(description="Taxa de rejeição média das páginas (0 a 1)")
    ExitRates: Rate = Field(description="Taxa de saída média das páginas (0 a 1)")
    PageValues: NonNegativeFloat = Field(
        description="Valor médio das páginas visitadas"
    )
    SpecialDay: Rate = Field(description="Proximidade de uma data especial (0 a 1)")
    Month: MonthName = Field(description="Mês da sessão")
    OperatingSystems: CategoryCode = Field(description="Código do sistema operacional")
    Browser: CategoryCode = Field(description="Código do navegador")
    Region: CategoryCode = Field(description="Código da região")
    TrafficType: CategoryCode = Field(description="Código do tipo de tráfego")
    VisitorType: VisitorTypeName = Field(description="Tipo de visitante")
    Weekend: bool = Field(description="Sessão em fim de semana")


class PredictRequest(BaseModel):
    """Lote de sessões para classificar."""

    sessions: list[SessionFeatures] = Field(
        min_length=1, max_length=MAX_SESSIONS_PER_REQUEST
    )


class PredictionItem(BaseModel):
    """Resultado para uma sessão, na mesma ordem da entrada."""

    probability: float = Field(ge=0, le=1, description="Probabilidade de compra")
    predicted_class: Literal[0, 1] = Field(
        description="1 se a probabilidade atinge o threshold do modelo, senão 0"
    )


class PredictResponse(BaseModel):
    """Resposta de `/predict`."""

    model_version: int
    threshold: float
    predictions: list[PredictionItem]


class ModelInfo(BaseModel):
    """Qual modelo está carregado e de onde ele veio."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    name: str
    version: int
    requested: str = Field(description="Valor de MODEL_VERSION (champion ou número)")
    run_id: str | None
    threshold: float
    data_md5: str | None = Field(description="Hash do dado que treinou o modelo")
    git_commit: str | None = Field(description="Commit do código que o treinou")
    test_average_precision: float | None
    loaded_at: datetime
