"""Esquema do dataset Online Shoppers Purchasing Intention."""

TARGET = "Revenue"

NUMERIC_FEATURES: tuple[str, ...] = (
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
)

# Os códigos (OperatingSystems, Browser, Region, TrafficType) e Weekend são
# categorias, não números: não existe ordem entre os valores.
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "Month",
    "VisitorType",
    "Weekend",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
)

FEATURE_COLUMNS: tuple[str, ...] = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
ALL_COLUMNS: tuple[str, ...] = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES, TARGET)
