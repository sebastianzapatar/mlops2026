"""Adaptación determinista del CSV para PyCaret y la API de inferencia."""

import pandas as pd


# Fijar los códigos evita depender del orden de categorías de cada lote.
OCEAN_CODES = {
    "<1H OCEAN": 0,
    "INLAND": 1,
    "ISLAND": 2,
    "NEAR BAY": 3,
    "NEAR OCEAN": 4,
}

# Este orden coincide con las columnas utilizadas al entrenar el pipeline.
PYCARET_FEATURES = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "ocean_proximity_code",
    "bedrooms_missing",
]


def prepare_pycaret_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Convierte entradas originales en columnas numéricas de esquema fijo.

    El valor -1 representa dormitorios desconocidos; la columna indicadora
    permite distinguirlo de un conteo observado. No se calculan estadísticas
    con el lote de inferencia, así que el resultado es igual en train y serve.
    """
    # Copiar los datos evita modificar el DataFrame recibido por el llamador.
    prepared = frame.copy()

    # Rechazar categorías desconocidas permite detectar errores de entrada.
    unknown = set(prepared["ocean_proximity"].dropna()) - set(OCEAN_CODES)
    if unknown:
        raise ValueError(f"ocean_proximity desconocido: {sorted(unknown)}")

    # La ausencia de dormitorios se conserva en una variable explícita.
    prepared["bedrooms_missing"] = prepared["total_bedrooms"].isna().astype(int)

    # El centinela evita la incompatibilidad del preprocesador de PyCaret 4a8
    # con columnas ausentes en este conjunto de datos.
    # Forzar tipo numérico evita conversiones implícitas de pandas al rellenar.
    prepared["total_bedrooms"] = pd.to_numeric(prepared["total_bedrooms"]).fillna(-1)

    # La categoría usa un código estable que se reproduce en FastAPI.
    prepared["ocean_proximity_code"] = prepared["ocean_proximity"].map(OCEAN_CODES)

    # No enviar la cadena original al pipeline numérico exportado.
    return prepared[PYCARET_FEATURES]
