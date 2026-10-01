"""API de inferencia: valida entradas y expone predicción y diagnóstico."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
from monitor import ModelMonitor
from feature_store import FeatureStore

app = FastAPI(
    title="California Housing API",
    description="API para predecir el valor de viviendas con monitoreo integrado",
    version="2.0.0",
)

# Se carga una vez al importar el módulo; hay que reiniciar para usar otro modelo.
try:
    model = joblib.load("best_model.pkl")
except Exception as e:
    model = None
    print(f"Error cargando el modelo: {e}")

# Ambos objetos viven en memoria durante el proceso de la API.
monitor = ModelMonitor()
feature_store = FeatureStore()


class HousingData(BaseModel):
    """Columnas originales que requiere el modelo de viviendas."""
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str


@app.get("/health")
def health():
    """Indica si el proceso está listo para servir predicciones."""
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo no está disponible.")
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(data: HousingData):
    """Predice el valor de una vivienda y registra la predicción en el monitor."""
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo no está disponible.")

    # Convertir el payload a DataFrame y agregar las features derivadas del
    # Feature Store (el modelo de retrain.py las necesita; el de train.py las ignora)
    input_data = feature_store.add_derived_features(pd.DataFrame([data.model_dump()]))

    # La salida de sklearn es un arreglo incluso para una sola vivienda.
    prediction = model.predict(input_data)[0]

    # El monitor conserva la entrada original y avisa sobre valores atípicos.
    monitor.log_prediction(data.model_dump(), float(prediction))

    return {"predicted_median_house_value": float(prediction)}


@app.get("/monitor/summary")
def monitor_summary():
    """Retorna el resumen de predicciones y métricas del monitor."""
    return monitor.get_summary()


@app.post("/monitor/check-drift")
def check_drift(data: HousingData):
    """Señala valores numéricos alejados de la referencia de entrenamiento."""
    return monitor.check_drift(data.model_dump())


@app.get("/features/info")
def features_info():
    """Retorna información del Feature Store."""
    return feature_store.get_info()


@app.get("/features/versions")
def features_versions():
    """Lista todas las versiones disponibles en el Feature Store."""
    return feature_store.list_versions()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
