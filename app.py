from fastapi import FastAPI
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

# Cargar el modelo entrenado
try:
    model = joblib.load("best_model.pkl")
except Exception as e:
    model = None
    print(f"Error cargando el modelo: {e}")

# Inicializar monitor y feature store
monitor = ModelMonitor()
feature_store = FeatureStore()


class HousingData(BaseModel):
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
    """Endpoint de salud para Docker healthcheck y load balancers."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(data: HousingData):
    """Predice el valor de una vivienda y registra la predicción en el monitor."""
    if not model:
        return {"error": "El modelo no está disponible."}

    # Convertir el payload a DataFrame
    input_data = pd.DataFrame([data.dict()])

    # Predecir
    prediction = model.predict(input_data)[0]

    # Registrar en el monitor (data drift + logging)
    monitor.log_prediction(data.dict(), float(prediction))

    return {"predicted_median_house_value": float(prediction)}


@app.get("/monitor/summary")
def monitor_summary():
    """Retorna el resumen de predicciones y métricas del monitor."""
    return monitor.get_summary()


@app.post("/monitor/check-drift")
def check_drift(data: HousingData):
    """Verifica si los datos de entrada presentan data drift."""
    return monitor.check_drift(data.dict())


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
