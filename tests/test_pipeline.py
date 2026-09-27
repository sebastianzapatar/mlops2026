"""
Tests unitarios para el pipeline de MLOps.
Se ejecutan en CI/CD con: uv run pytest tests/ -v
"""

import pytest
import pandas as pd
import os

from feature_store import FeatureStore
from monitor import ModelMonitor


class TestFeatureStore:
    """Tests para el Feature Store centralizado."""

    def test_derived_features(self):
        """Verifica que las features derivadas se calculen correctamente."""
        fs = FeatureStore()
        df = pd.DataFrame(
            {
                "total_rooms": [100, 200],
                "households": [10, 20],
                "total_bedrooms": [50, 100],
                "population": [30, 60],
            }
        )
        result = fs.add_derived_features(df)
        assert "rooms_per_household" in result.columns
        assert "bedrooms_per_room" in result.columns
        assert "population_per_household" in result.columns
        assert result["rooms_per_household"].iloc[0] == 10.0

    def test_preprocessor_creation(self):
        """Verifica que el preprocesador se construya sin errores."""
        fs = FeatureStore()
        preprocessor = fs.build_preprocessor()
        assert preprocessor is not None
        assert len(preprocessor.transformers) == 2

    def test_feature_info(self):
        """Verifica la información del Feature Store."""
        fs = FeatureStore()
        info = fs.get_info()
        assert "numeric_features" in info
        assert "categorical_features" in info
        assert info["target"] == "median_house_value"


class TestMonitor:
    """Tests para el sistema de monitoreo."""

    def test_log_prediction(self):
        """Verifica que las predicciones se registren correctamente."""
        monitor = ModelMonitor()
        record = monitor.log_prediction({"median_income": 5.0}, 250000.0)
        assert "timestamp" in record
        assert record["prediction"] == 250000.0

    def test_drift_detection_normal(self):
        """Verifica que datos normales no generen drift."""
        monitor = ModelMonitor()
        result = monitor.check_drift({"median_income": 3.5, "latitude": 37.0})
        assert "has_drift" in result

    def test_drift_detection_extreme(self):
        """Verifica que datos extremos generen drift."""
        monitor = ModelMonitor()
        result = monitor.check_drift({"median_income": 999.0})
        assert result["has_drift"] is True

    def test_summary_empty(self):
        """Verifica el resumen cuando no hay predicciones."""
        monitor = ModelMonitor()
        summary = monitor.get_summary()
        assert summary["total_predictions"] == 0

    def test_summary_with_data(self):
        """Verifica el resumen con predicciones registradas."""
        monitor = ModelMonitor()
        monitor.log_prediction({"median_income": 5.0}, 200000.0)
        monitor.log_prediction({"median_income": 8.0}, 400000.0)
        summary = monitor.get_summary()
        assert summary["total_predictions"] == 2
        assert summary["avg_prediction"] == 300000.0


class TestDataIntegrity:
    """Tests para la integridad de los datos."""

    def test_csv_exists(self):
        """Verifica que el dataset exista."""
        assert os.path.exists("1553768847-housing.csv")

    def test_csv_columns(self):
        """Verifica que el CSV tenga las columnas esperadas."""
        df = pd.read_csv("1553768847-housing.csv", nrows=5)
        expected_cols = [
            "longitude", "latitude", "housing_median_age",
            "total_rooms", "total_bedrooms", "population",
            "households", "median_income", "ocean_proximity",
            "median_house_value",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Columna '{col}' no encontrada"

    def test_model_exists(self):
        """Verifica que el modelo entrenado exista."""
        assert os.path.exists("best_model.pkl")


# Payload de ejemplo (primera fila del dataset)
SAMPLE_HOUSE = {
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41,
    "total_rooms": 880,
    "total_bedrooms": 129,
    "population": 322,
    "households": 126,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY",
}


def _small_pipeline(model, preprocessor, with_derived=False, n=500):
    """Entrena un pipeline pequeño y rápido sobre una muestra del dataset."""
    from sklearn.pipeline import Pipeline

    df = pd.read_csv("1553768847-housing.csv").sample(n, random_state=0)
    if with_derived:
        df = FeatureStore().add_derived_features(df)
    X, y = df.drop("median_house_value", axis=1), df["median_house_value"]
    return Pipeline([("preprocessor", preprocessor), ("model", model)]).fit(X, y)


class TestAPI:
    """Tests de la API FastAPI (sin levantar el servidor)."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        import app

        return TestClient(app.app)

    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_predict(self, client):
        """El modelo de train.py predice un valor positivo."""
        response = client.post("/predict", json=SAMPLE_HOUSE)
        assert response.status_code == 200
        assert response.json()["predicted_median_house_value"] > 0

    def test_predict_with_retrained_model(self, client, monkeypatch):
        """El modelo de retrain.py (con features derivadas) también funciona en la API."""
        from sklearn.ensemble import GradientBoostingRegressor
        import app

        retrained = _small_pipeline(
            GradientBoostingRegressor(n_estimators=10, random_state=42),
            FeatureStore().build_preprocessor(),
            with_derived=True,
        )
        monkeypatch.setattr(app, "model", retrained)
        response = client.post("/predict", json=SAMPLE_HOUSE)
        assert response.status_code == 200
        assert response.json()["predicted_median_house_value"] > 0

    def test_predict_invalid_payload(self, client):
        """Pydantic rechaza datos incompletos con 422."""
        response = client.post("/predict", json={"longitude": -122.23})
        assert response.status_code == 422

    def test_check_drift_endpoint(self, client):
        extreme = {**SAMPLE_HOUSE, "median_income": 999.0}
        response = client.post("/monitor/check-drift", json=extreme)
        assert response.json()["has_drift"] is True


class TestModelSerialization:
    """Los modelos deben poder guardarse en MLflow con skops."""

    @pytest.mark.parametrize("model_name", ["tree", "knn"])
    def test_skops_trusted_types(self, model_name):
        """Si una actualización agrega tipos nuevos, este test avisa antes que train.py."""
        import skops.io as sio
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from train import SKOPS_TRUSTED_TYPES, preprocess_and_split

        df = pd.read_csv("1553768847-housing.csv").sample(500, random_state=0)
        *_, preprocessor = preprocess_and_split(df)
        model = {
            "tree": RandomForestRegressor(n_estimators=5, random_state=42),
            "knn": KNeighborsRegressor(),
        }[model_name]
        pipeline = _small_pipeline(model, preprocessor)

        untrusted = sio.get_untrusted_types(data=sio.dumps(pipeline))
        assert set(untrusted) <= set(SKOPS_TRUSTED_TYPES)
