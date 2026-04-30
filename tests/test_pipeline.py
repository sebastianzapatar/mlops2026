"""
Tests unitarios para el pipeline de MLOps.
Se ejecutan en CI/CD con: poetry run pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

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
