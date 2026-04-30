"""
feature_store.py - Feature Store Centralizado
==============================================
Este módulo implementa un Feature Store local que:
- Centraliza las transformaciones de datos
- Versiona los conjuntos de features
- Permite reutilizar features entre entrenamiento y serving
- Garantiza consistencia entre el pipeline de training y la API
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import json
from datetime import datetime
import logging

logger = logging.getLogger("FeatureStore")


class FeatureStore:
    """
    Feature Store centralizado para el proyecto de Housing.

    Responsabilidades:
    - Definir y almacenar features de forma centralizada
    - Versionar transformaciones para reproducibilidad
    - Servir features consistentes para training y serving
    - Registrar metadatos de cada versión de features
    """

    STORE_DIR = "feature_store"
    METADATA_FILE = "feature_store/metadata.json"

    # Definición centralizada de features
    NUMERIC_FEATURES = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
    ]

    CATEGORICAL_FEATURES = [
        "ocean_proximity",
    ]

    TARGET = "median_house_value"

    # Features derivadas (ingeniería de features)
    DERIVED_FEATURES = {
        "rooms_per_household": ("total_rooms", "households"),
        "bedrooms_per_room": ("total_bedrooms", "total_rooms"),
        "population_per_household": ("population", "households"),
    }

    def __init__(self):
        """Inicializa el Feature Store y crea el directorio si no existe."""
        os.makedirs(self.STORE_DIR, exist_ok=True)
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        """Carga los metadatos del Feature Store o crea uno nuevo."""
        if os.path.exists(self.METADATA_FILE):
            with open(self.METADATA_FILE, "r") as f:
                return json.load(f)
        return {"versions": [], "current_version": None}

    def _save_metadata(self):
        """Guarda los metadatos del Feature Store a disco."""
        with open(self.METADATA_FILE, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula features derivadas a partir de las originales.

        Estas features capturan relaciones como:
        - rooms_per_household: tamaño promedio de vivienda
        - bedrooms_per_room: proporción de dormitorios
        - population_per_household: densidad de ocupación

        Args:
            df: DataFrame con las columnas originales.

        Returns:
            DataFrame con las columnas derivadas añadidas.
        """
        df = df.copy()
        for name, (numerator, denominator) in self.DERIVED_FEATURES.items():
            if numerator in df.columns and denominator in df.columns:
                df[name] = df[numerator] / df[denominator].replace(0, np.nan)
        return df

    def build_preprocessor(self) -> ColumnTransformer:
        """
        Construye el preprocesador estándar del Feature Store.

        Define las transformaciones canónicas:
        - Numéricas: Imputación por mediana + Escalado StandardScaler
        - Categóricas: Imputación por moda + OneHotEncoder

        Returns:
            ColumnTransformer configurado.
        """
        all_numeric = self.NUMERIC_FEATURES + list(self.DERIVED_FEATURES.keys())

        num_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        cat_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_transformer, all_numeric),
                ("cat", cat_transformer, self.CATEGORICAL_FEATURES),
            ]
        )

        return preprocessor

    def register_version(self, df: pd.DataFrame, description: str = "") -> str:
        """
        Registra una nueva versión de features en el store.

        Guarda los datos transformados como parquet y registra
        metadatos incluyendo estadísticas de cada feature.

        Args:
            df: DataFrame con los datos de features.
            description: Descripción de esta versión.

        Returns:
            ID de la versión registrada.
        """
        version_id = datetime.now().strftime("v_%Y%m%d_%H%M%S")
        version_path = os.path.join(self.STORE_DIR, f"{version_id}.parquet")

        # Guardar features como parquet
        df.to_parquet(version_path, index=False)

        # Registrar metadatos
        version_info = {
            "version_id": version_id,
            "created_at": datetime.now().isoformat(),
            "description": description,
            "num_rows": len(df),
            "num_features": len(df.columns),
            "features": list(df.columns),
            "path": version_path,
        }

        self.metadata["versions"].append(version_info)
        self.metadata["current_version"] = version_id
        self._save_metadata()

        logger.info(f"Feature version '{version_id}' registrada ({len(df)} filas)")
        return version_id

    def get_features(self, version_id: str = None) -> pd.DataFrame:
        """
        Recupera features de una versión específica del store.

        Args:
            version_id: ID de la versión. Si es None, usa la más reciente.

        Returns:
            DataFrame con las features de esa versión.
        """
        if version_id is None:
            version_id = self.metadata.get("current_version")

        if version_id is None:
            raise ValueError("No hay versiones registradas en el Feature Store")

        for v in self.metadata["versions"]:
            if v["version_id"] == version_id:
                return pd.read_parquet(v["path"])

        raise ValueError(f"Versión '{version_id}' no encontrada")

    def list_versions(self) -> list:
        """Lista todas las versiones disponibles en el store."""
        return self.metadata["versions"]

    def get_info(self) -> dict:
        """Retorna información general del Feature Store."""
        return {
            "total_versions": len(self.metadata["versions"]),
            "current_version": self.metadata.get("current_version"),
            "numeric_features": self.NUMERIC_FEATURES,
            "categorical_features": self.CATEGORICAL_FEATURES,
            "derived_features": list(self.DERIVED_FEATURES.keys()),
            "target": self.TARGET,
        }
