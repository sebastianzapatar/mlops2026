"""
retrain.py - Sistema de Reentrenamiento Automático
====================================================
Este script puede ejecutarse:
- Manualmente: poetry run python retrain.py
- Por GitHub Actions: workflow_dispatch o schedule
- Por el monitor: cuando detecta data drift significativo

Flujo:
1. Carga los datos más recientes
2. Usa el Feature Store para transformar
3. Reentrena el mejor modelo (Gradient Boosting)
4. Compara con el modelo actual
5. Solo reemplaza si el nuevo es mejor
6. Registra todo en MLFlow
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import mlflow.sklearn
import joblib
import os
import json
from datetime import datetime
from feature_store import FeatureStore

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AutoRetrain")


def load_current_model_metrics() -> dict:
    """
    Carga las métricas del modelo actualmente en producción.

    Returns:
        dict con rmse y r2 del modelo actual, o None si no existe.
    """
    metrics_path = "model_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return None


def save_model_metrics(metrics: dict):
    """Guarda las métricas del modelo actual a disco."""
    with open("model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def retrain(data_path: str = "1553768847-housing.csv"):
    """
    Ejecuta el pipeline de reentrenamiento automático.

    Pasos:
    1. Inicializa el Feature Store
    2. Carga y transforma datos con features derivadas
    3. Divide en entrenamiento/prueba
    4. Entrena Gradient Boosting con hiperparámetros optimizados
    5. Compara con el modelo actual
    6. Reemplaza solo si el nuevo modelo es mejor
    7. Registra todo en MLFlow
    """
    logger.info("=" * 60)
    logger.info("INICIO DE REENTRENAMIENTO AUTOMÁTICO")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # 1. Inicializar Feature Store
    fs = FeatureStore()
    logger.info("Feature Store inicializado")

    # 2. Cargar datos
    df = pd.read_csv(data_path)
    logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

    # 3. Agregar features derivadas
    df = fs.add_derived_features(df)
    logger.info(f"Features derivadas agregadas. Total columnas: {df.shape[1]}")

    # 4. Registrar versión en el Feature Store
    version_id = fs.register_version(
        df, description=f"Reentrenamiento automático {datetime.now().isoformat()}"
    )
    logger.info(f"Features registradas como versión: {version_id}")

    # 5. Separar features y target
    X = df.drop(fs.TARGET, axis=1)
    y = df[fs.TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6. Construir pipeline con Feature Store
    preprocessor = fs.build_preprocessor()
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42,
                ),
            ),
        ]
    )

    # 7. Entrenar
    logger.info("Entrenando Gradient Boosting (200 estimadores)...")
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    # 8. Calcular métricas
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae = float(mean_absolute_error(y_test, predictions))
    r2 = float(r2_score(y_test, predictions))

    new_metrics = {
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "r2": round(r2, 4),
        "trained_at": datetime.now().isoformat(),
        "feature_version": version_id,
        "n_samples": len(df),
    }

    logger.info(f"Nuevo modelo - RMSE: {rmse:,.2f} | MAE: {mae:,.2f} | R²: {r2:.4f}")

    # 9. Registrar en MLFlow
    mlflow.set_experiment("California_Housing_Retrain")
    with mlflow.start_run(run_name=f"retrain_{version_id}"):
        mlflow.log_param("model_type", "GradientBoosting")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("feature_version", version_id)
        mlflow.log_param("trigger", "automatic_retrain")

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(pipeline, "model")

    # 10. Comparar con modelo actual
    current_metrics = load_current_model_metrics()

    if current_metrics is None:
        logger.info("No hay modelo previo. Guardando nuevo modelo.")
        joblib.dump(pipeline, "best_model.pkl")
        save_model_metrics(new_metrics)
        logger.info("✅ Nuevo modelo guardado como best_model.pkl")
    elif rmse < current_metrics.get("rmse", float("inf")):
        improvement = current_metrics["rmse"] - rmse
        logger.info(
            f"✅ Nuevo modelo es MEJOR (mejora de ${improvement:,.2f} en RMSE)"
        )
        joblib.dump(pipeline, "best_model.pkl")
        save_model_metrics(new_metrics)
        logger.info("✅ Modelo reemplazado exitosamente")
    else:
        logger.info(
            f"❌ Nuevo modelo NO supera al actual "
            f"(actual: {current_metrics['rmse']:,.2f} vs nuevo: {rmse:,.2f})"
        )
        logger.info("Modelo actual se mantiene sin cambios.")

    logger.info("=" * 60)
    logger.info("REENTRENAMIENTO COMPLETADO")
    logger.info("=" * 60)

    return new_metrics


if __name__ == "__main__":
    retrain()
