"""
monitor.py - Sistema de Monitoreo del Modelo en Producción
==========================================================
Este módulo implementa:
- Detección de Data Drift (cambio en la distribución de datos)
- Logging de predicciones para auditoría
- Métricas de rendimiento en tiempo real
- Alertas cuando el modelo se degrada
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/monitor.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ModelMonitor")


class ModelMonitor:
    """
    Monitor de modelo en producción.

    Registra cada predicción, detecta data drift comparando
    las distribuciones de entrada contra los datos de entrenamiento,
    y genera alertas cuando el comportamiento cambia.
    """

    def __init__(self, reference_data_path: str = "1553768847-housing.csv"):
        """
        Inicializa el monitor cargando los datos de referencia
        (distribución original del entrenamiento).
        """
        self.predictions_log = []
        self.alert_threshold = 0.3  # Umbral de drift (30%)

        # Cargar estadísticas de referencia del dataset original
        os.makedirs("logs", exist_ok=True)
        try:
            ref_data = pd.read_csv(reference_data_path)
            numeric_cols = ref_data.select_dtypes(include=[np.number]).columns
            self.reference_stats = {
                col: {"mean": ref_data[col].mean(), "std": ref_data[col].std()}
                for col in numeric_cols
            }
            logger.info(
                f"Monitor inicializado con {len(numeric_cols)} features de referencia"
            )
        except Exception as e:
            logger.warning(f"No se pudo cargar datos de referencia: {e}")
            self.reference_stats = {}

    def log_prediction(self, input_data: dict, prediction: float):
        """
        Registra una predicción individual para auditoría y análisis.

        Args:
            input_data: Diccionario con las características de entrada.
            prediction: Valor predicho por el modelo.
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "input": input_data,
            "prediction": prediction,
        }
        self.predictions_log.append(record)

        # Guardar en archivo cada 100 predicciones
        if len(self.predictions_log) % 100 == 0:
            self._flush_logs()

        # Verificar drift en cada predicción
        drift_report = self.check_drift(input_data)
        if drift_report["has_drift"]:
            logger.warning(
                f"⚠️ DATA DRIFT DETECTADO en: {drift_report['drifted_features']}"
            )

        return record

    def check_drift(self, input_data: dict) -> dict:
        """
        Verifica si los datos de entrada se desvían significativamente
        de la distribución de referencia (entrenamiento).

        Usa el z-score: si |valor - media| > threshold * std,
        se considera drift.

        Args:
            input_data: Diccionario con las características de entrada.

        Returns:
            dict con has_drift (bool) y lista de features con drift.
        """
        drifted = []

        for feature, stats in self.reference_stats.items():
            if feature in input_data and stats["std"] > 0:
                z_score = abs(input_data[feature] - stats["mean"]) / stats["std"]
                if z_score > 3:  # Más de 3 desviaciones estándar
                    drifted.append(
                        {"feature": feature, "z_score": round(z_score, 2)}
                    )

        return {
            "has_drift": len(drifted) > 0,
            "drifted_features": drifted,
            "checked_at": datetime.now().isoformat(),
        }

    def get_summary(self) -> dict:
        """
        Retorna un resumen de las predicciones realizadas.

        Returns:
            dict con estadísticas de predicciones, predicción promedio,
            mínima, máxima y cantidad total.
        """
        if not self.predictions_log:
            return {"total_predictions": 0, "message": "Sin predicciones registradas"}

        predictions = [r["prediction"] for r in self.predictions_log]
        return {
            "total_predictions": len(predictions),
            "avg_prediction": round(float(np.mean(predictions)), 2),
            "min_prediction": round(float(np.min(predictions)), 2),
            "max_prediction": round(float(np.max(predictions)), 2),
            "std_prediction": round(float(np.std(predictions)), 2),
            "last_prediction_at": self.predictions_log[-1]["timestamp"],
        }

    def _flush_logs(self):
        """Guarda las predicciones acumuladas a disco."""
        log_file = f"logs/predictions_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a") as f:
            for record in self.predictions_log[-100:]:
                f.write(json.dumps(record, default=str) + "\n")
        logger.info(f"Logs guardados en {log_file}")
