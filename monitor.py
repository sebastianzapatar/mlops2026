"""Diagnóstico básico de entradas y resumen de predicciones del proceso.

El z-score identifica valores individuales atípicos frente al CSV de referencia;
no compara distribuciones ni mide degradación del rendimiento del modelo.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import logging

# Configuración de logging (logs/ está en .gitignore: crearla antes de abrir el archivo)
os.makedirs("logs", exist_ok=True)
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
    """Guarda predicciones en memoria y alerta sobre entradas atípicas."""

    Z_SCORE_THRESHOLD = 3

    def __init__(self, reference_data_path: str = "1553768847-housing.csv"):
        """Calcula media y desviación de columnas numéricas de referencia."""
        self.predictions_log = []

        # La columna objetivo también está en el CSV, pero no en la entrada.
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

        # Persistir por lotes reduce escrituras; el lote pendiente vive en memoria.
        if len(self.predictions_log) % 100 == 0:
            self._flush_logs()

        # Esta comprobación es por registro; una alerta no prueba drift poblacional.
        drift_report = self.check_drift(input_data)
        if drift_report["has_drift"]:
            logger.warning(
                f"⚠️ DATA DRIFT DETECTADO en: {drift_report['drifted_features']}"
            )

        return record

    def check_drift(self, input_data: dict) -> dict:
        """
        Marca entradas con |valor - media| / desviación > 3.

        El nombre del método se mantiene por compatibilidad con la API, aunque
        esta regla detecta valores atípicos y no drift de una distribución.

        Args:
            input_data: Diccionario con las características de entrada.

        Returns:
            dict con has_drift (bool) y lista de features con drift.
        """
        drifted = []

        for feature, stats in self.reference_stats.items():
            if feature in input_data and stats["std"] > 0:
                z_score = abs(input_data[feature] - stats["mean"]) / stats["std"]
                if z_score > self.Z_SCORE_THRESHOLD:
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
        Retorna un resumen de las predicciones de este proceso.

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
