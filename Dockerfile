# ──────────────────────────────────────────────
# Dockerfile para el API de predicción
# Imagen multi-stage para menor tamaño
# ──────────────────────────────────────────────

# Stage 1: Builder - Instala dependencias
FROM python:3.12-slim AS builder

WORKDIR /app

# Instalar Poetry
RUN pip install --no-cache-dir poetry

# Copiar archivos de dependencias
COPY pyproject.toml poetry.lock ./

# Exportar dependencias a requirements.txt (sin dev)
RUN poetry export -f requirements.txt --without-hashes -o requirements.txt

# Stage 2: Runner - Imagen final ligera
FROM python:3.12-slim AS runner

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements del builder
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY app.py .
COPY monitor.py .
COPY feature_store.py .

# Copiar el modelo entrenado
COPY best_model.pkl .

# Copiar datos para el feature store
COPY 1553768847-housing.csv .

# Exponer puerto del API
EXPOSE 8000

# Healthcheck para monitoreo
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Ejecutar API con Uvicorn
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
