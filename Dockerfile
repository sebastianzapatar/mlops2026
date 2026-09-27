# ──────────────────────────────────────────────
# Dockerfile para el API de predicción
# Imagen multi-stage para menor tamaño
# ──────────────────────────────────────────────

# Stage 1: Builder - Instala dependencias con uv
FROM python:3.12-slim AS builder

# Copiar el binario de uv desde su imagen oficial (no hace falta pip install)
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv

# Compilar .pyc para arrancar más rápido; usar el Python de la imagen (no descargar otro)
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Copiar archivos de dependencias
COPY pyproject.toml uv.lock .python-version ./

# Crear /app/.venv exactamente con las versiones de uv.lock
# --no-default-groups: sin dev (pytest) ni eda (Jupyter, matplotlib)
RUN uv sync --locked --no-default-groups

# Stage 2: Runner - Imagen final ligera
FROM python:3.12-slim AS runner

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar el entorno virtual del builder (sin uv ni caché)
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

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
