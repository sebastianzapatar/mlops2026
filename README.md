# 🏠 MLOps Pipeline — California Housing Prediction

<div align="center">

![Miku MLOps](miku_mlops.png)

*✨ Pipeline completo de Machine Learning Operations ✨*

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![Poetry](https://img.shields.io/badge/Poetry-Dependency%20Manager-60A5FA?logo=poetry)](https://python-poetry.org)
[![MLFlow](https://img.shields.io/badge/MLFlow-Experiment%20Tracking-0194E2?logo=mlflow)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Models-F7931E?logo=scikit-learn)](https://scikit-learn.org)

</div>

---

## 📋 Descripción

Pipeline end-to-end de MLOps para predecir el **valor medio de viviendas en California** usando el dataset [California Housing](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) (20,640 registros).

El proyecto cubre todo el ciclo de vida de un modelo de ML:

```
Poetry → EDA → Pipeline → Cross-Validation → MLFlow → Parquet → Docker → FastAPI
```

---

## 🗂️ Estructura del Proyecto

```
mlops/
├── train.py                    # 🤖 Entrenamiento de 10 modelos con CV 5-Fold
├── app.py                      # 🚀 API REST con FastAPI
├── monitor.py                  # 📈 Monitoreo de drift y logging
├── feature_store.py            # 🧮 Feature Store centralizado
├── retrain.py                  # 🔄 Reentrenamiento automático
├── eda.ipynb                   # 📊 Notebook de análisis exploratorio
├── 1553768847-housing.csv      # 📁 Dataset original
├── pyproject.toml              # 📦 Dependencias (Poetry)
├── poetry.lock                 # 🔒 Versiones exactas
├── Dockerfile                  # 🐳 Imagen multi-stage
├── compose.yml                 # 🐳 Docker Compose (API + MLFlow)
├── model_metrics.json          # 📋 Métricas del modelo actual
├── tests/
│   └── test_pipeline.py        # 🧪 11 tests unitarios
├── .github/workflows/
│   ├── ci_cd.yml               # ⚙️ CI/CD: test → train → docker
│   └── retrain.yml             # 🔄 Reentrenamiento semanal
├── presentacion_mlops.html     # 🎓 Presentación principal (Reveal.js)
└── presentacion_mlops_avanzado.html  # 🎓 Presentación avanzada
```

---

## 🚀 Inicio Rápido

### 1. Instalar dependencias

```bash
# Instalar Poetry (si no lo tienes)
pip install poetry

# Instalar dependencias del proyecto
poetry install
```

### 2. Entrenar los modelos

```bash
poetry run python train.py
```

Esto entrena **10 modelos de regresión** con **validación cruzada 5-Fold** y registra todo en MLFlow:

| Modelo | RMSE Test ($) | CV RMSE μ ($) | CV σ ($) | R² |
|--------|--------------|---------------|----------|------|
| 🏆 **Gradient Boosting** | **48,077** | **48,383** | **950** | **0.8236** |
| Random Forest | 48,942 | 49,262 | 624 | 0.8172 |
| Extra Trees | 52,281 | 52,374 | 667 | 0.7914 |
| KNN | 61,310 | 61,732 | 742 | 0.7132 |
| Decision Tree | 66,557 | 67,226 | 1,254 | 0.6620 |
| Linear Regression | 70,059 | 68,623 | 1,439 | 0.6254 |
| Ridge | 70,066 | 68,622 | 1,440 | 0.6254 |
| Lasso | 70,060 | 68,623 | 1,439 | 0.6254 |
| ElasticNet | 78,389 | 78,285 | 1,030 | 0.5311 |
| AdaBoost | 90,406 | 89,439 | 3,264 | 0.3763 |

### 3. Ver experimentos en MLFlow

```bash
poetry run mlflow ui
# Abre http://localhost:5000
```

### 4. Levantar la API

```bash
poetry run uvicorn app:app --reload
# Abre http://localhost:8000/docs para Swagger UI
```

### 5. Hacer una predicción

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41,
    "total_rooms": 880,
    "total_bedrooms": 129,
    "population": 322,
    "households": 126,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY"
  }'
```

---

## 🔬 Análisis Exploratorio (EDA)

El notebook `eda.ipynb` contiene **13 secciones** de análisis:

1. **Información general** del dataset
2. **Estadísticas descriptivas** con heatmap
3. **Valores nulos** — `total_bedrooms` tiene 207 (~1%)
4. **Variable objetivo** — histograma, boxplot, QQ-plot
5. **Distribución** de todas las variables numéricas
6. **Asimetría (skewness)** — detectar variables sesgadas
7. **Matriz de correlación** triangular
8. **Ingreso vs Valor** — scatter + mapa de densidad
9. **Proximidad al océano** — pie, boxplot, violin
10. **Distribución geográfica** — mapas de California
11. **Detección de outliers** con IQR
12. **Feature Engineering** — variables derivadas
13. **Conclusiones** del análisis

```bash
poetry run jupyter notebook eda.ipynb
```

---

## 🔄 Validación Cruzada (Cross-Validation)

Cada modelo se evalúa con **5-Fold Cross-Validation** sobre el set de entrenamiento:

```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    pipeline, X_train, y_train,
    cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)
cv_rmse = np.sqrt(-cv_scores)
```

Las métricas de CV se registran en MLFlow:
- `cv_rmse_mean` — RMSE promedio de los 5 folds
- `cv_rmse_std` — Desviación estándar (estabilidad)
- `cv_rmse_fold_1..5` — RMSE individual por fold

---

## ⚙️ Pipeline de Preprocesamiento

Se usa `ColumnTransformer` + `Pipeline` para **evitar data leakage**:

```python
# Numérico: impute con mediana → escalar
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categórico: impute con moda → one-hot encoding
cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combinar ambos
preprocessor = ColumnTransformer([
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])
```

> ⚠️ **Data Leakage**: Si escalas antes de dividir, el `StandardScaler` aprende estadísticas del test set, generando métricas falsamente optimistas.

---

## 📈 Monitoreo en Producción

El módulo `monitor.py` implementa:

- **Logging** de cada predicción con timestamp
- **Data drift detection** usando z-score (umbral: 3σ)
- **Resumen estadístico** en tiempo real

```bash
# Verificar drift
curl -X POST http://localhost:8000/monitor/check-drift \
  -H "Content-Type: application/json" \
  -d '{"longitude":-122.23, "latitude":37.88, ...}'

# Ver resumen
curl http://localhost:8000/monitor/summary
```

---

## 🐳 Docker

```bash
# Construir y levantar
docker compose up -d

# Ver logs
docker compose logs -f api
```

El Dockerfile usa **multi-stage build** para mantener la imagen ligera (~400MB vs ~1.5GB).

---

## 🧪 Tests

```bash
poetry run pytest tests/ -v
```

11 tests unitarios que cubren:
- ✅ Feature Store (features derivadas, preprocesador, info)
- ✅ Monitor (logging, drift normal, drift extremo, resumen)
- ✅ Integridad de datos (CSV existe, columnas correctas, modelo existe)

---

## 🎓 Presentaciones

Abrir en el navegador:
- `presentacion_mlops.html` — Pipeline básico (EDA → Train → API)
- `presentacion_mlops_avanzado.html` — Avanzado (CI/CD → Docker → Monitoreo)

---

## 📊 Stack Tecnológico

| Categoría | Herramientas |
|-----------|-------------|
| **Entorno** | Poetry, pyproject.toml |
| **Exploración** | Pandas, Seaborn, Matplotlib, Jupyter |
| **Modelado** | Scikit-Learn (10 modelos) |
| **Validación** | Cross-Validation 5-Fold |
| **Tracking** | MLFlow |
| **Almacenamiento** | PyArrow (Parquet) |
| **API** | FastAPI + Uvicorn |
| **Contenedores** | Docker + Docker Compose |
| **CI/CD** | GitHub Actions |
| **Monitoreo** | Data drift (z-score) |

---

<div align="center">

**Universidad EIA** — Ingeniería con propósito 🎓

</div>
