import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import mlflow.sklearn
import joblib
import pyarrow as pa
import pyarrow.parquet as pq
import os
import json


def load_data(path):
    return pd.read_csv(path)


def preprocess_and_split(df):
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    # Identify numerical and categorical columns
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    # Preprocessing for numerical data
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Preprocessing for categorical data
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, preprocessor


def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor):
    models = {
        "LinearRegression": {
            "model": LinearRegression(),
            "params": {},
        },
        "Ridge": {
            "model": Ridge(alpha=1.0),
            "params": {"alpha": 1.0},
        },
        "Lasso": {
            "model": Lasso(alpha=1.0, max_iter=5000),
            "params": {"alpha": 1.0, "max_iter": 5000},
        },
        "ElasticNet": {
            "model": ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000),
            "params": {"alpha": 1.0, "l1_ratio": 0.5},
        },
        "DecisionTree": {
            "model": DecisionTreeRegressor(max_depth=15, random_state=42),
            "params": {"max_depth": 15},
        },
        "RandomForest": {
            "model": RandomForestRegressor(n_estimators=100, random_state=42),
            "params": {"n_estimators": 100},
        },
        "ExtraTrees": {
            "model": ExtraTreesRegressor(n_estimators=100, random_state=42),
            "params": {"n_estimators": 100},
        },
        "GradientBoosting": {
            "model": GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
            ),
            "params": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 5},
        },
        "AdaBoost": {
            "model": AdaBoostRegressor(n_estimators=100, random_state=42),
            "params": {"n_estimators": 100},
        },
        "KNN": {
            "model": KNeighborsRegressor(n_neighbors=5),
            "params": {"n_neighbors": 5},
        },
    }

    mlflow.set_experiment("California_Housing_Prediction")

    best_model = None
    best_rmse = float("inf")
    best_model_name = ""
    cv_folds = 5

    print("=" * 90)
    print(
        f"{'Modelo':<22} {'RMSE Test':>12} {'MAE Test':>12} {'R² Test':>10}"
        f"  {'CV RMSE (μ±σ)':>20}"
    )
    print("=" * 90)

    for name, config in models.items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline(
                steps=[("preprocessor", preprocessor), ("model", config["model"])]
            )

            # ── Cross-Validation (5-fold) on training set ──
            cv_scores_neg_mse = cross_val_score(
                pipeline, X_train, y_train,
                cv=cv_folds,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
            )
            cv_rmse_scores = np.sqrt(-cv_scores_neg_mse)
            cv_rmse_mean = cv_rmse_scores.mean()
            cv_rmse_std = cv_rmse_scores.std()

            # ── Train on full training set & evaluate on test ──
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            print(
                f"{name:<22} {rmse:>12,.2f} {mae:>12,.2f} {r2:>10.4f}"
                f"  {cv_rmse_mean:>9,.2f} ± {cv_rmse_std:>7,.2f}"
            )

            # Log parameters
            mlflow.log_param("model_type", name)
            mlflow.log_param("cv_folds", cv_folds)
            for param_name, param_value in config["params"].items():
                mlflow.log_param(param_name, param_value)

            # Log test metrics
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mse", mse)

            # Log cross-validation metrics
            mlflow.log_metric("cv_rmse_mean", cv_rmse_mean)
            mlflow.log_metric("cv_rmse_std", cv_rmse_std)
            for i, fold_rmse in enumerate(cv_rmse_scores):
                mlflow.log_metric(f"cv_rmse_fold_{i+1}", fold_rmse)

            # Log model
            mlflow.sklearn.log_model(pipeline, "model")

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = pipeline
                best_model_name = name

    print("=" * 90)
    print(f"\n🏆 Mejor modelo: {best_model_name} con RMSE: {best_rmse:,.2f}\n")

    # Guardar el mejor modelo para la API
    joblib.dump(best_model, "best_model.pkl")

    # Guardar resultados en Parquet
    test_results = X_test.copy()
    test_results["actual_median_house_value"] = y_test
    test_results["predicted_median_house_value"] = best_model.predict(X_test)

    table = pa.Table.from_pandas(test_results)
    pq.write_table(table, "test_predictions.parquet")
    print("Resultados guardados en test_predictions.parquet")
    print(
        f"\n📊 Para ver los experimentos en MLFlow ejecuta:\n"
        f"   poetry run mlflow ui\n"
        f"   Luego abre http://localhost:5000 en tu navegador\n"
    )


if __name__ == "__main__":
    df = load_data("1553768847-housing.csv")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(df)
    train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor)
