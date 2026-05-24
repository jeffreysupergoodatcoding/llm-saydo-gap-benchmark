"""Logistic regression and LightGBM tabular models."""

from __future__ import annotations
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb


def _to_matrix(df: pl.DataFrame, drop_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    keep = [c for c in df.columns if c not in drop_cols]
    arr = df.select(keep).to_numpy().astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr, keep


def train_logistic(X_train, y_train, seed: int = 42):
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed, n_jobs=-1)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_lgbm(X_train, y_train, X_val=None, y_val=None, seed: int = 42, n_estimators: int = 500, class_weight: str | None = None):
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=20,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        random_state=seed,
        n_jobs=-1,
        class_weight=class_weight,
        verbose=-1,
    )
    if X_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
    else:
        model.fit(X_train, y_train)
    return model


def predict_proba(model, X):
    p = model.predict_proba(X)[:, 1]
    return p
