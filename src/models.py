"""
Machine-learning models for stock price prediction.

Provides classifiers (direction prediction) and regressors (price prediction)
with time-series-aware splitting, walk-forward validation, and comprehensive
evaluation metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Result containers
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ClassificationResult:
    """Container for classification model outputs."""

    model: Any
    model_name: str
    predictions: np.ndarray
    probabilities: Optional[np.ndarray]
    y_test: np.ndarray
    test_dates: pd.DatetimeIndex
    metrics: Dict[str, float]
    feature_importances: Optional[pd.Series]
    confusion: np.ndarray
    classification_report_text: str
    scaler: Optional[StandardScaler] = None


@dataclass
class RegressionResult:
    """Container for regression model outputs."""

    model: Any
    model_name: str
    predictions: np.ndarray
    y_test: np.ndarray
    test_dates: pd.DatetimeIndex
    metrics: Dict[str, float]
    feature_importances: Optional[pd.Series]
    scaler: Optional[StandardScaler] = None


@dataclass
class WalkForwardResult:
    """Container for walk-forward validation outputs."""

    predictions: List[int]
    actuals: List[int]
    dates: List
    accuracies: List[float]
    mean_accuracy: float


# ──────────────────────────────────────────────────────────────────────────────
# Time-series split utility
# ──────────────────────────────────────────────────────────────────────────────


def time_series_split(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    test_ratio: float = 0.2,
    scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex, Optional[StandardScaler]]:
    """Split data chronologically (NO shuffling).

    Parameters
    ----------
    df : pd.DataFrame
        ML-ready DataFrame (must not contain NaN in feature/target cols).
    feature_cols : list[str]
        Column names to use as features.
    target_col : str
        Target column name.
    test_ratio : float, default 0.2
        Fraction of data used for testing (from the end).
    scale : bool, default True
        Whether to standardise features.

    Returns
    -------
    tuple
        ``(X_train, X_test, y_train, y_test, test_dates, scaler)``
    """
    clean = df.dropna(subset=feature_cols + [target_col]).copy()
    n = len(clean)
    split_idx = int(n * (1 - test_ratio))

    train = clean.iloc[:split_idx]
    test = clean.iloc[split_idx:]

    X_train = train[feature_cols].values
    X_test = test[feature_cols].values
    y_train = train[target_col].values
    y_test = test[target_col].values
    test_dates = test.index

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, test_dates, scaler


# ──────────────────────────────────────────────────────────────────────────────
# Classification models
# ──────────────────────────────────────────────────────────────────────────────


def _evaluate_classifier(
    model,
    model_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_dates: pd.DatetimeIndex,
    feature_cols: List[str],
    scaler: Optional[StandardScaler],
) -> ClassificationResult:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    }

    fi = None
    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, target_names=["Down", "Up"], zero_division=0)

    return ClassificationResult(
        model=model,
        model_name=model_name,
        predictions=preds,
        probabilities=proba,
        y_test=y_test,
        test_dates=test_dates,
        metrics=metrics,
        feature_importances=fi,
        confusion=cm,
        classification_report_text=report,
        scaler=scaler,
    )


def train_random_forest_classifier(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "Target_Direction",
    test_ratio: float = 0.2,
    **rf_kwargs,
) -> ClassificationResult:
    """Train a Random Forest classifier for direction prediction.

    Parameters
    ----------
    df : pd.DataFrame
        ML-ready DataFrame.
    feature_cols : list[str]
        Feature column names.
    target_col : str
        Binary target column.
    test_ratio : float
        Test set fraction.
    **rf_kwargs
        Forwarded to ``RandomForestClassifier``.

    Returns
    -------
    ClassificationResult
    """
    X_train, X_test, y_train, y_test, test_dates, scaler = time_series_split(
        df, feature_cols, target_col, test_ratio
    )

    defaults = dict(n_estimators=200, max_depth=10, min_samples_leaf=20, random_state=42, n_jobs=-1)
    defaults.update(rf_kwargs)
    model = RandomForestClassifier(**defaults)
    model.fit(X_train, y_train)

    return _evaluate_classifier(model, "Random Forest Classifier", X_test, y_test, test_dates, feature_cols, scaler)


def train_gradient_boosting_classifier(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "Target_Direction",
    test_ratio: float = 0.2,
    **gb_kwargs,
) -> ClassificationResult:
    """Train a Gradient Boosting classifier for direction prediction.

    Returns
    -------
    ClassificationResult
    """
    X_train, X_test, y_train, y_test, test_dates, scaler = time_series_split(
        df, feature_cols, target_col, test_ratio
    )

    defaults = dict(n_estimators=200, max_depth=5, learning_rate=0.05, min_samples_leaf=20, random_state=42)
    defaults.update(gb_kwargs)
    model = GradientBoostingClassifier(**defaults)
    model.fit(X_train, y_train)

    return _evaluate_classifier(model, "Gradient Boosting Classifier", X_test, y_test, test_dates, feature_cols, scaler)


# ──────────────────────────────────────────────────────────────────────────────
# Regression models
# ──────────────────────────────────────────────────────────────────────────────


def _evaluate_regressor(
    model,
    model_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_dates: pd.DatetimeIndex,
    feature_cols: List[str],
    scaler: Optional[StandardScaler],
) -> RegressionResult:
    preds = model.predict(X_test)

    # Avoid MAPE when actuals contain zero
    mask = y_test != 0
    mape_val = mean_absolute_percentage_error(y_test[mask], preds[mask]) if mask.sum() > 0 else np.nan

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
        "mape": float(mape_val),
    }

    fi = None
    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    elif hasattr(model, "coef_"):
        fi = pd.Series(np.abs(model.coef_), index=feature_cols).sort_values(ascending=False)

    return RegressionResult(
        model=model,
        model_name=model_name,
        predictions=preds,
        y_test=y_test,
        test_dates=test_dates,
        metrics=metrics,
        feature_importances=fi,
        scaler=scaler,
    )


def train_linear_regression(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "Target_Price_1d",
    test_ratio: float = 0.2,
) -> RegressionResult:
    """Train a Linear Regression model for price prediction.

    Returns
    -------
    RegressionResult
    """
    X_train, X_test, y_train, y_test, test_dates, scaler = time_series_split(
        df, feature_cols, target_col, test_ratio
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    return _evaluate_regressor(model, "Linear Regression", X_test, y_test, test_dates, feature_cols, scaler)


def train_random_forest_regressor(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "Target_Price_1d",
    test_ratio: float = 0.2,
    **rf_kwargs,
) -> RegressionResult:
    """Train a Random Forest regressor for price prediction.

    Returns
    -------
    RegressionResult
    """
    X_train, X_test, y_train, y_test, test_dates, scaler = time_series_split(
        df, feature_cols, target_col, test_ratio
    )

    defaults = dict(n_estimators=200, max_depth=15, min_samples_leaf=10, random_state=42, n_jobs=-1)
    defaults.update(rf_kwargs)
    model = RandomForestRegressor(**defaults)
    model.fit(X_train, y_train)

    return _evaluate_regressor(model, "Random Forest Regressor", X_test, y_test, test_dates, feature_cols, scaler)


# ──────────────────────────────────────────────────────────────────────────────
# Walk-forward validation
# ──────────────────────────────────────────────────────────────────────────────


def walk_forward_validation(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "Target_Direction",
    n_splits: int = 5,
    train_ratio: float = 0.6,
) -> WalkForwardResult:
    """Expanding-window walk-forward validation.

    Parameters
    ----------
    df : pd.DataFrame
        ML-ready DataFrame.
    feature_cols : list[str]
        Feature columns.
    target_col : str
        Target column.
    n_splits : int
        Number of forward steps.
    train_ratio : float
        Minimum fraction of data used for the first training window.

    Returns
    -------
    WalkForwardResult
    """
    clean = df.dropna(subset=feature_cols + [target_col]).copy()
    n = len(clean)
    min_train = int(n * train_ratio)
    step = (n - min_train) // n_splits

    if step < 10:
        logger.warning("Walk-forward step too small (%d rows). Returning empty result.", step)
        return WalkForwardResult([], [], [], [], 0.0)

    all_preds: List[int] = []
    all_actuals: List[int] = []
    all_dates: List = []
    accuracies: List[float] = []

    scaler = StandardScaler()

    for i in range(n_splits):
        train_end = min_train + i * step
        test_end = min(train_end + step, n)
        if train_end >= n:
            break

        train_data = clean.iloc[:train_end]
        test_data = clean.iloc[train_end:test_end]

        if len(test_data) == 0:
            break

        X_tr = scaler.fit_transform(train_data[feature_cols].values)
        X_te = scaler.transform(test_data[feature_cols].values)
        y_tr = train_data[target_col].values
        y_te = test_data[target_col].values

        model = RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_leaf=20, random_state=42, n_jobs=-1
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)

        acc = accuracy_score(y_te, preds)
        accuracies.append(acc)
        all_preds.extend(preds.tolist())
        all_actuals.extend(y_te.tolist())
        all_dates.extend(test_data.index.tolist())

    mean_acc = float(np.mean(accuracies)) if accuracies else 0.0
    return WalkForwardResult(
        predictions=all_preds,
        actuals=all_actuals,
        dates=all_dates,
        accuracies=accuracies,
        mean_accuracy=mean_acc,
    )
