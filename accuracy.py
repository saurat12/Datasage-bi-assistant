"""
Accuracy / backtesting for DataSage.

Two ways to score a forecasting model on historical data:

1. `backtest(df, horizon)`
   Simple holdout. Hide the last `horizon` points, train on the rest,
   forecast, compare to what actually happened. Returns one set of metrics.
   Fast (1 model fit). Good for "is this model reasonable?"

2. `walk_forward_backtest(df, horizon, n_splits)`
   Rolling-origin (a.k.a. expanding-window) cross-validation. Repeats the
   holdout exercise `n_splits` times, advancing the cutoff each time, and
   averages the metrics. Slow (n_splits model fits). Good for "what
   accuracy can I actually trust?"

Both reuse `_fit_and_forecast` and frequency inference from forecasting.py
so we score the same model the user sees in the chart.

Metrics
-------
MAE   — mean absolute error, in the data's units.
RMSE  — root mean squared error, penalizes large misses more.
MAPE  — mean absolute percentage error. The headline metric. Unstable
        when actuals are near zero, so we filter zeros from the divisor.
SMAPE — symmetric MAPE. Safer when actuals can be zero / negative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from forecasting import (
    _coerce_series,
    _fit_and_forecast,
    _infer_frequency,
)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class AccuracyMetrics:
    """A single set of error metrics from one backtest split (or averaged)."""
    mae: float
    rmse: float
    mape: float           # percent (e.g. 8.3 means 8.3%)
    smape: float          # percent
    n_points: int         # how many forecast/actual pairs went into the score
    n_splits: int = 1     # how many splits were averaged into these numbers

  

# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _compute_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    n_splits: int = 1,
) -> AccuracyMetrics:
    """Compute MAE / RMSE / MAPE / SMAPE between two equal-length arrays."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    n = len(actual)

    if n == 0:
        return AccuracyMetrics(mae=np.nan, rmse=np.nan, mape=np.nan,
                               smape=np.nan, n_points=0, n_splits=n_splits)

    errors = actual - predicted
    abs_err = np.abs(errors)

    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # MAPE: skip points where actual is zero (would divide by zero).
    nonzero = actual != 0
    if nonzero.any():
        mape = float(np.mean(abs_err[nonzero] / np.abs(actual[nonzero])) * 100)
    else:
        mape = float("nan")

    # SMAPE: symmetric, defined even when actual or predicted is zero.
    denom = (np.abs(actual) + np.abs(predicted)) / 2
    safe = denom != 0
    if safe.any():
        smape = float(np.mean(abs_err[safe] / denom[safe]) * 100)
    else:
        smape = float("nan")

    return AccuracyMetrics(
        mae=mae, rmse=rmse, mape=mape, smape=smape,
        n_points=n, n_splits=n_splits,
    )


# ---------------------------------------------------------------------------
# Simple holdout backtest
# ---------------------------------------------------------------------------

def backtest(
    df: pd.DataFrame,
    horizon: int = 6,
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
) -> AccuracyMetrics:
    """
    Hide the last `horizon` points, train on the rest, forecast, score.

    Returns AccuracyMetrics. Raises ValueError if the series is too short
    to leave both a training set (≥4 points) and a holdout set.
    """
    history = _coerce_series(df, date_col=date_col, value_col=value_col)

    if len(history) < horizon + 4:
        raise ValueError(
            f"Series too short for backtest: need at least {horizon + 4} points "
            f"({horizon} for holdout + 4 for training), got {len(history)}."
        )

    train = history.iloc[:-horizon]
    holdout = history.iloc[-horizon:]

    _, seasonal_period = _infer_frequency(train["date"])
    point, _, _, _ = _fit_and_forecast(
        train["value"], periods=horizon, seasonal_period=seasonal_period
    )

    return _compute_metrics(holdout["value"].values, point, n_splits=1)


# ---------------------------------------------------------------------------
# Walk-forward (rolling-origin) backtest
# ---------------------------------------------------------------------------

def walk_forward_backtest(
    df: pd.DataFrame,
    horizon: int = 3,
    n_splits: int = 5,
    min_train_size: int = 12,
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
) -> AccuracyMetrics:
    """
    Rolling-origin cross-validation.

    Performs `n_splits` backtests. Split k trains on points
    [0 : end - (n_splits - k) * horizon] and forecasts the next `horizon`
    points. Errors from all splits are pooled, then metrics are computed
    on the combined arrays.

    Parameters
    ----------
    horizon : forecast horizon for each split.
    n_splits : how many splits to run. More = more reliable estimate, slower.
    min_train_size : smallest training window allowed. Splits that would
                     leave fewer than this many training points are skipped.

    Returns
    -------
    AccuracyMetrics with `n_splits` set to the actual number of splits used
    (which may be smaller than requested if the series is short).
    """
    history = _coerce_series(df, date_col=date_col, value_col=value_col)
    n = len(history)

    # We need: min_train_size + n_splits * horizon points minimum
    needed = min_train_size + n_splits * horizon
    if n < min_train_size + horizon:
        raise ValueError(
            f"Series too short for walk-forward: need at least "
            f"{min_train_size + horizon} points, got {n}."
        )

    # Reduce splits if the series doesn't have room for all of them.
    max_possible_splits = (n - min_train_size) // horizon
    actual_splits = min(n_splits, max_possible_splits)
    if actual_splits < 1:
        raise ValueError("Series too short for any walk-forward split.")

    all_actual: list[float] = []
    all_pred: list[float] = []

    for k in range(actual_splits):
        # Split k: train on everything before the kth-from-last horizon block.
        # Newest split (k = actual_splits - 1) trains on the most data.
        end_offset = (actual_splits - 1 - k) * horizon
        train_end = n - end_offset - horizon
        train = history.iloc[:train_end]
        holdout = history.iloc[train_end:train_end + horizon]

        if len(train) < min_train_size or len(holdout) < horizon:
            continue

        try:
            _, seasonal_period = _infer_frequency(train["date"])
            point, _, _, _ = _fit_and_forecast(
                train["value"], periods=horizon, seasonal_period=seasonal_period
            )
        except Exception:
            # If a particular split fails to fit, skip it rather than crash.
            continue

        all_actual.extend(holdout["value"].tolist())
        all_pred.extend(np.asarray(point).tolist())

    if not all_actual:
        raise ValueError("No walk-forward splits produced a usable forecast.")

    return _compute_metrics(
        np.array(all_actual), np.array(all_pred), n_splits=actual_splits
    )


# ---------------------------------------------------------------------------
# Convenience wrapper used by the app
# ---------------------------------------------------------------------------

def quick_accuracy(
    df: pd.DataFrame,
    horizon: int = 3,
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
) -> Optional[AccuracyMetrics]:
    """
    Best-effort accuracy estimate for the inline UI footer.

    Tries walk-forward validation first (more honest); falls back to a
    simple holdout when the series is too short. Returns None if neither
    is feasible — the app then just hides the accuracy line.
    """
    try:
        return walk_forward_backtest(
            df, horizon=horizon, n_splits=5,
            date_col=date_col, value_col=value_col,
        )
    except ValueError:
        pass

    try:
        return backtest(
            df, horizon=horizon,
            date_col=date_col, value_col=value_col,
        )
    except ValueError:
        return None