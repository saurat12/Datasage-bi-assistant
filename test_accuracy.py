"""Tests for Accuracy.py."""

import numpy as np
import pandas as pd
import pytest

from accuracy import (
    AccuracyMetrics,
    _compute_metrics,
    backtest,
    quick_accuracy,
    walk_forward_backtest,
)


@pytest.fixture
def long_clean_series():
    """36 months of trending + seasonal data with low noise."""
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=36, freq="MS")
    trend = np.linspace(1000, 1800, 36)
    seasonality = 150 * np.sin(np.arange(36) * 2 * np.pi / 12)
    noise = np.random.normal(0, 30, 36)
    return pd.DataFrame({"period": dates, "total_sales": trend + seasonality + noise})


# ---------------------------------------------------------------------------
# Metric math
# ---------------------------------------------------------------------------

class TestMetricMath:

    def test_mae_is_mean_absolute_error(self):
        # errors = [-10, 20, -30], abs = [10, 20, 30], mean = 20
        m = _compute_metrics(np.array([100, 200, 300]), np.array([110, 180, 330]))
        assert m.mae == pytest.approx(20.0)

    def test_rmse_penalizes_large_errors_more_than_mae(self):
        # Same total error magnitude, but concentrated in one big miss
        m = _compute_metrics(np.array([100, 100, 100]), np.array([100, 100, 130]))
        assert m.rmse > m.mae

    def test_mape_is_in_percent(self):
        # 10% off on each → MAPE should be 10
        m = _compute_metrics(np.array([100, 200, 300]), np.array([110, 220, 330]))
        assert m.mape == pytest.approx(10.0)

    def test_mape_skips_zero_actuals_safely(self):
        m = _compute_metrics(np.array([0, 100, 200]), np.array([5, 110, 190]))
        assert np.isfinite(m.mape)
        assert np.isfinite(m.smape)

    def test_perfect_forecast_has_zero_error(self):
        actual = np.array([100, 200, 300])
        m = _compute_metrics(actual, actual.copy())
        assert m.mae == 0
        assert m.rmse == 0
        assert m.mape == 0


# ---------------------------------------------------------------------------
# Simple holdout backtest
# ---------------------------------------------------------------------------

class TestBacktest:

    def test_returns_metrics_for_normal_series(self, long_clean_series):
        m = backtest(long_clean_series, horizon=6)
        assert isinstance(m, AccuracyMetrics)
        assert m.n_points == 6
        assert m.n_splits == 1

    def test_clean_seasonal_data_yields_low_mape(self, long_clean_series):
        m = backtest(long_clean_series, horizon=6)
        # Clean simulated data with strong seasonality should be very predictable
        assert m.mape < 15

    def test_too_short_series_raises(self):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5, freq="MS"),
            "value": [10, 20, 30, 40, 50],
        })
        with pytest.raises(ValueError, match="too short"):
            backtest(df, horizon=6)


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------

class TestWalkForwardBacktest:

    def test_runs_all_requested_splits_when_series_long_enough(self, long_clean_series):
        m = walk_forward_backtest(long_clean_series, horizon=3, n_splits=5)
        assert m.n_splits == 5
        assert m.n_points == 15  # 5 splits × 3 horizon points

    def test_reduces_splits_when_series_too_short_for_all(self):
        # 18 months can't fit 5 × 3 splits with 12-month minimum train,
        # but should still produce *some* splits.
        np.random.seed(1)
        df = pd.DataFrame({
            "period": pd.date_range("2023-01-01", periods=18, freq="MS"),
            "total_sales": 1000 + np.arange(18) * 20,
        })
        m = walk_forward_backtest(df, horizon=3, n_splits=5)
        assert m.n_splits >= 1
        assert m.n_splits <= 5

    def test_too_short_raises(self):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="MS"),
            "value": np.arange(10),
        })
        with pytest.raises(ValueError, match="too short"):
            walk_forward_backtest(df, horizon=3, n_splits=5, min_train_size=12)


# ---------------------------------------------------------------------------
# quick_accuracy convenience wrapper
# ---------------------------------------------------------------------------

class TestQuickAccuracy:

    def test_uses_walk_forward_when_possible(self, long_clean_series):
        m = quick_accuracy(long_clean_series, horizon=3)
        assert m is not None
        assert m.n_splits >= 2  # walk-forward, not single holdout

    def test_returns_none_on_tiny_series(self):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=4, freq="MS"),
            "value": [10, 20, 30, 40],
        })
        assert quick_accuracy(df, horizon=3) is None