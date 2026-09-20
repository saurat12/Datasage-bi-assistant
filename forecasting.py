"""
Forecasting module for DataSage.

Takes a time-series dataframe and produces forecast values with confidence
intervals. Uses Exponential Smoothing (ETS / Holt-Winters)
as the primary model with an ARIMA fallback for short or irregular series.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class ForecastResult:
    """Model output returned by `forecast_series`."""
    history: pd.DataFrame              # columns: date, value
    forecast: pd.DataFrame             # columns: date, value, lower, upper
    method: str                        # which model was used
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Intent detection (auto-trigger from natural language)
# ---------------------------------------------------------------------------

_FORECAST_KEYWORDS = [
    "forecast", "forecasting", "predict", "prediction", "projection",
    "project ", "projected", "expected", "estimate next",
    "what will", "what's next", "future", "outlook", "trend going forward",
    "next week", "next month", "next quarter", "next year",
    "coming weeks", "coming months", "coming quarter",
]

_HORIZON_PATTERN = re.compile(
    r"next\s+(?:(\d+)\s+)?(days?|weeks?|months?|quarters?|years?)\b",
    re.IGNORECASE,
)

# Absolute date targets the user might reference, e.g. "Q1 2018", "March 2024",
# "first quarter of 2018", "H1 2025", "in 2026".
_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_ORDINAL_QUARTERS = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2,
    "third": 3, "3rd": 3, "fourth": 4, "4th": 4,
}

_QUARTER_RE = re.compile(
    r"\b(?:q([1-4])|([1-4])(?:st|nd|rd|th)?\s+quarter|"
    r"(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter)\s*"
    r"(?:of\s+|,\s*|\s+(?:in|for)\s+)?(\d{4})\b",
    re.IGNORECASE,
)
_QUARTER_RE_REV = re.compile(  # "in 2018 Q1", "for 2018 first quarter"
    r"\b(\d{4})\s*(?:q([1-4])|([1-4])(?:st|nd|rd|th)?\s+quarter|"
    r"(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter)\b",
    re.IGNORECASE,
)
_HALF_RE = re.compile(r"\b(?:h([12])|(first|second|1st|2nd)\s+half)\s*(?:of\s+)?(\d{4})\b", re.IGNORECASE)
_MONTH_YEAR_RE = re.compile(
    r"\b(" + "|".join(_MONTHS.keys()) + r")\s+(\d{4})\b", re.IGNORECASE
)
_YEAR_ONLY_RE = re.compile(r"\b(?:in|for|during)\s+(\d{4})\b", re.IGNORECASE)


def detect_forecast_intent(question: str) -> bool:
    """Return True if the question implies the user wants a forecast."""
    if not question:
        return False
    q = question.lower()
    if any(kw in q for kw in _FORECAST_KEYWORDS):
        return True
    if _HORIZON_PATTERN.search(q):
        return True
    return False


def extract_horizon(question: str, default: int = 12) -> tuple[int, str]:
    """
    Try to extract a forecast horizon from the question.
    Returns (n_periods, unit_label). Falls back to (default, 'periods').
    """
    if not question:
        return default, "periods"
    m = _HORIZON_PATTERN.search(question.lower())
    if not m:
        return default, "periods"
    n = int(m.group(1) or 1)
    unit = m.group(2).rstrip("s")
    return n, unit + ("s" if n != 1 else "")


def _quarter_bounds(q: int, year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_month = (q - 1) * 3 + 1
    start = pd.Timestamp(year=year, month=start_month, day=1)
    end_month = start_month + 2
    end = pd.Timestamp(year=year, month=end_month, day=1) + pd.offsets.MonthEnd(0)
    return start, end


def extract_target_window(
    question: str,
) -> Optional[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """
    Detect an absolute date range the user is asking about, e.g.
    "Q1 2018", "first quarter of 2018", "March 2024", "H1 2025", "in 2026".
    Returns (start_date, end_date_inclusive, human_label) or None.
    """
    if not question:
        return None
    q = question

    # Quarter forms
    for regex in (_QUARTER_RE, _QUARTER_RE_REV):
        m = regex.search(q)
        if m:
            groups = m.groups()
            if regex is _QUARTER_RE:
                qn_digit, ord_digit, ord_word, year_str = groups
            else:
                year_str, qn_digit, ord_digit, ord_word = groups
            if qn_digit:
                qn = int(qn_digit)
            elif ord_digit:
                qn = int(ord_digit)
            else:
                qn = _ORDINAL_QUARTERS[ord_word.lower()]
            year = int(year_str)
            start, end = _quarter_bounds(qn, year)
            return start, end, f"Q{qn} {year}"

    # Half-year forms
    m = _HALF_RE.search(q)
    if m:
        h_digit, h_word, year_str = m.groups()
        if h_digit:
            h = int(h_digit)
        else:
            h = 1 if h_word.lower() in ("first", "1st") else 2
        year = int(year_str)
        start = pd.Timestamp(year=year, month=1 if h == 1 else 7, day=1)
        end = pd.Timestamp(year=year, month=6 if h == 1 else 12, day=1) + pd.offsets.MonthEnd(0)
        return start, end, f"H{h} {year}"

    # Specific month
    m = _MONTH_YEAR_RE.search(q)
    if m:
        month_name, year_str = m.groups()
        month = _MONTHS[month_name.lower()]
        year = int(year_str)
        start = pd.Timestamp(year=year, month=month, day=1)
        end = start + pd.offsets.MonthEnd(0)
        return start, end, f"{month_name.title()} {year}"

    # Whole year ("in 2026", "for 2025")
    m = _YEAR_ONLY_RE.search(q)
    if m:
        year = int(m.group(1))
        start = pd.Timestamp(year=year, month=1, day=1)
        end = pd.Timestamp(year=year, month=12, day=31)
        return start, end, str(year)

    return None



# ---------------------------------------------------------------------------
# Series preparation
# ---------------------------------------------------------------------------

def _coerce_series(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Normalize an arbitrary 2-column-ish dataframe into [date, value].
    Auto-detects the date and numeric columns when not specified.
    """
    if df is None or len(df) == 0:
        raise ValueError("Cannot forecast: empty dataset.")

    work = df.copy()

    # Auto-detect date column
    if date_col is None:
        for c in work.columns:
            if pd.api.types.is_datetime64_any_dtype(work[c]):
                date_col = c
                break
        if date_col is None:
            for c in work.columns:
                try:
                    parsed = pd.to_datetime(work[c], errors="raise")
                    work[c] = parsed
                    date_col = c
                    break
                except Exception:
                    continue
    if date_col is None:
        raise ValueError("Could not find a date/time column to forecast on.")

    # Auto-detect numeric value column
    if value_col is None:
        for c in work.columns:
            if c == date_col:
                continue
            if pd.api.types.is_numeric_dtype(work[c]):
                value_col = c
                break
    if value_col is None:
        raise ValueError("Could not find a numeric column to forecast.")

    work[date_col] = pd.to_datetime(work[date_col])
    work = work[[date_col, value_col]].rename(
        columns={date_col: "date", value_col: "value"}
    )
    work = work.dropna().sort_values("date").reset_index(drop=True)

    # If duplicate dates exist (e.g. category not aggregated), sum them
    if work["date"].duplicated().any():
        work = work.groupby("date", as_index=False)["value"].sum()

    return work


def _infer_frequency(dates: pd.Series) -> tuple[str, int]:
    """
    Infer pandas frequency string and an associated seasonal period.
    Returns (freq_alias, seasonal_period). seasonal_period=1 means non-seasonal.
    """
    if len(dates) < 2:
        return "D", 1

    diffs = dates.diff().dropna().dt.days
    if len(diffs) == 0:
        return "D", 1
    median_gap = float(diffs.median())

    if median_gap <= 1.5:
        return "D", 7        # daily -> weekly seasonality
    if median_gap <= 8:
        return "W", 52       # weekly -> yearly seasonality
    if median_gap <= 16:
        return "2W", 26
    if median_gap <= 45:
        return "MS", 12      # monthly -> yearly seasonality
    if median_gap <= 100:
        return "QS", 4       # quarterly -> yearly seasonality
    return "YS", 1


# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------

def _fit_and_forecast(
    series: pd.Series,
    periods: int,
    seasonal_period: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Try ETS (Holt-Winters) with a small AIC-selected grid; fall back to ARIMA,
    then to a naive drift model.

    The grid covers:
      - trend: additive (linear) vs damped
      - seasonality: additive vs multiplicative
    Multiplicative requires strictly positive data and is skipped otherwise.

    Returns (point_forecast, lower95, upper95, method_name).
    """
    n = len(series)
    values = series.values.astype(float)

    # --- Try ETS (Holt-Winters) with grid search ---
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        use_seasonal = seasonal_period > 1 and n >= 2 * seasonal_period
        all_positive = bool(np.all(values > 0))

        # Candidate (trend, seasonal, damped) configs to try.
        # Multiplicative seasonality only makes sense for strictly positive data.
        if use_seasonal:
            candidates = [
                ("add", "add", False),
                ("add", "add", True),
            ]
            if all_positive:
                candidates.extend([
                    ("add", "mul", False),
                    ("add", "mul", True),
                ])
        else:
            candidates = [
                ("add", None, False),
                ("add", None, True),
            ]

        best = None  # tuple of (aic, fit, config_label)
        for trend, seasonal, damped in candidates:
            try:
                kwargs = {
                    "trend": trend,
                    "seasonal": seasonal,
                    "damped_trend": damped,
                    "initialization_method": "estimated",
                }
                if seasonal is not None:
                    kwargs["seasonal_periods"] = seasonal_period
                model = ExponentialSmoothing(values, **kwargs)
                fit = model.fit(optimized=True)
                aic = float(fit.aic) if np.isfinite(fit.aic) else float("inf")
                if best is None or aic < best[0]:
                    seasonal_label = seasonal or "none"
                    damped_label = "damped" if damped else "linear"
                    label = (
                        f"ETS ({damped_label} trend, {seasonal_label} seasonality"
                        + (f", period={seasonal_period}" if seasonal else "")
                        + ")"
                    )
                    best = (aic, fit, label)
            except Exception:
                continue

        if best is not None:
            _, fit, method = best
            point = np.asarray(fit.forecast(periods))
            resid = values - np.asarray(fit.fittedvalues)
            sigma = float(np.nanstd(resid, ddof=1)) if len(resid) > 1 else 0.0
            steps = np.arange(1, periods + 1)
            band = 1.96 * sigma * np.sqrt(steps)
            return point, point - band, point + band, method

    except Exception:
        pass

    except Exception:
        pass

    # --- Fallback: ARIMA ---
    try:
        from statsmodels.tsa.arima.model import ARIMA

        model = ARIMA(values, order=(1, 1, 1))
        fit = model.fit()
        fc = fit.get_forecast(steps=periods)
        point = np.asarray(fc.predicted_mean)
        ci = fc.conf_int(alpha=0.05)
        lower = np.asarray(ci[:, 0])
        upper = np.asarray(ci[:, 1])
        return point, lower, upper, "ARIMA(1,1,1)"
    except Exception:
        pass

    # --- Last-resort: naive drift ---
    if n >= 2:
        slope = (values[-1] - values[0]) / (n - 1)
    else:
        slope = 0.0
    point = np.array([values[-1] + slope * (i + 1) for i in range(periods)])
    sigma = float(np.std(values, ddof=1)) if n > 1 else 0.0
    steps = np.arange(1, periods + 1)
    band = 1.96 * sigma * np.sqrt(steps)
    return point, point - band, point + band, "Naive drift"


def _future_dates(last_date: pd.Timestamp, freq: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]


# ---------------------------------------------------------------------------
# Grouped (multi-series) helpers
# ---------------------------------------------------------------------------

# Bound model-fitting cost and prevent very small groups from dominating output.
_MAX_GROUPS = 8


def _detect_category_column(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
) -> Optional[str]:
    """Return the name of a non-date, non-numeric column suitable for grouping."""
    for c in df.columns:
        if c in (date_col, value_col):
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        # categorical / string column
        return c
    return None


def _resolve_columns(
    df: pd.DataFrame,
    date_col: Optional[str],
    value_col: Optional[str],
) -> tuple[str, str]:
    """Auto-detect the date and numeric value columns when not supplied."""
    work = df

    if date_col is None:
        for c in work.columns:
            if pd.api.types.is_datetime64_any_dtype(work[c]):
                date_col = c
                break
        if date_col is None:
            for c in work.columns:
                try:
                    pd.to_datetime(work[c], errors="raise")
                    date_col = c
                    break
                except Exception:
                    continue
    if date_col is None:
        raise ValueError("Could not find a date/time column to forecast on.")

    if value_col is None:
        for c in work.columns:
            if c == date_col:
                continue
            if pd.api.types.is_numeric_dtype(work[c]):
                value_col = c
                break
    if value_col is None:
        raise ValueError("Could not find a numeric column to forecast.")

    return date_col, value_col


def _trim_groups(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    max_groups: int = _MAX_GROUPS,
) -> pd.DataFrame:
    """Keep top-N groups by total value, merge the rest into 'Other'."""
    totals = df.groupby(group_col)[value_col].sum().sort_values(ascending=False)
    if len(totals) <= max_groups:
        return df
    keep = set(totals.head(max_groups - 1).index)
    work = df.copy()
    work[group_col] = work[group_col].where(work[group_col].isin(keep), other="Other")
    return work


def _forecast_grouped(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    group_col: str,
    periods: int,
    horizon_label: str,
    horizon_unit: Optional[str] = None,
) -> ForecastResult:
    """Fit one model per group and combine into a single ForecastResult."""
    work = df[[date_col, group_col, value_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work = work.dropna().sort_values(date_col)

    # Sum any same-day duplicates within a group
    work = work.groupby([date_col, group_col], as_index=False)[value_col].sum()
    work = _trim_groups(work, group_col, value_col)

    # Use the union frequency from the most populous group for future dates
    pivot_dates = work.groupby(group_col)[date_col].apply(list)
    largest_group = work.groupby(group_col)[value_col].sum().idxmax()
    largest_dates = pd.Series(sorted(pivot_dates[largest_group]))
    freq, seasonal_period = _infer_frequency(largest_dates)

    history_frames: list[pd.DataFrame] = []
    forecast_frames: list[pd.DataFrame] = []
    methods_used: list[str] = []
    skipped: list[str] = []
    group_steps: dict[str, int] = {}

    for group_name, group_df in work.groupby(group_col):
        g = group_df.sort_values(date_col).reset_index(drop=True)
        if len(g) < 4:
            skipped.append(str(group_name))
            continue

        steps = _relative_periods(g[date_col].iloc[-1], freq, periods, horizon_unit)
        group_steps[str(group_name)] = steps
        point, lower, upper, method = _fit_and_forecast(
            g[value_col], periods=steps, seasonal_period=seasonal_period
        )
        future_idx = _future_dates(g[date_col].iloc[-1], freq, steps)

        hist_part = pd.DataFrame({
            "date": g[date_col].values,
            "value": g[value_col].values,
            "group": str(group_name),
        })
        fc_part = pd.DataFrame({
            "date": future_idx,
            "value": point,
            "lower": lower,
            "upper": upper,
            "group": str(group_name),
        })
        history_frames.append(hist_part)
        forecast_frames.append(fc_part)
        methods_used.append(method)

    if not forecast_frames:
        raise ValueError(
            "No group had enough history to forecast (need at least 4 points each)."
        )

    history_all = pd.concat(history_frames, ignore_index=True)
    forecast_all = pd.concat(forecast_frames, ignore_index=True)

    groups = sorted(forecast_all["group"].unique().tolist())

    primary_method = max(set(methods_used), key=methods_used.count)

    return ForecastResult(
        history=history_all,
        forecast=forecast_all,
        method=primary_method,
        meta={
            "freq": freq,
            "seasonal_period": seasonal_period,
            "periods": max(group_steps.values(), default=periods),
            "periods_by_group": group_steps,
            "groups": groups,
            "group_col": group_col,
            "skipped_groups": skipped,
            "horizon_label": horizon_label,
        },
    )


# ---------------------------------------------------------------------------
# Target window helpers
# ---------------------------------------------------------------------------

def _periods_to_reach(
    last_history_date: pd.Timestamp,
    target_end: pd.Timestamp,
    freq: str,
) -> int:
    """How many forecast steps (at `freq`) we need to cover up to target_end."""
    if target_end <= last_history_date:
        return 0
    # Approximate step length in days, conservatively sized so we always
    # reach past target_end.
    step_days = {"D": 1, "W": 7, "2W": 14, "MS": 31, "QS": 92, "YS": 366}.get(freq, 31)
    delta_days = (target_end - last_history_date).days
    return max(1, int(np.ceil(delta_days / step_days)) + 1)


def _clip_forecast_to_window(
    forecast_df: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Restrict forecast rows to [window_start, window_end].
    Returns only forecast rows within the requested window.
    """
    mask = (forecast_df["date"] >= window_start) & (forecast_df["date"] <= window_end)
    return forecast_df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _relative_periods(last_date, freq, amount, unit):
    """Translate a calendar horizon to steps at the actual series frequency."""
    if type(amount) is not int or amount < 1:
        raise ValueError("Forecast horizon must be a positive integer.")
    unit = (unit or 'periods').lower().rstrip('s')
    if unit == 'period':
        return amount
    offsets = {'day': ('days', 1), 'week': ('weeks', 1),
               'month': ('months', 1), 'quarter': ('months', 3), 'year': ('years', 1)}
    if unit not in offsets:
        raise ValueError(f"Unsupported forecast horizon unit: {unit}")
    field, multiplier = offsets[unit]
    start = pd.Timestamp(last_date)
    end = start + pd.DateOffset(**{field: amount * multiplier})
    # _periods_to_reach deliberately overestimates; clip to the calendar endpoint.
    candidates = _future_dates(start, freq, _periods_to_reach(start, end, freq))
    steps = int((candidates <= end).sum())
    if steps < 1:
        raise ValueError("The requested horizon is shorter than one data period; use finer-grained history.")
    return steps


def forecast_series(
    df: pd.DataFrame,
    periods: int = 12,
    horizon_label: Optional[str] = None,
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
    group_col: Optional[str] = None,
    target_window: Optional[tuple[pd.Timestamp, pd.Timestamp, str]] = None,
    horizon_unit: Optional[str] = None,
) -> ForecastResult:
    """
    Run a forecast on a dataframe.

    If the dataframe contains a categorical column (e.g. region, segment) in
    addition to a date and a numeric column — or one is passed via `group_col` —
    a separate model is fitted per group.

    Parameters
    ----------
    df : DataFrame with at least a date column and a numeric column.
    periods : number of future periods to forecast (used when no target_window).
    horizon_label : descriptive horizon label stored in result metadata.
    horizon_unit : calendar unit for periods (days/weeks/months/quarters/years).
                   Omit to preserve periods as native model steps.
    date_col, value_col : optional explicit column names; auto-detected.
    group_col : optional category column for per-group forecasting.
    target_window : optional (start_date, end_date, label) tuple. When supplied,
                    the forecaster auto-extends `periods` until it reaches
                    end_date, then clips the displayed forecast to that window.
    """
    if df is None or len(df) == 0:
        raise ValueError("Cannot forecast: empty dataset.")

    resolved_date, resolved_value = _resolve_columns(df, date_col, value_col)

    if group_col is None:
        group_col = _detect_category_column(df, resolved_date, resolved_value)

    # If a target window was supplied, extend the horizon to reach it.
    # (We need the data's last date and frequency for this, which means
    # doing a quick coerce up front.)
    if target_window is not None:
        target_start, target_end, target_label = target_window
        if group_col is not None:
            tmp = df[[resolved_date, resolved_value]].copy()
            tmp[resolved_date] = pd.to_datetime(tmp[resolved_date])
            last_hist = tmp[resolved_date].max()
            freq_probe, _ = _infer_frequency(tmp[resolved_date].drop_duplicates().sort_values())
        else:
            tmp_hist = _coerce_series(df, date_col=resolved_date, value_col=resolved_value)
            last_hist = tmp_hist["date"].max()
            freq_probe, _ = _infer_frequency(tmp_hist["date"])

        needed = _periods_to_reach(last_hist, target_end, freq_probe)
        # Use whichever is larger so we always cover the window
        periods = max(periods, needed) if needed > 0 else periods
        horizon_label = target_label
    else:
        target_start = target_end = target_label = None
        horizon_label = horizon_label or f"next {periods} periods"

    # Grouped path
    if group_col is not None:
        result = _forecast_grouped(
            df=df,
            date_col=resolved_date,
            value_col=resolved_value,
            group_col=group_col,
            periods=periods,
            horizon_label=horizon_label,
            horizon_unit=horizon_unit if target_window is None else None,
        )
        if target_window is not None:
            mask = (
                (result.forecast["date"] >= target_start)
                & (result.forecast["date"] <= target_end)
            )
            clipped = result.forecast[mask].reset_index(drop=True)
            result.forecast = clipped
            result.meta["target_window"] = target_label
        return result

    # Single-series path
    history = _coerce_series(df, date_col=resolved_date, value_col=resolved_value)
    if len(history) < 4:
        raise ValueError(
            f"Need at least 4 historical points to forecast (got {len(history)})."
        )

    freq, seasonal_period = _infer_frequency(history["date"])
    if target_window is None:
        periods = _relative_periods(history["date"].iloc[-1], freq, periods, horizon_unit)
    point, lower, upper, method = _fit_and_forecast(
        history["value"], periods=periods, seasonal_period=seasonal_period
    )
    future_index = _future_dates(history["date"].iloc[-1], freq, periods)
    forecast_df = pd.DataFrame(
        {"date": future_index, "value": point, "lower": lower, "upper": upper}
    )

    if target_window is not None:
        clipped = _clip_forecast_to_window(
            forecast_df, target_start, target_end
        )
        return ForecastResult(
            history=history,
            forecast=clipped,
            method=method,
            meta={"freq": freq, "seasonal_period": seasonal_period,
                  "periods": periods, "target_window": target_label},
        )

    return ForecastResult(
        history=history,
        forecast=forecast_df,
        method=method,
        meta={"freq": freq, "seasonal_period": seasonal_period, "periods": periods,
              "horizon_label": horizon_label},
    )
