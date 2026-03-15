"""
Feature engineering module — technical indicators and ML-ready features.

All indicators are computed from raw OHLCV data using pure pandas/numpy.
No external TA library is required.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Technical indicator functions
# ──────────────────────────────────────────────────────────────────────────────


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing).

    Parameters
    ----------
    series : pd.Series
        Price series (typically Close).
    window : int, default 14
        Look-back window.

    Returns
    -------
    pd.Series
        RSI values in [0, 100].
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD, Signal line, and Histogram.

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]
        (macd_line, signal_line, histogram)
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands.

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]
        (upper_band, middle_band, lower_band)
    """
    middle = sma(series, window)
    rolling_std = series.rolling(window=window, min_periods=window).std()
    upper = middle + num_std * rolling_std
    lower = middle - num_std * rolling_std
    return upper, middle, lower


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Average True Range.

    Parameters
    ----------
    high, low, close : pd.Series
        OHLC price series.
    window : int, default 14
        Smoothing window.

    Returns
    -------
    pd.Series
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Main feature builder
# ──────────────────────────────────────────────────────────────────────────────


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Augment an OHLCV DataFrame with a comprehensive set of technical indicators.

    The input DataFrame must contain at least the columns
    ``Open``, ``High``, ``Low``, ``Close``, ``Volume``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV data (index = Date).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with additional indicator columns appended.
    """
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # ── Moving Averages ───────────────────────────────────────────────────
    df["SMA_20"] = sma(close, 20)
    df["SMA_50"] = sma(close, 50)
    df["SMA_200"] = sma(close, 200)
    df["EMA_12"] = ema(close, 12)
    df["EMA_26"] = ema(close, 26)

    # ── RSI ───────────────────────────────────────────────────────────────
    df["RSI_14"] = rsi(close, 14)

    # ── MACD ──────────────────────────────────────────────────────────────
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = macd(close)

    # ── Bollinger Bands ───────────────────────────────────────────────────
    df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = bollinger_bands(close)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    df["BB_Pct"] = (close - df["BB_Lower"]) / (
        (df["BB_Upper"] - df["BB_Lower"]).replace(0, np.nan)
    )

    # ── ATR ───────────────────────────────────────────────────────────────
    df["ATR_14"] = atr(high, low, close, 14)

    # ── Volume ────────────────────────────────────────────────────────────
    df["Volume_SMA_20"] = sma(volume, 20)
    df["Volume_Ratio"] = volume / df["Volume_SMA_20"].replace(0, np.nan)

    # ── Momentum / Returns ────────────────────────────────────────────────
    df["Return_1d"] = close.pct_change(1)
    df["Return_5d"] = close.pct_change(5)
    df["Return_21d"] = close.pct_change(21)

    # ── Volatility ────────────────────────────────────────────────────────
    df["Volatility_21d"] = df["Return_1d"].rolling(21).std()

    # ── Price position relative to range ──────────────────────────────────
    df["High_Low_Pct"] = (high - low) / close
    df["Close_Open_Pct"] = (close - df["Open"]) / df["Open"]

    return df


# ──────────────────────────────────────────────────────────────────────────────
# ML-ready feature / target builder
# ──────────────────────────────────────────────────────────────────────────────


def prepare_ml_features(
    df: pd.DataFrame,
    target_horizon: int = 1,
    lag_periods: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Build ML-ready dataset with lagged features and target variables.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame **already containing** technical indicators
        (i.e., output of :func:`add_technical_indicators`).
    target_horizon : int, default 1
        Number of trading days ahead for the *classification* target.
        ``1`` means next-day.
    lag_periods : list[int], optional
        Which lag shifts to create for indicator columns.
        Defaults to ``[1, 2, 3, 5]``.

    Returns
    -------
    pd.DataFrame
        DataFrame with feature columns, lag features, and targets:

        * ``Target_Direction`` — 1 if future return > 0 else 0
        * ``Target_Return_1d`` — next-day return (regression target)
        * ``Target_Return_5d`` — next-5-day return (regression target)
        * ``Target_Price_1d`` — next-day close price
    """
    if lag_periods is None:
        lag_periods = [1, 2, 3, 5]

    df = df.copy()

    # ── Target variables (shifted BACK, so row t has the FUTURE value) ────
    df["Target_Return_1d"] = df["Close"].pct_change(1).shift(-1)
    df["Target_Return_5d"] = df["Close"].pct_change(5).shift(-5)
    df["Target_Price_1d"] = df["Close"].shift(-target_horizon)
    df["Target_Direction"] = (df["Target_Return_1d"] > 0).astype(int)

    # ── Lag features ──────────────────────────────────────────────────────
    indicator_cols = [
        "RSI_14",
        "MACD",
        "MACD_Hist",
        "BB_Pct",
        "ATR_14",
        "Volume_Ratio",
        "Return_1d",
        "Volatility_21d",
    ]
    for col in indicator_cols:
        if col not in df.columns:
            continue
        for lag in lag_periods:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the list of feature column names (excludes targets and raw OHLCV).

    Parameters
    ----------
    df : pd.DataFrame
        ML-ready DataFrame from :func:`prepare_ml_features`.

    Returns
    -------
    list[str]
    """
    exclude_prefixes = ("Target_",)
    raw_cols = {"Open", "High", "Low", "Close", "Adj Close", "Volume", "Date"}
    return [
        c
        for c in df.columns
        if c not in raw_cols and not any(c.startswith(p) for p in exclude_prefixes)
    ]


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Produce a simple rule-based signal summary for the most recent row.

    Returns a DataFrame with columns ``Indicator``, ``Value``, ``Signal``
    where *Signal* is one of ``Bullish``, ``Bearish``, ``Neutral``.
    """
    if df.empty:
        return pd.DataFrame(columns=["Indicator", "Value", "Signal"])

    last = df.iloc[-1]
    signals = []

    def _add(name: str, value, signal: str) -> None:
        signals.append({"Indicator": name, "Value": round(float(value), 4), "Signal": signal})

    # RSI
    if "RSI_14" in df.columns and pd.notna(last.get("RSI_14")):
        v = last["RSI_14"]
        sig = "Oversold (Bullish)" if v < 30 else ("Overbought (Bearish)" if v > 70 else "Neutral")
        _add("RSI (14)", v, sig)

    # MACD
    if "MACD_Hist" in df.columns and pd.notna(last.get("MACD_Hist")):
        v = last["MACD_Hist"]
        _add("MACD Histogram", v, "Bullish" if v > 0 else "Bearish")

    # Bollinger %B
    if "BB_Pct" in df.columns and pd.notna(last.get("BB_Pct")):
        v = last["BB_Pct"]
        sig = "Oversold (Bullish)" if v < 0.2 else ("Overbought (Bearish)" if v > 0.8 else "Neutral")
        _add("Bollinger %B", v, sig)

    # SMA cross-overs
    if all(c in df.columns for c in ["SMA_20", "SMA_50"]):
        if pd.notna(last.get("SMA_20")) and pd.notna(last.get("SMA_50")):
            sig = "Bullish" if last["SMA_20"] > last["SMA_50"] else "Bearish"
            _add("SMA 20/50 Cross", last["SMA_20"] - last["SMA_50"], sig)

    if all(c in df.columns for c in ["SMA_50", "SMA_200"]):
        if pd.notna(last.get("SMA_50")) and pd.notna(last.get("SMA_200")):
            sig = "Golden Cross (Bullish)" if last["SMA_50"] > last["SMA_200"] else "Death Cross (Bearish)"
            _add("SMA 50/200 Cross", last["SMA_50"] - last["SMA_200"], sig)

    # Price vs SMA 200
    if "SMA_200" in df.columns and pd.notna(last.get("SMA_200")):
        sig = "Bullish" if last["Close"] > last["SMA_200"] else "Bearish"
        _add("Price vs SMA 200", last["Close"] - last["SMA_200"], sig)

    # Momentum
    if "Return_21d" in df.columns and pd.notna(last.get("Return_21d")):
        v = last["Return_21d"]
        _add("21-Day Momentum", v, "Bullish" if v > 0 else "Bearish")

    # Volume
    if "Volume_Ratio" in df.columns and pd.notna(last.get("Volume_Ratio")):
        v = last["Volume_Ratio"]
        sig = "High Volume" if v > 1.5 else ("Low Volume" if v < 0.5 else "Normal")
        _add("Volume Ratio", v, sig)

    return pd.DataFrame(signals)
