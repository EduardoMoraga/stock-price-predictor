"""
Data fetching module using yfinance.

Provides robust stock data retrieval with caching, validation,
and comprehensive error handling.  No API key required.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_YEARS = 5

POPULAR_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA",
    "META", "NVDA", "JPM", "V", "JNJ",
    "WMT", "PG", "UNH", "HD", "DIS",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cache_key(ticker: str, start: str, end: str) -> str:
    """Return a deterministic filename for a given query."""
    raw = f"{ticker}_{start}_{end}"
    hsh = hashlib.md5(raw.encode()).hexdigest()[:10]
    return f"{ticker}_{hsh}.parquet"


def validate_ticker(ticker: str) -> bool:
    """Check whether *ticker* corresponds to a real instrument on Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. ``"AAPL"``).

    Returns
    -------
    bool
        ``True`` when the ticker is valid; ``False`` otherwise.
    """
    try:
        info = yf.Ticker(ticker).info
        # yfinance returns a dict even for invalid tickers, but some keys
        # will be missing or set to None.
        return info is not None and info.get("regularMarketPrice") is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main fetcher
# ---------------------------------------------------------------------------


def fetch_stock_data(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    years: int = DEFAULT_YEARS,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch historical OHLCV data for *ticker* from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. ``"AAPL"``).
    start : str, optional
        Start date in ``YYYY-MM-DD`` format.  Defaults to *years* before *end*.
    end : str, optional
        End date in ``YYYY-MM-DD`` format.  Defaults to today.
    years : int, default 5
        Lookback window used when *start* is ``None``.
    use_cache : bool, default True
        When ``True``, results are cached as Parquet files under ``data/``.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``Date`` with columns:
        ``Open``, ``High``, ``Low``, ``Close``, ``Adj Close``, ``Volume``.

    Raises
    ------
    ValueError
        If the ticker is invalid or no data is returned.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker symbol cannot be empty.")

    # Default date range --------------------------------------------------
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    if start is None:
        start = (datetime.today() - timedelta(days=365 * years)).strftime("%Y-%m-%d")

    # Cache lookup --------------------------------------------------------
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / _cache_key(ticker, start, end)

    if use_cache and cache_file.exists():
        logger.info("Loading cached data for %s from %s", ticker, cache_file)
        df = pd.read_parquet(cache_file)
        if not df.empty:
            return df

    # Fetch from yfinance -------------------------------------------------
    logger.info("Downloading %s data from %s to %s", ticker, start, end)
    try:
        df: pd.DataFrame = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
        )
    except Exception as exc:
        raise ValueError(
            f"Failed to download data for '{ticker}': {exc}"
        ) from exc

    if df is None or df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}'. "
            "Please verify the symbol is valid."
        )

    # yfinance may return MultiIndex columns when downloading a single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure expected columns are present ---------------------------------
    expected = {"Open", "High", "Low", "Close", "Volume"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"Downloaded data for '{ticker}' is missing columns: {missing}"
        )

    # Add Adj Close if missing (auto_adjust=False should include it)
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    # Sort by date (ascending) and drop any rows with all-NaN prices ------
    df = df.sort_index()
    df = df.dropna(subset=["Close"])

    # Persist to cache ----------------------------------------------------
    if use_cache:
        try:
            df.to_parquet(cache_file)
            logger.info("Cached data saved to %s", cache_file)
        except Exception as exc:
            logger.warning("Could not write cache file: %s", exc)

    return df


def get_ticker_info(ticker: str) -> dict:
    """Return summary information about a ticker (name, sector, etc.).

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.

    Returns
    -------
    dict
        Dictionary with keys such as ``shortName``, ``sector``, ``industry``,
        ``marketCap``, ``currentPrice``.
    """
    ticker = ticker.strip().upper()
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}

    return {
        "shortName": info.get("shortName", ticker),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "marketCap": info.get("marketCap"),
        "currentPrice": info.get("regularMarketPrice"),
        "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
        "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
        "currency": info.get("currency", "USD"),
    }
