"""
Simple backtesting engine for ML-based trading strategies.

Compares a signal-driven strategy against a passive buy-and-hold benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """Container for backtest outputs."""

    portfolio: pd.DataFrame
    total_return_strategy: float
    total_return_buyhold: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    summary: dict


def run_backtest(
    prices: pd.Series,
    predictions: np.ndarray,
    dates: pd.DatetimeIndex,
    initial_capital: float = 10_000.0,
    transaction_cost: float = 0.001,
) -> BacktestResult:
    """Execute a simple long-only backtest.

    Rules
    -----
    * If the model predicts **UP** (1), hold the stock (fully invested).
    * If the model predicts **DOWN** (0), hold cash.
    * Transaction costs are deducted on each position change.

    Parameters
    ----------
    prices : pd.Series
        Close prices aligned with *dates*.
    predictions : np.ndarray
        Binary predictions (1 = UP, 0 = DOWN) aligned with *dates*.
    dates : pd.DatetimeIndex
        Trading dates corresponding to each prediction.
    initial_capital : float, default 10_000
        Starting portfolio value.
    transaction_cost : float, default 0.001
        Proportional transaction cost per trade (0.1 %).

    Returns
    -------
    BacktestResult
    """
    # Align prices with prediction dates -----------------------------------
    price_df = prices.reindex(dates).dropna()
    valid_idx = price_df.index
    preds = pd.Series(predictions[: len(valid_idx)], index=valid_idx)

    daily_returns = price_df.pct_change().fillna(0)

    # Strategy returns: hold stock when pred==1, else 0 --------------------
    position = preds.shift(1).fillna(0)  # Use *previous* day's signal
    trade_signal = position.diff().abs().fillna(0)  # 1 when position changes
    costs = trade_signal * transaction_cost

    strategy_returns = position * daily_returns - costs
    buyhold_returns = daily_returns

    # Cumulative portfolio values ------------------------------------------
    strategy_value = initial_capital * (1 + strategy_returns).cumprod()
    buyhold_value = initial_capital * (1 + buyhold_returns).cumprod()

    portfolio = pd.DataFrame(
        {
            "Strategy": strategy_value,
            "Buy & Hold": buyhold_value,
            "Position": position,
            "Daily_Return_Strategy": strategy_returns,
            "Daily_Return_BuyHold": buyhold_returns,
        },
        index=valid_idx,
    )

    # ── Metrics ───────────────────────────────────────────────────────────
    total_return_strategy = float((strategy_value.iloc[-1] / initial_capital) - 1) if len(strategy_value) > 0 else 0.0
    total_return_buyhold = float((buyhold_value.iloc[-1] / initial_capital) - 1) if len(buyhold_value) > 0 else 0.0

    # Sharpe ratio (annualised, assuming 252 trading days)
    excess = strategy_returns
    sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0

    # Max drawdown
    cummax = strategy_value.cummax()
    drawdown = (strategy_value - cummax) / cummax
    max_dd = float(drawdown.min())

    # Win rate (fraction of positive daily returns when invested)
    invested_returns = strategy_returns[position == 1]
    win_rate = float((invested_returns > 0).mean()) if len(invested_returns) > 0 else 0.0

    total_trades = int(trade_signal.sum())

    summary = {
        "Initial Capital": f"${initial_capital:,.2f}",
        "Final Value (Strategy)": f"${strategy_value.iloc[-1]:,.2f}" if len(strategy_value) else "N/A",
        "Final Value (Buy & Hold)": f"${buyhold_value.iloc[-1]:,.2f}" if len(buyhold_value) else "N/A",
        "Total Return (Strategy)": f"{total_return_strategy:.2%}",
        "Total Return (Buy & Hold)": f"{total_return_buyhold:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_dd:.2%}",
        "Win Rate": f"{win_rate:.2%}",
        "Total Trades": total_trades,
    }

    return BacktestResult(
        portfolio=portfolio,
        total_return_strategy=total_return_strategy,
        total_return_buyhold=total_return_buyhold,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        total_trades=total_trades,
        summary=summary,
    )
