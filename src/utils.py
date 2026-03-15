"""
Shared utility helpers — formatting, plotting templates, constants.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette (dark theme)
# ──────────────────────────────────────────────────────────────────────────────

COLORS = {
    "bg": "#0f1419",
    "card_bg": "#1a1f2e",
    "text": "#e6e6e6",
    "accent": "#00d4aa",
    "accent2": "#00b4d8",
    "red": "#ff4757",
    "green": "#2ed573",
    "orange": "#ffa502",
    "purple": "#a855f7",
    "grid": "#2a2f3e",
}

# ──────────────────────────────────────────────────────────────────────────────
# Plotly dark layout template
# ──────────────────────────────────────────────────────────────────────────────


def dark_layout(**overrides: Any) -> Dict[str, Any]:
    """Return a Plotly layout dict for the project's dark theme.

    Any keyword argument overrides the default value.
    """
    base = dict(
        template="plotly_dark",
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"], family="Inter, sans-serif"),
        xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    base.update(overrides)
    return base


def format_large_number(n: float | int | None) -> str:
    """Human-readable formatting for large numbers (e.g. market cap)."""
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "N/A"
    abs_n = abs(n)
    if abs_n >= 1e12:
        return f"${n / 1e12:.2f}T"
    if abs_n >= 1e9:
        return f"${n / 1e9:.2f}B"
    if abs_n >= 1e6:
        return f"${n / 1e6:.2f}M"
    if abs_n >= 1e3:
        return f"${n / 1e3:.1f}K"
    return f"${n:,.2f}"


def pct_color(value: float) -> str:
    """Return green hex for positive values, red for negative."""
    return COLORS["green"] if value >= 0 else COLORS["red"]
