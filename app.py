"""
Stock Price Predictor — Interactive Streamlit Application.

A Bloomberg-terminal-inspired dashboard combining real-time stock data,
technical analysis, ML-based direction/price prediction, and backtesting.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.backtester import run_backtest
from src.data_fetcher import fetch_stock_data, get_ticker_info
from src.feature_engineering import (
    add_technical_indicators,
    generate_signals,
    get_feature_columns,
    prepare_ml_features,
)
from src.models import (
    train_gradient_boosting_classifier,
    train_linear_regression,
    train_random_forest_classifier,
    train_random_forest_regressor,
    walk_forward_validation,
)
from src.utils import COLORS, dark_layout, format_large_number, pct_color

# ══════════════════════════════════════════════════════════════════════════════
# Page config & custom CSS
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = f"""
<style>
    /* ── Global ───────────────────────────────────────── */
    .stApp {{
        background-color: {COLORS['bg']};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {COLORS['card_bg']};
    }}

    /* ── Metric cards ─────────────────────────────────── */
    .metric-card {{
        background: {COLORS['card_bg']};
        border: 1px solid {COLORS['grid']};
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }}
    .metric-card h3 {{
        color: {COLORS['accent']};
        margin: 0 0 6px 0;
        font-size: 1.6rem;
    }}
    .metric-card p {{
        color: {COLORS['text']};
        margin: 0;
        font-size: 0.85rem;
        opacity: 0.7;
    }}

    /* ── Tabs ─────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {COLORS['card_bg']};
        border-radius: 6px;
        color: {COLORS['text']};
        padding: 8px 16px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {COLORS['accent']} !important;
        color: {COLORS['bg']} !important;
    }}

    /* ── Footer ───────────────────────────────────────── */
    .footer {{
        text-align: center;
        padding: 30px 0 10px 0;
        color: {COLORS['text']};
        opacity: 0.5;
        font-size: 0.8rem;
    }}
    .footer a {{
        color: {COLORS['accent']};
        text-decoration: none;
    }}

    /* ── Disclaimer ───────────────────────────────────── */
    .disclaimer {{
        background: {COLORS['card_bg']};
        border-left: 4px solid {COLORS['orange']};
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        font-size: 0.8rem;
        color: {COLORS['text']};
        opacity: 0.8;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Cached helpers
# ══════════════════════════════════════════════════════════════════════════════


@st.cache_data(show_spinner="Fetching stock data...", ttl=3600)
def _fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    return fetch_stock_data(ticker, start=start, end=end, use_cache=True)


@st.cache_data(show_spinner="Loading ticker info...", ttl=3600)
def _info(ticker: str) -> dict:
    return get_ticker_info(ticker)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f"## <span style='color:{COLORS['accent']}'>Stock Price Predictor</span>", unsafe_allow_html=True)
    st.markdown("ML-powered stock analysis")
    st.markdown("---")

    ticker = st.text_input("Ticker Symbol", value="AAPL").strip().upper()

    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input("Start", value=dt.date.today() - dt.timedelta(days=5 * 365))
    with col_e:
        end_date = st.date_input("End", value=dt.date.today())

    st.markdown("---")
    classifier_choice = st.selectbox(
        "Classification Model",
        ["Random Forest", "Gradient Boosting"],
    )
    regressor_choice = st.selectbox(
        "Regression Model",
        ["Random Forest Regressor", "Linear Regression"],
    )
    test_pct = st.slider("Test Set %", 10, 40, 20, 5)

    st.markdown("---")
    st.markdown(
        '<div class="disclaimer">'
        "<strong>Disclaimer:</strong> This tool is for educational and research "
        "purposes only. It does <strong>not</strong> constitute financial advice. "
        "Past performance does not guarantee future results."
        "</div>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════════

try:
    raw_df = _fetch(ticker, str(start_date), str(end_date))
except Exception as exc:
    st.error(f"Could not fetch data for **{ticker}**: {exc}")
    st.stop()

if raw_df.empty:
    st.warning("No data returned. Please check the ticker and date range.")
    st.stop()

info = _info(ticker)

# Feature engineering
df_ind = add_technical_indicators(raw_df)
df_ml = prepare_ml_features(df_ind)
feature_cols = get_feature_columns(df_ml)

# ══════════════════════════════════════════════════════════════════════════════
# Header metrics
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"# {info.get('shortName', ticker)}  <span style='color:{COLORS['accent']}'>({ticker})</span>", unsafe_allow_html=True)

last_close = float(raw_df["Close"].iloc[-1])
prev_close = float(raw_df["Close"].iloc[-2]) if len(raw_df) > 1 else last_close
change = last_close - prev_close
change_pct = change / prev_close if prev_close else 0

m1, m2, m3, m4, m5 = st.columns(5)

def _metric_card(label: str, value: str) -> str:
    return f'<div class="metric-card"><h3>{value}</h3><p>{label}</p></div>'

m1.markdown(_metric_card("Last Close", f"${last_close:,.2f}"), unsafe_allow_html=True)
m2.markdown(
    _metric_card("Change", f"<span style='color:{pct_color(change)}'>{change:+.2f} ({change_pct:+.2%})</span>"),
    unsafe_allow_html=True,
)
m3.markdown(_metric_card("Market Cap", format_large_number(info.get("marketCap"))), unsafe_allow_html=True)
m4.markdown(_metric_card("Sector", info.get("sector", "N/A")), unsafe_allow_html=True)
m5.markdown(_metric_card("Data Points", f"{len(raw_df):,}"), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════

tab_price, tab_ml, tab_forecast, tab_backtest, tab_ta = st.tabs(
    ["Price & Indicators", "ML Predictions", "Price Forecast", "Backtest Results", "Technical Analysis"]
)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — Price & Indicators
# ──────────────────────────────────────────────────────────────────────────────
with tab_price:
    # Overlay toggles
    ov_col1, ov_col2, ov_col3, ov_col4 = st.columns(4)
    show_sma = ov_col1.checkbox("SMA (20/50/200)", True)
    show_ema = ov_col2.checkbox("EMA (12/26)", False)
    show_bb = ov_col3.checkbox("Bollinger Bands", False)
    show_vol = ov_col4.checkbox("Volume", True)

    n_rows = 2 + int(show_vol)
    heights = [0.5, 0.25, 0.25] if show_vol else [0.65, 0.35]
    subplot_titles = ["Price", "RSI"] + (["Volume"] if show_vol else [])

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=heights,
        subplot_titles=subplot_titles,
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=raw_df.index,
            open=raw_df["Open"],
            high=raw_df["High"],
            low=raw_df["Low"],
            close=raw_df["Close"],
            name="OHLC",
            increasing_line_color=COLORS["green"],
            decreasing_line_color=COLORS["red"],
        ),
        row=1,
        col=1,
    )

    if show_sma:
        for col, color in [("SMA_20", COLORS["accent"]), ("SMA_50", COLORS["accent2"]), ("SMA_200", COLORS["orange"])]:
            if col in df_ind.columns:
                fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind[col], name=col, line=dict(width=1, color=color)), row=1, col=1)

    if show_ema:
        for col, color in [("EMA_12", COLORS["purple"]), ("EMA_26", "#e84393")]:
            if col in df_ind.columns:
                fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind[col], name=col, line=dict(width=1, color=color)), row=1, col=1)

    if show_bb:
        if "BB_Upper" in df_ind.columns:
            fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["BB_Upper"], name="BB Upper", line=dict(width=1, dash="dot", color=COLORS["accent"])), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["BB_Lower"], name="BB Lower", line=dict(width=1, dash="dot", color=COLORS["accent"]), fill="tonexty", fillcolor="rgba(0,212,170,0.08)"), row=1, col=1)

    # RSI subplot
    if "RSI_14" in df_ind.columns:
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["RSI_14"], name="RSI 14", line=dict(color=COLORS["accent"], width=1)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color=COLORS["red"], opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=COLORS["green"], opacity=0.5, row=2, col=1)

    # Volume subplot
    if show_vol:
        vol_colors = [COLORS["green"] if c >= o else COLORS["red"] for c, o in zip(raw_df["Close"], raw_df["Open"])]
        fig.add_trace(go.Bar(x=raw_df.index, y=raw_df["Volume"], name="Volume", marker_color=vol_colors, opacity=0.6), row=3, col=1)

    fig.update_layout(**dark_layout(height=700, showlegend=True, xaxis_rangeslider_visible=False, title_text=f"{ticker} — Price & Indicators"))
    st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — ML Predictions (classification)
# ──────────────────────────────────────────────────────────────────────────────
with tab_ml:
    st.subheader("Direction Prediction (Up / Down)")
    test_ratio = test_pct / 100

    with st.spinner("Training classification model..."):
        try:
            if classifier_choice == "Random Forest":
                cls_result = train_random_forest_classifier(df_ml, feature_cols, test_ratio=test_ratio)
            else:
                cls_result = train_gradient_boosting_classifier(df_ml, feature_cols, test_ratio=test_ratio)
        except Exception as exc:
            st.error(f"Model training failed: {exc}")
            st.stop()

    # Metrics row
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Accuracy", f"{cls_result.metrics['accuracy']:.2%}")
    mc2.metric("Precision", f"{cls_result.metrics['precision']:.2%}")
    mc3.metric("Recall", f"{cls_result.metrics['recall']:.2%}")
    mc4.metric("F1 Score", f"{cls_result.metrics['f1']:.2%}")

    col_left, col_right = st.columns(2)

    # Confusion matrix
    with col_left:
        st.markdown("#### Confusion Matrix")
        cm = cls_result.confusion
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Pred Down", "Pred Up"],
                y=["Actual Down", "Actual Up"],
                text=cm,
                texttemplate="%{text}",
                colorscale=[[0, COLORS["bg"]], [1, COLORS["accent"]]],
                showscale=False,
            )
        )
        fig_cm.update_layout(**dark_layout(height=350, title_text="Confusion Matrix"))
        st.plotly_chart(fig_cm, use_container_width=True)

    # Feature importance
    with col_right:
        st.markdown("#### Feature Importance")
        if cls_result.feature_importances is not None:
            top_fi = cls_result.feature_importances.head(15)
            fig_fi = go.Figure(
                go.Bar(
                    x=top_fi.values[::-1],
                    y=top_fi.index[::-1],
                    orientation="h",
                    marker_color=COLORS["accent"],
                )
            )
            fig_fi.update_layout(**dark_layout(height=350, title_text="Top 15 Features"))
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Feature importances not available for this model.")

    # Predictions vs actual
    st.markdown("#### Predictions vs Actual (Test Set)")
    pred_df = pd.DataFrame(
        {"Actual": cls_result.y_test, "Predicted": cls_result.predictions},
        index=cls_result.test_dates,
    )
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=pred_df.index, y=pred_df["Actual"], name="Actual", mode="lines", line=dict(color=COLORS["accent"])))
    fig_pred.add_trace(go.Scatter(x=pred_df.index, y=pred_df["Predicted"], name="Predicted", mode="lines", line=dict(color=COLORS["orange"], dash="dot")))
    fig_pred.update_layout(**dark_layout(height=300, title_text="Direction: Actual vs Predicted", yaxis_title="Direction (1=Up, 0=Down)"))
    st.plotly_chart(fig_pred, use_container_width=True)

    with st.expander("Classification Report"):
        st.code(cls_result.classification_report_text)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — Price Forecast (regression)
# ──────────────────────────────────────────────────────────────────────────────
with tab_forecast:
    st.subheader("Price Prediction (Regression)")

    with st.spinner("Training regression model..."):
        try:
            if regressor_choice == "Linear Regression":
                reg_result = train_linear_regression(df_ml, feature_cols, test_ratio=test_ratio)
            else:
                reg_result = train_random_forest_regressor(df_ml, feature_cols, test_ratio=test_ratio)
        except Exception as exc:
            st.error(f"Regression model failed: {exc}")
            st.stop()

    mr1, mr2, mr3, mr4 = st.columns(4)
    mr1.metric("RMSE", f"${reg_result.metrics['rmse']:.2f}")
    mr2.metric("MAE", f"${reg_result.metrics['mae']:.2f}")
    mr3.metric("R\u00b2", f"{reg_result.metrics['r2']:.4f}")
    mr4.metric("MAPE", f"{reg_result.metrics['mape']:.2%}")

    # Predicted vs actual price
    reg_df = pd.DataFrame(
        {"Actual": reg_result.y_test, "Predicted": reg_result.predictions},
        index=reg_result.test_dates,
    )

    # Confidence band (simple: +/- 1 RMSE)
    rmse = reg_result.metrics["rmse"]
    reg_df["Upper"] = reg_df["Predicted"] + rmse
    reg_df["Lower"] = reg_df["Predicted"] - rmse

    fig_reg = go.Figure()
    fig_reg.add_trace(go.Scatter(x=reg_df.index, y=reg_df["Upper"], mode="lines", line=dict(width=0), showlegend=False))
    fig_reg.add_trace(go.Scatter(x=reg_df.index, y=reg_df["Lower"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(0,212,170,0.15)", name="Confidence Band"))
    fig_reg.add_trace(go.Scatter(x=reg_df.index, y=reg_df["Actual"], name="Actual Price", line=dict(color=COLORS["accent"], width=2)))
    fig_reg.add_trace(go.Scatter(x=reg_df.index, y=reg_df["Predicted"], name="Predicted Price", line=dict(color=COLORS["orange"], width=2, dash="dot")))
    fig_reg.update_layout(**dark_layout(height=450, title_text=f"{ticker} — Predicted vs Actual Price", yaxis_title="Price ($)"))
    st.plotly_chart(fig_reg, use_container_width=True)

    # Residuals
    residuals = reg_df["Actual"] - reg_df["Predicted"]
    fig_res = go.Figure(go.Histogram(x=residuals, nbinsx=50, marker_color=COLORS["accent"], opacity=0.7))
    fig_res.update_layout(**dark_layout(height=300, title_text="Residual Distribution", xaxis_title="Residual ($)", yaxis_title="Count"))
    st.plotly_chart(fig_res, use_container_width=True)

    if reg_result.feature_importances is not None:
        st.markdown("#### Feature Importance (Regression)")
        top_fi_r = reg_result.feature_importances.head(15)
        fig_fi_r = go.Figure(go.Bar(x=top_fi_r.values[::-1], y=top_fi_r.index[::-1], orientation="h", marker_color=COLORS["accent2"]))
        fig_fi_r.update_layout(**dark_layout(height=350, title_text="Top 15 Features"))
        st.plotly_chart(fig_fi_r, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — Backtest Results
# ──────────────────────────────────────────────────────────────────────────────
with tab_backtest:
    st.subheader("Backtest: ML Strategy vs Buy & Hold")

    with st.spinner("Running backtest..."):
        try:
            bt = run_backtest(
                prices=raw_df["Close"].reindex(cls_result.test_dates).dropna(),
                predictions=cls_result.predictions,
                dates=cls_result.test_dates,
            )
        except Exception as exc:
            st.error(f"Backtest failed: {exc}")
            st.stop()

    b1, b2, b3, b4 = st.columns(4)
    b1.markdown(
        _metric_card("Strategy Return", f"<span style='color:{pct_color(bt.total_return_strategy)}'>{bt.total_return_strategy:.2%}</span>"),
        unsafe_allow_html=True,
    )
    b2.markdown(
        _metric_card("Buy & Hold Return", f"<span style='color:{pct_color(bt.total_return_buyhold)}'>{bt.total_return_buyhold:.2%}</span>"),
        unsafe_allow_html=True,
    )
    b3.markdown(_metric_card("Sharpe Ratio", f"{bt.sharpe_ratio:.2f}"), unsafe_allow_html=True)
    b4.markdown(_metric_card("Max Drawdown", f"<span style='color:{COLORS['red']}'>{bt.max_drawdown:.2%}</span>"), unsafe_allow_html=True)

    b5, b6, _, _ = st.columns(4)
    b5.markdown(_metric_card("Win Rate", f"{bt.win_rate:.2%}"), unsafe_allow_html=True)
    b6.markdown(_metric_card("Total Trades", f"{bt.total_trades}"), unsafe_allow_html=True)

    # Portfolio value chart
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=bt.portfolio.index, y=bt.portfolio["Strategy"], name="ML Strategy", line=dict(color=COLORS["accent"], width=2)))
    fig_bt.add_trace(go.Scatter(x=bt.portfolio.index, y=bt.portfolio["Buy & Hold"], name="Buy & Hold", line=dict(color=COLORS["orange"], width=2, dash="dash")))
    fig_bt.update_layout(**dark_layout(height=450, title_text="Portfolio Value Over Time", yaxis_title="Value ($)"))
    st.plotly_chart(fig_bt, use_container_width=True)

    with st.expander("Backtest Summary Table"):
        st.table(pd.DataFrame(bt.summary.items(), columns=["Metric", "Value"]))

# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 — Technical Analysis Dashboard
# ──────────────────────────────────────────────────────────────────────────────
with tab_ta:
    st.subheader("Technical Analysis Dashboard")

    # Signal summary
    signals_df = generate_signals(df_ind)
    if not signals_df.empty:
        bullish = signals_df["Signal"].str.contains("Bullish", case=False).sum()
        bearish = signals_df["Signal"].str.contains("Bearish", case=False).sum()
        neutral = len(signals_df) - bullish - bearish

        s1, s2, s3 = st.columns(3)
        s1.markdown(_metric_card("Bullish Signals", f"<span style='color:{COLORS['green']}'>{bullish}</span>"), unsafe_allow_html=True)
        s2.markdown(_metric_card("Bearish Signals", f"<span style='color:{COLORS['red']}'>{bearish}</span>"), unsafe_allow_html=True)
        s3.markdown(_metric_card("Neutral Signals", f"{neutral}"), unsafe_allow_html=True)

        st.markdown("#### Signal Details")
        st.dataframe(signals_df, use_container_width=True, hide_index=True)

    # MACD chart
    st.markdown("#### MACD")
    fig_macd = make_subplots(rows=1, cols=1)
    if "MACD" in df_ind.columns:
        fig_macd.add_trace(go.Scatter(x=df_ind.index, y=df_ind["MACD"], name="MACD", line=dict(color=COLORS["accent"])))
        fig_macd.add_trace(go.Scatter(x=df_ind.index, y=df_ind["MACD_Signal"], name="Signal", line=dict(color=COLORS["orange"])))
        hist_colors = [COLORS["green"] if v >= 0 else COLORS["red"] for v in df_ind["MACD_Hist"].fillna(0)]
        fig_macd.add_trace(go.Bar(x=df_ind.index, y=df_ind["MACD_Hist"], name="Histogram", marker_color=hist_colors, opacity=0.5))
    fig_macd.update_layout(**dark_layout(height=300, title_text="MACD"))
    st.plotly_chart(fig_macd, use_container_width=True)

    # Bollinger Bands
    st.markdown("#### Bollinger Bands")
    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(x=df_ind.index, y=df_ind["BB_Upper"], name="Upper", line=dict(color=COLORS["accent"], dash="dot", width=1)))
    fig_bb.add_trace(go.Scatter(x=df_ind.index, y=df_ind["BB_Lower"], name="Lower", line=dict(color=COLORS["accent"], dash="dot", width=1), fill="tonexty", fillcolor="rgba(0,212,170,0.08)"))
    fig_bb.add_trace(go.Scatter(x=df_ind.index, y=df_ind["Close"], name="Close", line=dict(color=COLORS["text"], width=1)))
    fig_bb.update_layout(**dark_layout(height=350, title_text="Bollinger Bands"))
    st.plotly_chart(fig_bb, use_container_width=True)

    # ATR
    st.markdown("#### ATR & Volatility")
    col_atr, col_vol = st.columns(2)
    with col_atr:
        fig_atr = go.Figure(go.Scatter(x=df_ind.index, y=df_ind["ATR_14"], name="ATR 14", line=dict(color=COLORS["accent2"])))
        fig_atr.update_layout(**dark_layout(height=250, title_text="Average True Range (14)"))
        st.plotly_chart(fig_atr, use_container_width=True)
    with col_vol:
        fig_vol = go.Figure(go.Scatter(x=df_ind.index, y=df_ind["Volatility_21d"], name="Volatility 21d", line=dict(color=COLORS["purple"])))
        fig_vol.update_layout(**dark_layout(height=250, title_text="21-Day Rolling Volatility"))
        st.plotly_chart(fig_vol, use_container_width=True)

    # Walk-forward validation
    st.markdown("#### Walk-Forward Validation")
    with st.spinner("Running walk-forward validation..."):
        wf = walk_forward_validation(df_ml, feature_cols, n_splits=5)
    if wf.accuracies:
        st.metric("Mean Walk-Forward Accuracy", f"{wf.mean_accuracy:.2%}")
        fig_wf = go.Figure(go.Bar(x=[f"Fold {i+1}" for i in range(len(wf.accuracies))], y=wf.accuracies, marker_color=COLORS["accent"]))
        fig_wf.add_hline(y=0.5, line_dash="dash", line_color=COLORS["red"], annotation_text="Random baseline")
        fig_wf.update_layout(**dark_layout(height=300, title_text="Walk-Forward Accuracy by Fold", yaxis_title="Accuracy"))
        st.plotly_chart(fig_wf, use_container_width=True)
    else:
        st.info("Not enough data for walk-forward validation with current settings.")

# ══════════════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    '<div class="footer">'
    'Built by <strong>Eduardo Moraga</strong> | '
    '<a href="https://eduardomoraga.github.io" target="_blank">eduardomoraga.github.io</a>'
    "</div>",
    unsafe_allow_html=True,
)
