"""Nifty 50 next-day close predictor (Streamlit app).

Run:  streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nifty import data, model, strategy

st.set_page_config(page_title="Nifty 50 Predictor", page_icon="📈", layout="wide")

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9aa0a6"


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    period = st.selectbox("History to train on", ["5y", "10y", "15y", "max"], index=1)
    model_name = st.selectbox("Model", list(model.MODELS), index=1)
    test_days = st.slider("Backtest window (trading days)", 60, 500, 250, step=10)
    use_sp500 = st.toggle(
        "Use S&P 500 overnight move",
        value=True,
        help="The US market closes after NSE, so the S&P 500's latest move is "
        "known before Nifty opens the next day.",
    )
    st.divider()
    uploaded = st.file_uploader(
        "Or use your own CSV",
        type="csv",
        help="Yahoo format (Date,Open,High,Low,Close,Volume) or "
        "Investing.com format (Date,Price,Open,High,Low,Vol.,Change %).",
    )
    if st.button("Refresh live data", width="stretch"):
        st.cache_data.clear()
        st.rerun()


# ---------------------------------------------------------------- data
@st.cache_data(ttl=300, show_spinner="Fetching Nifty 50 from Yahoo Finance…")
def live_nifty(period: str) -> pd.DataFrame:
    return data.fetch_yahoo("^NSEI", period)


@st.cache_data(ttl=300, show_spinner="Fetching S&P 500…")
def live_sp500(period: str) -> pd.DataFrame:
    return data.fetch_sp500(period)


@st.cache_data(show_spinner="Training the model and running the backtest…")
def forecast(nifty: pd.DataFrame, model_name: str, test_days: int, sp500: pd.DataFrame | None):
    return model.run(nifty, model_name, test_days, sp500)


if uploaded is not None:
    nifty = data.load_csv(uploaded)
    source = f"your file **{uploaded.name}**"
    live = False
else:
    error = "no rows returned"
    try:
        nifty = live_nifty(period)
    except Exception as e:  # network error, rate limit
        nifty, error = pd.DataFrame(), str(e)
    live = not nifty.empty
    source = "Yahoo Finance (**^NSEI**), live"
    if not live:
        st.warning(
            "Couldn't reach Yahoo Finance, so this uses the bundled history file "
            f"(ends {data.load_csv().index[-1]:%d %b %Y}). Forecasts won't be current. ({error})"
        )
        nifty = data.last_years(data.load_csv(), period)
        source = "the bundled CSV (offline)"

if nifty.empty:
    st.error("No price data. Upload a CSV in the sidebar.")
    st.stop()

sp500 = None
if use_sp500:
    try:
        sp500 = live_sp500(period)
    except Exception:
        sp500 = None
    if sp500 is None or sp500.empty or sp500.index[-1] < nifty.index[-1] - pd.Timedelta(days=5):
        st.info("S&P 500 data isn't available for these dates, so the model runs without it.")
        sp500 = None

try:
    result = forecast(nifty, model_name, test_days, sp500)
except ValueError as e:
    st.error(str(e))
    st.stop()

bt, m = result.backtest, result.metrics


# ---------------------------------------------------------------- header
st.title("📈 Nifty 50 next-day close")
st.caption(f"Data from {source}. Last daily bar: {result.last_date:%a %d %b %Y}.")

change = result.next_close - result.last_close
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    f"Predicted close, {result.next_date:%a %d %b}",
    f"₹{result.next_close:,.2f}",
    f"{change:+,.2f} ({change / result.last_close:+.2%})",
)
c2.metric(f"Last close, {result.last_date:%d %b}", f"₹{result.last_close:,.2f}")
c3.metric(
    "Average error (backtest)",
    f"₹{m['MAE']:,.1f}",
    f"{m['Better than naive %']:+.1f}% vs 'no change' guess",
    help=f"The 'no change' guess (tomorrow closes where today did) is off by ₹{m['Naive MAE']:,.1f} on average.",
)
c4.metric(
    "Direction right",
    f"{m['Direction accuracy %']:.1f}%",
    help=f"Share of days the model got up/down right. {m['Up days %']:.0f}% of days were up days.",
)

tab_forecast, tab_strategy, tab_details = st.tabs(["Forecast", "Strategy", "Data & model"])


def _layout(fig: go.Figure, height: int = 420, **kw) -> go.Figure:
    style = dict(
        height=height, hovermode="x unified", margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_layout(**{**style, **kw})
    return fig


# ---------------------------------------------------------------- forecast tab
with tab_forecast:
    st.subheader("Predicted vs actual close")
    fig = go.Figure()
    fig.add_scatter(x=bt["Date"], y=bt["Actual"], name="Actual", line=dict(color=BLUE, width=2))
    fig.add_scatter(x=bt["Date"], y=bt["Predicted"], name="Predicted", line=dict(color=ORANGE, width=1.5))
    fig.add_scatter(
        x=[result.last_date, result.next_date], y=[result.last_close, result.next_close],
        name="Next-day forecast", mode="lines+markers",
        line=dict(color=ORANGE, dash="dot"), marker=dict(size=[0, 10]),
    )
    st.plotly_chart(_layout(fig, yaxis=dict(title="₹", tickformat=",.0f")), width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("**Error by day** (predicted minus actual)")
        err = bt["Predicted"] - bt["Actual"]
        fig_err = go.Figure(go.Bar(
            x=bt["Date"], y=err, marker_color=[ORANGE if e > 0 else BLUE for e in err],
            hovertemplate="%{x|%d %b %Y}: ₹%{y:+,.1f}<extra></extra>",
        ))
        st.plotly_chart(_layout(fig_err, 320, yaxis_title="₹"), width="stretch")
    with right:
        st.markdown("**Predicted vs actual** (dots on the line are perfect)")
        lo, hi = bt[["Actual", "Predicted"]].min().min(), bt[["Actual", "Predicted"]].max().max()
        fig_sc = go.Figure()
        fig_sc.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=GREY, dash="dash"), showlegend=False)
        fig_sc.add_scatter(
            x=bt["Actual"], y=bt["Predicted"], mode="markers", marker=dict(color=ORANGE, size=5, opacity=0.6),
            showlegend=False, hovertemplate="Actual ₹%{x:,.0f}<br>Predicted ₹%{y:,.0f}<extra></extra>",
        )
        st.plotly_chart(_layout(fig_sc, 320, xaxis_title="Actual ₹", yaxis_title="Predicted ₹", hovermode="closest"), width="stretch")

    st.caption(
        f"Walk-forward backtest over the last {len(bt)} trading days: the model is retrained "
        f"every {model.RETRAIN_EVERY} days using only earlier data, so it never sees the answer."
    )


# ---------------------------------------------------------------- strategy tab
with tab_strategy:
    st.subheader("Trade the predictions vs buy and hold")
    s1, s2, s3, s4 = st.columns(4)
    signal = s1.selectbox(
        "Signal",
        ["Nifty only (trade at close)", "With S&P 500 (trade at next open)"],
        disabled=sp500 is None,
        help="The S&P 500 closes after NSE, so that signal can only be acted on at the "
        "next morning's open. The Nifty-only signal is known at NSE's close.",
    )
    positions = s2.selectbox("Positions", ["Long or cash", "Long or short"])
    threshold = s3.number_input(
        "Min. predicted move (%)", 0.0, 2.0, 0.0, step=0.05,
        help="Stay in cash when the predicted move is smaller than this.",
    )
    cost = s4.number_input(
        "Cost per buy or sell (%)", 0.0, 0.5, 0.03, step=0.01, format="%.2f",
        help="Brokerage, taxes and slippage per side. About 0.03% is typical for Nifty futures.",
    )

    if sp500 is not None and signal.startswith("With"):
        strat_bt, entry = bt, "open"
    elif sp500 is None:
        strat_bt, entry = bt, "close"
    else:
        strat_bt, entry = forecast(nifty, model_name, test_days, None).backtest, "close"

    mode = "long_cash" if positions == "Long or cash" else "long_short"
    curves, stats = strategy.backtest(strat_bt, mode, entry, threshold, cost)

    fig_eq = go.Figure()
    fig_eq.add_scatter(x=curves.index, y=curves["Buy & hold"], name="Buy & hold", line=dict(color=BLUE, width=2))
    fig_eq.add_scatter(x=curves.index, y=curves["Strategy"], name="Strategy", line=dict(color=ORANGE, width=2))
    st.plotly_chart(_layout(fig_eq, 380, yaxis=dict(title="Value of ₹1 invested", tickformat=".2f")), width="stretch")

    st.dataframe(
        stats.style.format("{:,.1f}", na_rep="–")
        .format("{:,.2f}", subset=pd.IndexSlice[["Sharpe ratio"], :])
        .format("{:,.0f}", subset=pd.IndexSlice[["Trades"], :]),
        width="stretch",
    )
    if entry == "open":
        st.caption(
            "Enters at each day's open and exits at its close. Much of the S&P 500's move is "
            "usually already in Nifty's opening gap, so this version tends to do worse."
        )
    st.caption("Past backtests don't predict future returns. Educational use only, not investment advice.")


# ---------------------------------------------------------------- details tab
with tab_details:
    sp_note = "the S&P 500's latest 1-day and 5-day returns" if sp500 is not None else "no S&P 500 data (off)"
    st.markdown(
        f"""
- **Model:** {model_name}, trained on {len(nifty):,} daily bars ({nifty.index[0]:%b %Y} to {nifty.index[-1]:%b %Y}).
- **What it predicts:** tomorrow's % return. The price is today's close × e^(predicted return).
- **Inputs:** the last 5 daily returns, distance from the 5/10/20/50-day averages, 5- and 20-day
  volatility, the day's range and candle body, RSI(14), volume, weekday, and {sp_note}.
- **Accuracy:** RMSE ₹{m['RMSE']:,.1f}, MAPE {m['MAPE %']:.2f}%.
- **Calendar:** the next trading day is the next weekday; NSE holidays aren't skipped.
"""
    )
    if result.importance is not None:
        imp = result.importance
        fig_imp = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=BLUE))
        st.plotly_chart(_layout(fig_imp, 420, xaxis_title="Importance", hovermode="closest"), width="stretch")

    table = bt[["Date", "PrevClose", "Actual", "Predicted"]].iloc[::-1].reset_index(drop=True)
    table["Error"] = table["Predicted"] - table["Actual"]
    st.dataframe(
        table.style.format({"Date": "{:%d %b %Y}", "PrevClose": "₹{:,.2f}", "Actual": "₹{:,.2f}",
                            "Predicted": "₹{:,.2f}", "Error": "{:+,.2f}"}),
        width="stretch", height=360,
    )
    st.download_button("Download backtest CSV", table.to_csv(index=False), "nifty_backtest.csv")

st.caption(
    "Daily index moves are close to random, so a next-day forecast is usually within a fraction "
    "of a percent of today's close. Educational use only, not investment advice."
)
