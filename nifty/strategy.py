"""Trading the backtest's predictions one day at a time, against buy and hold."""

from __future__ import annotations

import numpy as np
import pandas as pd


def backtest(
    bt: pd.DataFrame,
    mode: str = "long_cash",
    entry: str = "close",
    threshold_pct: float = 0.0,
    cost_pct: float = 0.03,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (equity curves, stats table).

    mode: "long_cash"  long when an up day is predicted, otherwise in cash.
          "long_short" long on predicted up days, short on predicted down days.
    entry: "close" holds from the previous close to the day's close. Only
           possible when the signal is known at NSE's close (Nifty-only model).
           "open" enters at the day's open and exits at its close. Needed for
           the S&P 500 model, whose signal arrives overnight.
    threshold_pct: stay in cash when the predicted move is smaller than this.
    cost_pct: cost of each buy or sell, as % of the position.
    """
    d = bt.set_index("Date")
    start = d["Open"] if entry == "open" else d["PrevClose"]
    day_return = np.log(d["Actual"] / start)
    predicted_move = d["Predicted"] / start - 1

    position = np.sign(predicted_move).where(predicted_move.abs() * 100 >= threshold_pct, 0.0)
    if mode == "long_cash":
        position = position.clip(lower=0)
    position = position.fillna(0.0)

    if entry == "open":
        trades = position.abs() * 2  # in at the open, out at the close, every day
    else:
        trades = position.diff().abs().fillna(position.abs())

    strat = position * day_return - trades * cost_pct / 100
    hold = np.log(d["Actual"] / d["PrevClose"])

    curves = pd.DataFrame(
        {"Strategy": np.exp(strat.cumsum()), "Buy & hold": np.exp(hold.cumsum()), "Position": position}
    )
    stats = pd.DataFrame(
        {
            "Strategy": _stats(strat, position, int(trades.sum())),
            "Buy & hold": _stats(hold, pd.Series(1.0, index=hold.index), 1),
        }
    )
    return curves, stats


def _stats(r: pd.Series, position: pd.Series, trades: int) -> dict:
    r = r.fillna(0.0)
    equity = np.exp(r.cumsum())
    years = len(r) / 252
    vol = r.std() * np.sqrt(252)
    in_market = r[position != 0]
    return {
        "Total return %": (equity.iloc[-1] - 1) * 100,
        "Return per year %": (equity.iloc[-1] ** (1 / years) - 1) * 100,
        "Sharpe ratio": r.mean() * 252 / vol if vol else np.nan,
        "Worst drop %": (equity / equity.cummax() - 1).min() * 100,
        "Winning days %": (in_market > 0).mean() * 100 if len(in_market) else np.nan,
        "Days in market %": (position != 0).mean() * 100,
        "Trades": trades,
    }
