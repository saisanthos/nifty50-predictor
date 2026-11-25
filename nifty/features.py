"""Turning prices into model inputs.

Every feature on day t uses only what is known at that day's close.
The target is the next day's return: log(close[t+1] / close[t]).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "target"


def build(nifty: pd.DataFrame, sp500: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per Nifty trading day: feature columns, plus "target" (next
    day's log return, empty on the last day) and "Close"."""
    close = nifty["Close"]
    ret = np.log(close).diff()
    f = pd.DataFrame(index=nifty.index)

    for lag in range(1, 6):
        f[f"return_{lag}d_ago"] = ret.shift(lag - 1)
    for window in (5, 10, 20, 50):
        f[f"vs_avg_{window}d"] = close / close.rolling(window).mean() - 1
    f["volatility_5d"] = ret.rolling(5).std()
    f["volatility_20d"] = ret.rolling(20).std()
    f["day_range"] = (nifty["High"] - nifty["Low"]) / close
    f["day_body"] = (close - nifty["Open"]) / nifty["Open"]
    f["rsi_14"] = _rsi(close, 14) / 100
    volume = nifty["Volume"].replace(0, np.nan)
    if volume.notna().mean() > 0.8:
        f["volume_z_20d"] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
    f["weekday"] = nifty.index.dayofweek

    if sp500 is not None and not sp500.empty:
        f = f.join(_sp500_features(sp500, nifty.index))

    f[TARGET] = ret.shift(-1)
    f["Close"] = close
    return f.replace([np.inf, -np.inf], np.nan)


def columns(feats: pd.DataFrame) -> list[str]:
    """The feature column names (everything except target and Close)."""
    return [c for c in feats.columns if c not in (TARGET, "Close")]


def _sp500_features(sp500: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """S&P 500 1-day and 5-day returns, as of each Nifty date.

    The US closes after NSE. So for a Nifty date t we take the latest S&P close
    dated on or before t: it is known before NSE opens on t+1. Values older than
    5 days (e.g. data that stops early) are dropped rather than carried forward.
    """
    log_close = np.log(sp500["Close"].dropna())
    s = pd.DataFrame({"sp500_return_1d": log_close.diff(), "sp500_return_5d": log_close.diff(5)})
    s["seen"] = s.index
    merged = s.reindex(s.index.union(dates)).ffill().reindex(dates)
    stale = (dates.to_series() - merged["seen"]).dt.days > 5
    merged.loc[stale.values, ["sp500_return_1d", "sp500_return_5d"]] = np.nan
    return merged.drop(columns="seen")


def _rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))
