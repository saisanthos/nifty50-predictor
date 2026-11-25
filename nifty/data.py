"""Getting daily price data: live from Yahoo Finance, or from a CSV file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
BUNDLED_CSV = Path(__file__).resolve().parent.parent / "data" / "nifty50_history.csv"


def fetch_yahoo(ticker: str = "^NSEI", period: str = "10y") -> pd.DataFrame:
    """Daily OHLCV bars from Yahoo Finance, indexed by date."""
    import yfinance as yf

    raw = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
    return _tidy(raw)


def fetch_sp500(period: str = "10y") -> pd.DataFrame:
    """Daily S&P 500 bars, leaving out today's bar while the US market is
    still open (a half-finished day would be a different signal)."""
    df = fetch_yahoo("^GSPC", period)
    if df.empty:
        return df
    now = pd.Timestamp.now(tz="America/New_York")
    still_open = now.hour * 60 + now.minute < 16 * 60 + 15
    if df.index[-1] == now.normalize().tz_localize(None) and still_open:
        df = df.iloc[:-1]
    return df


def load_csv(source=BUNDLED_CSV) -> pd.DataFrame:
    """Read a CSV in Yahoo format (Date,Open,High,Low,Close,Volume) or
    Investing.com format (Date,Price,Open,High,Low,Vol.,Change %)."""
    raw = pd.read_csv(source)
    raw.columns = [c.strip() for c in raw.columns]
    if "Price" in raw.columns and "Close" not in raw.columns:
        raw = raw.rename(columns={"Price": "Close"})
    if "Vol." in raw.columns:
        raw["Volume"] = raw["Vol."].map(_parse_volume)
    if "Volume" not in raw.columns:
        raw["Volume"] = np.nan
    for col in ["Open", "High", "Low", "Close"]:
        raw[col] = pd.to_numeric(raw[col].astype(str).str.replace(",", ""), errors="coerce")
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    return _tidy(raw.dropna(subset=["Date"]).set_index("Date"))


def last_years(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Trim to the last N years, e.g. period="10y". "max" keeps everything."""
    if period == "max" or df.empty:
        return df
    return df.loc[df.index[-1] - pd.DateOffset(years=int(period.rstrip("y"))) :]


def _parse_volume(value) -> float:
    """'25.36M' -> 25_360_000."""
    if pd.isna(value):
        return np.nan
    text = str(value).replace(",", "").strip()
    multiplier = {"K": 1e3, "M": 1e6, "B": 1e9}.get(text[-1:].upper(), 1)
    try:
        return float(text.rstrip("KMBkmb")) * multiplier
    except ValueError:
        return np.nan


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Same shape for every source: date index (no time zone), OHLCV columns,
    sorted, no duplicate days."""
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    df.index.name = "Date"
    df = df[COLUMNS].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna(subset=["Open", "High", "Low", "Close"])
