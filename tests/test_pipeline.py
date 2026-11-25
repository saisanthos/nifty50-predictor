import numpy as np
import pandas as pd
import pytest

from nifty import data, features, model, strategy


@pytest.fixture(scope="module")
def nifty():
    return data.last_years(data.load_csv(), "5y")


def test_bundled_csv_loads():
    df = data.load_csv()
    assert list(df.columns) == data.COLUMNS
    assert df.index.is_monotonic_increasing
    assert df["Close"].iloc[0] == pytest.approx(1592.20)
    assert df["Volume"].iloc[0] == pytest.approx(25.36e6)


def test_features_use_no_future_data(nifty):
    """Changing tomorrow's prices must not change today's features."""
    a = features.build(nifty)
    changed = nifty.copy()
    changed.iloc[-1, :4] *= 1.5
    b = features.build(changed)
    cols = features.columns(a)
    pd.testing.assert_frame_equal(a[cols].iloc[:-1], b[cols].iloc[:-1])


def test_sp500_alignment_uses_last_close_on_or_before(nifty):
    sp = nifty.copy()  # any series works for the alignment check
    sp.index = sp.index - pd.Timedelta(days=1)
    f = features.build(nifty, sp)
    assert "sp500_return_1d" in f
    d = nifty.index[100]
    expected = np.log(sp["Close"]).diff().loc[: d].iloc[-1]
    assert f.loc[d, "sp500_return_1d"] == pytest.approx(expected)


def test_backtest_and_forecast(nifty):
    r = model.run(nifty, "Ridge regression", 120)
    assert len(r.backtest) == 120
    assert r.backtest["Date"].iloc[-1] == nifty.index[-1]
    assert (r.backtest["Date"] > r.backtest["Date"].shift()).iloc[1:].all()
    assert r.next_date > r.last_date
    assert abs(r.next_close / r.last_close - 1) < 0.05
    assert {"MAE", "Naive MAE", "Direction accuracy %"} <= set(r.metrics)


def test_strategy(nifty):
    bt = model.run(nifty, "Ridge regression", 120).backtest
    curves, stats = strategy.backtest(bt, "long_cash", cost_pct=0)
    assert set(curves["Position"].unique()) <= {0.0, 1.0}
    hold = bt["Actual"].iloc[-1] / bt["PrevClose"].iloc[0]
    assert curves["Buy & hold"].iloc[-1] == pytest.approx(hold)
    _, stats_ls = strategy.backtest(bt, "long_short", entry="open", cost_pct=0.05)
    assert stats_ls.loc["Trades", "Strategy"] > 0
