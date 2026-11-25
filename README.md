# Nifty 50 next-day close predictor

A Streamlit app that pulls Nifty 50 prices from Yahoo Finance, trains a model,
and predicts the next trading day's closing price. It shows how the model would
have done in the past (a walk-forward backtest) and whether trading its
predictions would have beaten simply buying and holding.

## Run it on your computer

```bash
pip install -r requirements.txt
streamlit run app.py
```

It opens at http://localhost:8501. To run the tests: `pip install pytest && pytest`.

## Put it online (Streamlit Community Cloud, free)

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app**, then **Deploy a public app from GitHub**.
3. Pick this repository, branch `main`, main file `app.py`.
4. Click **Deploy**. It installs `requirements.txt` by itself and gives you a URL.

Every push to `main` redeploys the app automatically.

## What's in the app

- **Top row:** tomorrow's predicted close, the last close, the average backtest error
  (and how it compares with guessing "no change"), and how often the model got the
  direction right.
- **Forecast tab:** predicted vs actual closes over the backtest, error by day, and a
  predicted-vs-actual scatter.
- **Strategy tab:** trade the predictions one day at a time. Choose long-or-cash or
  long-or-short, a minimum predicted move, and a cost per trade. Shows the equity curve
  against buy and hold, with return, Sharpe ratio, worst drop and trade count.
- **Data & model tab:** model inputs, feature importance, and the full backtest table
  (downloadable as CSV).

Sidebar: how much history to train on, the model (Ridge, Random forest, Gradient
boosting), backtest length, the S&P 500 toggle, and an optional CSV upload.

## How the code is organised

```
app.py               the Streamlit page (layout and charts only)
nifty/data.py        getting prices: Yahoo Finance, or a CSV file
nifty/features.py    turning prices into model inputs
nifty/model.py       training, the walk-forward backtest, tomorrow's forecast
nifty/strategy.py    trading the predictions vs buy and hold
data/                Nifty 50 daily history, 2000 to Apr 2025 (offline fallback)
tests/               checks that run without internet
```

Everything in `nifty/` is plain Python with no Streamlit, so it can be used or tested
on its own.

## How it works

- **Predicts the return, not the price.** The model predicts tomorrow's % change; the
  price is today's close × e^(predicted change). This keeps tree models valid at index
  levels they never saw in training.
- **Honest backtest.** Each block of 21 trading days is predicted by a model trained only
  on days before it, so the backtest never sees the answer.
- **"No change" baseline.** Guessing that tomorrow closes where today did is hard to beat
  on daily index data, so the app always shows the model's error next to it.
- **S&P 500 overnight move.** The US market closes after NSE, so the S&P 500's latest
  1-day and 5-day moves are known before Nifty opens the next day. Today's S&P bar is
  ignored while the US session is still open.
- **Strategy timing.** A Nifty-only signal is known at NSE's close, so it trades close to
  close. A signal that uses the S&P 500 only arrives overnight, so it can only trade from
  the next open to the close. Much of the US move is usually already in Nifty's opening
  gap, so that version tends to do worse.
- **Offline fallback.** If Yahoo Finance can't be reached, the app uses
  `data/nifty50_history.csv` and says so. The S&P 500 feature is off in that case.
- **Calendar.** The next trading day is taken as the next weekday; NSE holidays aren't
  skipped.

## Using your own data

Upload a CSV in the sidebar, in either Yahoo format (`Date,Open,High,Low,Close,Volume`)
or Investing.com format (`Date,Price,Open,High,Low,Vol.,Change %`).

---

Educational use only, not investment advice. Daily index moves are close to random, so
a next-day forecast is usually within a fraction of a percent of today's close.
