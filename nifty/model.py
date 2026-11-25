"""Training, the walk-forward backtest, and tomorrow's forecast.

The model predicts tomorrow's % return, not the price. The price is then
today's close x e^(predicted return). Predicting returns keeps tree models
valid at index levels they never saw in training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import features

MODELS = {
    "Ridge regression": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "Random forest": lambda: RandomForestRegressor(
        n_estimators=300, max_depth=6, min_samples_leaf=5, n_jobs=-1, random_state=42
    ),
    "Gradient boosting": lambda: GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=42
    ),
}

RETRAIN_EVERY = 21  # trading days, about a month
MIN_TRAIN_DAYS = 200


@dataclass
class Forecast:
    backtest: pd.DataFrame  # one row per predicted day, see walk_forward()
    metrics: dict
    last_date: pd.Timestamp
    last_close: float
    next_date: pd.Timestamp
    next_close: float
    importance: pd.Series | None


def run(nifty: pd.DataFrame, model_name: str, test_days: int, sp500: pd.DataFrame | None = None) -> Forecast:
    """Backtest the model over the last `test_days`, then predict the next day."""
    feats = features.build(nifty, sp500)
    cols = features.columns(feats)
    labelled = feats.dropna(subset=cols + [features.TARGET])
    if len(labelled) < test_days + MIN_TRAIN_DAYS:
        raise ValueError(
            f"Need at least {test_days + MIN_TRAIN_DAYS} usable days of history, "
            f"got {len(labelled)}. Pick more history or a shorter backtest."
        )

    bt = walk_forward(nifty, labelled, cols, model_name, test_days)

    final = MODELS[model_name]()
    final.fit(labelled[cols], labelled[features.TARGET])
    today = feats.dropna(subset=cols).iloc[[-1]]
    last_close = float(today["Close"].iloc[0])
    next_return = float(final.predict(today[cols])[0])

    return Forecast(
        backtest=bt,
        metrics=score(bt),
        last_date=today.index[0],
        last_close=last_close,
        next_date=next_weekday(today.index[0]),
        next_close=last_close * np.exp(next_return),
        importance=_importance(final, cols),
    )


def walk_forward(nifty, labelled, cols, model_name, test_days) -> pd.DataFrame:
    """Predict each of the last `test_days` using a model trained only on
    earlier days, retraining every RETRAIN_EVERY days.

    Columns: Date (the day being predicted), PrevClose, Open, Actual,
    Predicted, Naive (= PrevClose, i.e. "no change").
    """
    test = labelled.iloc[-test_days:]
    predicted = []
    for start in range(0, len(test), RETRAIN_EVERY):
        block = test.iloc[start : start + RETRAIN_EVERY]
        # Train on rows strictly before the block. The last of those rows'
        # target is the move into the block's first day, known by then.
        train = labelled.loc[labelled.index < block.index[0]]
        m = MODELS[model_name]()
        m.fit(train[cols], train[features.TARGET])
        predicted.append(pd.Series(m.predict(block[cols]), index=block.index))
    pred_return = pd.concat(predicted)

    # Each row is dated by the day it predicts (the next trading day).
    pos = nifty.index.get_indexer(test.index)
    next_dates = nifty.index[pos + 1]
    prev_close = test["Close"].to_numpy()
    return pd.DataFrame(
        {
            "Date": next_dates,
            "PrevClose": prev_close,
            "Open": nifty["Open"].reindex(next_dates).to_numpy(),
            "Actual": prev_close * np.exp(test[features.TARGET].to_numpy()),
            "Predicted": prev_close * np.exp(pred_return.to_numpy()),
            "Naive": prev_close,
        }
    )


def score(bt: pd.DataFrame) -> dict:
    err = bt["Predicted"] - bt["Actual"]
    naive_err = bt["Naive"] - bt["Actual"]
    up_actual = np.sign(bt["Actual"] - bt["PrevClose"])
    up_pred = np.sign(bt["Predicted"] - bt["PrevClose"])
    mae, naive_mae = err.abs().mean(), naive_err.abs().mean()
    return {
        "MAE": float(mae),
        "RMSE": float(np.sqrt((err**2).mean())),
        "MAPE %": float((err.abs() / bt["Actual"]).mean() * 100),
        "Naive MAE": float(naive_mae),
        "Better than naive %": float((1 - mae / naive_mae) * 100),
        "Direction accuracy %": float((up_actual == up_pred).mean() * 100),
        "Up days %": float((up_actual > 0).mean() * 100),
    }


def next_weekday(d: pd.Timestamp) -> pd.Timestamp:
    """Next weekday. NSE holidays are not skipped."""
    return (pd.Timestamp(d) + pd.offsets.BDay(1)).normalize()


def _importance(model, cols) -> pd.Series | None:
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=cols).sort_values()
    if hasattr(model, "named_steps"):
        return pd.Series(np.abs(model.named_steps["ridge"].coef_), index=cols).sort_values()
    return None
