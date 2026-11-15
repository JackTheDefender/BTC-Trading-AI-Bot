"""BTC/USDT 15-minute research model and PAPER backtest. No live orders.

Install (Windows Python 3.12):
    python -m pip install numpy pandas requests scikit-learn tensorflow xgboost matplotlib
Run:
    python BTC_Ensemble_Trading_Revised.py

No prediction accuracy or trade win rate is guaranteed. All dates are UTC.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from xgboost import XGBClassifier

# ----------------------- SETTINGS -------------------------------------------
SYMBOL = "BTCUSDT"
INTERVAL = "15m"
CANDLES = 8000                    # about 83 days; longer history is preferable
API_URL = "https://api.binance.com/api/v3/klines"
LOOKBACK = 48
SEED = 42
EPOCHS = 8
BATCH_SIZE = 128
INITIAL_CAPITAL = 10_000.0
POSITION_FRACTION = 0.25          # use at most 25% of available cash
POSITION_STOP_LOSS = 0.01         # 1% adverse move from invested entry price
POSITION_TAKE_PROFIT = 0.05       # 5% favorable move from invested entry price
SESSION_LOSS_LIMIT = 0.01         # halt fold at -1% of starting fold capital
SESSION_PROFIT_TARGET = 0.05      # halt fold at +5% of starting fold capital
FEE_BPS = 10.0                    # illustrative per side; change to your fees
SLIPPAGE_BPS = 3.0                # illustrative per side
MIN_TRADE_CONFIDENCE = 0.70      # filter low-confidence trades
MIN_CONFIDENCE_MARGIN = 0.10     # best class probability minus runner-up
MIN_ROWS = 3500
OUTPUT_DIR = Path(__file__).resolve().parent / "btc_ensemble_results"
FEATURES = [
    "ret1", "ret2", "ret4", "ret8", "ret16", "range", "body",
    "vol_z", "rsi", "macd", "macd_signal", "ema_dist_12",
    "ema_dist_48", "atr_pct", "volatility", "time_sin", "time_cos"
]
# Class mapping: SELL=0, HOLD=1, BUY=2; spot backtest is long-only.
keras.utils.set_random_seed(SEED)


def get_candles(total: int = CANDLES) -> pd.DataFrame:
    """Fetch completed Binance spot candles backwards, preserving UTC open times."""
    session = requests.Session()
    blocks = []
    end_time = None
    while sum(len(b) for b in blocks) < total:
        remaining = total - sum(len(b) for b in blocks)
        params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": min(1000, remaining)}
        if end_time is not None:
            params["endTime"] = end_time
        response = session.get(API_URL, params=params, timeout=30)
        response.raise_for_status()
        chunk = response.json()
        if not isinstance(chunk, list) or not chunk:
            break
        blocks.append(chunk)
        end_time = int(chunk[0][0]) - 1
        print(f"Fetched {sum(map(len, blocks))} candles", flush=True)
        time.sleep(0.15)
    if not blocks:
        raise RuntimeError("No candles returned. Check Binance API availability.")
    columns = ["open_time", "open", "high", "low", "close", "volume", "close_time",
               "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    frame = pd.DataFrame([c for block in reversed(blocks) for c in block], columns=columns)
    frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    frame = frame.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    # Exclude a candle that has not yet completed.
    close_times = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    frame = frame.loc[close_times < pd.Timestamp.now(tz="UTC")].copy()
    if frame.empty or not frame.index.is_unique:
        raise ValueError("Missing or duplicate candle timestamps")
    return frame


def make_features(raw: pd.DataFrame) -> pd.DataFrame:
    x = raw.copy()
    close = x["close"]
    x["ret1"] = np.log(close).diff()
    for lag in [2, 4, 8, 16]:
        x[f"ret{lag}"] = np.log(close / close.shift(lag))
    x["range"] = (x["high"] - x["low"]) / close
    x["body"] = (close - x["open"]) / x["open"]
    logvol = np.log1p(x["volume"])
    x["vol_z"] = (logvol - logvol.rolling(48).mean()) / (logvol.rolling(48).std() + 1e-9)
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    x["rsi"] = (100 - 100 / (1 + up / (down + 1e-12))) / 100
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    x["macd"] = macd / close
    x["macd_signal"] = macd.ewm(span=9, adjust=False).mean() / close
    for n in [12, 48]:
        x[f"ema_dist_{n}"] = close / close.ewm(span=n, adjust=False).mean() - 1
    tr = pd.concat([x["high"] - x["low"], (x["high"] - close.shift()).abs(),
                    (x["low"] - close.shift()).abs()], axis=1).max(axis=1)
    x["atr_pct"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / close
    x["volatility"] = x["ret1"].rolling(48).std()
    minutes = x.index.hour * 60 + x.index.minute
    x["time_sin"] = np.sin(2 * np.pi * minutes / 1440)
    x["time_cos"] = np.cos(2 * np.pi * minutes / 1440)
    x["future_ret"] = close.shift(-1) / close - 1
    roundtrip_cost = 2 * (FEE_BPS + SLIPPAGE_BPS) / 10_000
    threshold = max(roundtrip_cost, 0.0015)
    x["target"] = np.select([x["future_ret"] < -threshold,
                             x["future_ret"] > threshold], [0, 2], default=1)
    x = x.replace([np.inf, -np.inf], np.nan)
    return x.dropna(subset=FEATURES + ["future_ret"]).copy()


def build_fold(data: pd.DataFrame, tr_end: int, va_end: int, te_end: int):
    """Scale on training only; sequences end at signal candle i (no future features).

    Purge the final training and validation LABEL because its next-close target
    falls in the following split. Context features from preceding splits are okay.
    """
    scaler = StandardScaler().fit(data.iloc[:tr_end][FEATURES])
    features = scaler.transform(data[FEATURES]).astype("float32")
    targets = data["target"].to_numpy(dtype="int64")

    def part(start: int, end: int):
        xs, ys, ts = [], [], []
        for i in range(max(start, LOOKBACK - 1), end):
            xs.append(features[i - LOOKBACK + 1:i + 1])
            ys.append(targets[i])
            ts.append(data.index[i])
        return np.asarray(xs, dtype="float32"), np.asarray(ys), pd.DatetimeIndex(ts)

    # A label at i uses close(i+1); exclude boundary i=end-1 from train/val.
    return scaler, part(0, tr_end - 1), part(tr_end, va_end - 1), part(va_end, te_end)


def compile_network(model):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_lstm():
    inp = keras.Input(shape=(LOOKBACK, len(FEATURES)))
    x = layers.LSTM(48, return_sequences=True)(inp)
    x = layers.LSTM(24)(x)
    x = layers.Dropout(0.25)(x)
    return compile_network(keras.Model(inp, layers.Dense(3, activation="softmax")(x)))


def build_transformer():
    inp = keras.Input(shape=(LOOKBACK, len(FEATURES)))
    x = layers.Dense(48)(inp)
    x = x + layers.Embedding(input_dim=LOOKBACK, output_dim=48)(
        keras.ops.arange(0, LOOKBACK, 1))
    for _ in range(2):
        z = layers.LayerNormalization()(x)
        x = x + layers.Dropout(0.1)(layers.MultiHeadAttention(
            num_heads=4, key_dim=12, dropout=0.1)(z, z))
        z = layers.LayerNormalization()(x)
        z = layers.Dense(48)(layers.Dense(96, activation="relu")(z))
        x = x + layers.Dropout(0.1)(z)
    x = layers.GlobalAveragePooling1D()(layers.LayerNormalization()(x))
    return compile_network(keras.Model(inp, layers.Dense(3, activation="softmax")(x)))


def aligned_xgb_prob(clf, x):
    raw = clf.predict_proba(x.reshape(len(x), -1))
    out = np.zeros((len(x), 3), dtype=float)
    for col, cls in enumerate(clf.classes_):
        out[:, int(cls)] = raw[:, col]
    return out


def select_weights(validation_probs, labels):
    """Select weights on validation cross-entropy, never test outcomes."""
    best_loss, best_weights = float("inf"), None
    for a in np.arange(0, 1.001, 0.25):
        for b in np.arange(0, 1.001 - a, 0.25):
            c = 1 - a - b
            if c < -1e-9:
                continue
            w = np.array([a, b, max(c, 0)])
            prob = np.clip(np.tensordot(w, validation_probs, axes=(0, 0)), 1e-7, 1)
            loss = log_loss(labels, prob, labels=[0, 1, 2])
            if loss < best_loss:
                best_loss, best_weights = loss, w
    return best_weights, best_loss


def make_signals(frame):
    p = frame[["p_sell", "p_hold", "p_buy"]].to_numpy(dtype=float)
    if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1, atol=0.02):
        raise ValueError("Invalid ensemble probabilities")
    ranks = np.sort(p, axis=1)
    best = p.argmax(axis=1)
    confident = (ranks[:, -1] >= MIN_TRADE_CONFIDENCE) & (
        ranks[:, -1] - ranks[:, -2] >= MIN_CONFIDENCE_MARGIN)
    # 0=SELL (close long), 1=HOLD, 2=BUY (open long)
    frame["signal"] = np.where(confident, best, 1).astype(int)
    frame["confidence"] = ranks[:, -1]
    return frame


def backtest(predictions: pd.DataFrame, market: pd.DataFrame):
    """Spot long-only, next-bar-open fills, OHLC stops, no live trading.

    When both stop and target are reached within one candle, assume STOP first.
    Exchange stop orders and session loss limits can slip during price gaps.
    """
    fee, slip = FEE_BPS / 10_000, SLIPPAGE_BPS / 10_000
    cash, qty = INITIAL_CAPITAL, 0.0
    entry_cost, entry_price, entry_time = 0.0, np.nan, None
    stop_price, take_price = np.nan, np.nan
    records, trades = [], []
    halted = False
    prev_execution_pos = None

    def close_trade(ts, price, reason):
        nonlocal cash, qty, entry_cost
        proceeds = qty * price * (1 - fee)
        cash += proceeds
        trades.append({"entry_time": entry_time, "exit_time": ts,
                       "entry": entry_price, "exit": price,
                       "net_pnl": proceeds - entry_cost, "reason": reason})
        qty = 0.0
        entry_cost = 0.0

    for signal_ts, row in predictions.sort_index().iterrows():
        if halted:
            break
        if signal_ts not in market.index:
            raise ValueError(f"Signal timestamp not in market: {signal_ts}")
        candle_pos = market.index.get_loc(signal_ts)
        if not isinstance(candle_pos, (int, np.integer)):
            raise ValueError("Market index must be unique")
        exec_pos = candle_pos + 1  # signal is known at candle close
        if exec_pos >= len(market):
            break
        bar_ts = market.index[exec_pos]
        bar = market.iloc[exec_pos]
        if prev_execution_pos is not None and exec_pos != prev_execution_pos + 1:
            # No carryover across gaps in test prediction coverage.
            if qty > 0:
                close_trade(bar_ts, float(bar["open"]) * (1 - slip), "data gap")
        prev_execution_pos = exec_pos

        signal = int(row["signal"])
        # Trade at NEXT candle open. A SELL signal closes a spot long.
        if qty > 0 and signal == 0:
            close_trade(bar_ts, float(bar["open"]) * (1 - slip), "SELL signal")
        if qty == 0 and signal == 2:
            fill = float(bar["open"]) * (1 + slip)
            budget = cash * POSITION_FRACTION
            buy_qty = budget / (fill * (1 + fee))
            if buy_qty > 0:
                entry_cost = buy_qty * fill * (1 + fee)
                cash -= entry_cost
                qty = buy_qty
                entry_price, entry_time = fill, bar_ts
                stop_price = fill * (1 - POSITION_STOP_LOSS)
                take_price = fill * (1 + POSITION_TAKE_PROFIT)

        # Conservative OHLC fill assumption: stop wins if both levels hit.
        if qty > 0:
            if float(bar["open"]) <= stop_price:
                close_trade(bar_ts, float(bar["open"]) * (1 - slip), "stop gap")
            elif float(bar["low"]) <= stop_price:
                close_trade(bar_ts, stop_price * (1 - slip), "stop")
            elif float(bar["open"]) >= take_price:
                close_trade(bar_ts, float(bar["open"]) * (1 - slip), "target gap")
            elif float(bar["high"]) >= take_price:
                close_trade(bar_ts, take_price * (1 - slip), "target")

        equity = cash + qty * float(bar["close"])
        if equity <= INITIAL_CAPITAL * (1 - SESSION_LOSS_LIMIT):
            reason = "session -1% loss limit"
        elif equity >= INITIAL_CAPITAL * (1 + SESSION_PROFIT_TARGET):
            reason = "session +5% profit target"
        else:
            reason = None
        if reason:
            if qty > 0:
                close_trade(bar_ts, float(bar["close"]) * (1 - slip), reason)
            equity = cash
            halted = True
        records.append({"timestamp": bar_ts, "equity": equity,
                        "cash": cash, "quantity": qty})

    if qty > 0 and records:
        # Close remaining exposure at the last processed candle close.
        ts = records[-1]["timestamp"]
        close_trade(ts, float(market.loc[ts, "close"]) * (1 - slip), "end of fold")
        records[-1]["equity"] = cash
        records[-1]["cash"] = cash
        records[-1]["quantity"] = 0.0
    curve = pd.DataFrame(records).set_index("timestamp") if records else pd.DataFrame(
        columns=["equity", "cash", "quantity"])
    return curve, pd.DataFrame(trades), halted


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = get_candles()
    data = make_features(raw)
    if len(data) < MIN_ROWS:
        raise ValueError(f"Only {len(data)} usable rows; need at least {MIN_ROWS}")
    print("Data:", len(data), "from", data.index.min(), "to", data.index.max())
    print("Class distribution:", data["target"].value_counts().sort_index().to_dict())

    n = len(data)
    folds = [(int(n * .50), int(n * .60), int(n * .70)),
             (int(n * .60), int(n * .70), int(n * .80)),
             (int(n * .70), int(n * .80), int(n * .90)),
             (int(n * .80), int(n * .90), n)]
    all_predictions, all_trades, fold_stats = [], [], []
    for fold_id, (tr_end, va_end, te_end) in enumerate(folds, 1):
        print(f"\n==== FOLD {fold_id} ====", flush=True)
        _, tr, va, te = build_fold(data, tr_end, va_end, te_end)
        Xtr, ytr, _ = tr
        Xva, yva, _ = va
        Xte, yte, ts_te = te
        if len(Xtr) == 0 or len(Xva) == 0 or len(Xte) == 0:
            raise ValueError("Empty split")
        if set(np.unique(ytr)) != {0, 1, 2}:
            raise ValueError("Training split missing a class; get more history")
        print("Samples:", len(Xtr), len(Xva), len(Xte))
        callbacks = [keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True)]
        lstm = build_lstm()
        lstm.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS,
                 batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1)
        transformer = build_transformer()
        transformer.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=EPOCHS,
                        batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1)
        clf = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            objective="multi:softprob", num_class=3,
                            eval_metric="mlogloss", random_state=SEED, n_jobs=2)
        clf.fit(Xtr.reshape(len(Xtr), -1), ytr)
        pv = np.stack([lstm.predict(Xva, verbose=0),
                       transformer.predict(Xva, verbose=0),
                       aligned_xgb_prob(clf, Xva)])
        pt = np.stack([lstm.predict(Xte, verbose=0),
                       transformer.predict(Xte, verbose=0),
                       aligned_xgb_prob(clf, Xte)])
        weights, val_loss = select_weights(pv, yva)
        ensemble = np.tensordot(weights, pt, axes=(0, 0))
        assert ensemble.shape == (len(ts_te), 3)
        frame = pd.DataFrame(ensemble, index=ts_te,
                             columns=["p_sell", "p_hold", "p_buy"])
        frame["fold"] = fold_id
        frame["actual"] = yte
        frame = make_signals(frame)
        all_predictions.append(frame)
        curve, trades, halted = backtest(frame, raw)
        if not trades.empty:
            trades["fold"] = fold_id
            all_trades.append(trades)
        accuracy = accuracy_score(yte, ensemble.argmax(axis=1))
        wins = int((trades["net_pnl"] > 0).sum()) if not trades.empty else 0
        count = len(trades)
        win_rate = 100 * wins / count if count else np.nan
        ending = float(curve["equity"].iloc[-1]) if not curve.empty else INITIAL_CAPITAL
        fold_stats.append({"fold": fold_id, "test_samples": len(yte),
                           "classification_accuracy": accuracy,
                           "validation_log_loss": val_loss,
                           "lstm_weight": weights[0], "transformer_weight": weights[1],
                           "xgb_weight": weights[2], "trades": count,
                           "trade_win_rate_pct": win_rate,
                           "net_return_pct": 100 * (ending / INITIAL_CAPITAL - 1),
                           "session_halted": halted})
        print(f"Fold {fold_id}: accuracy={accuracy:.2%}, trades={count}, "
              f"win rate={win_rate:.1f}%, return={100*(ending/INITIAL_CAPITAL-1):.2f}%")
        keras.backend.clear_session()

    predictions = pd.concat(all_predictions).sort_index()
    if not predictions.index.is_unique:
        raise ValueError("Overlapping test folds produced duplicate timestamps")
    trade_frame = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    stats = pd.DataFrame(fold_stats)
    predictions.to_csv(OUTPUT_DIR / "out_of_sample_predictions.csv", index_label="timestamp")
    trade_frame.to_csv(OUTPUT_DIR / "paper_trades.csv", index=False)
    stats.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
    print("\n=== OUT-OF-SAMPLE SUMMARY ===")
    print(stats.to_string(index=False))
    print("\nPooled classification report (SELL=0, HOLD=1, BUY=2):")
    print(classification_report(predictions["actual"],
                                predictions[["p_sell", "p_hold", "p_buy"]].to_numpy().argmax(axis=1),
                                labels=[0, 1, 2], zero_division=0))
    print("\nPaper trade count:", len(trade_frame))
    if not trade_frame.empty:
        print("Pooled trade win rate:", f"{100*(trade_frame.net_pnl>0).mean():.2f}%")
        print("Total paper P&L across independently reset folds:",
              f"${trade_frame.net_pnl.sum():,.2f}")
    print("Saved results to:", OUTPUT_DIR)
    print("PAPER RESEARCH ONLY. No live trading orders are submitted.")


if __name__ == "__main__":
    main()
