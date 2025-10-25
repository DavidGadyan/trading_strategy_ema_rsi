# backtesting.py
import argparse
from binance.client import Client
import pandas as pd
import numpy as np
import talib
from datetime import timedelta

from config import (
    API, SECRET, QUOTE_ASSET, TRADE_QUOTE_AMOUNT
)
from Util import binanceToPandas
from Strategy import strategyCalculator  # exact entry/exit rules

# -----------------------------
# Interval mapping
# -----------------------------
INTERVAL_MAP = {
    "1h": Client.KLINE_INTERVAL_1HOUR,
    "30m": Client.KLINE_INTERVAL_30MINUTE,
    "15m": Client.KLINE_INTERVAL_15MINUTE,
    "5m": Client.KLINE_INTERVAL_5MINUTE,
}

# For subtracting lookback windows
INTERVAL_TO_MINUTES = {
    "1h": 60,
    "30m": 30,
    "15m": 15,
    "5m": 5,
}

def fetch_klines_df(client: Client, symbol: str, interval_key: str, start: str, end: str, lookback_bars: int) -> pd.DataFrame:
    """
    Fetch historical klines with an extra `lookback_bars` window BEFORE `start`,
    to avoid look-ahead bias when computing indicators.
    """
    start_dt = pd.to_datetime(start)
    minutes = INTERVAL_TO_MINUTES[interval_key]
    adjusted_start = start_dt - timedelta(minutes=minutes * lookback_bars)

    raw = client.get_historical_klines(
        symbol=symbol,
        interval=INTERVAL_MAP[interval_key],
        start_str=adjusted_start.strftime("%Y-%m-%d %H:%M:%S"),
    )
    if not raw:
        raise RuntimeError("No klines returned. Check symbol/period/interval.")
    df = binanceToPandas(raw)
    # Ensure types
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)
    return df

def compute_indicators(df: pd.DataFrame):
    """
    Compute the same indicators as Strategy.calculateIndicators, vectorized.
    """
    close = df["Close"].values
    high  = df["High"].values
    low   = df["Low"].values

    ema8  = talib.EMA(close, timeperiod=8)
    ema13 = talib.EMA(close, timeperiod=13)
    ema21 = talib.EMA(close, timeperiod=21)
    ema34 = talib.EMA(close, timeperiod=34)
    ema55 = talib.EMA(close, timeperiod=55)

    rsi = talib.RSI(close, timeperiod=14)
    kFast, _ = talib.STOCHF(high, low, close, fastk_period=14)

    return ema8, ema13, ema21, ema34, ema55, rsi, kFast

def backtest(df: pd.DataFrame, symbol: str, interval_key: str, start_trade_time: pd.Timestamp, close_on_end: bool, fee_bps: int) -> pd.DataFrame:
    """
    Long-only backtest:
      - Compute indicators across the WHOLE df (which includes lookback).
      - Start evaluating/acting ONLY from `start_trade_time` onward (no look-ahead).
      - Enter on Strategy enterLong at bar close when flat.
      - Exit on Strategy exitLong at bar close when holding.
      - Size = TRADE_QUOTE_AMOUNT (e.g., 10 USDT) per entry.
      - Fees modeled as taker: fee_bps on entry and fee_bps on exit (default 10 bps = 0.1% each side).
      - P/L computed in quote terms and percent after fees.
      - Optionally force-close at final candle if still holding.
    """
    ema8, ema13, ema21, ema34, ema55, rsi, kFast = compute_indicators(df)

    # Determine index to begin acting
    trade_start_idx = int(np.searchsorted(df["Open Time"].values, np.datetime64(start_trade_time)))
    valid_mask = ~(
        np.isnan(ema8) | np.isnan(ema13) | np.isnan(ema21) |
        np.isnan(ema34) | np.isnan(ema55) | np.isnan(rsi) | np.isnan(kFast)
    )
    if not valid_mask.any():
        raise RuntimeError("Indicators are NaN; not enough data for selected period/interval.")

    first_valid_idx = int(np.argmax(valid_mask))
    start_idx = max(trade_start_idx, first_valid_idx)

    holding     = False
    entry_price = None
    entry_time  = None

    trade_rows = []
    Q = float(TRADE_QUOTE_AMOUNT)
    f = float(fee_bps) / 10_000.0  # e.g., 10 bps = 0.001

    for i in range(start_idx, len(df)):
        e8, e13, e21, e34, e55 = ema8[i], ema13[i], ema21[i], ema34[i], ema55[i]
        rrsi = rsi[i]
        kf   = kFast[i]
        if np.isnan([e8, e13, e21, e34, e55, rrsi, kf]).any():
            continue

        enterLong, exitLong = strategyCalculator(e8, e13, e21, e34, e55, rrsi, kf)

        close_price = float(df["Close"].iloc[i])
        ts = pd.to_datetime(df["Open Time"].iloc[i])

        if holding:
            if exitLong:
                exit_price = close_price
                gross_mult = exit_price / entry_price  # exit/entry

                # Fee-aware PnL:
                entry_fee = Q * f
                gross_proceeds = Q * gross_mult
                exit_fee = gross_proceeds * f
                net_proceeds = gross_proceeds - exit_fee
                pnl_quote = net_proceeds - (Q + entry_fee)
                pnl_pct   = (pnl_quote / Q) * 100.0

                trade_rows.append({
                    "symbol": symbol,
                    "interval": interval_key,
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "exit_reason": "signal",
                    "entry_fee_quote": entry_fee,
                    "exit_fee_quote": exit_fee,
                    "pnl_pct": pnl_pct,
                    "pnl_quote": pnl_quote
                })
                holding = False
                entry_price = None
                entry_time  = None
        else:
            if enterLong:
                entry_price = close_price
                entry_time  = ts
                holding = True

    # Optionally force-close any open position on the last available bar
    if holding and close_on_end:
        exit_price = float(df["Close"].iloc[-1])
        ts = pd.to_datetime(df["Open Time"].iloc[-1])
        gross_mult = exit_price / entry_price

        entry_fee = Q * f
        gross_proceeds = Q * gross_mult
        exit_fee = gross_proceeds * f
        net_proceeds = gross_proceeds - exit_fee
        pnl_quote = net_proceeds - (Q + entry_fee)
        pnl_pct   = (pnl_quote / Q) * 100.0

        trade_rows.append({
            "symbol": symbol,
            "interval": interval_key,
            "entry_time": entry_time,
            "exit_time": ts,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "exit_reason": "end",
            "entry_fee_quote": entry_fee,
            "exit_fee_quote": exit_fee,
            "pnl_pct": pnl_pct,
            "pnl_quote": pnl_quote
        })

    return pd.DataFrame(trade_rows)

def summarize_trades(trades: pd.DataFrame):
    if trades.empty:
        return {
            "num_trades": 0,
            "win_rate": 0.0,
            "total_pnl_quote": 0.0,
            "avg_pnl_pct": 0.0,
            "median_pnl_pct": 0.0,
        }
    wins = (trades["pnl_pct"] > 0).sum()
    num = len(trades)
    total_quote = trades["pnl_quote"].sum()
    return {
        "num_trades": int(num),
        "win_rate": float(wins) / num * 100.0,
        "total_pnl_quote": float(total_quote),
        "avg_pnl_pct": float(trades["pnl_pct"].mean()),
        "median_pnl_pct": float(trades["pnl_pct"].median()),
    }

def parse_args():
    p = argparse.ArgumentParser(description="Backtest EMA/RSI/Stoch strategy on Binance klines with lookback and fees.")
    p.add_argument("--symbol", required=True, help="Trading pair, e.g., BTCUSDT")
    p.add_argument("--period_start", required=True, help="Start time, e.g., '2025-10-24 00:00:00'")
    p.add_argument("--period_end", required=True, help="End time, e.g., '2025-10-25 00:00:00'")
    p.add_argument("--interval", required=True, choices=list(INTERVAL_MAP.keys()), help="Candle interval: 1h, 30m, 15m, 5m")
    p.add_argument("--lookback", type=int, default=400, help="Bars to include BEFORE period_start (default: 400).")
    p.add_argument("--fee_bps", type=int, default=10, help="Taker fee in basis points PER SIDE (default: 10 = 0.1%).")
    p.add_argument("--no-close-on-end", dest="close_on_end", action="store_false",
                   help="Do NOT close an open position at the final candle.")
    p.set_defaults(close_on_end=True)
    return p.parse_args()

def main():
    args = parse_args()

    sym = args.symbol.upper()
    if not sym.endswith(QUOTE_ASSET):
        raise SystemExit(f"--symbol must end with {QUOTE_ASSET} (e.g., BTCUSDT). Got: {args.symbol}")

    client = Client(API, SECRET)

    # Fetch data with lookback, then backtest from period_start
    df = fetch_klines_df(client, sym, args.interval, args.period_start, args.period_end, args.lookback)
    if df.empty:
        raise SystemExit("No candles fetched for the specified period.")

    start_trade_time = pd.to_datetime(args.period_start)
    trades = backtest(df, sym, args.interval, start_trade_time, args.close_on_end, args.fee_bps)

    if trades.empty:
        print("No completed trades in the given period/interval.")
        return

    pd.set_option("display.max_columns", None)
    trades = trades.copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades["exit_time"]  = pd.to_datetime(trades["exit_time"])

    print("\n=== Trades ===")
    print(trades[[
        "symbol","interval","entry_time","exit_time",
        "entry_price","exit_price","exit_reason",
        "entry_fee_quote","exit_fee_quote","pnl_pct","pnl_quote"
    ]].to_string(index=False, justify="left", col_space=12, formatters={
        "entry_price": "{:.6f}".format,
        "exit_price": "{:.6f}".format,
        "entry_fee_quote": "{:.6f}".format,
        "exit_fee_quote": "{:.6f}".format,
        "pnl_pct": "{:.2f}".format,
        "pnl_quote": "{:.4f}".format,
    }))

    summary = summarize_trades(trades)
    print("\n=== Summary ===")
    print(f"Trades:               {summary['num_trades']}")
    print(f"Win rate:             {summary['win_rate']:.2f}%")
    print(f"Total PnL ({QUOTE_ASSET}): {summary['total_pnl_quote']:.4f}")
    print(f"Avg PnL %:            {summary['avg_pnl_pct']:.2f}%")
    print(f"Median PnL %:         {summary['median_pnl_pct']:.2f}%")

if __name__ == "__main__":
    main()
