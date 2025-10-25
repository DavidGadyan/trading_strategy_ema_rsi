# Bot.py
from binance.client import Client
from decimal import Decimal
from time import sleep
import logging

import pandas as pd
import numpy as np
import talib

from config import (
    API, SECRET, markets, tick_interval,
    QUOTE_ASSET, TRADE_QUOTE_AMOUNT,
    DEFAULT_KLINES_LIMIT, ALIGN_TO_CANDLE,
    BASE_POLL_SECONDS, MAX_BACKOFF_SECONDS,
    STABLE_ONLY, REGIME_VOL_WINDOW, REGIME_VOL_LOW_PCT,
    REGIME_VOL_HIGH_PCT, REGIME_ADX_THRESHOLD, REGIME_MIN_HISTORY
)
from Utils import binanceToPandas, savePickle, openPickle
from Strategy import calculateIndicators, strategyDecision

# ------------------------------
# Local-only settings
# ------------------------------
SYMBOL_FILTERS_CACHE = "SymbolFilters.pickle"  # cache for exchange filters

# Map kline intervals to milliseconds for candle-close alignment
INTERVAL_MS = {
    Client.KLINE_INTERVAL_1MINUTE:   60_000,
    Client.KLINE_INTERVAL_3MINUTE:   180_000,
    Client.KLINE_INTERVAL_5MINUTE:   300_000,
    Client.KLINE_INTERVAL_15MINUTE:  900_000,
    Client.KLINE_INTERVAL_30MINUTE:  1_800_000,
    Client.KLINE_INTERVAL_1HOUR:     3_600_000,
    Client.KLINE_INTERVAL_2HOUR:     7_200_000,
    Client.KLINE_INTERVAL_4HOUR:     14_400_000,
    Client.KLINE_INTERVAL_6HOUR:     21_600_000,
    Client.KLINE_INTERVAL_8HOUR:     28_800_000,
    Client.KLINE_INTERVAL_12HOUR:    43_200_000,
    Client.KLINE_INTERVAL_1DAY:      86_400_000,
    Client.KLINE_INTERVAL_3DAY:      259_200_000,
    Client.KLINE_INTERVAL_1WEEK:     604_800_000,
    Client.KLINE_INTERVAL_1MONTH:    2_592_000_000,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ============================================================
# Embedded Regime Detection (stable / high_vol / other)
# ============================================================
def _rolling_volatility(close: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation of simple returns."""
    rets = close.pct_change()
    return rets.rolling(window=window, min_periods=window).std()

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ADX on the given OHLC series."""
    arr = talib.ADX(high.values, low.values, close.values, timeperiod=period)
    return pd.Series(arr, index=close.index)

def detect_regime_series(
    df: pd.DataFrame,
    vol_window: int = REGIME_VOL_WINDOW,
    adx_period: int = 14,
    vol_low_pct: float = REGIME_VOL_LOW_PCT,
    vol_high_pct: float = REGIME_VOL_HIGH_PCT,
    adx_trend_threshold: float = REGIME_ADX_THRESHOLD,
    min_history: int = REGIME_MIN_HISTORY,
) -> pd.DataFrame:
    """
    Label each bar as 'stable' / 'high_vol' / 'other' using rolling vol percentiles + ADX.
    - stable: vol <= vol_low AND ADX >= threshold
    - high_vol: vol >= vol_high
    - other: otherwise
    """
    out = pd.DataFrame(index=pd.to_datetime(df["Open Time"]))
    close = pd.Series(df["Close"].values, index=out.index, dtype=float)
    high  = pd.Series(df["High"].values,  index=out.index, dtype=float)
    low   = pd.Series(df["Low"].values,   index=out.index, dtype=float)

    vol = _rolling_volatility(close, vol_window)
    out["vol"] = vol

    # Expanding data-adaptive percentiles
    try:
        vol_low_series  = vol.expanding(min_periods=min_history).quantile(vol_low_pct)
        vol_high_series = vol.expanding(min_periods=min_history).quantile(vol_high_pct)
    except Exception:
        # Safe fallback if expanding.quantile is unavailable
        vol_low_vals, vol_high_vals = [], []
        vals = []
        for v in vol.values:
            vals.append(v)
            arr = np.array([x for x in vals if pd.notna(x)])
            if len(arr) >= min_history:
                vol_low_vals.append(np.quantile(arr, vol_low_pct))
                vol_high_vals.append(np.quantile(arr, vol_high_pct))
            else:
                vol_low_vals.append(np.nan)
                vol_high_vals.append(np.nan)
        vol_low_series  = pd.Series(vol_low_vals, index=out.index)
        vol_high_series = pd.Series(vol_high_vals, index=out.index)

    out["vol_low"]  = vol_low_series
    out["vol_high"] = vol_high_series

    adx = _adx(high, low, close, adx_period)
    out["adx"] = adx

    regime = []
    for i in range(len(out)):
        v  = out["vol"].iloc[i]
        vl = out["vol_low"].iloc[i]
        vh = out["vol_high"].iloc[i]
        ax = out["adx"].iloc[i]

        if pd.isna(v) or pd.isna(vl) or pd.isna(vh) or pd.isna(ax):
            regime.append(np.nan)
            continue

        if (v <= vl) and (ax >= adx_trend_threshold):
            regime.append("stable")
        elif v >= vh:
            regime.append("high_vol")
        else:
            regime.append("other")

    out["regime"] = regime
    out["stable_flag"] = (out["regime"] == "stable")
    return out

def allow_entry_at(ts: pd.Timestamp, regime_df: pd.DataFrame) -> bool:
    """Return True if timestamp ts is 'stable' (or the nearest prior bar is)."""
    ts = pd.to_datetime(ts)
    try:
        return bool(regime_df.loc[ts, "stable_flag"])
    except KeyError:
        idx = regime_df.index.searchsorted(ts, side="right") - 1
        if idx < 0:
            return False
        return bool(regime_df.iloc[idx]["stable_flag"])

# ============================================================
# Bot
# ============================================================
class Bot:
    def __init__(self):
        logging.info("Initializing Bot...")
        self.client = Client(API, SECRET)
        logging.info("Loaded API keys")

        # Balances and positions
        self.usdt = Decimal("0")
        self.balance = []                # raw balance list from Binance (cast to Decimal where needed)
        self.available_currencies = []   # non-zero balance assets
        self.positions = {}              # symbol -> {base_qty, avg_entry, quote_spent, fee, order_id}

        # Exchange filters per symbol (stepSize/tickSize/minNotional)
        self.symbol_filters = {}

        # Initialize account state
        self.refreshBalance()
        logging.info("Fetched account balance")

        # Prepare symbol filters (precision + min notional)
        self.load_or_build_symbol_filters()

        # Init empty positions (no inference from last order to avoid false state)
        for base in markets:
            symbol = base + QUOTE_ASSET
            self.positions[symbol] = None

        logging.info(f"Ready. Free {QUOTE_ASSET}: {self.usdt:.2f}")

    # ------------------------------
    # Public API
    # ------------------------------
    def run(self):
        logging.info("Bot is running\n--------TRADES-------\n")
        backoff = 1
        while True:
            for base in markets:
                symbol = base + QUOTE_ASSET
                try:
                    if ALIGN_TO_CANDLE:
                        self.wait_for_candle_close(symbol, tick_interval)

                    klines = self.getKlines(symbol)

                    # Compute trading signals on CLOSED candles
                    ema8, ema13, ema21, ema34, ema55, rsi, kFast = calculateIndicators(klines)
                    enterLong, exitLong = strategyDecision(ema8, ema13, ema21, ema34, ema55, rsi, kFast)

                    # ---- Regime gate (entries only) ----
                    allow_entry = True
                    if STABLE_ONLY:
                        regime_df = detect_regime_series(
                            klines,
                            vol_window=REGIME_VOL_WINDOW,
                            adx_period=14,
                            vol_low_pct=REGIME_VOL_LOW_PCT,
                            vol_high_pct=REGIME_VOL_HIGH_PCT,
                            adx_trend_threshold=REGIME_ADX_THRESHOLD,
                            min_history=REGIME_MIN_HISTORY,
                        )
                        # Use the timestamp of the last CLOSED bar
                        ts_last = pd.to_datetime(klines["Open Time"].iloc[-1])
                        allow_entry = allow_entry_at(ts_last, regime_df)

                    holding = self.is_holding(symbol)

                    if holding and exitLong:
                        self.sell(symbol, klines)
                    elif not holding and enterLong and allow_entry:
                        self.buy(symbol, klines)

                    sleep(BASE_POLL_SECONDS)
                    backoff = 1  # reset backoff on success
                except KeyboardInterrupt:
                    logging.info("Stopping bot (KeyboardInterrupt).")
                    return
                except Exception as ex:
                    logging.error(f"{symbol} error: {ex}")
                    sleep(min(backoff, MAX_BACKOFF_SECONDS))
                    backoff *= 2

    # ------------------------------
    # Account & market data helpers
    # ------------------------------
    def refreshBalance(self):
        """Refresh account balances; store free USDT and non-zero assets."""
        acct = self.client.get_account()
        balances = acct.get("balances") or []
        self.available_currencies = []
        self.balance = []

        free_usdt = Decimal("0")
        for item in balances:
            asset = item["asset"]
            free_amt = Decimal(item["free"])
            locked_amt = Decimal(item["locked"])
            if asset == QUOTE_ASSET:
                free_usdt = free_amt
            elif free_amt > 0:
                self.available_currencies.append(asset)
                self.balance.append({"asset": asset, "free": free_amt, "locked": locked_amt})

        self.usdt = free_usdt

    def getKlines(self, symbol):
        raw_klines = self.client.get_klines(symbol=symbol, interval=tick_interval, limit=DEFAULT_KLINES_LIMIT)
        return binanceToPandas(raw_klines)

    def wait_for_candle_close(self, symbol, interval):
        """Sleep until the current kline is supposed to close, using server time."""
        ms = INTERVAL_MS.get(interval)
        if not ms:
            return  # unknown interval mapping; skip alignment
        server_time = self.client.get_server_time()["serverTime"]  # ms
        remainder = server_time % ms
        sleep_ms = (ms - remainder) + 1500  # small safety buffer
        if sleep_ms > 0:
            sleep(sleep_ms / 1000.0)

    # ------------------------------
    # Exchange filters / precision / notional
    # ------------------------------
    def load_or_build_symbol_filters(self):
        """Load filters from cache or build them from exchange info."""
        try:
            cached = openPickle(SYMBOL_FILTERS_CACHE)
            # Validate symbols are present; otherwise rebuild
            want = {base + QUOTE_ASSET for base in markets}
            if set(cached.keys()) >= want:
                self.symbol_filters = cached
                logging.info("Loaded symbol filters from cache")
                return
            else:
                logging.info("Cache missing symbols; rebuilding filters")
        except Exception:
            logging.info("No symbol filter cache found; building filters")

        for base in markets:
            symbol = base + QUOTE_ASSET
            self.symbol_filters[symbol] = self.load_symbol_filters(symbol)

        savePickle(self.symbol_filters, SYMBOL_FILTERS_CACHE)
        logging.info("Saved symbol filters to cache")

    def load_symbol_filters(self, symbol):
        """Fetch and parse LOT_SIZE, PRICE_FILTER, and MIN_NOTIONAL/NOTIONAL."""
        info = self.client.get_symbol_info(symbol)
        if not info:
            raise RuntimeError(f"No symbol info for {symbol}")

        lot = next((f for f in info["filters"] if f["filterType"] in ("LOT_SIZE", "MARKET_LOT_SIZE")), None)
        pricef = next((f for f in info["filters"] if f["filterType"] == "PRICE_FILTER"), None)
        notion = next((f for f in info["filters"] if f["filterType"] in ("MIN_NOTIONAL", "NOTIONAL")), None)

        if not lot or not pricef or not notion:
            raise RuntimeError(f"Incomplete filters for {symbol}")

        stepSize = Decimal(lot["stepSize"])
        tickSize = Decimal(pricef["tickSize"])
        minNotional = Decimal(notion.get("minNotional") or notion.get("notional"))

        return {"stepSize": stepSize, "tickSize": tickSize, "minNotional": minNotional}

    @staticmethod
    def quantize_qty(qty: Decimal, step: Decimal) -> Decimal:
        """Round quantity down to the nearest LOT_SIZE step."""
        if step == 0:
            return qty
        return (qty // step) * step

    @staticmethod
    def quantize_price(price: Decimal, tick: Decimal) -> Decimal:
        """Round price down to the nearest PRICE_FILTER tick."""
        if tick == 0:
            return price
        return (price // tick) * tick

    # ------------------------------
    # Position state
    # ------------------------------
    def is_holding(self, symbol) -> bool:
        pos = self.positions.get(symbol)
        return bool(pos and pos.get("base_qty", Decimal("0")) > 0)

    def record_entry(self, symbol, order):
        """Record position from order fills; fallback to cummulativeQuoteQty."""
        fills = order.get("fills", [])
        if fills:
            quote_spent = sum(Decimal(f["price"]) * Decimal(f["qty"]) for f in fills)
            base_qty = sum(Decimal(f["qty"]) for f in fills)
            fee = sum(Decimal(f["commission"]) for f in fills)  # fee asset may be BNB or quote
            avg_price = (quote_spent / base_qty) if base_qty > 0 else Decimal("0")
        else:
            quote_spent = Decimal(order["cummulativeQuoteQty"])
            base_qty = Decimal(order["executedQty"])
            fee = Decimal("0")
            avg_price = (quote_spent / base_qty) if base_qty > 0 else Decimal("0")

        self.positions[symbol] = {
            "base_qty": base_qty,
            "avg_entry": avg_price,
            "quote_spent": quote_spent,
            "fee": fee,
            "order_id": order["orderId"],
        }

    # ------------------------------
    # Trading actions
    # ------------------------------
    def buy(self, symbol, df):
        """
        Enter position with FIXED quote sizing using TRADE_QUOTE_AMOUNT.
        Uses market BUY with quoteOrderQty to spend exactly that quote amount,
        after checking exchange minNotional.
        """
        self.refreshBalance()

        price = Decimal(str(df["Close"].iloc[-1]))
        f = self.symbol_filters[symbol]

        # Ensure we meet exchange min notional (in quote terms)
        quote_to_spend = Decimal(str(TRADE_QUOTE_AMOUNT))
        if quote_to_spend < f["minNotional"]:
            logging.info(f"{symbol} | Configured TRADE_QUOTE_AMOUNT ({quote_to_spend}) < minNotional ({f['minNotional']}). Skipping.")
            return

        # Also ensure we actually have the quote balance
        if self.usdt < quote_to_spend:
            logging.info(f"{symbol} | Not enough {QUOTE_ASSET}: have {self.usdt}, need {quote_to_spend}.")
            return

        logging.info(f"Buying ~{symbol} with {quote_to_spend} {QUOTE_ASSET} @ ~{price}")
        # Market buy by quote amount; Binance enforces LOT_SIZE internally
        order = self.client.order_market_buy(symbol=symbol, quoteOrderQty=str(quote_to_spend))

        # Track position from actual fills
        self.record_entry(symbol, order)

    def sell(self, symbol, df):
        """Exit entire position; compute realized PnL using stored entry."""
        pos = self.positions.get(symbol)
        if not pos or pos["base_qty"] <= 0:
            logging.info(f"{symbol} | No recorded position to sell")
            return

        price = Decimal(str(df["Close"].iloc[-1]))
        f = self.symbol_filters[symbol]

        qty = pos["base_qty"]  # sell full executed base qty
        # Ensure LOT_SIZE compliance (round down if necessary)
        qty = self.quantize_qty(qty, f["stepSize"])
        notional = qty * price
        if notional < f["minNotional"]:
            logging.info(f"{symbol} | Sell notional {notional} < minNotional {f['minNotional']}. Skipping.")
            return

        logging.info(f"Selling {qty} {symbol} @ ~{price}")
        order = self.client.order_market_sell(symbol=symbol, quantity=str(qty))

        proceeds = Decimal(order["cummulativeQuoteQty"])  # may include multiple fills
        pnl = proceeds - pos["quote_spent"]               # fees in non-quote assets not netted here
        logging.info(f"{symbol} | Realized PnL: {pnl} {QUOTE_ASSET}\n")

        # Clear the position
        self.positions[symbol] = None
        # Refresh balances after trade
        self.refreshBalance()
