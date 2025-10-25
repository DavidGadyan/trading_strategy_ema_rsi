# Bot.py
from binance.client import Client
from decimal import Decimal, ROUND_DOWN
from time import sleep
import logging
import math

from config import API, SECRET, markets, tick_interval
from Utils import binanceToPandas, savePickle, openPickle
from Strategy import calculateIndicators, strategyDecision

# ------------------------------
# Configuration (centralize knobs here; avoid hardcoding throughout)
# ------------------------------
QUOTE_ASSET = "USDT"           # Quote currency (kept here to avoid changing config.py)
DEFAULT_KLINES_LIMIT = 500     # How many candles to fetch for indicators
ALIGN_TO_CANDLE = True         # Trade on closed candles only (prevents repaint)
BASE_POLL_SECONDS = 5          # Small idle sleep between symbols
RISK_BUY_PCT = Decimal("0.25") # Allocate 25% of free USDT per entry
MAX_BACKOFF_SECONDS = 60       # Backoff cap on errors
SYMBOL_FILTERS_CACHE = "SymbolFilters.pickle"  # cache for exchange filters

# Map common kline intervals to milliseconds for candle-close alignment
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

class Bot:
    def __init__(self):
        logging.info("Initializing Bot...")
        self.client = Client(API, SECRET)
        logging.info("Loaded API keys")

        # Balances and positions
        self.usdt = Decimal("0")
        self.balance = []                # raw balance list from Binance (with floats cast to Decimal where needed)
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
                    ema8, ema13, ema21, ema34, ema55, rsi, kFast = calculateIndicators(klines)
                    enterLong, exitLong = strategyDecision(ema8, ema13, ema21, ema34, ema55, rsi, kFast)

                    holding = self.is_holding(symbol)

                    if holding and exitLong:
                        self.sell(symbol, klines)
                    elif not holding and enterLong:
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
        # time remaining to next boundary, with a small safety buffer
        remainder = server_time % ms
        sleep_ms = (ms - remainder) + 1500
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

        # Some markets use LOT_SIZE for both; some include MARKET_LOT_SIZE
        lot = next((f for f in info["filters"] if f["filterType"] in ("LOT_SIZE", "MARKET_LOT_SIZE")), None)
        pricef = next((f for f in info["filters"] if f["filterType"] == "PRICE_FILTER"), None)
        # MIN_NOTIONAL renamed to NOTIONAL on some markets
        notion = next((f for f in info["filters"] if f["filterType"] in ("MIN_NOTIONAL", "NOTIONAL")), None)

        if not lot or not pricef or not notion:
            raise RuntimeError(f"Incomplete filters for {symbol}")

        stepSize = Decimal(lot["stepSize"])
        tickSize = Decimal(pricef["tickSize"])
        # field may be 'minNotional' or 'notional'
        minNotional = Decimal(notion.get("minNotional") or notion.get("notional"))

        return {
            "stepSize": stepSize,
            "tickSize": tickSize,
            "minNotional": minNotional
        }

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
            # fee asset may be BNB or quote asset - not netted here
            fee = sum(Decimal(f["commission"]) for f in fills)
            avg_price = (quote_spent / base_qty) if base_qty > 0 else Decimal("0")
        else:
            # Fallback using order summary
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
        """Enter position with percent-of-cash sizing and proper min-notional checks."""
        self.refreshBalance()

        price = Decimal(str(df["Close"].iloc[-1]))
        free_q = self.usdt

        # capital allocation for this entry
        alloc = (free_q * RISK_BUY_PCT)
        if alloc <= 0:
            logging.info(f"{symbol} | No {QUOTE_ASSET} available for allocation")
            return

        f = self.symbol_filters[symbol]
        # Use quantity-based market buy to enforce LOT_SIZE
        qty = (alloc / price)
        qty = self.quantize_qty(qty, f["stepSize"])

        # Re-check notional after rounding
        notional = qty * price
        if notional < f["minNotional"]:
            logging.info(f"{symbol} | Notional {notional} < minNotional {f['minNotional']}. Skipping.")
            return

        logging.info(f"Buying {qty} {symbol} @ ~{price} (alloc {alloc} {QUOTE_ASSET})")
        order = self.client.order_market_buy(symbol=symbol, quantity=str(qty))

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

        qty = self.quantize_qty(pos["base_qty"], f["stepSize"])
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
