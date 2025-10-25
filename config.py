import os
from binance.client import Client
from dotenv import load_dotenv

load_dotenv()

# --- API keys ---
API = os.getenv("API")
SECRET = os.getenv("SECRET")

# --- Markets & timeframe ---
# Symbols are BASE assets; bot will trade BASE + QUOTE_ASSET (e.g., BTCUSDT)
markets = ['PEPE', 'BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA']
tick_interval = Client.KLINE_INTERVAL_30MINUTE  # evaluate signals on closed 30m candles

# --- Trading currency & sizing ---
QUOTE_ASSET = "USDT"          # Quote currency to trade against
TRADE_QUOTE_AMOUNT = 10.0     # Fixed quote spend per entry (e.g., 10 USDT via quoteOrderQty)

# --- Data / loop behavior ---
DEFAULT_KLINES_LIMIT = 500    # Candles fetched each loop for indicators/regime detection
ALIGN_TO_CANDLE = True        # Wait for candle close before evaluating signals
BASE_POLL_SECONDS = 5         # Small idle sleep between symbols
MAX_BACKOFF_SECONDS = 60      # Cap for exponential backoff on errors

# --- Regime detector (stable-only entries) ---
# Matches your CLI example: --stable-only --regime-vol-window 30 --regime-vol-low-pct 0.8
#                           --regime-vol-high-pct 0.8 --regime-adx 20 --regime-min-history 200
STABLE_ONLY = True
REGIME_VOL_WINDOW = 30
REGIME_VOL_LOW_PCT = 0.80
REGIME_VOL_HIGH_PCT = 0.80
REGIME_ADX_THRESHOLD = 20.0
REGIME_MIN_HISTORY = 200
