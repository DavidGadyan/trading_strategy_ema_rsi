# regime.py
import numpy as np
import pandas as pd
import talib

# -----------------------------
# Core helpers
# -----------------------------
def _rolling_volatility(close: pd.Series, window: int) -> pd.Series:
    """
    Rolling standard deviation of simple returns.
    Returns a series aligned with `close`.
    """
    rets = close.pct_change()
    return rets.rolling(window=window, min_periods=window).std()

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    ADX on the given OHLC series. Returns a pandas Series aligned with inputs.
    """
    arr = talib.ADX(high.values, low.values, close.values, timeperiod=period)
    return pd.Series(arr, index=close.index)

# -----------------------------
# Regime detection
# -----------------------------
def detect_regime_series(
    df: pd.DataFrame,
    vol_window: int = 30,
    adx_period: int = 14,
    vol_low_pct: float = 0.40,
    vol_high_pct: float = 0.70,
    adx_trend_threshold: float = 20.0,
    min_history: int = 200,
) -> pd.DataFrame:
    """
    Detect regimes per bar using rolling volatility and ADX.

    Regime logic (data-adaptive):
      1) Compute rolling volatility (std of returns) with window=vol_window
      2) Compute two percentile thresholds of that rolling volatility *up to each bar*:
            vol_low = rolling percentile(vol, vol_low_pct)
            vol_high = rolling percentile(vol, vol_high_pct)
      3) Compute ADX (trend strength)
      4) Classify:
            if vol <= vol_low and ADX >= adx_trend_threshold  -> 'stable'   (low vol + directional)
            elif vol >= vol_high                              -> 'high_vol' (turbulent)
            else                                              -> 'other'    (mid-vol / uncertain)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ['Open Time','Open','High','Low','Close'] with numeric OHLC and
        'Open Time' as a datetime-like column (as produced by your Utils.binanceToPandas).
    vol_window : int
        Lookback for rolling volatility (default 30 bars).
    adx_period : int
        ADX period (default 14).
    vol_low_pct : float
        Lower percentile (0..1) for volatility threshold (default 0.40 = 40th pct).
    vol_high_pct : float
        Upper percentile (0..1) for volatility threshold (default 0.70 = 70th pct).
    adx_trend_threshold : float
        ADX minimum to consider trend "strong enough" (default 20).
    min_history : int
        Require at least this many rows before producing a non-NaN regime (for stability).

    Returns
    -------
    out : pd.DataFrame with index df['Open Time'] and columns:
        - 'vol'         : rolling volatility
        - 'vol_low'     : rolling low-percentile threshold of vol
        - 'vol_high'    : rolling high-percentile threshold of vol
        - 'adx'         : ADX value
        - 'regime'      : one of {'stable','high_vol','other'} (string)
        - 'stable_flag' : bool (True if regime == 'stable')
    """
    # Ensure datetime index for alignment
    out = pd.DataFrame(index=pd.to_datetime(df["Open Time"]))
    close = pd.Series(df["Close"].values, index=out.index, dtype=float)
    high  = pd.Series(df["High"].values,  index=out.index, dtype=float)
    low   = pd.Series(df["Low"].values,   index=out.index, dtype=float)

    # 1) Rolling volatility
    vol = _rolling_volatility(close, vol_window)
    out["vol"] = vol

    # 2) Rolling percentile thresholds (expanding to avoid look-ahead)
    #    We compute percentiles on the historical window up to each time.
    #    Use expanding quantile for data-adaptive thresholds.
    # NOTE: pandas expanding.quantile requires pandas >= 1.4.0. Fallback implemented if needed.
    try:
        vol_low_series  = vol.expanding(min_periods=min_history).quantile(vol_low_pct)
        vol_high_series = vol.expanding(min_periods=min_history).quantile(vol_high_pct)
    except Exception:
        # Fallback: manual expanding percentile (slower) if pandas lacks expanding.quantile
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

    # 3) ADX
    adx = _adx(high, low, close, adx_period)
    out["adx"] = adx

    # 4) Classification
    regime = []
    for i in range(len(out)):
        v   = out["vol"].iloc[i]
        vl  = out["vol_low"].iloc[i]
        vh  = out["vol_high"].iloc[i]
        ax  = out["adx"].iloc[i]

        # Not enough data yet
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

# -----------------------------
# Convenience gate for entries
# -----------------------------
def allow_entry_at(ts: pd.Timestamp, regime_df: pd.DataFrame) -> bool:
    """
    Returns True if the timestamp `ts` is classified as 'stable'.
    Assumes regime_df is indexed by 'Open Time'.
    """
    ts = pd.to_datetime(ts)
    try:
        return bool(regime_df.loc[ts, "stable_flag"])
    except KeyError:
        # If exact timestamp missing (rounding issues), try the nearest previous bar
        idx = regime_df.index.searchsorted(ts, side="right") - 1
        if idx < 0:
            return False
        return bool(regime_df.iloc[idx]["stable_flag"])
