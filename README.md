# Cryptocurrency Trader

<p align="center">
  <img src="binance.jpg" alt="Binance" width="120">
</p>


This is a fairly simple cryptocurrency trading bot that is meant to be left running and will make you a profit over time. It has about a 65% win rate which means that it may make some losing trades but will still be profitable in the long run. 

## Installation

You need to have a Binance spot trading account to use this project (I tried my best to avoid the image validation thing, try use 2FA).

1. Generate a Binance API and SECRET key

    **WRITE THESE DOWN IN A SAFE PLACE**. The secret key will not be visible if you come back to the page a second time

2. Pull this repository to your local machine (or wherever you want to run the bot)

    ```
    git clone https://github.com/DavidGadyan/trading_strategy_ema_rsi.git
    cd trading_strategy_ema_rsi
    ```

3. Create a file called `.env`

    This is where you will store you keys. Make sure that this file is secure so that nobody can see it. Create two variables and insert your keys like this:

    ```
    API=YOUR-API-KEY-HERE
    SECRET=YOUR-SECRET-KEY-HERE
    ```

4. Install dependencies

    - [ ] Write a script and requirements.txt file to do this

    I know, I need to improve this. For now, make sure you do the following.

    1. Install TA-Lib for your OS

        Follow this instructions on the [TA-Lib website](https://www.ta-lib.org/) to install TA-Lib locally

    2. Install Python dependencies

        Install the following with a `pip install` command (requirements.txt coming soon)

        - numpy
        - matplotlib
        - TA-Lib
        - python-binance
        - python-dotenv
        - pandas
5. Customize your configuration

    Edit the `config.py` to your liking. You can modify the list of coins that the bot will monitor and change the time interval for the klines (although I have found that 30 minutes works best)

6. Start the bot

    - [ ] TODO: Make this into a Linux service so that it runs on boot
    
    ```
    python3 main.py
    ```


## Strategy

The bot has a relatively simple strategy that I have blatently stolen from a Trading View users Pine Script. I highly recommend that you check it out on [Trading View](https://www.tradingview.com/script/QoNKoXwW-Simple-profitable-trading-strategy/). Every 30 seconds, the bot will fetch the klines from Binance and compute the following:

- Exponential Moving Averages (8, 13, 21, 34 and 55)
- Relative Strength Index 14
- Stochastic

It will enter long if *all* of the following is true:

- Each EMA signal being larger than the next (ema8 > ema13, ema13 > ema21 etc)
- RSI < 65
- Stochastic < 80

It will exit this trade if *any* of the following is true:

- RSI > 70
- EMA21 < EMA55
- Stochastic > 95


## Backtest command & results

Arguments
Flag	Required	Type	Example	Default	Description
--symbol	✅	str	BTCUSDT	—	Trading pair. Must end with the quote asset (e.g., USDT).
--period_start	✅	str (YYYY-MM-DD HH:MM:SS)	"2025-10-24 00:00:00"	—	Inclusive start timestamp for the test window.
--period_end	✅	str (YYYY-MM-DD HH:MM:SS)	"2025-10-25 00:00:00"	—	Exclusive end timestamp for the test window.
--interval	✅	enum	1h, 30m, 15m, 5m	—	Candlestick interval.
--lookback	❌	int	400	400	Number of candles fetched before period_start to warm up indicators (prevents look-ahead bias).
--fee_bps	❌	int	10	10	Per-side taker fee in basis points. 10 = 0.10% on entry and 0.10% on exit (market orders).
--no-close-on-end	❌	flag	--no-close-on-end	(off)	By default, any open trade is force-closed at the final candle; pass this flag to leave it open.


```python
python backtest.py --symbol BTCUSDT --period_start "2025-10-01 00:00:00" --period_end "2024-10-25 12:00:00" --interval 30m --fee_bps 10
```

```
=== Trades ===
symbol       interval     entry_time          exit_time           entry_price   exit_price    exit_reason  entry_fee_quote exit_fee_quote pnl_pct      pnl_quote
BTCUSDT      30m          2025-10-01 00:00:00 2025-10-01 04:00:00 114021.860000 114587.500000 signal       0.010000        0.010050        0.30         0.0296
BTCUSDT      30m          2025-10-01 04:30:00 2025-10-01 08:00:00 114289.010000 115089.820000 signal       0.010000        0.010070        0.50         0.0500
BTCUSDT      30m          2025-10-01 17:30:00 2025-10-01 23:00:00 116768.070000 118352.590000 signal       0.010000        0.010136        1.16         0.1156
BTCUSDT      30m          2025-10-02 04:00:00 2025-10-02 12:30:00 118568.180000 119402.420000 signal       0.010000        0.010070        0.50         0.0503
BTCUSDT      30m          2025-10-02 13:00:00 2025-10-02 13:30:00 119165.050000 119621.200000 signal       0.010000        0.010038        0.18         0.0182
BTCUSDT      30m          2025-10-02 14:30:00 2025-10-02 15:30:00 118814.090000 119910.780000 signal       0.010000        0.010092        0.72         0.0722
BTCUSDT      30m          2025-10-02 16:30:00 2025-10-03 10:00:00 119731.400000 120468.260000 signal       0.010000        0.010062        0.41         0.0415
BTCUSDT      30m          2025-10-03 12:00:00 2025-10-03 13:00:00 120215.040000 120529.700000 signal       0.010000        0.010026        0.06         0.0061
BTCUSDT      30m          2025-10-03 13:30:00 2025-10-03 14:30:00 120374.730000 120917.850000 signal       0.010000        0.010045        0.25         0.0251
BTCUSDT      30m          2025-10-03 17:30:00 2025-10-04 21:00:00 122073.250000 122327.330000 signal       0.010000        0.010021        0.01         0.0008
BTCUSDT      30m          2025-10-05 00:30:00 2025-10-05 01:30:00 122157.700000 122377.720000 signal       0.010000        0.010018       -0.02        -0.0020
BTCUSDT      30m          2025-10-05 09:00:00 2025-10-05 23:00:00 124013.400000 123465.280000 signal       0.010000        0.009956       -0.64        -0.0642
BTCUSDT      30m          2025-10-06 00:30:00 2025-10-06 11:00:00 123347.290000 124365.910000 signal       0.010000        0.010083        0.62         0.0625
BTCUSDT      30m          2025-10-06 11:30:00 2025-10-06 13:00:00 124215.000000 124848.000000 signal       0.010000        0.010051        0.31         0.0309
BTCUSDT      30m          2025-10-06 14:30:00 2025-10-06 16:30:00 124530.000000 125455.240000 signal       0.010000        0.010074        0.54         0.0542
BTCUSDT      30m          2025-10-06 17:30:00 2025-10-07 07:30:00 125284.000000 123667.990000 signal       0.010000        0.009871       -1.49        -0.1489
BTCUSDT      30m          2025-10-08 18:30:00 2025-10-09 03:00:00 123460.220000 121969.040000 signal       0.010000        0.009879       -1.41        -0.1407
BTCUSDT      30m          2025-10-13 02:00:00 2025-10-13 19:30:00 115210.690000 115792.840000 signal       0.010000        0.010051        0.30         0.0305
BTCUSDT      30m          2025-10-13 22:30:00 2025-10-14 04:00:00 115507.190000 113495.900000 signal       0.010000        0.009826       -1.94        -0.1940
BTCUSDT      30m          2025-10-19 11:30:00 2025-10-19 14:00:00 107783.470000 108396.570000 signal       0.010000        0.010057        0.37         0.0368
BTCUSDT      30m          2025-10-19 19:30:00 2025-10-20 04:30:00 108908.430000 110422.100000 signal       0.010000        0.010139        1.19         0.1188
BTCUSDT      30m          2025-10-20 09:00:00 2025-10-21 04:00:00 110800.480000 108234.680000 signal       0.010000        0.009768       -2.51        -0.2513
BTCUSDT      30m          2025-10-21 17:30:00 2025-10-22 00:30:00 111973.420000 107986.140000 signal       0.010000        0.009644       -3.76        -0.3757
BTCUSDT      30m          2025-10-23 07:30:00 2025-10-23 17:30:00 109269.900000 111241.710000 signal       0.010000        0.010180        1.60         0.1603
BTCUSDT      30m          2025-10-23 18:00:00 2025-10-24 04:00:00 110534.010000 110962.210000 signal       0.010000        0.010039        0.19         0.0187
BTCUSDT      30m          2025-10-24 05:00:00 2025-10-24 06:30:00 111001.940000 111464.360000 signal       0.010000        0.010042        0.22         0.0216
BTCUSDT      30m          2025-10-24 07:00:00 2025-10-24 21:30:00 111215.990000 111085.570000 signal       0.010000        0.009988       -0.32        -0.0317
BTCUSDT      30m          2025-10-25 00:00:00 2025-10-25 15:00:00 110850.040000 111456.300000    end       0.010000        0.010055        0.35         0.0346

=== Summary ===
Trades:               28
Win rate:             71.43%
Total PnL (USDT): -0.2301
Avg PnL %:            -0.08%
Median PnL %:         0.27%
```

python backtest_with_regime.py --symbol BTCUSDT --period_start "2025-10-01 00:00:00" --period_end   "2025-10-25 12:00:00" --interval 30m --fee_bps 10 --stable-only --regime-vol-window 30 --regime-vol-low-pct 0.7  --regime-vol-high-pct 0.7  --regime-adx 20  --regime-min-history 200



As you can observe win rate was around 70% for BTC/USDT during selected period but it varies significantly. To have more stable results this strategy can be improved by adding market regimes which improve PnL and stablity of returns over same period 

```python
python backtest_with_regime.py --symbol BTCUSDT --period_start "2025-10-01 00:00:00" --period_end   "2025-10-25 12:00:00" --interval 30m --fee_bps 10 --stable-only --regime-vol-window 30 --regime-vol-low-pct 0.8  --regime-vol-high-pct 0.8  --regime-adx 20  --regime-min-history 200
```

```
=== Trades ===
symbol       interval     entry_time          exit_time           entry_price   exit_price    exit_reason  entry_fee_quote exit_fee_quote pnl_pct      pnl_quote
BTCUSDT      30m          2025-10-01 00:00:00 2025-10-01 04:00:00 114021.860000 114587.500000 signal       0.010000        0.010050        0.30         0.0296
BTCUSDT      30m          2025-10-01 04:30:00 2025-10-01 08:00:00 114289.010000 115089.820000 signal       0.010000        0.010070        0.50         0.0500
BTCUSDT      30m          2025-10-02 04:00:00 2025-10-02 12:30:00 118568.180000 119402.420000 signal       0.010000        0.010070        0.50         0.0503
BTCUSDT      30m          2025-10-02 13:00:00 2025-10-02 13:30:00 119165.050000 119621.200000 signal       0.010000        0.010038        0.18         0.0182
BTCUSDT      30m          2025-10-02 14:30:00 2025-10-02 15:30:00 118814.090000 119910.780000 signal       0.010000        0.010092        0.72         0.0722
BTCUSDT      30m          2025-10-02 16:30:00 2025-10-03 10:00:00 119731.400000 120468.260000 signal       0.010000        0.010062        0.41         0.0415
BTCUSDT      30m          2025-10-06 00:30:00 2025-10-06 11:00:00 123347.290000 124365.910000 signal       0.010000        0.010083        0.62         0.0625
BTCUSDT      30m          2025-10-06 15:30:00 2025-10-06 16:30:00 125000.000000 125455.240000 signal       0.010000        0.010036        0.16         0.0164
BTCUSDT      30m          2025-10-06 17:30:00 2025-10-07 07:30:00 125284.000000 123667.990000 signal       0.010000        0.009871       -1.49        -0.1489
BTCUSDT      30m          2025-10-08 18:30:00 2025-10-09 03:00:00 123460.220000 121969.040000 signal       0.010000        0.009879       -1.41        -0.1407
BTCUSDT      30m          2025-10-13 02:00:00 2025-10-13 19:30:00 115210.690000 115792.840000 signal       0.010000        0.010051        0.30         0.0305
BTCUSDT      30m          2025-10-19 11:30:00 2025-10-19 14:00:00 107783.470000 108396.570000 signal       0.010000        0.010057        0.37         0.0368
BTCUSDT      30m          2025-10-19 19:30:00 2025-10-20 04:30:00 108908.430000 110422.100000 signal       0.010000        0.010139        1.19         0.1188
BTCUSDT      30m          2025-10-20 12:00:00 2025-10-21 04:00:00 111063.100000 108234.680000 signal       0.010000        0.009745       -2.74        -0.2744
BTCUSDT      30m          2025-10-23 07:30:00 2025-10-23 17:30:00 109269.900000 111241.710000 signal       0.010000        0.010180        1.60         0.1603
BTCUSDT      30m          2025-10-23 18:00:00 2025-10-24 04:00:00 110534.010000 110962.210000 signal       0.010000        0.010039        0.19         0.0187
BTCUSDT      30m          2025-10-24 05:00:00 2025-10-24 06:30:00 111001.940000 111464.360000 signal       0.010000        0.010042        0.22         0.0216
BTCUSDT      30m          2025-10-24 07:00:00 2025-10-24 21:30:00 111215.990000 111085.570000 signal       0.010000        0.009988       -0.32        -0.0317
BTCUSDT      30m          2025-10-25 07:30:00 2025-10-25 12:00:00 111489.600000 111700.000000    end       0.010000        0.010019       -0.01        -0.0011

=== Summary ===
Trades:               19
Win rate:             73.68%
Total PnL (USDT): 0.1306
Avg PnL %:            0.07%
Median PnL %:         0.30%
```