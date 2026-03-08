"""
signals.py - 5M Signal Detection and Confluence Engine
Evaluates MACD, CMF, VWAP, RSI confluence for long/short entries.
Merges 4H regime context to gate and scale signals.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from regime import classify_regime, get_leverage

# Indicator params
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIG    = 9
CMF_PERIOD  = 20
RSI_PERIOD  = 14
ATR_PERIOD  = 14

# Confluence thresholds
CMF_LONG_THRESH  =  0.05
CMF_SHORT_THRESH = -0.05
RSI_LONG_LO, RSI_LONG_HI   = 45, 65
RSI_SHORT_LO, RSI_SHORT_HI = 35, 55

# Exit params
HARD_STOP_PCT  = 0.015   # 1.5% hard stop
ATR_TRAIL_MULT = 1.5     # trailing stop ATR multiplier
T1_PCT         = 0.015   # +1.5% target 1
T2_PCT         = 0.030   # +3.0% target 2
TIME_EXIT_BARS = 4       # 4 x 5M bars = 20 minutes
MIN_MOVE_PCT   = 0.005   # 0.5% minimum directional move


def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all 5M entry indicators.
    Input: OHLCV DataFrame on 5M timeframe
    Output: df with macd_hist, macd_cross_long, macd_cross_short,
            cmf, vwap, rsi, atr added
    """
    df = df.copy()

    # -- MACD -----------------------------------------------------------------
    # Histogram sign change = cross signal
    macd = ta.macd(df['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIG)
    hist_col = f'MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIG}'
    df['macd_hist']       = macd[hist_col]
    df['macd_cross_long'] = (df['macd_hist'] > 0) & (df['macd_hist'].shift(1) <= 0)
    df['macd_cross_short']= (df['macd_hist'] < 0) & (df['macd_hist'].shift(1) >= 0)

    # -- CMF(20) — Chaikin Money Flow -----------------------------------------
    # Money Flow Volume = ((C-L)-(H-C)) / (H-L) * Volume
    hl_range = (df['high'] - df['low']).replace(0, np.nan)
    mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / hl_range * df['volume']
    df['cmf'] = mfv.rolling(CMF_PERIOD).sum() / df['volume'].rolling(CMF_PERIOD).sum()

    # -- VWAP — daily reset ---------------------------------------------------
    # Typical price * volume, cumulative sum reset each UTC day
    tp = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (
        (tp * df['volume']).groupby(df.index.date).cumsum() /
        df['volume'].groupby(df.index.date).cumsum()
    )

    # -- RSI(14) --------------------------------------------------------------
    df['rsi'] = ta.rsi(df['close'], length=RSI_PERIOD)

    # -- ATR(14) — for stop sizing --------------------------------------------
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=ATR_PERIOD)

    return df


def merge_regime(df5: pd.DataFrame, df4: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill 4H regime data into 5M DataFrame.
    Blocks entries during:
      - First 5M bar after each 4H candle close (regime recalculation)
      - First 30 minutes of UTC day (VWAP noise window)
    """
    df5 = df5.copy()
    df4c = classify_regime(df4)

    # Forward-fill 4H regime columns into 5M index
    regime_cols = ['regime', 'leverage', 'adx']
    df4_ri = df4c[regime_cols].reindex(df5.index, method='ffill')
    df5['regime']   = df4_ri['regime']
    df5['leverage'] = df4_ri['leverage']
    df5['adx_4h']   = df4_ri['adx']

    # Block first bar after each 4H close
    blocked = pd.Series(False, index=df5.index)
    for t in df4.index:
        mask = (df5.index > t) & (df5.index <= t + pd.Timedelta('5min'))
        blocked[mask] = True

    # Block 00:00-00:30 UTC (VWAP is meaningless in first 30 min of day)
    utc_block = (df5.index.hour == 0) & (df5.index.minute < 30)
    df5['blocked'] = blocked | utc_block

    return df5


def score_confluence(df5: pd.DataFrame) -> pd.DataFrame:
    """
    Score long and short confluence (0-4) for each 5M bar.
    Applies regime-adjusted thresholds for entry gating.
    """
    df5 = df5.copy()

    # -- LONG conditions ------------------------------------------------------
    c_macd_l = df5['macd_cross_long']
    c_cmf_l  = df5['cmf']  >  CMF_LONG_THRESH
    c_vwap_l = df5['close'] > df5['vwap']
    c_rsi_l  = (df5['rsi'] >= RSI_LONG_LO) & (df5['rsi'] <= RSI_LONG_HI)
    df5['long_score'] = (c_macd_l.astype(int) + c_cmf_l.astype(int) +
                         c_vwap_l.astype(int) + c_rsi_l.astype(int))

    # -- SHORT conditions -----------------------------------------------------
    c_macd_s = df5['macd_cross_short']
    c_cmf_s  = df5['cmf']  <  CMF_SHORT_THRESH
    c_vwap_s = df5['close'] < df5['vwap']
    c_rsi_s  = (df5['rsi'] >= RSI_SHORT_LO) & (df5['rsi'] <= RSI_SHORT_HI)
    df5['short_score'] = (c_macd_s.astype(int) + c_cmf_s.astype(int) +
                          c_vwap_s.astype(int) + c_rsi_s.astype(int))

    # -- Entry gates (regime-adjusted) ----------------------------------------
    # TRENDING: 3/4 confluence required
    # TRANSITIONAL: 4/4 required, leverage capped at 5x
    # RANGING: no entries
    is_trend_up   = df5['regime'] == 'TRENDING_UP'
    is_trend_dn   = df5['regime'] == 'TRENDING_DOWN'
    is_transit    = df5['regime'] == 'TRANSITIONAL'
    not_blocked   = ~df5['blocked']

    df5['signal_long'] = (
        ((is_trend_up  & (df5['long_score']  >= 3)) |
         (is_transit   & (df5['long_score']  >= 4)))
        & not_blocked
    )
    df5['signal_short'] = (
        ((is_trend_dn  & (df5['short_score'] >= 3)) |
         (is_transit   & (df5['short_score'] >= 4)))
        & not_blocked
    )

    # Cap leverage at 5x in transitional regime
    df5.loc[is_transit, 'leverage'] = df5.loc[is_transit, 'leverage'].clip(upper=5)

    return df5


def get_stop_target(entry_price: float, direction: str, atr: float) -> dict:
    """
    Calculate stop loss and target levels for a given entry.
    Returns dict with: stop_loss, target1, target2, atr_trail_stop
    """
    if direction == 'long':
        stop_loss     = entry_price * (1 - HARD_STOP_PCT)
        target1       = entry_price * (1 + T1_PCT)
        target2       = entry_price * (1 + T2_PCT)
        atr_trail     = entry_price - (atr * ATR_TRAIL_MULT)
    else:  # short
        stop_loss     = entry_price * (1 + HARD_STOP_PCT)
        target1       = entry_price * (1 - T1_PCT)
        target2       = entry_price * (1 - T2_PCT)
        atr_trail     = entry_price + (atr * ATR_TRAIL_MULT)

    return {
        'stop_loss':  round(stop_loss, 4),
        'target1':    round(target1, 4),
        'target2':    round(target2, 4),
        'atr_trail':  round(atr_trail, 4),
    }


def evaluate_current_bar(df5: pd.DataFrame, df4: pd.DataFrame) -> dict | None:
    """
    Evaluate the most recent completed 5M bar for a trade signal.
    Returns signal dict if valid entry found, else None.
    """
    df5 = calc_indicators(df5)
    df5 = merge_regime(df5, df4)
    df5 = score_confluence(df5)

    last = df5.iloc[-1]
    entry_price = float(last['close'])
    atr         = float(last['atr'])

    if last['signal_long']:
        levels = get_stop_target(entry_price, 'long', atr)
        return {
            'direction':   'LONG',
            'entry_price': entry_price,
            'regime':      last['regime'],
            'leverage':    int(last['leverage']),
            'adx':         round(float(last['adx_4h']), 2),
            'long_score':  int(last['long_score']),
            'short_score': 0,
            'cmf':         round(float(last['cmf']), 4),
            'rsi':         round(float(last['rsi']), 2),
            'vwap':        round(float(last['vwap']), 4),
            'macd_hist':   round(float(last['macd_hist']), 6),
            **levels,
        }
    elif last['signal_short']:
        levels = get_stop_target(entry_price, 'short', atr)
        return {
            'direction':   'SHORT',
            'entry_price': entry_price,
            'regime':      last['regime'],
            'leverage':    int(last['leverage']),
            'adx':         round(float(last['adx_4h']), 2),
            'long_score':  0,
            'short_score': int(last['short_score']),
            'cmf':         round(float(last['cmf']), 4),
            'rsi':         round(float(last['rsi']), 2),
            'vwap':        round(float(last['vwap']), 4),
            'macd_hist':   round(float(last['macd_hist']), 6),
            **levels,
        }

    return None
