"""
regime.py - 4H Regime Detection Module
Detects SOL market regime: TRENDING_UP, TRENDING_DOWN, TRANSITIONAL, RANGING
Uses ADX(14), EMA50, EMA200, and EMA50 slope.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta

# ADX thresholds
ADX_TREND      = 25   # above this = trending
ADX_TRANSIT_LO = 20   # 20-25 = transitional
ADX_PERIOD     = 14
EMA_FAST       = 50
EMA_SLOW       = 200
SLOPE_LOOKBACK = 3    # candles for EMA slope calculation

# ADX -> leverage map: list of (adx_upper_bound, leverage)
LEVERAGE_MAP = [
    (20,   2),
    (25,   5),
    (35,  10),
    (999, 20),
]


def get_leverage(adx_val: float) -> int:
    """Map ADX reading to max allowed leverage."""
    for upper, lev in LEVERAGE_MAP:
        if adx_val < upper:
            return lev
    return 2


def classify_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: 4H OHLCV DataFrame with columns [open, high, low, close, volume]
    Output: same DataFrame with added columns:
        adx, ema50, ema200, ema50_slope, regime, leverage
    """
    df = df.copy()

    # ADX(14) - measures trend strength regardless of direction
    adx_result = ta.adx(df['high'], df['low'], df['close'], length=ADX_PERIOD)
    df['adx'] = adx_result[f'ADX_{ADX_PERIOD}']

    # EMA 50 and 200 - directional bias
    df['ema50']  = ta.ema(df['close'], length=EMA_FAST)
    df['ema200'] = ta.ema(df['close'], length=EMA_SLOW)

    # EMA50 slope over last 3 candles - confirms direction
    df['ema50_slope'] = df['ema50'].diff(SLOPE_LOOKBACK)

    # Classify regime using vectorised np.select
    conditions = [
        (df['adx'] > ADX_TREND) & (df['ema50'] > df['ema200']) & (df['ema50_slope'] > 0),
        (df['adx'] > ADX_TREND) & (df['ema50'] < df['ema200']) & (df['ema50_slope'] < 0),
        (df['adx'] >= ADX_TRANSIT_LO) & (df['adx'] <= ADX_TREND),
    ]
    choices = ['TRENDING_UP', 'TRENDING_DOWN', 'TRANSITIONAL']
    df['regime']   = np.select(conditions, choices, default='RANGING')
    df['leverage'] = df['adx'].apply(get_leverage)

    return df


def get_current_regime(df4h: pd.DataFrame) -> dict:
    """
    Given a classified 4H DataFrame, return the latest regime snapshot.
    Returns dict with: regime, adx, leverage, ema50, ema200, ema50_slope
    """
    df4h = classify_regime(df4h)
    last = df4h.iloc[-1]
    return {
        'regime':      last['regime'],
        'adx':         round(float(last['adx']), 2),
        'leverage':    int(last['leverage']),
        'ema50':       round(float(last['ema50']), 4),
        'ema200':      round(float(last['ema200']), 4),
        'ema50_slope': round(float(last['ema50_slope']), 4),
    }
