"""
live.py - Live Signal Detection Engine
Main loop for the SOL/USDT 5M scalping bot.
Polls for new 5M and 4H candle closes, evaluates signals,
manages Sharpe guard, and dispatches Telegram alerts.

Usage:
    export TELEGRAM_BOT_TOKEN=your_token
    export TELEGRAM_CHAT_ID=your_chat_id
    python live.py

Required packages: ccxt, pandas, pandas_ta, python-telegram-bot
"""

import os
import time
import ccxt
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from collections import deque

from regime import get_current_regime, classify_regime
from signals import calc_indicators, merge_regime, score_confluence, evaluate_current_bar
from alerts import (
    send_signal_alert,
    send_daily_summary,
    send_sharpe_halt_alert,
    sharpe_guard_status,
)

# -----------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------
SYMBOL        = 'SOL/USDT'
TF_5M         = '5m'
TF_4H         = '4h'
POLL_INTERVAL = 10         # seconds between candle-close checks
CANDLES_5M    = 500        # how many 5M candles to load for indicators
CANDLES_4H    = 300        # how many 4H candles to load for regime
MAX_OPEN_POS  = 2          # never stack more than 2 positions
SHARPE_WINDOW = 20         # rolling Sharpe window (trade count)
ANNUAL_FACTOR = (252 * 6) ** 0.5  # ~6 round trips/day annualisation

# Telegram credentials from environment
BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
CHAT_ID   = os.environ.get('TELEGRAM_CHAT_ID', '')


# -----------------------------------------------------------------
# STATE
# -----------------------------------------------------------------
class BotState:
    """Mutable bot state shared across the main loop."""
    def __init__(self):
        self.last_5m_ts    = None    # timestamp of last processed 5M bar
        self.last_4h_ts    = None    # timestamp of last processed 4H bar
        self.open_positions = 0      # count of currently open positions
        self.trade_returns  = deque(maxlen=SHARPE_WINDOW)  # last N trade PnL %
        self.signals_today  = 0      # signals fired in current UTC day
        self.wins_today     = 0      # winning trades today
        self.halted         = False  # True when Sharpe guard halts trading
        self.last_regime    = None   # last known regime dict
        self.daily_summary_sent = False  # reset at UTC midnight


# -----------------------------------------------------------------
# DATA FETCHING
# -----------------------------------------------------------------
def fetch_recent(exchange: ccxt.Exchange, timeframe: str, n: int) -> pd.DataFrame:
    """Fetch the most recent N candles for SYMBOL on given timeframe."""
    bars = exchange.fetch_ohlcv(SYMBOL, timeframe, limit=n)
    df = pd.DataFrame(bars, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df.set_index('ts', inplace=True)
    df = df[~df.index.duplicated()].sort_index()
    # Drop the last (incomplete) candle -- only process closed bars
    return df.iloc[:-1]


# -----------------------------------------------------------------
# SHARPE GUARD
# -----------------------------------------------------------------
def compute_current_sharpe(trade_returns: deque) -> float:
    """
    Compute current rolling Sharpe from recent trade returns.
    Uses simple mean/std when window is full; returns 1.0 as neutral default
    when fewer than 5 trades have been recorded (not enough data).
    """
    arr = np.array(list(trade_returns))
    if len(arr) < 5:
        return 1.0   # neutral -- insufficient data, don't penalise early
    mean = np.mean(arr)
    std  = np.std(arr, ddof=1)
    if std == 0:
        return 1.0
    return float((mean / std) * ANNUAL_FACTOR)


def record_trade_result(state: BotState, pnl_pct: float, is_win: bool):
    """Record a completed trade result and update daily counters."""
    state.trade_returns.append(pnl_pct)
    if is_win:
        state.wins_today += 1


# -----------------------------------------------------------------
# DAILY RESET
# -----------------------------------------------------------------
def check_daily_reset(state: BotState, exchange: ccxt.Exchange,
                      df4: pd.DataFrame):
    """
    At UTC 00:00, send daily summary and reset daily counters.
    Triggered once per day when the clock crosses midnight UTC.
    """
    now = datetime.now(timezone.utc)
    if now.hour == 0 and now.minute < 1 and not state.daily_summary_sent:
        # Compute stats for summary
        total_today = state.signals_today
        win_rate    = (state.wins_today / total_today) if total_today > 0 else 0.0
        sharpe      = compute_current_sharpe(state.trade_returns)
        guard       = sharpe_guard_status(sharpe)
        regime_info = get_current_regime(df4)

        warnings = []
        if state.halted:
            warnings.append('Sharpe guard halt was triggered during this session')
        if total_today < 3:
            warnings.append('Low signal count -- check regime conditions')

        if BOT_TOKEN and CHAT_ID:
            send_daily_summary(
                BOT_TOKEN, CHAT_ID,
                regime_info, total_today, win_rate,
                sharpe, guard, warnings or None
            )

        # Reset daily counters
        state.signals_today       = 0
        state.wins_today          = 0
        state.daily_summary_sent  = True
        print(f"  [DAILY RESET] {now.strftime('%Y-%m-%d %H:%M UTC')}")

    elif now.hour != 0:
        state.daily_summary_sent = False   # allow next day's summary


# -----------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------
def main():
    print("\n" + "="*55)
    print("  SOL/USDT 5M SCALPER -- LIVE SIGNAL ENGINE")
    print("="*55)

    if not BOT_TOKEN or not CHAT_ID:
        print("  WARNING: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set.")
        print("           Alerts will be printed to console only.\n")

    exchange = ccxt.binance({'enableRateLimit': True})
    state    = BotState()

    print(f"  Symbol       : {SYMBOL}")
    print(f"  Timeframes   : {TF_5M} (signals)  {TF_4H} (regime)")
    print(f"  Max positions: {MAX_OPEN_POS}")
    print(f"  Sharpe window: {SHARPE_WINDOW} trades")
    print(f"  Poll interval: {POLL_INTERVAL}s")
    print("  Starting main loop...\n")

    while True:
        try:
            now = datetime.now(timezone.utc)

            # -- Fetch latest candles -----------------------------------------
            df5_raw = fetch_recent(exchange, TF_5M, CANDLES_5M)
            df4_raw = fetch_recent(exchange, TF_4H, CANDLES_4H)

            latest_5m = df5_raw.index[-1]
            latest_4h = df4_raw.index[-1]

            # -- Update 4H regime when new 4H candle closes -------------------
            if latest_4h != state.last_4h_ts:
                state.last_4h_ts  = latest_4h
                regime_info       = get_current_regime(df4_raw)
                state.last_regime = regime_info
                print(f"  [4H CLOSE] {latest_4h}  Regime: {regime_info['regime']}"
                      f"  ADX: {regime_info['adx']}  Lev: {regime_info['leverage']}x")

            # -- Check daily summary / reset ----------------------------------
            check_daily_reset(state, exchange, df4_raw)

            # -- Process new 5M candle ----------------------------------------
            if latest_5m != state.last_5m_ts:
                state.last_5m_ts = latest_5m

                # Compute current Sharpe and guard status
                sharpe = compute_current_sharpe(state.trade_returns)
                guard  = sharpe_guard_status(sharpe)

                # Halt alert -- send once when guard triggers halt
                if guard['halt'] and not state.halted:
                    state.halted = True
                    print(f"  *** SHARPE GUARD HALT *** Sharpe: {sharpe:.3f}")
                    if BOT_TOKEN and CHAT_ID:
                        send_sharpe_halt_alert(BOT_TOKEN, CHAT_ID, sharpe)

                elif not guard['halt'] and state.halted:
                    # Sharpe recovered above 0 -- resume
                    state.halted = False
                    print(f"  [GUARD LIFTED] Sharpe recovered: {sharpe:.3f}")

                # Skip signal evaluation if halted or max positions reached
                if state.halted:
                    print(f"  [{latest_5m}] HALTED (Sharpe: {sharpe:.3f})")
                    time.sleep(POLL_INTERVAL)
                    continue

                if state.open_positions >= MAX_OPEN_POS:
                    time.sleep(POLL_INTERVAL)
                    continue

                # -- Run signal engine ----------------------------------------
                df5_ind = calc_indicators(df5_raw)
                df5_reg = merge_regime(df5_ind, df4_raw)
                df5_sig = score_confluence(df5_reg)

                signal = evaluate_current_bar(df5_sig, df4_raw)

                if signal:
                    # Apply Sharpe guard size multiplier to leverage
                    adjusted_lev = max(2, int(signal['leverage'] * guard['multiplier']))
                    signal['leverage'] = adjusted_lev

                    state.signals_today  += 1
                    state.open_positions += 1

                    print(f"\n  [SIGNAL] {signal['direction']}  "
                          f"Entry: {signal['entry_price']}  "
                          f"Lev: {signal['leverage']}x  "
                          f"Regime: {signal['regime']}  "
                          f"Score: {signal.get('long_score') or signal.get('short_score')}/4")
                    print(f"           SL: {signal['stop_loss']}  "
                          f"T1: {signal['target1']}  T2: {signal['target2']}")
                    print(f"           Sharpe: {sharpe:.2f}  Guard: {guard['label']}")

                    if BOT_TOKEN and CHAT_ID:
                        send_signal_alert(BOT_TOKEN, CHAT_ID, signal, sharpe, guard)
                    else:
                        # Console fallback when Telegram not configured
                        from alerts import format_signal_alert
                        print("\n" + format_signal_alert(signal, sharpe, guard) + "\n")

                else:
                    regime_label = (state.last_regime or {}).get('regime', 'UNKNOWN')
                    print(f"  [{latest_5m}] No signal  Regime: {regime_label}  "
                          f"Sharpe: {sharpe:.2f}")

            time.sleep(POLL_INTERVAL)

        except ccxt.NetworkError as e:
            print(f"  [NETWORK ERROR] {e}  Retrying in 30s...")
            time.sleep(30)
        except ccxt.ExchangeError as e:
            print(f"  [EXCHANGE ERROR] {e}  Retrying in 60s...")
            time.sleep(60)
        except KeyboardInterrupt:
            print("\n  [STOPPED] Bot shut down by user.")
            break
        except Exception as e:
            print(f"  [ERROR] {e}")
            time.sleep(POLL_INTERVAL)


if __name__ == '__main__':
    main()
