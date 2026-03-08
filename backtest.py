"""
backtest.py - VectorBT Backtesting Module for SOL/USDT Scalper
Fetches data via CCXT, runs regime-filtered signal backtest,
tests long-only, short-only, and combined strategies.
Outputs metrics and equity curve chart.
"""

import ccxt
import numpy as np
import pandas as pd
import vectorbt as vbt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timezone

from regime import classify_regime
from signals import calc_indicators, merge_regime, score_confluence

# Config
SYMBOL          = 'SOL/USDT'
BACKTEST_DAYS   = 90
INITIAL_CAPITAL = 10_000   # USD
FEES            = 0.0004   # 0.04% taker (Binance perps)
SLIPPAGE        = 0.0003   # 3bps estimated slippage
HARD_STOP_PCT   = 0.015    # 1.5%
T1_PCT          = 0.015    # take profit at T1


def fetch_ohlcv(timeframe: str, days: int) -> pd.DataFrame:
    """
    Fetch OHLCV data from Binance via CCXT.
    Paginates automatically to cover full date range.
    Returns clean DataFrame indexed by UTC timestamp.
    """
    exchange = ccxt.binance({'enableRateLimit': True})
    since_ms = int(
        (datetime.now(timezone.utc).timestamp() - days * 86400) * 1000
    )
    all_bars = []
    while True:
        bars = exchange.fetch_ohlcv(
            SYMBOL, timeframe, since=since_ms, limit=1000
        )
        if not bars:
            break
        all_bars.extend(bars)
        since_ms = bars[-1][0] + 1
        if len(bars) < 1000:
            break

    df = pd.DataFrame(
        all_bars, columns=['ts', 'open', 'high', 'low', 'close', 'volume']
    )
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df.set_index('ts', inplace=True)
    df = df[~df.index.duplicated()].sort_index()
    print(f"  [{timeframe}] {len(df)} candles fetched")
    return df


def compute_rolling_sharpe(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Exponentially-weighted rolling Sharpe ratio on trade returns.
    Recent trades are weighted more heavily than older ones.
    Annualised assuming ~6 trades/day on 5M scalper.
    """
    ewm_mean = returns.ewm(span=window).mean()
    ewm_std  = returns.ewm(span=window).std()
    ann      = np.sqrt(252 * 6)   # ~6 round trips/day
    sharpe   = (ewm_mean / ewm_std.replace(0, np.nan)) * ann
    return sharpe


def sharpe_guard_status(sharpe: float) -> dict:
    """Return size multiplier and status label for current Sharpe value."""
    if sharpe > 1.0:
        return {'multiplier': 1.00, 'label': 'NORMAL',     'halt': False, 'warn': False}
    elif sharpe >= 0.5:
        return {'multiplier': 0.50, 'label': 'REDUCED_50', 'halt': False, 'warn': False}
    elif sharpe >= 0.0:
        return {'multiplier': 0.25, 'label': 'REDUCED_75', 'halt': False, 'warn': True}
    else:
        return {'multiplier': 0.00, 'label': 'HALTED',     'halt': True,  'warn': True}


def run_backtest(df5: pd.DataFrame, direction: str = 'both') -> dict:
    """
    Execute VectorBT portfolio backtest.
    direction: 'long' | 'short' | 'both'
    Returns dict with portfolio object and stats.
    """
    price = df5['close']

    no_signal = pd.Series(False, index=df5.index)
    entries_long  = df5['signal_long']  if direction in ('long',  'both') else no_signal
    entries_short = df5['signal_short'] if direction in ('short', 'both') else no_signal

    # Normalise leverage to [0,1] range for VectorBT size scaling
    lev_arr = df5['leverage'].fillna(2).values.astype(float)
    size    = lev_arr / lev_arr.max()

    portfolio = vbt.Portfolio.from_signals(
        close         = price,
        entries       = entries_long,
        short_entries = entries_short,
        sl_stop       = HARD_STOP_PCT,
        tp_stop       = T1_PCT,
        init_cash     = INITIAL_CAPITAL,
        fees          = FEES,
        slippage      = SLIPPAGE,
        size          = size,
        size_type     = 'valuepercent',
        accumulate    = False,
    )

    return {
        'portfolio': portfolio,
        'stats':     portfolio.stats(),
        'direction': direction,
    }


def print_stats(result: dict):
    """Pretty-print backtest metrics."""
    s   = result['stats']
    lbl = result['direction'].upper()
    sep = '=' * 55
    print(f"\n{sep}")
    print(f"  BACKTEST RESULTS  --  {lbl}")
    print(sep)
    for key, fmt in [
        ('Total Return [%]',        '.2f'),
        ('Sharpe Ratio',            '.3f'),
        ('Sortino Ratio',           '.3f'),
        ('Max Drawdown [%]',        '.2f'),
        ('Win Rate [%]',            '.1f'),
        ('Total Trades',            'd'),
        ('Profit Factor',           '.2f'),
        ('Expectancy',              '.4f'),
    ]:
        val = s.get(key, 'N/A')
        try:
            label = key.replace(' [%]', '').replace('[', '').replace(']', '')
            if fmt == 'd':
                print(f"  {label:<28} {int(val)}")
            else:
                print(f"  {label:<28} {val:{fmt}}")
        except (TypeError, ValueError):
            print(f"  {key:<28} {val}")
    print(sep)


def plot_equity_curves(results: list, output_path: str = 'equity_curves.png'):
    """Save a 3-panel equity curve chart (long / short / combined)."""
    colors = {'long': '#00b894', 'short': '#e17055', 'both': '#0984e3'}
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        'SOL/USDT 5M Scalper -- Equity Curves (90-Day Backtest)',
        fontsize=13, fontweight='bold'
    )
    for ax, result in zip(axes, results):
        pf    = result['portfolio']
        lbl   = result['direction'].upper()
        col   = colors[result['direction']]
        s     = result['stats']
        equity = pf.value()
        ax.plot(equity.index, equity.values, color=col, linewidth=1.2, label=lbl)
        ax.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--',
                   linewidth=0.8, alpha=0.6)
        ax.set_title(
            f"{lbl}  |  Return: {s.get('Total Return [%]', 0):.1f}%  "
            f"Sharpe: {s.get('Sharpe Ratio', 0):.2f}  "
            f"MaxDD: {s.get('Max Drawdown [%]', 0):.1f}%",
            fontsize=10
        )
        ax.set_ylabel('USD')
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel('Date (UTC)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Chart saved: {output_path}")


if __name__ == '__main__':
    print("\n[1/5] Fetching 5M OHLCV data...")
    df5 = fetch_ohlcv('5m', BACKTEST_DAYS)

    print("[2/5] Fetching 4H OHLCV data...")
    df4 = fetch_ohlcv('4h', BACKTEST_DAYS)

    print("[3/5] Calculating indicators...")
    df5 = calc_indicators(df5)
    df4 = classify_regime(df4)

    print("[4/5] Merging regime and scoring confluence...")
    df5 = merge_regime(df5, df4)
    df5 = score_confluence(df5)

    n_long  = int(df5['signal_long'].sum())
    n_short = int(df5['signal_short'].sum())
    print(f"  Signals: {n_long} longs, {n_short} shorts")
    print(f"  Regime distribution:\n{df5['regime'].value_counts().to_string()}")

    print("[5/5] Running backtests (long / short / combined)...")
    r_long  = run_backtest(df5, 'long')
    r_short = run_backtest(df5, 'short')
    r_both  = run_backtest(df5, 'both')

    print_stats(r_long)
    print_stats(r_short)
    print_stats(r_both)

    plot_equity_curves([r_long, r_short, r_both])

    # Sharpe guard check on combined portfolio
    combined_rets  = r_both['portfolio'].returns()
    rolling_sharpe = compute_rolling_sharpe(combined_rets)
    cur_sharpe     = float(rolling_sharpe.dropna().iloc[-1])
    guard          = sharpe_guard_status(cur_sharpe)

    print(f"\n  Rolling Sharpe (EWM-20): {cur_sharpe:.3f}")
    print(f"  Sharpe Guard: {guard['label']}")
    if guard['halt']:
        print("  *** ALERT: ALL ENTRIES HALTED -- Sharpe < 0 ***")
    elif guard['warn']:
        print(f"  *** WARNING: Size reduced to {guard['multiplier']*100:.0f}% ***")
