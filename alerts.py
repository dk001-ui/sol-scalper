"""
alerts.py - Telegram Alert Module
Formats and sends trade signals and daily summary reports.
Uses python-telegram-bot (v20+ async API).
"""

import asyncio
from datetime import datetime, timezone
from telegram import Bot
from telegram.constants import ParseMode

# These are loaded from environment variables in live.py
# Pass them in when calling send_* functions
# BOT_TOKEN = os.environ['TELEGRAM_BOT_TOKEN']
# CHAT_ID   = os.environ['TELEGRAM_CHAT_ID']


REGIME_EMOJI = {
    'TRENDING_UP':   'UP',
    'TRENDING_DOWN': 'DOWN',
    'TRANSITIONAL':  'TRANSITIONAL',
    'RANGING':       'RANGING',
}

DIRECTION_LABEL = {
    'LONG':  'LONG  [BUY]',
    'SHORT': 'SHORT [SELL]',
}


def format_signal_alert(signal: dict, sharpe: float, guard: dict) -> str:
    """
    Format a trade signal into a structured Telegram message.

    signal dict keys:
        direction, entry_price, stop_loss, target1, target2,
        atr_trail, leverage, regime, adx, long_score, short_score,
        cmf, rsi, vwap, macd_hist

    sharpe: current rolling Sharpe (float)
    guard:  dict from sharpe_guard_status()
    """
    d         = signal['direction']
    score     = signal['long_score'] if d == 'LONG' else signal['short_score']
    regime    = signal['regime']
    lev       = signal['leverage']
    ts        = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    # Confluence checklist (4 conditions)
    # We infer which fired from score and direction
    if d == 'LONG':
        checks = [
            ('MACD golden cross (5M)',    signal['macd_hist'] > 0),
            ('CMF > 0.05 (bullish flow)', signal['cmf'] > 0.05),
            ('Price above VWAP',          signal['entry_price'] > signal['vwap']),
            ('RSI 45-65 (entry zone)',    45 <= signal['rsi'] <= 65),
        ]
    else:
        checks = [
            ('MACD death cross (5M)',     signal['macd_hist'] < 0),
            ('CMF < -0.05 (sell flow)',   signal['cmf'] < -0.05),
            ('Price below VWAP',          signal['entry_price'] < signal['vwap']),
            ('RSI 35-55 (entry zone)',    35 <= signal['rsi'] <= 55),
        ]

    confluence_lines = '\n'.join(
        f"  {'[X]' if fired else '[ ]'} {label}"
        for label, fired in checks
    )

    # Sharpe guard section
    guard_line = ''
    if guard['halt']:
        guard_line = '\n*** SHARPE GUARD ACTIVE: HALTED -- no new entries ***\n'
    elif guard['warn']:
        guard_line = f"\n*** WARNING: Size reduced to {guard['multiplier']*100:.0f}% (Sharpe guard) ***\n"

    msg = f"""
SOL/USDT PERP  |  {ts}
{'='*42}
DIRECTION : {DIRECTION_LABEL[d]}
REGIME    : {REGIME_EMOJI.get(regime, regime)}  (ADX: {signal['adx']:.1f})
LEVERAGE  : {lev}x  (ADX-scaled, max 20x)
{'='*42}
ENTRY     : ${signal['entry_price']:.4f}  (market)
STOP LOSS : ${signal['stop_loss']:.4f}  (-1.5% hard stop)
TARGET 1  : ${signal['target1']:.4f}  (+1.5% -- close 50%)
TARGET 2  : ${signal['target2']:.4f}  (+3.0% -- trail remainder)
ATR TRAIL : ${signal['atr_trail']:.4f}  (1.5x ATR -- activates after T1)
{'='*42}
CONFLUENCE CHECKLIST ({score}/4 fired):
{confluence_lines}
{'='*42}
ROLLING SHARPE (last 20 trades): {sharpe:.2f}
SHARPE GUARD  : {guard['label']}
{guard_line}
[Risk: max 1% capital per trade. Never exceed 20x.]
""".strip()

    return msg


def format_daily_summary(
    regime_info: dict,
    signal_count_24h: int,
    win_rate_24h: float,
    rolling_sharpe: float,
    guard: dict,
    warnings: list[str] | None = None,
) -> str:
    """
    Format the 00:00 UTC daily summary alert.

    regime_info: dict from get_current_regime()
    signal_count_24h: number of signals fired in last 24H
    win_rate_24h: win rate as fraction (e.g. 0.62 = 62%)
    rolling_sharpe: current Sharpe float
    guard: dict from sharpe_guard_status()
    warnings: optional list of warning strings
    """
    ts      = datetime.now(timezone.utc).strftime('%Y-%m-%d UTC')
    regime  = regime_info['regime']
    adx     = regime_info['adx']
    lev_rec = regime_info['leverage']

    # Recommended leverage range based on current regime
    lev_ranges = {
        'TRENDING_UP':   f"{lev_rec}x  (trending, full scale)",
        'TRENDING_DOWN': f"{lev_rec}x  (trending short, full scale)",
        'TRANSITIONAL':  '3x-5x  (capped -- transitional regime)',
        'RANGING':       '2x MAX  (chop filter active)',
    }
    lev_range_str = lev_ranges.get(regime, f'{lev_rec}x')

    warnings_section = ''
    if warnings:
        warnings_section = '\nWARNINGS:\n' + '\n'.join(f'  - {w}' for w in warnings)

    guard_note = ''
    if guard['halt']:
        guard_note = '*** SHARPE GUARD: HALTED -- review strategy before next session ***'
    elif guard['warn']:
        guard_note = f'*** SHARPE GUARD: Active -- size at {guard["multiplier"]*100:.0f}% ***'

    msg = f"""
SOL/USDT DAILY SUMMARY  |  {ts}
{'='*42}
CURRENT REGIME     : {REGIME_EMOJI.get(regime, regime)}
ADX (4H)           : {adx:.1f}
{'='*42}
24H SIGNALS        : {signal_count_24h}
24H WIN RATE       : {win_rate_24h*100:.1f}%
ROLLING SHARPE     : {rolling_sharpe:.2f}  (last 20 trades)
{'='*42}
LEVERAGE RANGE     : {lev_range_str}
SHARPE GUARD       : {guard['label']}
{guard_note}
{warnings_section}
{'='*42}
[Next regime check at next 4H candle close]
""".strip()

    return msg


async def _send_message(bot_token: str, chat_id: str, text: str):
    """Internal async sender."""
    bot = Bot(token=bot_token)
    # Telegram message limit is 4096 chars; split if needed
    for i in range(0, len(text), 4000):
        await bot.send_message(
            chat_id=chat_id,
            text=text[i:i+4000],
            parse_mode=None,   # plain text -- avoid markdown escaping issues
        )


def send_signal_alert(bot_token: str, chat_id: str, signal: dict,
                      sharpe: float, guard: dict):
    """Synchronous wrapper -- formats and sends a trade signal alert."""
    text = format_signal_alert(signal, sharpe, guard)
    asyncio.run(_send_message(bot_token, chat_id, text))
    print(f"  [ALERT SENT] {signal['direction']} @ {signal['entry_price']}")


def send_daily_summary(bot_token: str, chat_id: str, regime_info: dict,
                       signal_count_24h: int, win_rate_24h: float,
                       rolling_sharpe: float, guard: dict,
                       warnings: list[str] | None = None):
    """Synchronous wrapper -- formats and sends the daily summary."""
    text = format_daily_summary(
        regime_info, signal_count_24h, win_rate_24h,
        rolling_sharpe, guard, warnings
    )
    asyncio.run(_send_message(bot_token, chat_id, text))
    print(f"  [DAILY SUMMARY SENT]")


def send_sharpe_halt_alert(bot_token: str, chat_id: str, sharpe: float):
    """Emergency alert when Sharpe drops below zero and trading is halted."""
    ts  = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    msg = f"""
*** SHARPE GUARD EMERGENCY HALT ***
{ts}
Rolling Sharpe has dropped below 0.0: {sharpe:.3f}
ALL NEW ENTRIES SUSPENDED.
Review last 20 trades before resuming.
SOL/USDT Scalper -- Alert System
""".strip()
    asyncio.run(_send_message(bot_token, chat_id, msg))
    print(f"  [HALT ALERT SENT] Sharpe: {sharpe:.3f}")
