# SOL/USDT 5M Scalping Bot

Production-ready signal engine for Solana perpetual futures. Regime-filtered, leverage-scaled, Sharpe-optimized.

## Architecture

| Module | File | Purpose |
|---|---|---|
| Regime Detection | `regime.py` | 4H ADX + EMA50/200 -> TRENDING / TRANSITIONAL / RANGING |
| Signal Engine | `signals.py` | 5M MACD + CMF + VWAP + RSI confluence scoring |
| Backtester | `backtest.py` | VectorBT 90-day backtest, long/short/combined |
| Alert System | `alerts.py` | Telegram signal + daily summary formatting |
| Live Loop | `live.py` | 10s polling loop, Sharpe guard, position limits |

## Strategy Logic

### Regime Filter (4H)
- **TRENDING_UP**: ADX > 25, EMA50 > EMA200, positive slope -> long bias, full leverage
- **TRENDING_DOWN**: ADX > 25, EMA50 < EMA200, negative slope -> short bias, full leverage
- **TRANSITIONAL**: ADX 20-25 -> 4/4 confluence required, 5x leverage cap
- **RANGING**: ADX < 20 -> all entries suppressed

### Entry Confluence (5M, minimum 3/4)
1. MACD(12,26,9) histogram cross
2. CMF(20) > 0.05 (long) or < -0.05 (short)
3. Price above/below daily VWAP
4. RSI(14) in 45-65 (long) or 35-55 (short)

### Dynamic Leverage
| ADX | Leverage |
|---|---|
| < 20 | 2x |
| 20-25 | 5x |
| 25-35 | 10x |
| > 35 | 20x |

### Exit Rules (priority order)
1. Hard stop: -1.5% from entry
2. ATR trailing stop: 1.5x ATR(14), activates after T1
3. MACD cross opposite direction
4. Time exit: no 0.5% move within 20 minutes -> close

### Sharpe Guard
| Rolling Sharpe (last 20 trades) | Action |
|---|---|
| > 1.0 | Normal operation |
| 0.5-1.0 | Size reduced 50% |
| 0.0-0.5 | Size reduced 75%, warning alert |
| < 0.0 | All entries halted, emergency Telegram alert |

## Quick Start

```bash
pip install -r requirements.txt

# Run 90-day backtest (no API keys needed)
python backtest.py

# Run live signal engine
export TELEGRAM_BOT_TOKEN=your_token
export TELEGRAM_CHAT_ID=your_chat_id
python live.py
```

## Telegram Alert Format

Each signal alert includes:
- Direction (LONG/SHORT), entry price, leverage recommendation
- Stop loss (-1.5%), Target 1 (+1.5%), Target 2 (+3.0%), ATR trail level
- Confluence checklist (which of 4 conditions fired)
- Current rolling Sharpe and guard status
- Regime classification with ADX reading

Daily summary at 00:00 UTC: regime, 24H signal count, win rate, Sharpe, leverage range.

## Key Design Decisions

- **85th percentile vol threshold** for SOL-specific regime detection (not generic 60%)
- **EWM-weighted Sharpe** -- recent trades count more than old ones (avoids false guard triggers)
- **VWAP blocked 00:00-00:30 UTC** -- VWAP is noise in first 30 min of day
- **First 5M bar after 4H close blocked** -- regime recalculation period
- **Transitional regime caps leverage at 5x** -- not 5x-10x as per raw ADX scaling

## Research

Full strategy research: [Solana Trading Strategy Research](research/solana-trading-strategies.md)
