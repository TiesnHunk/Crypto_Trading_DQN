# MODEL COMPARISON REPORT

Date: 2025-12-15 13:23:40
Coin: BTC
Test Period: 30 days (720 hours)
Test Range: 2025-10-02 00:00:00 to 2025-10-31 23:00:00

## MODELS TESTED
- DQN: Loaded
- DQN+PSO+LSTM: Not Available
- LSTM: Loaded
- PSO+LSTM: Not Available
- PPO+PSO+LSTM: Not Available

## PERFORMANCE METRICS

| Model | Total Return (%) | Final Balance ($) | Num Trades | Buy | Sell | Hold |
|-------|------------------|-------------------|------------|-----|------|------|
| DQN | -76.35% | $2364.73 | 664 | 2 | 662 | 56 |
| LSTM | -99.99% | $0.58 | 509 | 262 | 247 | 211 |

## ANALYSIS

**Best Performer:** DQN
- Total Return: -76.35%
- Final Balance: $2364.73

**Market Condition:**
- Trend: Bearish (Giảm)
- Price Change: -7.44%
- Initial Price: $118428.46
- Final Price: $109613.24

## OBSERVATIONS

1. **Action Strategy:**
   - DQN: Hold 7.8% of time
   - LSTM: Hold 29.3% of time

2. **Trading Frequency:**
   - DQN: 664 trades
   - LSTM: 509 trades