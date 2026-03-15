"""
Tìm các ngày đại diện cho Bull, Bear, và Sideways market
"""

import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/raw/multi_coin_1h.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df_btc = df[df['coin'] == 'BTC'].copy()
df_btc = df_btc.sort_values('timestamp').reset_index(drop=True)

print(f"Total BTC data: {len(df_btc)} rows")
print(f"Date range: {df_btc['timestamp'].min()} to {df_btc['timestamp'].max()}")

# Group by date and calculate daily statistics
df_btc['date'] = df_btc['timestamp'].dt.date
daily_stats = df_btc.groupby('date').agg({
    'close': ['first', 'last', 'min', 'max'],
    'trend': 'mean',
    'rsi': 'mean',
    'macd_hist': 'mean',
    'volatility': 'mean'
}).reset_index()

daily_stats.columns = ['date', 'open', 'close', 'min', 'max', 'trend_avg', 'rsi_avg', 'macd_avg', 'volatility_avg']
daily_stats['price_change_pct'] = (daily_stats['close'] - daily_stats['open']) / daily_stats['open'] * 100
daily_stats['count'] = df_btc.groupby('date').size().values

# Filter days with full 24 hours
daily_stats = daily_stats[daily_stats['count'] >= 20].copy()

print(f"\nDays with >= 20 hours data: {len(daily_stats)}")

# Classify market regimes
def classify_market(row):
    trend = row['trend_avg']
    rsi = row['rsi_avg']
    price_change = row['price_change_pct']
    
    # Bull: trend > 0.3, RSI > 55, price change > 0.5%
    if trend > 0.3 and rsi > 55 and price_change > 0.5:
        return 'BULL'
    # Bear: trend < -0.3, RSI < 45, price change < -0.5%
    elif trend < -0.3 and rsi < 45 and price_change < -0.5:
        return 'BEAR'
    # Sideways: abs(trend) < 0.3, 45 < RSI < 55, abs(price_change) < 0.5%
    elif abs(trend) < 0.3 and 45 <= rsi <= 55 and abs(price_change) < 0.5:
        return 'SIDEWAYS'
    else:
        return 'MIXED'

daily_stats['market_type'] = daily_stats.apply(classify_market, axis=1)

# Count market types
print("\nMarket Type Distribution:")
print(daily_stats['market_type'].value_counts())

# Find representative days
print("\n" + "="*60)
print("BULL MARKET DAYS (Top 10)")
print("="*60)
bull_days = daily_stats[daily_stats['market_type'] == 'BULL'].sort_values('price_change_pct', ascending=False).head(10)
print(bull_days[['date', 'price_change_pct', 'trend_avg', 'rsi_avg', 'open', 'close']])

print("\n" + "="*60)
print("BEAR MARKET DAYS (Top 10 - most negative)")
print("="*60)
bear_days = daily_stats[daily_stats['market_type'] == 'BEAR'].sort_values('price_change_pct').head(10)
print(bear_days[['date', 'price_change_pct', 'trend_avg', 'rsi_avg', 'open', 'close']])

print("\n" + "="*60)
print("SIDEWAYS MARKET DAYS (Top 10 - least volatile)")
print("="*60)
sideways_days = daily_stats[daily_stats['market_type'] == 'SIDEWAYS'].copy()
sideways_days['abs_change'] = sideways_days['price_change_pct'].abs()
sideways_days = sideways_days.sort_values('abs_change').head(10)
print(sideways_days[['date', 'price_change_pct', 'trend_avg', 'rsi_avg', 'open', 'close']])

# Select best representatives
print("\n" + "="*60)
print("RECOMMENDED TEST DAYS")
print("="*60)

if len(bull_days) > 0:
    bull_rep = bull_days.iloc[0]
    print(f"\nBULL MARKET: {bull_rep['date']}")
    print(f"  Price change: {bull_rep['price_change_pct']:.2f}%")
    print(f"  Trend avg: {bull_rep['trend_avg']:.3f}")
    print(f"  RSI avg: {bull_rep['rsi_avg']:.2f}")

if len(bear_days) > 0:
    bear_rep = bear_days.iloc[0]
    print(f"\nBEAR MARKET: {bear_rep['date']}")
    print(f"  Price change: {bear_rep['price_change_pct']:.2f}%")
    print(f"  Trend avg: {bear_rep['trend_avg']:.3f}")
    print(f"  RSI avg: {bear_rep['rsi_avg']:.2f}")

if len(sideways_days) > 0:
    sideways_rep = sideways_days.iloc[0]
    print(f"\nSIDEWAYS MARKET: {sideways_rep['date']}")
    print(f"  Price change: {sideways_rep['price_change_pct']:.2f}%")
    print(f"  Trend avg: {sideways_rep['trend_avg']:.3f}")
    print(f"  RSI avg: {sideways_rep['rsi_avg']:.2f}")
