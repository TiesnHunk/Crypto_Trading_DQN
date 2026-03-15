"""
TÌM NGÀY TỐT NHẤT ĐỂ TEST DỰ BÁO
Phân tích dữ liệu lịch sử để tìm ngày có:
- Trend rõ ràng (Bull/Bear)
- Volatility vừa phải
- Nhiều tín hiệu trading
"""

import pandas as pd
import numpy as np
from datetime import datetime

def analyze_days(data_path='data/raw/multi_coin_1h.csv'):
    """Phân tích từng ngày để tìm ngày test tối ưu"""
    
    print("Loading data...")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter BTC only
    df_btc = df[df['coin'] == 'BTC'].copy()
    df_btc = df_btc.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Total BTC data: {len(df_btc)} rows")
    print(f"Date range: {df_btc['timestamp'].min()} to {df_btc['timestamp'].max()}")
    
    # Group by date
    df_btc['date'] = df_btc['timestamp'].dt.date
    
    # Analyze each day
    day_stats = []
    
    for date, group in df_btc.groupby('date'):
        if len(group) < 24:  # Skip incomplete days
            continue
        
        # Calculate metrics
        price_change = (group['close'].iloc[-1] - group['close'].iloc[0]) / group['close'].iloc[0] * 100
        avg_trend = group['trend'].mean()
        avg_rsi = group['rsi'].mean()
        avg_volatility = group['volatility'].mean()
        
        # Determine market type
        if price_change > 2 and avg_trend > 0:
            market_type = 'Strong Bull'
            score = 10
        elif price_change > 0.5 and avg_trend > 0:
            market_type = 'Bull'
            score = 8
        elif price_change < -2 and avg_trend < 0:
            market_type = 'Strong Bear'
            score = 10
        elif price_change < -0.5 and avg_trend < 0:
            market_type = 'Bear'
            score = 8
        else:
            market_type = 'Sideways'
            score = 5
        
        # Boost score for clear RSI signals
        if (market_type in ['Strong Bull', 'Bull'] and avg_rsi > 60) or \
           (market_type in ['Strong Bear', 'Bear'] and avg_rsi < 40):
            score += 2
        
        # Boost score for good volatility (not too low, not too high)
        if 0.02 < avg_volatility < 0.08:
            score += 2
        
        day_stats.append({
            'date': date,
            'price_change': price_change,
            'avg_trend': avg_trend,
            'avg_rsi': avg_rsi,
            'avg_volatility': avg_volatility,
            'market_type': market_type,
            'score': score,
            'num_hours': len(group)
        })
    
    # Convert to DataFrame
    df_stats = pd.DataFrame(day_stats)
    df_stats = df_stats.sort_values('score', ascending=False)
    
    print("\n" + "="*80)
    print("TOP 10 NGÀY TỐT NHẤT ĐỂ TEST (Có trend rõ ràng)")
    print("="*80)
    print(f"\n{'Rank':<6} {'Date':<12} {'Market':<14} {'Price %':<10} {'Trend':<10} {'RSI':<10} {'Score':<6}")
    print("-"*80)
    
    for idx, row in df_stats.head(10).iterrows():
        print(f"{idx+1:<6} {str(row['date']):<12} {row['market_type']:<14} "
              f"{row['price_change']:>8.2f}% {row['avg_trend']:>8.3f} "
              f"{row['avg_rsi']:>8.2f} {row['score']:>5.0f}")
    
    print("\n" + "="*80)
    print("TOP 5 NGÀY BULL MARKET")
    print("="*80)
    df_bull = df_stats[df_stats['market_type'].isin(['Strong Bull', 'Bull'])].head(5)
    for idx, row in df_bull.iterrows():
        print(f"  {str(row['date']):<12} | Price: +{row['price_change']:.2f}% | "
              f"Trend: {row['avg_trend']:.3f} | RSI: {row['avg_rsi']:.1f}")
    
    print("\n" + "="*80)
    print("TOP 5 NGÀY BEAR MARKET")
    print("="*80)
    df_bear = df_stats[df_stats['market_type'].isin(['Strong Bear', 'Bear'])].head(5)
    for idx, row in df_bear.iterrows():
        print(f"  {str(row['date']):<12} | Price: {row['price_change']:.2f}% | "
              f"Trend: {row['avg_trend']:.3f} | RSI: {row['avg_rsi']:.1f}")
    
    print("\n" + "="*80)
    print("KHUYẾN NGHỊ")
    print("="*80)
    
    best_day = df_stats.iloc[0]
    print(f"\nNgày tốt nhất để test: {best_day['date']}")
    print(f"  - Market type: {best_day['market_type']}")
    print(f"  - Price change: {best_day['price_change']:.2f}%")
    print(f"  - Average trend: {best_day['avg_trend']:.3f}")
    print(f"  - Average RSI: {best_day['avg_rsi']:.2f}")
    print(f"  - Score: {best_day['score']:.0f}/14")
    
    print(f"\nĐể test ngày này, sửa file predict_one_day_dqn.py:")
    print(f"  TEST_DATE = '{best_day['date']}'  # Dòng 876")
    
    # Save results
    df_stats.to_csv('results/reports/best_test_days.csv', index=False)
    print(f"\nĐã lưu kết quả vào: results/reports/best_test_days.csv")
    
    return df_stats

if __name__ == '__main__':
    analyze_days()
