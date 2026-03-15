"""
Module lấy dữ liệu Bitcoin lịch sử thực tế từ web
Get real Bitcoin historical data from web
"""

import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf


def get_bitcoin_historical_data(start_date, end_date):
    """
    Lấy dữ liệu Bitcoin lịch sử thực tế từ Yahoo Finance.
    Yahoo giới hạn 1h trong ~730 ngày gần nhất. Với khoảng dài hơn, fallback sang 1d.
    """
    try:
        print(f"📥 Fetching Bitcoin data from {start_date} to {end_date}...")

        btc_ticker = yf.Ticker("BTC-USD")

        # Decide interval based on range
        days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        interval = "1h" if days <= 720 else "1d"
        if interval == "1d":
            print("ℹ️ Range > 720 days detected. Falling back to daily interval (1d) for Yahoo limits.")

        data = btc_ticker.history(start=start_date, end=end_date, interval=interval)

        if data is None or len(data) == 0:
            print("⚠️ No data from Yahoo Finance")
            return None

        data.columns = [col.lower() for col in data.columns]
        print(f"✅ Got {len(data)} rows from Yahoo Finance (interval={interval})")
        print(f"   Date range: {data.index.min()} to {data.index.max()}")
        return data

    except Exception as e:
        # Specific hint for common 1h limitation
        msg = str(e)
        if "1h data not available" in msg and "last 730 days" in msg:
            try:
                print("ℹ️ Yahoo 1h range limit hit. Retrying with daily interval (1d)...")
                btc_ticker = yf.Ticker("BTC-USD")
                data = btc_ticker.history(start=start_date, end=end_date, interval="1d")
                if data is not None and len(data) > 0:
                    data.columns = [col.lower() for col in data.columns]
                    print(f"✅ Got {len(data)} rows from Yahoo Finance (interval=1d)")
                    return data
            except Exception as e2:
                print(f"⚠️ Fallback to 1d failed: {e2}")
        print(f"⚠️ Error fetching from Yahoo Finance: {e}")
        return None


def get_bitcoin_major_events():
    """
    Lấy danh sách sự kiện lớn của Bitcoin
    
    Returns:
        Dict với key là date và value là event description
    """
    events = {
        '2018-01-06': 'Bitcoin đạt đỉnh $17k (đầu năm 2018)',
        '2018-12-15': 'Đáy bear market $3.1k',
        '2019-06-26': 'Khởi phục đạt $13k',
        '2020-03-12': 'COVID-19 crash $3.8k',
        '2021-04-14': 'Đỉnh cao nhất $64k',
        '2021-11-10': 'Đỉnh ATH $69k',
        '2022-11-21': 'Đáy bear market $15.4k',
        '2023-01-01': 'Khởi phục $24k',
        '2024-03-05': 'Đỉnh mới $73k',
        '2024-07-05': 'Đáy pullback $53k',
    }
    
    return events


def analyze_period(data):
    """
    Phân tích period của data với lịch sử Bitcoin
    
    Args:
        data: DataFrame với index datetime
    
    Returns:
        Dict analysis
    """
    if data is None or len(data) == 0:
        return None
    
    start_date = data.index.min()
    end_date = data.index.max()
    price_start = data['close'].iloc[0]
    price_end = data['close'].iloc[-1]
    price_change = ((price_end - price_start) / price_start) * 100
    
    # Phân loại period
    events = get_bitcoin_major_events()
    
    # So sánh với các events
    closest_events = []
    for event_date, event_desc in events.items():
        event_dt = pd.to_datetime(event_date)
        if start_date <= event_dt <= end_date:
            closest_events.append((event_date, event_desc, event_dt))
    
    # Phân loại volatility
    returns = data['close'].pct_change()
    volatility = returns.std() * np.sqrt(24) * 365  # Annualized
    
    # Phân loại
    if price_change > 20:
        period_type = "Bullish (Giá tăng mạnh)"
        period_score = "Excellent"
    elif price_change > 5:
        period_type = "Moderate Growth (Tăng vừa)"
        period_score = "Good"
    elif price_change > -5:
        period_type = "Sideways (Đi ngang)"
        period_score = "Neutral"
    elif price_change > -20:
        period_type = "Correction (Điều chỉnh)"
        period_score = "Challenging"
    else:
        period_type = "Bearish (Giảm mạnh)"
        period_score = "Difficult"
    
    analysis = {
        'start_date': start_date,
        'end_date': end_date,
        'price_start': price_start,
        'price_end': price_end,
        'price_change': price_change,
        'volatility': volatility,
        'period_type': period_type,
        'period_score': period_score,
        'events_in_period': closest_events,
        'max_price': data['close'].max(),
        'min_price': data['close'].min(),
        'avg_price': data['close'].mean()
    }
    
    return analysis


if __name__ == "__main__":
    # Test
    start = datetime(2023, 1, 1)
    end = datetime.now()
    
    data = get_bitcoin_historical_data(start, end)
    
    if data is not None:
        print("\nSample data:")
        print(data.head())
        print("\nStatistics:")
        print(data.describe())
        
        # Analyze
        analysis = analyze_period(data)
        print("\nAnalysis:")
        for key, value in analysis.items():
            print(f"  {key}: {value}")

