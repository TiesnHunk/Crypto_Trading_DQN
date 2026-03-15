"""
Vẽ biểu đồ nến giống Binance
Binance-style candlestick chart
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import modules (use absolute package paths)
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.data.binance_data import BinanceDataFetcher
from src.data.kaggle_data import KaggleDataLoader

try:
    from mplfinance.original_flavor import candlestick_ohlc
    import matplotlib.dates as mdates
    HAS_MPLFINANCE = True
except ImportError:
    HAS_MPLFINANCE = False


def plot_binance_style_candlestick(df, save_path='binance_candlestick.png'):
    """
    Vẽ biểu đồ nến giống Binance
    
    Args:
        df: DataFrame với OHLCV data
        save_path: Đường dẫn lưu file
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), 
                                         height_ratios=[2, 1, 1])
    
    fig.patch.set_facecolor('#1e1e1e')
    
    # Set dark background cho tất cả axes
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('#1e1e1e')
        ax.spines['bottom'].set_color('#333')
        ax.spines['top'].set_color('#333')
        ax.spines['right'].set_color('#333')
        ax.spines['left'].set_color('#333')
        ax.tick_params(colors='white', labelsize=9)
    
    # Plot 1: Candlestick chart với MA
    ax1.set_facecolor('#1e1e1e')
    
    dates = df.index
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    
    # Calculate MA
    ma_fast = df['close'].rolling(window=20).mean()
    ma_slow = df['close'].rolling(window=50).mean()
    
    # Colors: xanh cho tăng, đỏ cho giảm
    colors = ['#26a69a' if closes[i] >= opens[i] else '#ef5350' 
              for i in range(len(df))]
    
    # Plot candles manually
    for i in range(len(df)):
        date = dates[i]
        open_price = opens[i]
        close_price = closes[i]
        high = highs[i]
        low = lows[i]
        
        # Body
        body_low = min(open_price, close_price)
        body_high = max(open_price, close_price)
        body_height = body_high - body_low
        is_green = close_price >= open_price
        
        color = '#26a69a' if is_green else '#ef5350'
        
        # Draw body
        if body_height > 0:
            rect = Rectangle((mdates.date2num(date) - 0.3, body_low), 
                           0.6, body_height, 
                           facecolor=color, edgecolor=color, linewidth=0.5)
            ax1.add_patch(rect)
        
        # Draw wicks
        ax1.plot([mdates.date2num(date), mdates.date2num(date)], 
                [low, high], color=color, linewidth=1)
        
        # Draw center line
        ax1.plot([mdates.date2num(date) - 0.3, mdates.date2num(date) + 0.3], 
                [open_price, open_price], color=color, linewidth=2, alpha=0.7)
    
    # Plot MA lines
    ax1.plot(dates, ma_fast.values, label='MA 20', color='#ffa726', 
            linewidth=1.5, alpha=0.8)
    ax1.plot(dates, ma_slow.values, label='MA 50', color='#ab47bc', 
            linewidth=1.5, alpha=0.8)
    
    # Format
    ax1.set_ylabel('Giá (USD)', color='white', fontsize=11)
    ax1.set_title('BTC/USDT - Binance Candlestick Chart', 
                 color='white', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, facecolor='#1e1e1e', 
              edgecolor='#333', labelcolor='white', fontsize=9)
    ax1.grid(True, alpha=0.2, color='#333')
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax1.xaxis.set_tick_params(rotation=45, colors='white')
    
    # Plot 2: Volume bars
    volumes = df['volume'].values
    volume_colors = ['#26a69a' if closes[i] >= opens[i] else '#ef5350' 
                     for i in range(len(df))]
    
    # Convert dates to matplotlib format
    dates_num = [mdates.date2num(d) for d in dates]
    
    ax2.bar(dates_num, volumes, width=0.8, color=volume_colors, alpha=0.6)
    ax2.set_ylabel('Volume', color='white', fontsize=11)
    ax2.set_title('Volume Chart', color='white', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.2, color='#333', axis='y')
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    
    # Plot 3: Indicators (RSI + MACD)
    ax3_twin = ax3.twinx()
    ax3_twin.set_facecolor('#1e1e1e')
    
    # RSI
    if 'rsi' in df.columns:
        rsi = df['rsi'].values
        ax3.plot(dates, rsi, color='#26a69a', linewidth=1.5, label='RSI')
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5, linewidth=0.8)
        ax3.fill_between(dates, 70, rsi, alpha=0.1, color='red')
        ax3.fill_between(dates, 30, rsi, alpha=0.1, color='green')
    
    # MACD
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        macd = df['macd'].values
        signal = df['macd_signal'].values
        ax3_twin.plot(dates, macd, color='#ffa726', linewidth=1.5, label='MACD', alpha=0.8)
        ax3_twin.plot(dates, signal, color='#ab47bc', linewidth=1.5, 
                     label='Signal', alpha=0.8)
    
    ax3.set_ylabel('RSI', color='white', fontsize=11)
    ax3_twin.set_ylabel('MACD', color='white', fontsize=11)
    ax3.set_title('Technical Indicators (RSI & MACD)', 
                 color='white', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.2, color='#333')
    ax3_twin.grid(True, alpha=0.2, color='#333')
    
    # Legends
    ax3.legend(loc='upper left', frameon=True, facecolor='#1e1e1e', 
              edgecolor='#333', labelcolor='white', fontsize=8)
    ax3_twin.legend(loc='upper right', frameon=True, facecolor='#1e1e1e', 
                   edgecolor='#333', labelcolor='white', fontsize=8)
    
    # Format x-axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax3.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax3.xaxis.set_tick_params(rotation=45, colors='white')
    
    # Hide x-axis for top 2 plots
    ax1.xaxis.set_ticklabels([])
    ax2.xaxis.set_ticklabels([])
    
    # Set common x-axis
    ax3.set_xlabel('Thời Gian', color='white', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#1e1e1e')
    print(f"✅ Saved Binance-style chart to: {save_path}")


if __name__ == "__main__":
    # Test
    print("Creating Binance-style candlestick chart...")
    
    # Load data
    fetcher = BinanceDataFetcher()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    df = fetcher.get_klines('BTCUSDT', '1h', start_date, end_date, limit=500)
    
    if df is None or len(df) == 0:
        print("Using sample data...")
        dates = pd.date_range(start='2023-01-01', periods=100, freq='h')
        np.random.seed(42)
        prices = 30000 + np.cumsum(np.random.randn(100) * 100)
        
        df = pd.DataFrame({
            'open': prices + np.random.randn(100) * 50,
            'high': prices + np.abs(np.random.randn(100) * 50),
            'low': prices - np.abs(np.random.randn(100) * 50),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
    
    # Add indicators
    from src.utils.indicators import TechnicalIndicators
    df = TechnicalIndicators.add_all_indicators(df)
    df = df.dropna()
    
    # Use only recent 100 rows for demo
    if len(df) > 100:
        df = df.tail(100)
    
    plot_binance_style_candlestick(df)

