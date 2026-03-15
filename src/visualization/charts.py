"""
Visualization Module - Improved Charts
Module vẽ biểu đồ cải tiến với nhiều thông tin hơn
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config.config import PLOT_STYLE, PLOT_DIR, SAVE_PLOTS


plt.style.use(PLOT_STYLE)


def plot_extended_price_chart(df, title="Bitcoin Price Chart (2018-2025)", save_path=None):
    """
    Vẽ biểu đồ giá mở rộng với indicators
    
    Args:
        df: DataFrame với price data
        title: Tiêu đề chart
        save_path: Đường dẫn lưu chart
    """
    fig, axes = plt.subplots(4, 1, figsize=(20, 16), 
                            gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    
    # Price and Moving Averages
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], label='Close Price', linewidth=1.5, color='blue')
    
    if 'ma_short' in df.columns:
        ax1.plot(df.index, df['ma_short'], label='MA20', linewidth=1, alpha=0.7, color='orange')
    if 'ma_long' in df.columns:
        ax1.plot(df.index, df['ma_long'], label='MA50', linewidth=1, alpha=0.7, color='red')
    
    # Bollinger Bands
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        ax1.fill_between(df.index, df['bb_upper'], df['bb_lower'], alpha=0.1, color='gray')
        ax1.plot(df.index, df['bb_upper'], '--', linewidth=0.5, alpha=0.5, color='gray', label='BB Upper')
        ax1.plot(df.index, df['bb_lower'], '--', linewidth=0.5, alpha=0.5, color='gray', label='BB Lower')
    
    ax1.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
    ax1.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(df.index[0], df.index[-1])
    
    # Volume
    ax2 = axes[1]
    colors = ['green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red' 
              for i in range(len(df))]
    ax2.bar(df.index, df['volume'], color=colors, alpha=0.5, width=0.8)
    if 'volume_ma' in df.columns:
        ax2.plot(df.index, df['volume_ma'], 'purple', linewidth=1, label='Volume MA20')
    ax2.set_ylabel('Volume', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(df.index[0], df.index[-1])
    
    # RSI
    ax3 = axes[2]
    if 'rsi' in df.columns:
        ax3.plot(df.index, df['rsi'], color='purple', linewidth=1.5)
        ax3.axhline(y=70, color='r', linestyle='--', linewidth=1, alpha=0.7, label='Overbought')
        ax3.axhline(y=30, color='g', linestyle='--', linewidth=1, alpha=0.7, label='Oversold')
        ax3.fill_between(df.index, 30, 70, alpha=0.1, color='gray')
    ax3.set_ylabel('RSI', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(df.index[0], df.index[-1])
    
    # Trend
    ax4 = axes[3]
    if 'trend' in df.columns:
        trend_colors = ['green' if t > 0 else 'red' if t < 0 else 'gray' for t in df['trend']]
        ax4.bar(df.index, df['trend'], color=trend_colors, alpha=0.5, width=0.8)
    ax4.set_ylabel('Trend', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax4.set_ylim(-1.5, 1.5)
    ax4.axhline(y=0, color='black', linewidth=0.5)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(df.index[0], df.index[-1])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Chart saved to {save_path}")
    
    if SAVE_PLOTS:
        os.makedirs(PLOT_DIR, exist_ok=True)
        default_path = os.path.join(PLOT_DIR, 'extended_price_chart.png')
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
        print(f"📊 Chart also saved to {default_path}")
    
    plt.show()
    
    return fig


def plot_trading_performance(history, mdp, save_path=None):
    """
    Vẽ biểu đồ performance của trading
    
    Args:
        history: Trading history
        mdp: MDP environment
        save_path: Đường dẫn lưu chart
    """
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    
    # Profit over time
    axes[0, 0].plot(history['episodes'], history['profits'], alpha=0.6)
    if len(history['profits']) > 50:
        window = 50
        ma = pd.Series(history['profits']).rolling(window=window).mean()
        axes[0, 0].plot(history['episodes'], ma, 'r-', linewidth=2, label=f'MA({window})')
    axes[0, 0].set_title('Profit Over Episodes', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Profit')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cumulative profit
    cumulative_profit = np.cumsum(history['profits'])
    axes[0, 1].plot(history['episodes'], cumulative_profit, 'g-', linewidth=2)
    axes[0, 1].set_title('Cumulative Profit', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Cumulative Profit')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Profit distribution
    axes[1, 0].hist(history['profits'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].axvline(np.mean(history['profits']), color='r', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(history["profits"]):.4f}')
    axes[1, 0].axvline(np.median(history['profits']), color='g', linestyle='--', 
                       linewidth=2, label=f'Median: {np.median(history["profits"]):.4f}')
    axes[1, 0].set_title('Profit Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Profit')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Epsilon decay
    axes[1, 1].plot(history['episodes'], history['epsilon'], 'purple', linewidth=2)
    axes[1, 1].set_title('Epsilon Decay', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Epsilon')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Rolling statistics
    if len(history['profits']) > 100:
        window = 100
        rolling_mean = pd.Series(history['profits']).rolling(window=window).mean()
        rolling_std = pd.Series(history['profits']).rolling(window=window).std()
        
        axes[2, 0].plot(history['episodes'], rolling_mean, 'b-', linewidth=2, label='Mean')
        axes[2, 0].fill_between(history['episodes'], 
                                rolling_mean - rolling_std, 
                                rolling_mean + rolling_std, 
                                alpha=0.3, label='±1 Std')
        axes[2, 0].set_title(f'Rolling Statistics (window={window})', fontsize=14, fontweight='bold')
        axes[2, 0].set_xlabel('Episode')
        axes[2, 0].set_ylabel('Profit')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
    
    # Win rate over time
    if len(history['profits']) > 50:
        window = 50
        profits_array = np.array(history['profits'])
        win_rate = []
        for i in range(window, len(profits_array)):
            wins = np.sum(profits_array[i-window:i] > 0)
            win_rate.append(wins / window * 100)
        
        axes[2, 1].plot(history['episodes'][window:], win_rate, 'g-', linewidth=2)
        axes[2, 1].axhline(y=50, color='r', linestyle='--', linewidth=1, alpha=0.7)
        axes[2, 1].set_title(f'Win Rate (window={window})', fontsize=14, fontweight='bold')
        axes[2, 1].set_xlabel('Episode')
        axes[2, 1].set_ylabel('Win Rate (%)')
        axes[2, 1].set_ylim(0, 100)
        axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Performance chart saved to {save_path}")
    
    if SAVE_PLOTS:
        os.makedirs(PLOT_DIR, exist_ok=True)
        default_path = os.path.join(PLOT_DIR, 'trading_performance.png')
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
        print(f"📊 Performance chart also saved to {default_path}")
    
    plt.show()
    
    return fig


def plot_comparison_chart(df, title="Bitcoin: Long-term View", save_path=None):
    """
    Vẽ biểu đồ so sánh dài hạn
    
    Args:
        df: DataFrame với price data
        title: Tiêu đề
        save_path: Đường dẫn lưu
    """
    fig, axes = plt.subplots(2, 1, figsize=(20, 10))
    
    # Linear scale
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], linewidth=2, color='blue')
    ax1.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{title} - Linear Scale', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(df.index[0], df.index[-1])
    
    # Log scale
    ax2 = axes[1]
    ax2.semilogy(df.index, df['close'], linewidth=2, color='green')
    ax2.set_ylabel('Price (USD, log scale)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_title(f'{title} - Log Scale', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(df.index[0], df.index[-1])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Comparison chart saved to {save_path}")
    
    if SAVE_PLOTS:
        os.makedirs(PLOT_DIR, exist_ok=True)
        default_path = os.path.join(PLOT_DIR, 'longterm_comparison.png')
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
        print(f"📊 Comparison chart also saved to {default_path}")
    
    plt.show()
    
    return fig
