"""
Advanced Trading Chart Visualization
Vẽ chart trading giống Binance với candlestick, indicators, và signals
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from mplfinance.original_flavor import candlestick_ohlc
import pickle
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class TradingChartVisualizer:
    """
    Vẽ chart trading chuyên nghiệp giống Binance
    """
    
    def __init__(self, figsize=(20, 12)):
        """
        Initialize visualizer
        
        Args:
            figsize: Kích thước figure (width, height)
        """
        self.figsize = figsize
        self.colors = {
            'up': '#26a69a',      # Green candle
            'down': '#ef5350',    # Red candle
            'buy': '#00ff00',     # Buy signal
            'sell': '#ff0000',    # Sell signal
            'hold': '#ffff00',    # Hold
            'grid': '#2a2e39',    # Grid color
            'bg': '#1e222d',      # Background
            'text': '#d1d4dc',    # Text
            'volume_up': '#26a69a80',
            'volume_down': '#ef535080'
        }
    
    def plot_full_trading_chart(
        self,
        df: pd.DataFrame,
        trades: list = None,
        portfolio_history: list = None,
        save_path: str = None,
        title: str = "Bitcoin Trading Chart",
        show_volume: bool = True,
        show_indicators: bool = True,
        show_signals: bool = True
    ):
        """
        Vẽ chart đầy đủ với candlestick, indicators, và trading signals
        
        Args:
            df: DataFrame với OHLCV và indicators
            trades: List các giao dịch (từ paper trading)
            portfolio_history: List portfolio history
            save_path: Đường dẫn lưu chart
            title: Tiêu đề chart
            show_volume: Hiển thị volume
            show_indicators: Hiển thị indicators
            show_signals: Hiển thị buy/sell signals
        """
        # Setup style
        plt.style.use('dark_background')
        
        # Create subplots
        n_subplots = 4 if show_volume else 3
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(n_subplots, 1, height_ratios=[3, 1, 1, 1][:n_subplots], hspace=0.05)
        
        ax_price = fig.add_subplot(gs[0])
        ax_rsi = fig.add_subplot(gs[1], sharex=ax_price)
        ax_macd = fig.add_subplot(gs[2], sharex=ax_price)
        if show_volume:
            ax_volume = fig.add_subplot(gs[3], sharex=ax_price)
        
        # Prepare data
        df_plot = df.copy()
        df_plot['date_num'] = mdates.date2num(df_plot.index)
        
        # 1. Plot Candlestick
        self._plot_candlestick(ax_price, df_plot)
        
        # 2. Plot Indicators on price chart
        if show_indicators:
            self._plot_price_indicators(ax_price, df_plot)
        
        # 3. Plot Trading Signals
        if show_signals and trades is not None:
            self._plot_trading_signals(ax_price, df_plot, trades)
        
        # 4. Plot RSI
        self._plot_rsi(ax_rsi, df_plot)
        
        # 5. Plot MACD
        self._plot_macd(ax_macd, df_plot)
        
        # 6. Plot Volume
        if show_volume:
            self._plot_volume(ax_volume, df_plot)
        
        # 7. Plot Portfolio Performance (if available)
        if portfolio_history is not None:
            self._add_portfolio_panel(fig, portfolio_history)
        
        # Format axes
        self._format_axes(ax_price, ax_rsi, ax_macd, 
                         ax_volume if show_volume else None, 
                         df_plot)
        
        # Add title and info
        ax_price.set_title(title, fontsize=16, fontweight='bold', pad=20)
        self._add_info_box(ax_price, df_plot, trades, portfolio_history)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='#1e222d', edgecolor='none')
            print(f"✅ Chart saved to {save_path}")
        
        return fig
    
    def _plot_candlestick(self, ax, df):
        """Vẽ candlestick chart"""
        # Prepare candlestick data
        ohlc = df[['date_num', 'open', 'high', 'low', 'close']].values
        
        # Custom candlestick plotting
        for i in range(len(ohlc)):
            date_num, o, h, l, c = ohlc[i]
            
            color = self.colors['up'] if c >= o else self.colors['down']
            
            # Draw high-low line
            ax.plot([date_num, date_num], [l, h], color=color, linewidth=1)
            
            # Draw open-close rectangle
            height = abs(c - o)
            bottom = min(o, c)
            
            if height < df['close'].std() * 0.001:  # Doji
                height = df['close'].std() * 0.001
            
            rect = Rectangle(
                (date_num - 0.3, bottom),
                0.6,
                height,
                facecolor=color,
                edgecolor=color,
                alpha=0.8
            )
            ax.add_patch(rect)
        
        ax.set_ylabel('Price (USDT)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    def _plot_price_indicators(self, ax, df):
        """Vẽ indicators trên price chart (MA, Bollinger Bands)"""
        # Simple Moving Averages
        if 'sma_20' in df.columns:
            ax.plot(df.index, df['sma_20'], label='SMA 20', 
                   color='#2196f3', linewidth=1.5, alpha=0.7)
        
        if 'sma_50' in df.columns:
            ax.plot(df.index, df['sma_50'], label='SMA 50', 
                   color='#ff9800', linewidth=1.5, alpha=0.7)
        
        # Bollinger Bands
        if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
            ax.plot(df.index, df['bb_upper'], label='BB Upper', 
                   color='#9c27b0', linewidth=1, alpha=0.5, linestyle='--')
            ax.plot(df.index, df['bb_middle'], label='BB Middle', 
                   color='#9c27b0', linewidth=1, alpha=0.5)
            ax.plot(df.index, df['bb_lower'], label='BB Lower', 
                   color='#9c27b0', linewidth=1, alpha=0.5, linestyle='--')
            
            # Fill between bands
            ax.fill_between(df.index, df['bb_upper'], df['bb_lower'], 
                           color='#9c27b0', alpha=0.1)
        
        # Legend
        ax.legend(loc='upper left', fontsize=9, framealpha=0.3)
    
    def _plot_trading_signals(self, ax, df, trades):
        """Vẽ buy/sell signals"""
        buy_signals = []
        sell_signals = []
        
        for trade in trades:
            timestamp = trade['timestamp']
            price = trade['price']
            action = trade['action']
            
            if timestamp in df.index:
                if action == 'BUY':
                    buy_signals.append((timestamp, price))
                elif action == 'SELL':
                    sell_signals.append((timestamp, price))
        
        # Plot BUY signals
        if buy_signals:
            buy_times, buy_prices = zip(*buy_signals)
            ax.scatter(buy_times, buy_prices, 
                      marker='^', s=200, c=self.colors['buy'], 
                      edgecolors='white', linewidths=2,
                      label='BUY', zorder=5)
        
        # Plot SELL signals
        if sell_signals:
            sell_times, sell_prices = zip(*sell_signals)
            ax.scatter(sell_times, sell_prices, 
                      marker='v', s=200, c=self.colors['sell'], 
                      edgecolors='white', linewidths=2,
                      label='SELL', zorder=5)
        
        # Update legend
        if buy_signals or sell_signals:
            ax.legend(loc='upper left', fontsize=9, framealpha=0.3)
    
    def _plot_rsi(self, ax, df):
        """Vẽ RSI indicator"""
        if 'rsi' not in df.columns:
            return
        
        ax.plot(df.index, df['rsi'], label='RSI', 
               color='#2196f3', linewidth=2)
        
        # Overbought/Oversold lines
        ax.axhline(y=70, color='#ef5350', linestyle='--', 
                  linewidth=1, alpha=0.5, label='Overbought (70)')
        ax.axhline(y=30, color='#26a69a', linestyle='--', 
                  linewidth=1, alpha=0.5, label='Oversold (30)')
        ax.axhline(y=50, color='#757575', linestyle=':', 
                  linewidth=1, alpha=0.3)
        
        # Fill zones
        ax.fill_between(df.index, 70, 100, color='#ef5350', alpha=0.1)
        ax.fill_between(df.index, 0, 30, color='#26a69a', alpha=0.1)
        
        ax.set_ylabel('RSI', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.legend(loc='upper left', fontsize=8, framealpha=0.3)
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    def _plot_macd(self, ax, df):
        """Vẽ MACD indicator"""
        if 'macd' not in df.columns:
            return
        
        # MACD line
        ax.plot(df.index, df['macd'], label='MACD', 
               color='#2196f3', linewidth=2)
        
        # Signal line
        if 'macd_signal' in df.columns:
            ax.plot(df.index, df['macd_signal'], label='Signal', 
                   color='#ff9800', linewidth=2)
        
        # Histogram
        if 'macd_hist' in df.columns:
            colors = [self.colors['up'] if val >= 0 else self.colors['down'] 
                     for val in df['macd_hist']]
            ax.bar(df.index, df['macd_hist'], 
                  color=colors, alpha=0.5, width=0.8, label='Histogram')
        
        # Zero line
        ax.axhline(y=0, color='#757575', linestyle='-', 
                  linewidth=1, alpha=0.5)
        
        ax.set_ylabel('MACD', fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8, framealpha=0.3)
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    def _plot_volume(self, ax, df):
        """Vẽ volume bar chart"""
        if 'volume' not in df.columns:
            return
        
        # Color based on price change
        colors = [self.colors['volume_up'] if row['close'] >= row['open'] 
                 else self.colors['volume_down'] 
                 for _, row in df.iterrows()]
        
        ax.bar(df.index, df['volume'], color=colors, width=0.8, alpha=0.8)
        
        ax.set_ylabel('Volume', fontsize=11, fontweight='bold')
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    def _format_axes(self, ax_price, ax_rsi, ax_macd, ax_volume, df):
        """Format all axes"""
        # Remove x-axis labels except bottom
        plt.setp(ax_price.get_xticklabels(), visible=False)
        plt.setp(ax_rsi.get_xticklabels(), visible=False)
        
        bottom_ax = ax_volume if ax_volume else ax_macd
        
        # Format x-axis (datetime)
        bottom_ax.xaxis.set_major_formatter(
            mdates.DateFormatter('%Y-%m-%d %H:%M')
        )
        bottom_ax.xaxis.set_major_locator(
            mdates.AutoDateLocator()
        )
        
        # Rotate labels
        plt.setp(bottom_ax.xaxis.get_majorticklabels(), 
                rotation=45, ha='right')
        
        bottom_ax.set_xlabel('Time', fontsize=11, fontweight='bold')
        
        # Set date limits
        ax_price.set_xlim(df.index[0], df.index[-1])
    
    def _add_info_box(self, ax, df, trades, portfolio_history):
        """Thêm info box với stats"""
        # Calculate stats
        start_price = df.iloc[0]['close']
        end_price = df.iloc[-1]['close']
        price_change = ((end_price / start_price) - 1) * 100
        
        info_text = f"Period: {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}\n"
        info_text += f"Start Price: ${start_price:,.2f}\n"
        info_text += f"End Price: ${end_price:,.2f}\n"
        info_text += f"Price Change: {price_change:+.2f}%\n"
        
        if trades:
            info_text += f"Total Trades: {len(trades)}\n"
            buy_count = len([t for t in trades if t['action'] == 'BUY'])
            sell_count = len([t for t in trades if t['action'] == 'SELL'])
            info_text += f"Buys: {buy_count} | Sells: {sell_count}\n"
        
        if portfolio_history:
            final_value = portfolio_history[-1]['portfolio_value']
            initial_value = portfolio_history[0]['portfolio_value']
            portfolio_return = ((final_value / initial_value) - 1) * 100
            info_text += f"Portfolio Return: {portfolio_return:+.2f}%"
        
        # Add text box
        ax.text(0.02, 0.98, info_text, 
               transform=ax.transAxes,
               fontsize=9,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='#1e222d', 
                        alpha=0.8, edgecolor='#424242'),
               color='white',
               family='monospace')
    
    def _add_portfolio_panel(self, fig, portfolio_history):
        """Thêm panel hiển thị portfolio performance"""
        # Create inset axes for portfolio
        ax_portfolio = fig.add_axes([0.7, 0.72, 0.28, 0.18])
        ax_portfolio.patch.set_facecolor('#1e222d')
        ax_portfolio.patch.set_alpha(0.8)
        
        # Extract data
        df_portfolio = pd.DataFrame(portfolio_history)
        
        # Plot portfolio value
        ax_portfolio.plot(df_portfolio['timestamp'], 
                         df_portfolio['portfolio_value'],
                         color='#4caf50', linewidth=2)
        
        # Fill area
        ax_portfolio.fill_between(df_portfolio['timestamp'],
                                  df_portfolio['portfolio_value'],
                                  alpha=0.3, color='#4caf50')
        
        # Format
        ax_portfolio.set_title('Portfolio Value', fontsize=10, 
                             fontweight='bold', color='white')
        ax_portfolio.tick_params(labelsize=8, colors='white')
        ax_portfolio.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        
        # Format y-axis
        ax_portfolio.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'${x:,.0f}')
        )
        
        # Remove x labels
        ax_portfolio.set_xticks([])


def visualize_trading_results(
    data_path: str = None,
    trades_path: str = None,
    portfolio_path: str = None,
    df: pd.DataFrame = None,
    trades: list = None,
    portfolio_history: list = None,
    save_path: str = 'results/charts/trading_chart.png',
    **kwargs
):
    """
    Wrapper function để vẽ trading chart từ files hoặc data
    
    Args:
        data_path: Path to OHLCV data CSV
        trades_path: Path to trades CSV
        portfolio_path: Path to portfolio CSV
        df: DataFrame (nếu không dùng file)
        trades: List trades (nếu không dùng file)
        portfolio_history: List portfolio (nếu không dùng file)
        save_path: Đường dẫn lưu chart
        **kwargs: Additional arguments for plot_full_trading_chart
    """
    # Load data from files if needed
    if df is None and data_path:
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    if trades is None and trades_path:
        trades_df = pd.read_csv(trades_path, parse_dates=['timestamp'])
        trades = trades_df.to_dict('records')
    
    if portfolio_history is None and portfolio_path:
        portfolio_df = pd.read_csv(portfolio_path, parse_dates=['timestamp'])
        portfolio_history = portfolio_df.to_dict('records')
    
    # Create visualizer
    viz = TradingChartVisualizer()
    
    # Plot
    fig = viz.plot_full_trading_chart(
        df=df,
        trades=trades,
        portfolio_history=portfolio_history,
        save_path=save_path,
        **kwargs
    )
    
    return fig


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Trading Chart Visualization')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to OHLCV data CSV')
    parser.add_argument('--trades', type=str,
                       help='Path to trades CSV')
    parser.add_argument('--portfolio', type=str,
                       help='Path to portfolio CSV')
    parser.add_argument('--output', type=str, 
                       default='results/charts/trading_chart.png',
                       help='Output path for chart')
    parser.add_argument('--title', type=str,
                       default='Bitcoin Trading Chart',
                       help='Chart title')
    
    args = parser.parse_args()
    
    # Visualize
    visualize_trading_results(
        data_path=args.data,
        trades_path=args.trades,
        portfolio_path=args.portfolio,
        save_path=args.output,
        title=args.title
    )
    
    print(f"✅ Chart created: {args.output}")
