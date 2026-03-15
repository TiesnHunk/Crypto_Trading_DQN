"""
Module implement logic giao dịch dựa theo xu hướng (Trend-based trading)
Trend-based trading logic module
"""

import pandas as pd
import numpy as np
from typing import Dict, List


class TrendBasedStrategy:
    """
    Class implement chiến lược giao dịch dựa trên xu hướng
    Kết hợp nhiều indicator để xác định xu hướng và tín hiệu giao dịch
    """
    
    def __init__(self, rsi_oversold: float = 30, rsi_overbought: float = 70,
                 adx_threshold: float = 25, trend_lookback: int = 10):
        """
        Khởi tạo strategy
        
        Args:
            rsi_oversold: Ngưỡng RSI oversold
            rsi_overbought: Ngưỡng RSI overbought
            adx_threshold: Ngưỡng ADX để xác định xu hướng mạnh
            trend_lookback: Số period để xác định xu hướng
        """
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.adx_threshold = adx_threshold
        self.trend_lookback = trend_lookback
    
    def get_trend_signal(self, row: pd.Series) -> Dict:
        """
        Tính toán tín hiệu giao dịch dựa trên xu hướng
        
        Args:
            row: Một row của DataFrame chứa indicators
        
        Returns:
            Dict chứa tín hiệu và các thông tin
        """
        signal = 0  # 0: Hold, 1: Buy, -1: Sell
        confidence = 0.0
        reasons = []
        
        # 1. Xác định xu hướng từ MA
        if pd.notna(row.get('sma_20')) and pd.notna(row.get('sma_50')):
            if row['sma_20'] > row['sma_50']:
                trend = 'uptrend'
                reasons.append("Price > SMA (Uptrend)")
            elif row['sma_20'] < row['sma_50']:
                trend = 'downtrend'
                reasons.append("Price < SMA (Downtrend)")
            else:
                trend = 'sideways'
                reasons.append("Price sideways")
        else:
            trend = 'unknown'
        
        # 2. RSI signals
        rsi_signal = 0
        if pd.notna(row.get('rsi')):
            if row['rsi'] < self.rsi_oversold:
                rsi_signal = 1  # Buy signal
                reasons.append(f"RSI Oversold ({row['rsi']:.1f})")
                confidence += 0.3
            elif row['rsi'] > self.rsi_overbought:
                rsi_signal = -1  # Sell signal
                reasons.append(f"RSI Overbought ({row['rsi']:.1f})")
                confidence += 0.3
        
        # 3. MACD signals
        macd_signal = 0
        if pd.notna(row.get('macd')) and pd.notna(row.get('macd_signal')):
            if row['macd'] > row['macd_signal'] and row['macd_hist'] > 0:
                macd_signal = 1
                reasons.append("MACD Bullish")
                confidence += 0.2
            elif row['macd'] < row['macd_signal'] and row['macd_hist'] < 0:
                macd_signal = -1
                reasons.append("MACD Bearish")
                confidence += 0.2
        
        # 4. Bollinger Bands
        bb_signal = 0
        if all(pd.notna(row.get(col)) for col in ['bb_lower', 'bb_upper', 'close']):
            bb_position = (row['close'] - row['bb_lower']) / (row['bb_upper'] - row['bb_lower'])
            
            if bb_position < 0.2:  # Gần lower band
                bb_signal = 1
                reasons.append("Price near BB Lower")
                confidence += 0.2
            elif bb_position > 0.8:  # Gần upper band
                bb_signal = -1
                reasons.append("Price near BB Upper")
                confidence += 0.2
        
        # 5. ADX - Strength of trend
        strong_trend = False
        if pd.notna(row.get('adx')):
            if row['adx'] > self.adx_threshold:
                strong_trend = True
                reasons.append(f"Strong Trend (ADX={row['adx']:.1f})")
                confidence += 0.3
        
        # Kết hợp tín hiệu
        # Tín hiệu mua
        if trend == 'uptrend' and (rsi_signal == 1 or macd_signal == 1) and strong_trend:
            signal = 1
            confidence = min(confidence + 0.5, 1.0)
            reasons.append("**BUY SIGNAL**")
        # Tín hiệu bán
        elif trend == 'downtrend' and (rsi_signal == -1 or macd_signal == -1) and strong_trend:
            signal = -1
            confidence = min(confidence + 0.5, 1.0)
            reasons.append("**SELL SIGNAL**")
        # Tín hiệu RSI cực đoan
        elif rsi_signal == 1 and macd_signal != -1:
            signal = 1
            confidence = min(confidence, 0.7)
            reasons.append("RSI Extreme Buy")
        elif rsi_signal == -1 and macd_signal != 1:
            signal = -1
            confidence = min(confidence, 0.7)
            reasons.append("RSI Extreme Sell")
        # Bollinger Bands reversal
        elif bb_signal == 1 and macd_signal != -1:
            signal = 1
            confidence = min(confidence, 0.6)
            reasons.append("BB Reversal Buy")
        elif bb_signal == -1 and macd_signal != 1:
            signal = -1
            confidence = min(confidence, 0.6)
            reasons.append("BB Reversal Sell")
        
        return {
            'signal': signal,
            'confidence': confidence,
            'trend': trend,
            'reasons': '; '.join(reasons) if reasons else 'No signal'
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo tín hiệu giao dịch cho toàn bộ DataFrame
        
        Args:
            df: DataFrame chứa dữ liệu và indicators
        
        Returns:
            DataFrame với các cột signal thêm vào
        """
        df = df.copy()
        
        signals = []
        for idx, row in df.iterrows():
            signal_dict = self.get_trend_signal(row)
            signals.append(signal_dict)
        
        # Thêm vào DataFrame
        df['trend_signal'] = [s['signal'] for s in signals]
        df['signal_confidence'] = [s['confidence'] for s in signals]
        df['trend'] = [s['trend'] for s in signals]
        df['signal_reasons'] = [s['reasons'] for s in signals]
        
        return df
    
    def get_optimized_parameters(self, df: pd.DataFrame) -> Dict:
        """
        Tối ưu hóa các tham số dựa trên dữ liệu lịch sử
        
        Args:
            df: DataFrame dữ liệu
        
        Returns:
            Dict chứa các parameters tối ưu
        """
        # Đơn giản: thử các giá trị khác nhau
        best_params = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'adx_threshold': 25,
            'trend_lookback': 10
        }
        
        # Có thể implement grid search ở đây
        # ...
        
        return best_params


if __name__ == "__main__":
    # Test trend-based strategy
    print("=== Test Trend-Based Strategy ===\n")
    
    # Tạo dữ liệu mẫu
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    prices = 100 + np.cumsum(np.random.randn(100) * 2)
    
    df = pd.DataFrame({
        'open': prices + np.random.randn(100) * 0.5,
        'high': prices + np.abs(np.random.randn(100) * 0.5),
        'low': prices - np.abs(np.random.randn(100) * 0.5),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Thêm indicators
    from indicators import TechnicalIndicators
    df = TechnicalIndicators.add_all_indicators(df)
    
    # Tạo strategy
    strategy = TrendBasedStrategy()
    
    # Generate signals
    df = strategy.generate_signals(df)
    
    # Hiển thị các tín hiệu
    print("Tín hiệu giao dịch tạo ra:")
    signals_df = df[df['trend_signal'] != 0][['close', 'trend', 'trend_signal', 'signal_confidence', 'signal_reasons']]
    
    if len(signals_df) > 0:
        print(signals_df)
    else:
        print("Không có tín hiệu giao dịch nào")

