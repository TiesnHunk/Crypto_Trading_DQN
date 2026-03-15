"""
Module tính toán các chỉ báo kỹ thuật
Technical indicators module
"""

import pandas as pd
import numpy as np


class TechnicalIndicators:
    """
    Class tính toán các chỉ báo kỹ thuật cho phân tích xu hướng
    """
    
    @staticmethod
    def calculate_rsi(data, period=14):
        """
        Tính RSI (Relative Strength Index)
        
        Args:
            data: Series giá đóng cửa
            period: Chu kỳ (mặc định 14)
        
        Returns:
            Series RSI
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_macd(data, fast=12, slow=26, signal=9):
        """
        Tính MACD (Moving Average Convergence Divergence)
        
        Args:
            data: Series giá đóng cửa
            fast: Chu kỳ MA nhanh
            slow: Chu kỳ MA chậm
            signal: Chu kỳ signal line
        
        Returns:
            DataFrame với MACD, Signal, Histogram
        """
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return pd.DataFrame({
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        })
    
    @staticmethod
    def calculate_sma(data, period):
        """
        Tính Simple Moving Average
        
        Args:
            data: Series giá
            period: Chu kỳ
        
        Returns:
            Series SMA
        """
        return data.rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(data, period):
        """
        Tính Exponential Moving Average
        
        Args:
            data: Series giá
            period: Chu kỳ
        
        Returns:
            Series EMA
        """
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_bollinger_bands(data, period=20, std_dev=2):
        """
        Tính Bollinger Bands
        
        Args:
            data: Series giá đóng cửa
            period: Chu kỳ
            std_dev: Số độ lệch chuẩn
        
        Returns:
            DataFrame với Upper, Middle, Lower bands
        """
        middle = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return pd.DataFrame({
            'upper': upper,
            'middle': middle,
            'lower': lower
        })
    
    @staticmethod
    def calculate_adx(high, low, close, period=14):
        """
        Tính ADX (Average Directional Index) - chỉ báo xu hướng
        
        Args:
            high: Series giá cao nhất
            low: Series giá thấp nhất
            close: Series giá đóng cửa
            period: Chu kỳ
        
        Returns:
            Series ADX
        """
        # Tính True Range
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Tính Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)
        
        # Smoothing
        atr = true_range.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Tính ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    @staticmethod
    def calculate_trend_indicator(data, short_period=10, long_period=30):
        """
        Tính toán chỉ báo xu hướng đơn giản
        
        Args:
            data: Series giá đóng cửa
            short_period: Chu kỳ MA ngắn
            long_period: Chu kỳ MA dài
        
        Returns:
            Series: 1 (uptrend), 0 (sideways), -1 (downtrend)
        """
        sma_short = data.rolling(window=short_period).mean()
        sma_long = data.rolling(window=long_period).mean()
        
        trend = np.where(sma_short > sma_long, 1, 
                        np.where(sma_short < sma_long, -1, 0))
        
        return pd.Series(trend, index=data.index)
    
    @staticmethod
    def add_all_indicators(df):
        """
        Thêm tất cả các chỉ báo vào DataFrame
        
        Args:
            df: DataFrame với các cột open, high, low, close, volume
        
        Returns:
            DataFrame với các chỉ báo đã thêm vào
        """
        df = df.copy()
        
        # RSI
        df['rsi'] = TechnicalIndicators.calculate_rsi(df['close'])
        
        # MACD
        macd_data = TechnicalIndicators.calculate_macd(df['close'])
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_hist'] = macd_data['histogram']
        
        # Moving Averages
        df['sma_20'] = TechnicalIndicators.calculate_sma(df['close'], 20)
        df['sma_50'] = TechnicalIndicators.calculate_sma(df['close'], 50)
        df['ema_12'] = TechnicalIndicators.calculate_ema(df['close'], 12)
        df['ema_26'] = TechnicalIndicators.calculate_ema(df['close'], 26)
        
        # Bollinger Bands
        bb = TechnicalIndicators.calculate_bollinger_bands(df['close'])
        df['bb_upper'] = bb['upper']
        df['bb_middle'] = bb['middle']
        df['bb_lower'] = bb['lower']
        
        # ADX
        df['adx'] = TechnicalIndicators.calculate_adx(df['high'], df['low'], df['close'])
        
        # Trend Indicator
        df['trend'] = TechnicalIndicators.calculate_trend_indicator(df['close'])
        
        # Price change
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['price_change'].rolling(window=20).std()
        
        return df


if __name__ == "__main__":
    # Test với dữ liệu mẫu
    print("=== Test Technical Indicators ===")
    
    # Tạo dữ liệu mẫu
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Tạo dữ liệu giá ngẫu nhiên
    prices = 100 + np.cumsum(np.random.randn(100) * 2)
    
    df = pd.DataFrame({
        'open': prices + np.random.randn(100) * 0.5,
        'high': prices + np.abs(np.random.randn(100) * 0.5),
        'low': prices - np.abs(np.random.randn(100) * 0.5),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Thêm các chỉ báo
    df = TechnicalIndicators.add_all_indicators(df)
    
    print("\nDữ liệu đã tính toán:")
    print(df.tail(10))

