"""
Module để lấy dữ liệu từ Binance API
Data fetching module for Binance API
"""

import pandas as pd
try:
    from binance.client import Client
    _BINANCE_AVAILABLE = True
except ImportError:  # Graceful fallback when python-binance isn't installed
    Client = None
    _BINANCE_AVAILABLE = False
from datetime import datetime, timedelta
import time
import os


class BinanceDataFetcher:
    """
    Class để lấy dữ liệu crypto từ Binance
    """
    
    def __init__(self, api_key=None, api_secret=None):
        """
        Khởi tạo client Binance
        
        Args:
            api_key: API key từ Binance (optional cho public data)
            api_secret: API secret từ Binance (optional cho public data)
        """
        # Try to load from config if not provided
        if api_key is None or api_secret is None:
            try:
                from config import BINANCE_API_KEY, BINANCE_API_SECRET
                api_key = BINANCE_API_KEY
                api_secret = BINANCE_API_SECRET
                print("✅ Đã load API credentials từ config.py")
            except (ImportError, AttributeError):
                # No config file, use public API
                pass
        
        if not _BINANCE_AVAILABLE:
            raise ImportError(
                "python-binance chưa được cài đặt. Cài đặt bằng: pip install python-binance "
                "hoặc pip install -r requirements.txt"
            )
        
        if api_key and api_secret:
            self.client = Client(api_key, api_secret)
            print("✅ Đã kết nối Binance API với API key")
        else:
            # Public API không cần key
            self.client = Client()
            print("ℹ️ Sử dụng Binance Public API (không cần key)")
    
    def load_kaggle_data(self, data_source: str = "binance"):
        """
        Load dữ liệu từ Kaggle hoặc Binance
        
        Args:
            data_source: "kaggle" hoặc "binance"
        
        Returns:
            DataFrame hoặc None
        """
        if data_source.lower() == "kaggle":
            try:
                from kaggle_data import load_kaggle_bitcoin_sample
                df = load_kaggle_bitcoin_sample()
                return df
            except ImportError:
                print("⚠️ Kaggle data loader not available")
                return None
        return None
    
    def get_klines(self, symbol, interval, start_date=None, end_date=None, limit=500):
        """
        Lấy dữ liệu kline (OHLCV) từ Binance
        
        Args:
            symbol: Cặp giao dịch, ví dụ 'BTCUSDT'
            interval: Khung thời gian (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            start_date: Ngày bắt đầu (string format 'YYYY-MM-DD' hoặc datetime)
            end_date: Ngày kết thúc (string format 'YYYY-MM-DD' hoặc datetime)
            limit: Số lượng candles tối đa (max 1000)
        
        Returns:
            DataFrame với các cột: Open, High, Low, Close, Volume
        """
        try:
            # Chuyển đổi start_date và end_date sang timestamp
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
            
            if start_date:
                start_str = start_date.strftime('%d %b %Y %H:%M:%S')
            else:
                start_str = None
            
            if end_date:
                end_str = end_date.strftime('%d %b %Y %H:%M:%S')
            else:
                end_str = None
            
            # Lấy dữ liệu
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_str,
                end_str=end_str,
                limit=limit
            )
            
            # Chuyển đổi sang DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Xử lý dữ liệu
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Chọn các cột quan trọng
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Lỗi khi lấy dữ liệu: {e}")
            return None
    
    def _interval_to_hours(self, interval):
        """
        Chuyển đổi interval string sang số giờ
        
        Args:
            interval: Interval string (1m, 1h, 1d, etc.)
        
        Returns:
            Số giờ tương ứng
        """
        interval_map = {
            '1m': 1/60, '3m': 3/60, '5m': 5/60, '15m': 15/60, '30m': 30/60,
            '1h': 1, '2h': 2, '4h': 4, '6h': 6, '8h': 8, '12h': 12,
            '1d': 24, '3d': 72, '1w': 168, '1M': 720
        }
        return interval_map.get(interval, 1)
    
    def get_historical_data_chunked(self, symbol, interval, start_date, end_date, chunk_days=None):
        """
        Lấy dữ liệu lịch sử cho period dài bằng cách chia nhỏ thành các chunk
        Tự động tính chunk_days tối ưu để lấy tối đa 1000 candles mỗi request
        
        Args:
            symbol: Cặp giao dịch, ví dụ 'BTCUSDT'
            interval: Khung thời gian (1h, 1d, etc.)
            start_date: Ngày bắt đầu (string 'YYYY-MM-DD' hoặc datetime)
            end_date: Ngày kết thúc (string 'YYYY-MM-DD' hoặc datetime)
            chunk_days: Số ngày mỗi chunk (auto-calculate nếu None)
        
        Returns:
            DataFrame với dữ liệu đầy đủ từ start_date đến end_date
        """
        # Chuyển đổi sang datetime nếu là string
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Tính chunk_days tối ưu nếu không được cung cấp
        if chunk_days is None:
            interval_hours = self._interval_to_hours(interval)
            max_candles = 1000  # Binance limit
            # Tính số ngày tương ứng với 1000 candles
            # Trừ đi 10 để đảm bảo không vượt quá limit
            chunk_days = int((max_candles * interval_hours) / 24) - 1
            chunk_days = max(1, chunk_days)  # Tối thiểu 1 ngày
        
        print(f"📊 Fetching data in chunks...")
        print(f"   Symbol: {symbol}")
        print(f"   Interval: {interval}")
        print(f"   Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"   Duration: {(end_date - start_date).days} days")
        print(f"   Chunk size: {chunk_days} days (~{int(chunk_days * 24 / self._interval_to_hours(interval))} candles)")
        
        all_data = []
        current_start = start_date
        chunk_count = 0
        
        while current_start < end_date:
            # Tính end của chunk hiện tại
            current_end = min(current_start + timedelta(days=chunk_days), end_date)
            
            chunk_count += 1
            print(f"\n🔄 Chunk {chunk_count}: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")
            
            try:
                # Lấy dữ liệu cho chunk này
                df_chunk = self.get_klines(
                    symbol=symbol,
                    interval=interval,
                    start_date=current_start,
                    end_date=current_end,
                    limit=1000  # Max limit cho mỗi request
                )
                
                if df_chunk is not None and len(df_chunk) > 0:
                    all_data.append(df_chunk)
                    print(f"   ✅ Retrieved {len(df_chunk)} candles")
                else:
                    print(f"   ⚠️ No data for this chunk")
                
                # Sleep để tránh rate limit
                time.sleep(0.5)
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                time.sleep(1)
            
            # Move to next chunk
            current_start = current_end
        
        # Kết hợp tất cả chunks
        if len(all_data) > 0:
            print(f"\n✅ Combining {len(all_data)} chunks...")
            df_combined = pd.concat(all_data)
            
            # Remove duplicates và sort
            df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
            df_combined = df_combined.sort_index()
            
            print(f"📊 Total data points: {len(df_combined)}")
            print(f"📅 Date range: {df_combined.index[0]} to {df_combined.index[-1]}")
            
            return df_combined
        else:
            print("❌ No data retrieved")
            return None
    
    def get_multiple_symbols(self, symbols, interval, start_date=None, end_date=None, limit=500):
        """
        Lấy dữ liệu cho nhiều symbols cùng lúc
        
        Args:
            symbols: List các symbols, ví dụ ['BTCUSDT', 'ETHUSDT']
            interval: Khung thời gian
            start_date: Ngày bắt đầu
            end_date: Ngày kết thúc
            limit: Số lượng candles
        
        Returns:
            Dict với key là symbol và value là DataFrame
        """
        data = {}
        for symbol in symbols:
            print(f"Đang lấy dữ liệu cho {symbol}...")
            df = self.get_klines(symbol, interval, start_date, end_date, limit)
            if df is not None:
                data[symbol] = df
            time.sleep(0.5)  # Tránh rate limit
        return data
    
    def get_latest_price(self, symbol):
        """
        Lấy giá hiện tại của một symbol
        
        Args:
            symbol: Cặp giao dịch
        
        Returns:
            Giá hiện tại
        """
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            print(f"Lỗi khi lấy giá: {e}")
            return None


# Ví dụ sử dụng
if __name__ == "__main__":
    # Khởi tạo fetcher (không cần API key cho public data)
    fetcher = BinanceDataFetcher()
    
    # Lấy dữ liệu BTC/USDT trong 30 ngày qua
    print("=== Ví dụ lấy dữ liệu từ Binance ===\n")
    
    # Lấy dữ liệu 1h cho BTC/USDT
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print("Lấy dữ liệu BTC/USDT (30 ngày, interval 1h)...")
    df = fetcher.get_klines('BTCUSDT', '1h', start_date, end_date, limit=500)
    
    if df is not None:
        print(f"\nDữ liệu đã tải về:")
        print(df.head())
        print(f"\nShape: {df.shape}")
        print(f"\nMô tả thống kê:")
        print(df.describe())
        
        # Lưu vào file CSV
        df.to_csv('btcusdt_data.csv')
        print("\nĐã lưu dữ liệu vào file 'btcusdt_data.csv'")
    
    # Lấy giá hiện tại
    print("\n=== Giá hiện tại ===")
    price = fetcher.get_latest_price('BTCUSDT')
    if price:
        print(f"BTC/USDT: ${price:,.2f}")

