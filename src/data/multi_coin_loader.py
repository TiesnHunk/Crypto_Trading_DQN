"""
Module để tải và xử lý data từ nhiều cryptocurrencies
Multi-cryptocurrency data loader and processor
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import kagglehub


class MultiCoinLoader:
    """
    Class để tải và xử lý data từ nhiều coins
    """
    
    # Mapping dataset names
    DATASETS = {
        'BTC': 'novandraanugrah/bitcoin-historical-datasets-2018-2024',
        'ETH': 'prasoonkottarathil/ethereum-historical-dataset',
        'BNB': 'imranbukhari/comprehensive-bnbusd-1m-data',
        'SOL': 'novandraanugrah/solana-price-data-binance-api-2020now',  # ✅ NEW: More comprehensive SOL data
        'ADA': 'imranbukhari/comprehensive-adausd-1h-data'  # ✅ NEW: Comprehensive ADA data
    }
    
    def __init__(self, data_dir: str = None):
        """
        Khởi tạo loader
        
        Args:
            data_dir: Thư mục chứa data (default: ../data/raw)
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_coin_data(self, coin: str, force_download: bool = False) -> str:
        """
        Tải data cho 1 coin từ Kaggle
        
        Args:
            coin: Tên coin (BTC, ETH, BNB, SOL, ADA)
            force_download: Có force download lại không
            
        Returns:
            Path đến dataset
        """
        if coin not in self.DATASETS:
            raise ValueError(f"Coin {coin} not supported. Available: {list(self.DATASETS.keys())}")
        
        print(f"\n📥 Downloading {coin} data...")
        
        try:
            # Kiểm tra xem đã download chưa
            local_path = self.data_dir / coin.lower()
            
            if local_path.exists() and not force_download:
                print(f"✅ {coin} data already exists at {local_path}")
                return str(local_path)
            
            # Download từ Kaggle
            path = kagglehub.dataset_download(self.DATASETS[coin])
            print(f"✅ {coin} data downloaded to: {path}")
            
            return path
            
        except Exception as e:
            print(f"❌ Error downloading {coin}: {e}")
            return None
    
    def load_coin_dataframe(self, coin: str, path: str = None) -> pd.DataFrame:
        """
        Load data từ 1 coin thành DataFrame chuẩn hóa
        
        Args:
            coin: Tên coin
            path: Path đến data (nếu None sẽ download)
            
        Returns:
            DataFrame với cột chuẩn [timestamp, open, high, low, close, volume, coin]
        """
        if path is None:
            path = self.download_coin_data(coin)
        
        if path is None:
            return None
        
        print(f"\n📊 Loading {coin} data from {path}...")
        
        path = Path(path)
        
        # Tìm CSV files
        csv_files = list(path.glob("*.csv"))
        
        if not csv_files:
            print(f"❌ No CSV files found in {path}")
            return None
        
        # Load file đầu tiên hoặc file lớn nhất
        if len(csv_files) == 1:
            csv_file = csv_files[0]
        else:
            # Chọn file lớn nhất
            csv_file = max(csv_files, key=lambda f: f.stat().st_size)
        
        print(f"   Loading file: {csv_file.name}")
        
        try:
            df = pd.read_csv(csv_file)
            print(f"   Raw data: {len(df)} rows, {len(df.columns)} columns")
            
            # Chuẩn hóa columns
            df = self._standardize_columns(df, coin)
            
            if df is not None:
                print(f"✅ {coin}: {len(df)} rows loaded")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading {coin}: {e}")
            return None
    
    def _standardize_columns(self, df: pd.DataFrame, coin: str) -> pd.DataFrame:
        """
        Chuẩn hóa tên columns thành format thống nhất
        
        Expected columns: timestamp, open, high, low, close, volume
        """
        # Mapping các tên column có thể có
        column_mappings = {
            'timestamp': ['timestamp', 'time', 'date', 'datetime', 'Date', 'Timestamp', 
                         'Open time', 'open_time', 'OpenTime', 'Datetime'],
            'open': ['open', 'Open', 'open_price', 'Open price'],
            'high': ['high', 'High', 'high_price', 'High price'],
            'low': ['low', 'Low', 'low_price', 'Low price'],
            'close': ['close', 'Close', 'close_price', 'Close price', 'Price(in dollars)', 'Price'],
            'volume': ['volume', 'Volume', 'vol', 'Vol', 'Vol.', 'Volume BTC', 'volume_btc']
        }
        
        # Tìm và rename columns
        new_columns = {}
        for standard_name, possible_names in column_mappings.items():
            for col in df.columns:
                if col in possible_names:
                    new_columns[col] = standard_name
                    break
        
        if not new_columns:
            print(f"⚠️  Could not map columns for {coin}")
            print(f"   Available columns: {list(df.columns)}")
            print(f"   Please check column names in the CSV file")
            return None
        
        # Print mapping info
        print(f"   Column mapping for {coin}:")
        for old_col, new_col in new_columns.items():
            print(f"      {old_col} → {new_col}")
        
        df = df.rename(columns=new_columns)
        
        # Kiểm tra có đủ columns cần thiết không
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            print(f"⚠️  {coin} missing columns: {missing}")
            return None
        
        # Giữ lại các columns cần thiết
        df = df[required].copy()
        
        # Convert timestamp
        df = self._convert_timestamp(df)
        
        # Add coin column
        df['coin'] = coin
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'], keep='last')
        
        return df
    
    def _convert_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert timestamp column to datetime"""
        try:
            # Try different timestamp formats
            if df['timestamp'].dtype == 'int64':
                # Unix timestamp (seconds or milliseconds)
                if df['timestamp'].iloc[0] > 1e12:
                    # Milliseconds
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                else:
                    # Seconds
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            else:
                # String timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            print(f"⚠️  Error converting timestamp: {e}")
            return df
    
    def resample_to_timeframe(self, df: pd.DataFrame, timeframe: str = '1h') -> pd.DataFrame:
        """
        Resample data về timeframe cụ thể
        
        Args:
            df: DataFrame with timestamp index
            timeframe: '1h', '4h', '1d', etc.
            
        Returns:
            Resampled DataFrame
        """
        if 'timestamp' not in df.columns:
            return df
        
        print(f"   Resampling to {timeframe}...")
        
        df = df.set_index('timestamp')
        
        # Resample OHLCV data
        resampled = df.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'coin': 'first'
        })
        
        # Remove NaN rows
        resampled = resampled.dropna()
        
        # Reset index
        resampled = resampled.reset_index()
        
        print(f"   After resampling: {len(resampled)} rows")
        
        return resampled
    
    def load_all_coins(self, coins: List[str] = None, 
                       timeframe: str = '1h',
                       force_download: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Load data cho tất cả coins
        
        Args:
            coins: List coin names (nếu None sẽ load tất cả)
            timeframe: Timeframe để resample
            force_download: Force download lại data
            
        Returns:
            Dict {coin_name: DataFrame}
        """
        if coins is None:
            coins = list(self.DATASETS.keys())
        
        print(f"\n{'='*60}")
        print(f"📥 Loading {len(coins)} coins: {', '.join(coins)}")
        print(f"⏰ Target timeframe: {timeframe}")
        print(f"{'='*60}")
        
        all_data = {}
        
        for coin in coins:
            try:
                # Download
                path = self.download_coin_data(coin, force_download)
                
                if path is None:
                    continue
                
                # Load
                df = self.load_coin_dataframe(coin, path)
                
                if df is None:
                    continue
                
                # Resample
                df = self.resample_to_timeframe(df, timeframe)
                
                all_data[coin] = df
                
            except Exception as e:
                print(f"❌ Error loading {coin}: {e}")
                continue
        
        print(f"\n✅ Successfully loaded {len(all_data)} coins")
        
        return all_data
    
    def combine_all_coins(self, coins_data: Dict[str, pd.DataFrame],
                         align_timestamps: bool = True) -> pd.DataFrame:
        """
        Kết hợp data của tất cả coins thành 1 DataFrame
        
        Args:
            coins_data: Dict {coin: DataFrame}
            align_timestamps: Có align timestamps không (chỉ giữ timestamps có ở tất cả coins)
            
        Returns:
            Combined DataFrame
        """
        if not coins_data:
            return None
        
        print(f"\n🔗 Combining {len(coins_data)} coins...")
        
        # Concatenate tất cả
        combined = pd.concat(coins_data.values(), ignore_index=True)
        
        print(f"   Total rows: {len(combined)}")
        print(f"   Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")
        print(f"   Coins: {combined['coin'].unique()}")
        
        # Sort by timestamp
        combined = combined.sort_values('timestamp').reset_index(drop=True)
        
        # Save distribution info
        print("\n📊 Distribution by coin:")
        for coin in combined['coin'].unique():
            count = len(combined[combined['coin'] == coin])
            print(f"   {coin}: {count:,} rows")
        
        return combined
    
    def save_combined_data(self, df: pd.DataFrame, filename: str = None):
        """
        Lưu combined data
        
        Args:
            df: DataFrame to save
            filename: Output filename (default: multi_coin_data.csv)
        """
        if filename is None:
            filename = 'multi_coin_data.csv'
        
        output_path = self.data_dir / filename
        
        print(f"\n💾 Saving to {output_path}...")
        df.to_csv(output_path, index=False)
        print(f"✅ Saved {len(df)} rows to {output_path}")
        
        return str(output_path)


def main():
    """Test function"""
    print("=== Multi-Coin Data Loader ===\n")
    
    # Initialize loader
    loader = MultiCoinLoader()
    
    # Load all coins
    coins_data = loader.load_all_coins(
        coins=['BTC', 'ETH', 'BNB', 'SOL', 'ADA'],
        timeframe='1h',
        force_download=False
    )
    
    if not coins_data:
        print("❌ No data loaded")
        return
    
    # Combine
    combined = loader.combine_all_coins(coins_data)
    
    # Save
    loader.save_combined_data(combined, 'multi_coin_1h.csv')
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
