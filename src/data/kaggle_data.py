"""
Module để lấy dữ liệu từ Kaggle
Kaggle data loading module
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import kagglehub


class KaggleDataLoader:
    """
    Class để download và load dữ liệu từ Kaggle
    """
    
    def __init__(self):
        """
        Khởi tạo Kaggle data loader
        """
        self.downloaded_datasets = {}
    
    def download_bitcoin_dataset(self, dataset_name: str = "novandraanugrah/bitcoin-historical-datasets-2018-2024"):
        """
        Download Bitcoin historical dataset từ Kaggle
        
        Args:
            dataset_name: Tên dataset trên Kaggle
        
        Returns:
            Path đến thư mục chứa dataset
        """
        try:
            print(f"Downloading dataset: {dataset_name}")
            print("This may take a few minutes...")
            
            # Download latest version
            path = kagglehub.dataset_download(dataset_name)
            
            print(f"✅ Dataset downloaded successfully!")
            print(f"Path: {path}")
            
            self.downloaded_datasets[dataset_name] = path
            
            return path
            
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            print("Make sure you have:")
            print("1. Installed kagglehub: pip install kagglehub")
            print("2. Authenticated with Kaggle (if dataset is private)")
            return None
    
    def load_data_from_path(self, data_path: str, timeframe: str = "1h") -> pd.DataFrame:
        """
        Load dữ liệu từ file CSV
        
        Args:
            data_path: Đường dẫn đến thư mục chứa files
            timeframe: "15m", "1h", "4h", "1d"
        
        Returns:
            DataFrame chứa dữ liệu
        """
        try:
            # Tìm file CSV trong thư mục
            path = Path(data_path)
            
            # Tìm file theo timeframe
            csv_files = list(path.glob(f"btc_{timeframe}_*.csv"))
            
            if not csv_files:
                # Fallback 1: Tìm file bitcoin_kaggle_data.csv
                bitcoin_file = path / "bitcoin_kaggle_data.csv"
                if bitcoin_file.exists():
                    csv_files = [bitcoin_file]
                else:
                    # Fallback 2: lấy file CSV đầu tiên
                    csv_files = list(path.glob("*.csv"))
            
            if not csv_files:
                print(f"❌ No CSV files found in {data_path}")
                return None
            
            print(f"Found {len(csv_files)} CSV file(s)")
            print(f"Loading file: {csv_files[0].name}")
            
            # Load file
            df = pd.read_csv(csv_files[0])
            
            print(f"✅ Loaded data: {len(df)} rows")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def preprocess_bitcoin_data(self, df: pd.DataFrame, 
                                date_col: str = 'Open time',
                                price_col: str = 'Close') -> pd.DataFrame:
        """
        Preprocess Bitcoin data từ Kaggle
        
        Args:
            df: Raw DataFrame
            date_col: Tên cột date
            price_col: Tên cột giá chính
        
        Returns:
            Processed DataFrame với format chuẩn
        """
        df = df.copy()
        
        # Convert date column
        if date_col in df.columns:
            df['timestamp'] = pd.to_datetime(df[date_col])
            df = df.set_index('timestamp')
        else:
            # Nếu không có date column, tạo index theo thứ tự
            df = df.reset_index(drop=True)
        
        # Đảm bảo có các cột cần thiết (open, high, low, close, volume)
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # Map common column names (Kaggle format)
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'close',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }
        
        # Rename columns
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]
        
        # Xử lý missing data
        df = df.dropna(subset=['close'])
        
        # Forward fill cho missing values
        df = df.ffill()
        
        # Sort by timestamp
        df = df.sort_index()
        
        # Select only required columns (if they exist)
        available_cols = [col for col in required_cols if col in df.columns]
        
        if len(available_cols) < 4:  # At least need close price
            print("⚠️ Warning: Missing required columns")
            print(f"Available columns: {df.columns.tolist()}")
        
        return df[available_cols]
    
    def get_kaggle_data(self, dataset_name: str = None, 
                       local_path: str = None,
                       timeframe: str = "1h") -> pd.DataFrame:
        """
        Download và load dữ liệu từ Kaggle
        
        Args:
            dataset_name: Tên dataset (nếu None, dùng Bitcoin dataset)
            local_path: Đường dẫn local (nếu đã download trước)
        
        Returns:
            Processed DataFrame
        """
        # Download dataset nếu chưa có local path
        if local_path is None:
            if dataset_name is None:
                dataset_name = "novandraanugrah/bitcoin-historical-datasets-2018-2024"
            
            data_path = self.download_bitcoin_dataset(dataset_name)
        else:
            data_path = local_path
        
        if data_path is None:
            return None
        
        # Load data
        df = self.load_data_from_path(data_path, timeframe=timeframe)
        
        if df is None:
            return None
        
        # Preprocess
        df = self.preprocess_bitcoin_data(df)
        
        return df
    
    def load_sample_data(self, sample_type: str = "recent") -> pd.DataFrame:
        """
        Load sample data với các options khác nhau
        
        Args:
            sample_type: 
                - "recent": Dữ liệu gần đây (1000 rows cuối)
                - "full": Toàn bộ dữ liệu
                - "2019": Chỉ năm 2019
                - "2023": Chỉ năm 2023
        
        Returns:
            DataFrame
        """
        df = self.get_kaggle_data()
        
        if df is None:
            return None
        
        if sample_type == "recent":
            df = df.tail(1000)
        elif sample_type == "full":
            pass
        elif sample_type == "2019":
            df = df[df.index.year == 2019]
        elif sample_type == "2023":
            df = df[df.index.year == 2023]
        
        return df


def load_kaggle_bitcoin_sample(start_date: str = None, end_date: str = None, timeframe: str = "1h") -> pd.DataFrame:
    """
    Helper function để load Bitcoin data từ Kaggle
    
    Args:
        start_date: Ngày bắt đầu (format: 'YYYY-MM-DD')
        end_date: Ngày kết thúc (format: 'YYYY-MM-DD')
    
    Returns:
        DataFrame với dữ liệu Bitcoin
    """
    loader = KaggleDataLoader()
    
    # Download và load data
    df = loader.get_kaggle_data(timeframe="1h")
    
    if df is None:
        print("⚠️ Cannot load data from Kaggle. Using fallback...")
        return None
    
    # Filter by date range if provided
    if start_date is not None:
        start = pd.to_datetime(start_date)
        df = df[df.index >= start]
    
    if end_date is not None:
        end = pd.to_datetime(end_date)
        df = df[df.index <= end]
    
    print(f"✅ Loaded {len(df)} rows from Kaggle dataset")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df


if __name__ == "__main__":
    print("=== Test Kaggle Data Loader ===\n")
    
    # Initialize loader
    loader = KaggleDataLoader()
    
    # Download và load dataset
    print("Downloading Bitcoin dataset from Kaggle...")
    df = loader.get_kaggle_data()
    
    if df is not None:
        print("\n✅ Data loaded successfully!")
        print(f"\nShape: {df.shape}")
        print(f"\nColumns: {df.columns.tolist()}")
        print(f"\nDate range: {df.index.min()} to {df.index.max()}")
        print(f"\nFirst few rows:")
        print(df.head())
        print(f"\nLast few rows:")
        print(df.tail())
        print(f"\nStatistics:")
        print(df.describe())
        
        # Save to CSV
        df.to_csv('bitcoin_kaggle_data.csv')
        print("\n✅ Saved to 'bitcoin_kaggle_data.csv'")
    else:
        print("\n❌ Failed to load data")
        print("Using sample data instead...")
        
        # Fallback to sample data
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='h')
        np.random.seed(42)
        prices = 30000 + np.cumsum(np.random.randn(1000) * 100)
        
        df = pd.DataFrame({
            'open': prices + np.random.randn(1000) * 50,
            'high': prices + np.abs(np.random.randn(1000) * 50),
            'low': prices - np.abs(np.random.randn(1000) * 50),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 1000)
        }, index=dates)
        
        print("✅ Sample data created")
        print(df.head())

