"""
Download ETH hourly data from Binance API
Fills the gap: 2020-04-17 to 2024-06-30
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
import time

def download_binance_klines(symbol: str, start_date: str, end_date: str, interval: str = '1h'):
    """
    Download historical kline/candlestick data from Binance
    
    Args:
        symbol: Trading pair symbol (e.g., 'ETHUSDT')
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'
        interval: Kline interval ('1h' for hourly)
        
    Returns:
        DataFrame with OHLCV data
    """
    print(f"\n{'='*70}")
    print(f"📥 DOWNLOADING {symbol} FROM BINANCE")
    print(f"{'='*70}")
    print(f"   Symbol: {symbol}")
    print(f"   Interval: {interval}")
    print(f"   Start: {start_date}")
    print(f"   End: {end_date}")
    print(f"{'='*70}\n")
    
    # Convert dates to timestamps (milliseconds)
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)
    
    # Binance API endpoint
    base_url = 'https://api.binance.com/api/v3/klines'
    
    all_data = []
    current_ts = start_ts
    batch_count = 0
    
    # Binance limit: 1000 candles per request
    # For 1h interval: 1000 hours = ~41 days
    limit = 1000
    
    print("🔽 Downloading data in batches...")
    
    while current_ts < end_ts:
        batch_count += 1
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_ts,
            'endTime': end_ts,
            'limit': limit
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                print(f"   Batch {batch_count}: No more data")
                break
            
            all_data.extend(data)
            
            # Update current timestamp to last candle's close time + 1ms
            last_timestamp = data[-1][6]  # Close time
            current_ts = last_timestamp + 1
            
            # Show progress
            current_dt = datetime.fromtimestamp(last_timestamp / 1000)
            progress_pct = ((current_ts - start_ts) / (end_ts - start_ts)) * 100
            print(f"   Batch {batch_count}: {len(data):4d} candles | Up to {current_dt.date()} | {progress_pct:.1f}% complete")
            
            # Respect rate limits (1200 requests/min = ~50ms per request)
            time.sleep(0.1)
            
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Error downloading batch {batch_count}: {e}")
            break
    
    print(f"\n✅ Downloaded {len(all_data):,} candles in {batch_count} batches\n")
    
    if not all_data:
        print("❌ No data downloaded")
        return None
    
    # Convert to DataFrame
    print("🔄 Converting to DataFrame...")
    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    
    # Convert price columns to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    # Keep only necessary columns
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"✅ Processed {len(df):,} rows")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   First close: ${df['close'].iloc[0]:,.2f}")
    print(f"   Last close: ${df['close'].iloc[-1]:,.2f}\n")
    
    return df

def merge_with_existing(new_data: pd.DataFrame, coin: str = 'ETH', output_dir: str = None):
    """
    Merge new hourly data with existing multi_coin_1h.csv
    
    Args:
        new_data: New DataFrame to merge
        coin: Coin symbol
        output_dir: Output directory
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'
    else:
        output_dir = Path(output_dir)
    
    multi_coin_file = output_dir / 'multi_coin_1h.csv'
    
    if not multi_coin_file.exists():
        print(f"❌ multi_coin_1h.csv not found at {multi_coin_file}")
        return None
    
    print(f"📂 Loading existing multi_coin_1h.csv...")
    df_existing = pd.read_csv(multi_coin_file)
    df_existing['timestamp'] = pd.to_datetime(df_existing['timestamp'])
    
    print(f"   Loaded {len(df_existing):,} rows")
    
    # Get existing coin data
    coin_existing = df_existing[df_existing['coin'] == coin].copy()
    print(f"\n   Existing {coin}: {len(coin_existing):,} rows")
    if len(coin_existing) > 0:
        print(f"   Range: {coin_existing['timestamp'].min()} to {coin_existing['timestamp'].max()}")
    
    # Add coin column to new data
    new_data['coin'] = coin
    
    # Find overlap and new data
    if len(coin_existing) > 0:
        last_existing_date = coin_existing['timestamp'].max()
        print(f"\n🔍 Finding data after {last_existing_date}...")
        new_only = new_data[new_data['timestamp'] > last_existing_date].copy()
        print(f"   New {coin} data: {len(new_only):,} rows")
    else:
        new_only = new_data.copy()
        print(f"\n🔍 No existing {coin} data, using all {len(new_only):,} rows")
    
    if len(new_only) == 0:
        print(f"ℹ️  No new {coin} data to add")
        return multi_coin_file
    
    print(f"   New range: {new_only['timestamp'].min()} to {new_only['timestamp'].max()}")
    
    # Remove old coin data and add combined new data
    print(f"\n🔗 Merging data...")
    df_other_coins = df_existing[df_existing['coin'] != coin].copy()
    
    # Combine old and new coin data
    coin_all = pd.concat([coin_existing, new_only], ignore_index=True)
    coin_all = coin_all.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')
    
    # Merge all
    df_merged = pd.concat([df_other_coins, coin_all], ignore_index=True)
    df_merged = df_merged.sort_values(['coin', 'timestamp']).reset_index(drop=True)
    
    print(f"✅ Merged data:")
    print(f"   Total rows: {len(df_merged):,}")
    for c in sorted(df_merged['coin'].unique()):
        c_data = df_merged[df_merged['coin'] == c]
        print(f"   {c}: {len(c_data):,} rows ({c_data['timestamp'].min().date()} to {c_data['timestamp'].max().date()})")
    
    # Backup
    backup_file = output_dir / 'multi_coin_1h_backup.csv'
    if not backup_file.exists():
        print(f"\n💾 Creating backup: {backup_file.name}")
        df_existing.to_csv(backup_file, index=False)
    
    # Save
    print(f"\n💾 Saving to {multi_coin_file}...")
    df_merged.to_csv(multi_coin_file, index=False)
    print(f"✅ Saved successfully!")
    
    # Summary
    coin_after = df_merged[df_merged['coin'] == coin]
    print(f"\n{'='*70}")
    print(f"✅ {coin} DATA UPDATE COMPLETE")
    print(f"{'='*70}")
    print(f"   Before: {len(coin_existing):,} rows")
    print(f"   Added:  {len(new_only):,} new rows")
    print(f"   After:  {len(coin_after):,} rows")
    print(f"   New range: {coin_after['timestamp'].min()} to {coin_after['timestamp'].max()}")
    print(f"{'='*70}\n")
    
    return multi_coin_file

def verify_training_coverage():
    """
    Verify ETH now has sufficient coverage for training period
    """
    print("\n🔍 VERIFYING TRAINING DATA COVERAGE...")
    
    data_file = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'multi_coin_1h.csv'
    
    if not data_file.exists():
        print(f"❌ File not found: {data_file}")
        return
    
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Training period
    train_start = pd.to_datetime('2020-01-01')
    train_end = pd.to_datetime('2023-12-31')
    
    df_train = df[(df['timestamp'] >= train_start) & (df['timestamp'] <= train_end)]
    
    print(f"\n📊 Training Period Coverage (2020-01-01 to 2023-12-31):")
    print(f"{'='*70}")
    
    for coin in sorted(df['coin'].unique()):
        coin_train = df_train[df_train['coin'] == coin]
        if len(coin_train) > 0:
            coverage = len(coin_train)
            date_range = f"{coin_train['timestamp'].min().date()} to {coin_train['timestamp'].max().date()}"
            print(f"   {coin:3s}: {coverage:6,} samples | {date_range}")
        else:
            print(f"   {coin:3s}: NO DATA ❌")
    
    print(f"{'='*70}\n")
    
    # Balance analysis
    eth_train = df_train[df_train['coin'] == 'ETH']
    other_train = df_train[df_train['coin'] != 'ETH'].groupby('coin').size()
    
    if len(eth_train) > 0 and len(other_train) > 0:
        avg_other = other_train.mean()
        eth_ratio = len(eth_train) / avg_other * 100
        
        print(f"📊 Training Balance Analysis:")
        print(f"   ETH samples: {len(eth_train):,}")
        print(f"   Avg other coins: {avg_other:,.0f}")
        print(f"   ETH ratio: {eth_ratio:.1f}%")
        
        if eth_ratio >= 80:
            print(f"   ✅ ETH is well-balanced (>80%)")
        elif eth_ratio >= 50:
            print(f"   ⚠️  ETH is acceptable (50-80%)")
        else:
            print(f"   ❌ ETH needs more data (<50%)")
        print()

def main():
    """
    Main function: Download ETH from Binance and merge
    """
    try:
        # Download ETH hourly data
        # From 2020-04-17 (where current ETH ends) to 2024-06-30 (validation end)
        eth_data = download_binance_klines(
            symbol='ETHUSDT',
            start_date='2020-04-17',
            end_date='2024-07-01',  # Include July 1st to get full June
            interval='1h'
        )
        
        if eth_data is None:
            print("❌ Failed to download ETH data")
            return
        
        # Merge with existing data
        output_file = merge_with_existing(eth_data, coin='ETH')
        
        if output_file:
            # Verify coverage
            verify_training_coverage()
            
            print("✅ SUCCESS! ETH hourly data has been added.")
            print(f"   Now you can run: python src/main_multi_coin_dqn.py")
        else:
            print("❌ Failed to merge ETH data")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
