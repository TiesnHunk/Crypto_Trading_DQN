"""
FINAL FIX: Replace ALL ETH data with hourly data from Binance
Remove Kaggle daily data, download complete hourly dataset
"""

import pandas as pd
import requests
from datetime import datetime
from pathlib import Path
import time

def download_binance_hourly(symbol: str, start_date: str, end_date: str):
    """Download complete hourly data from Binance"""
    
    print(f"\n{'='*70}")
    print(f"📥 DOWNLOADING COMPLETE {symbol} HOURLY DATA")
    print(f"{'='*70}")
    print(f"   Symbol: {symbol}")
    print(f"   Start: {start_date}")
    print(f"   End: {end_date}")
    print(f"{'='*70}\n")
    
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)
    
    base_url = 'https://api.binance.com/api/v3/klines'
    all_data = []
    current_ts = start_ts
    batch_count = 0
    
    print("🔽 Downloading hourly data...")
    
    while current_ts < end_ts:
        batch_count += 1
        
        params = {
            'symbol': symbol,
            'interval': '1h',
            'startTime': current_ts,
            'endTime': end_ts,
            'limit': 1000
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
            
            all_data.extend(data)
            last_timestamp = data[-1][6]
            current_ts = last_timestamp + 1
            
            current_dt = datetime.fromtimestamp(last_timestamp / 1000)
            progress_pct = ((current_ts - start_ts) / (end_ts - start_ts)) * 100
            print(f"   Batch {batch_count:3d}: {len(data):4d} candles | {current_dt.date()} | {progress_pct:5.1f}%")
            
            time.sleep(0.1)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            break
    
    print(f"\n✅ Downloaded {len(all_data):,} hourly candles\n")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"✅ Processed {len(df):,} rows")
    print(f"   Range: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
    
    return df

def replace_eth_completely():
    """Remove old ETH (daily), add new ETH (hourly)"""
    
    # Download complete ETH hourly data
    # Start from 2020-01-01 to cover training period properly
    eth_hourly = download_binance_hourly(
        symbol='ETHUSDT',
        start_date='2020-01-01',
        end_date='2024-07-01'
    )
    
    if eth_hourly is None or len(eth_hourly) == 0:
        print("❌ Failed to download ETH data")
        return
    
    eth_hourly['coin'] = 'ETH'
    
    # Load existing multi_coin data
    data_file = Path('data/raw/multi_coin_1h.csv')
    
    if not data_file.exists():
        print(f"❌ File not found: {data_file}")
        return
    
    print("📂 Loading multi_coin_1h.csv...")
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"   Total rows: {len(df):,}")
    print(f"   Coins: {df['coin'].unique()}")
    
    # REMOVE ALL OLD ETH DATA
    eth_old = df[df['coin'] == 'ETH']
    print(f"\n🗑️  Removing OLD ETH data:")
    print(f"   Old ETH rows: {len(eth_old):,}")
    print(f"   Old range: {eth_old['timestamp'].min()} to {eth_old['timestamp'].max()}")
    
    df_without_eth = df[df['coin'] != 'ETH'].copy()
    
    # ADD NEW ETH HOURLY DATA
    print(f"\n➕ Adding NEW ETH hourly data:")
    print(f"   New ETH rows: {len(eth_hourly):,}")
    print(f"   New range: {eth_hourly['timestamp'].min()} to {eth_hourly['timestamp'].max()}")
    
    # Merge
    df_final = pd.concat([df_without_eth, eth_hourly], ignore_index=True)
    df_final = df_final.sort_values(['coin', 'timestamp']).reset_index(drop=True)
    
    print(f"\n✅ Final merged data:")
    print(f"   Total rows: {len(df_final):,}")
    for coin in sorted(df_final['coin'].unique()):
        coin_data = df_final[df_final['coin'] == coin]
        print(f"   {coin}: {len(coin_data):,} rows ({coin_data['timestamp'].min().date()} to {coin_data['timestamp'].max().date()})")
    
    # Backup
    backup_file = Path('data/raw/multi_coin_1h_backup_before_eth_fix.csv')
    if not backup_file.exists():
        print(f"\n💾 Creating backup: {backup_file.name}")
        df.to_csv(backup_file, index=False)
    
    # Save
    print(f"\n💾 Saving to {data_file}...")
    df_final.to_csv(data_file, index=False)
    print("✅ Saved!\n")
    
    # Verify training period
    train_start = pd.to_datetime('2020-01-01')
    train_end = pd.to_datetime('2023-12-31')
    df_train = df_final[(df_final['timestamp'] >= train_start) & (df_final['timestamp'] <= train_end)]
    
    print(f"📊 Training Period (2020-2023):")
    print(f"{'='*70}")
    for coin in sorted(df_final['coin'].unique()):
        coin_train = df_train[df_train['coin'] == coin]
        if len(coin_train) > 0:
            print(f"   {coin}: {len(coin_train):,} samples")
    print(f"{'='*70}\n")
    
    # Balance check
    eth_train = df_train[df_train['coin'] == 'ETH']
    other_train = df_train[df_train['coin'] != 'ETH'].groupby('coin').size()
    
    if len(eth_train) > 0 and len(other_train) > 0:
        avg_other = other_train.mean()
        eth_ratio = len(eth_train) / avg_other * 100
        
        print(f"📊 Balance Check:")
        print(f"   ETH: {len(eth_train):,} samples")
        print(f"   Avg others: {avg_other:,.0f}")
        print(f"   ETH ratio: {eth_ratio:.1f}%")
        
        if eth_ratio >= 80:
            print(f"   ✅ PERFECT! ETH is well-balanced")
        elif eth_ratio >= 50:
            print(f"   ✅ GOOD! ETH is acceptable")
        else:
            print(f"   ❌ ETH still under-represented")
    
    print(f"\n{'='*70}")
    print(f"✅ ETH HOURLY DATA REPLACEMENT COMPLETE")
    print(f"{'='*70}")
    print(f"   Now run: python src/main_multi_coin_dqn.py")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    try:
        replace_eth_completely()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
