"""
Download ETH data from Kaggle and integrate into multi_coin_1h.csv
Fills the gap: 2020-04-17 to 2023-12-31 (and beyond)
"""

import kagglehub
import pandas as pd
import numpy as np
from pathlib import Path

def download_eth_kaggle():
    """
    Download ETH data from Kaggle dataset
    """
    print("\n" + "="*70)
    print("📥 DOWNLOADING ETH DATA FROM KAGGLE")
    print("="*70)
    print("   Dataset: ahmadwaleed1/ethereum-price-usd-2016-2023")
    print("   Time range: 2018-2023")
    print("="*70 + "\n")
    
    # Download from Kaggle
    print("🔽 Downloading from Kaggle...")
    path = kagglehub.dataset_download("ahmadwaleed1/ethereum-price-usd-2016-2023")
    print(f"✅ Downloaded to: {path}\n")
    
    return path

def process_eth_kaggle(kaggle_path: str, output_dir: str = None):
    """
    Process Kaggle ETH data and merge with existing multi_coin_1h.csv
    
    Args:
        kaggle_path: Path to downloaded Kaggle dataset
        output_dir: Output directory for processed data
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("📊 Processing ETH data...\n")
    
    # Find CSV file in Kaggle directory
    kaggle_dir = Path(kaggle_path)
    csv_files = list(kaggle_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"❌ No CSV files found in {kaggle_dir}")
        return None
    
    eth_file = csv_files[0]
    print(f"📁 Found ETH file: {eth_file.name}")
    
    # Read Kaggle ETH data
    print("   Reading Kaggle data...")
    eth_kaggle = pd.read_csv(eth_file)
    print(f"   Loaded {len(eth_kaggle):,} rows")
    print(f"   Columns: {list(eth_kaggle.columns)}")
    
    # Show sample
    print(f"\n   Sample data (first 3 rows):")
    print(eth_kaggle.head(3))
    
    # Convert to our format (assuming columns: timestamp, open, high, low, close, volume)
    # Adjust column names based on actual Kaggle dataset structure
    print("\n🔄 Converting to multi_coin format...")
    
    # Detect timestamp column
    timestamp_col = None
    for col in ['timestamp', 'date', 'Date', 'Timestamp', 'time']:
        if col in eth_kaggle.columns:
            timestamp_col = col
            break
    
    if timestamp_col is None:
        print("❌ Could not find timestamp column")
        return None
    
    eth_kaggle['timestamp'] = pd.to_datetime(eth_kaggle[timestamp_col])
    
    # Detect price columns (case-insensitive)
    price_cols = {}
    for col_name in ['open', 'high', 'low', 'close', 'volume']:
        found = False
        for col in eth_kaggle.columns:
            if col.lower() == col_name.lower():
                price_cols[col_name] = col
                found = True
                break
        if not found:
            # Try variations
            if col_name == 'open':
                for col in eth_kaggle.columns:
                    if 'open' in col.lower():
                        price_cols[col_name] = col
                        found = True
                        break
            # Add similar logic for other columns if needed
    
    print(f"   Detected columns: {price_cols}")
    
    # Create standardized dataframe
    eth_processed = pd.DataFrame({
        'timestamp': eth_kaggle['timestamp'],
        'open': eth_kaggle[price_cols.get('open', 'Open')],
        'high': eth_kaggle[price_cols.get('high', 'High')],
        'low': eth_kaggle[price_cols.get('low', 'Low')],
        'close': eth_kaggle[price_cols.get('close', 'Close')],
        'volume': eth_kaggle[price_cols.get('volume', 'Volume')],
        'coin': 'ETH'
    })
    
    # Sort by timestamp
    eth_processed = eth_processed.sort_values('timestamp').reset_index(drop=True)
    
    print(f"✅ Processed {len(eth_processed):,} ETH rows")
    print(f"   Date range: {eth_processed['timestamp'].min()} to {eth_processed['timestamp'].max()}")
    
    # Load existing multi_coin_1h.csv
    multi_coin_file = output_dir / 'multi_coin_1h.csv'
    
    if not multi_coin_file.exists():
        print(f"\n❌ multi_coin_1h.csv not found at {multi_coin_file}")
        print("   Cannot merge. Please ensure multi_coin_1h.csv exists.")
        return None
    
    print(f"\n📂 Loading existing multi_coin_1h.csv...")
    df_existing = pd.read_csv(multi_coin_file)
    df_existing['timestamp'] = pd.to_datetime(df_existing['timestamp'])
    
    print(f"   Loaded {len(df_existing):,} rows")
    print(f"   Existing coins: {df_existing['coin'].unique()}")
    
    # Get existing ETH data
    eth_existing = df_existing[df_existing['coin'] == 'ETH'].copy()
    print(f"\n   Existing ETH: {len(eth_existing):,} rows")
    if len(eth_existing) > 0:
        print(f"   Existing ETH range: {eth_existing['timestamp'].min()} to {eth_existing['timestamp'].max()}")
    
    # Find the gap: what's missing after existing ETH data ends
    if len(eth_existing) > 0:
        last_eth_date = eth_existing['timestamp'].max()
        print(f"\n🔍 Finding new ETH data after {last_eth_date}...")
        eth_new = eth_processed[eth_processed['timestamp'] > last_eth_date].copy()
        print(f"   New ETH data: {len(eth_new):,} rows")
    else:
        print(f"\n🔍 No existing ETH data, using all Kaggle data...")
        eth_new = eth_processed.copy()
    
    if len(eth_new) == 0:
        print("ℹ️  No new ETH data to add (Kaggle data overlaps with existing)")
        return multi_coin_file
    
    print(f"   New ETH range: {eth_new['timestamp'].min()} to {eth_new['timestamp'].max()}")
    
    # Merge new ETH data
    print(f"\n🔗 Merging new ETH data into multi_coin_1h.csv...")
    
    # Remove ETH from existing (we'll re-add all ETH data)
    df_other_coins = df_existing[df_existing['coin'] != 'ETH'].copy()
    
    # Combine: Other coins + Old ETH + New ETH
    eth_all = pd.concat([eth_existing, eth_new], ignore_index=True)
    eth_all = eth_all.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')
    
    df_merged = pd.concat([df_other_coins, eth_all], ignore_index=True)
    df_merged = df_merged.sort_values(['coin', 'timestamp']).reset_index(drop=True)
    
    print(f"✅ Merged data:")
    print(f"   Total rows: {len(df_merged):,}")
    print(f"   Coins: {sorted(df_merged['coin'].unique())}")
    for coin in sorted(df_merged['coin'].unique()):
        coin_data = df_merged[df_merged['coin'] == coin]
        print(f"   {coin}: {len(coin_data):,} rows ({coin_data['timestamp'].min()} to {coin_data['timestamp'].max()})")
    
    # Backup original file
    backup_file = output_dir / 'multi_coin_1h_backup.csv'
    if not backup_file.exists():
        print(f"\n💾 Creating backup: {backup_file.name}")
        df_existing.to_csv(backup_file, index=False)
    
    # Save merged data
    print(f"\n💾 Saving merged data to {multi_coin_file}...")
    df_merged.to_csv(multi_coin_file, index=False)
    print(f"✅ Saved successfully!")
    
    # Summary
    eth_after = df_merged[df_merged['coin'] == 'ETH']
    print(f"\n" + "="*70)
    print(f"✅ ETH DATA UPDATE COMPLETE")
    print(f"="*70)
    print(f"   Before: {len(eth_existing):,} ETH rows")
    print(f"   Added:  {len(eth_new):,} new rows")
    print(f"   After:  {len(eth_after):,} ETH rows")
    print(f"   New range: {eth_after['timestamp'].min()} to {eth_after['timestamp'].max()}")
    print(f"="*70 + "\n")
    
    return multi_coin_file

def verify_training_data():
    """
    Verify that ETH now has sufficient data for training period
    """
    print("\n🔍 VERIFYING TRAINING DATA...")
    
    data_file = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'multi_coin_1h.csv'
    
    if not data_file.exists():
        print(f"❌ File not found: {data_file}")
        return
    
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Training period: 2020-01-01 to 2023-12-31
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
            print(f"   {coin:3s}: NO DATA in training period ❌")
    
    print(f"{'='*70}\n")
    
    # Check if ETH is now balanced
    eth_train = df_train[df_train['coin'] == 'ETH']
    other_coins_train = df_train[df_train['coin'] != 'ETH'].groupby('coin').size()
    
    if len(eth_train) > 0 and len(other_coins_train) > 0:
        avg_other = other_coins_train.mean()
        eth_ratio = len(eth_train) / avg_other * 100
        
        print(f"📊 Training Balance Analysis:")
        print(f"   ETH samples: {len(eth_train):,}")
        print(f"   Avg other coins: {avg_other:,.0f}")
        print(f"   ETH ratio: {eth_ratio:.1f}%")
        
        if eth_ratio >= 80:
            print(f"   ✅ ETH is well-balanced (>80% of avg)")
        elif eth_ratio >= 50:
            print(f"   ⚠️  ETH is acceptable (50-80% of avg)")
        else:
            print(f"   ❌ ETH is under-represented (<50% of avg)")
        print()

def main():
    """
    Main function: Download and integrate ETH data
    """
    try:
        # Download from Kaggle
        kaggle_path = download_eth_kaggle()
        
        # Process and merge
        output_file = process_eth_kaggle(kaggle_path)
        
        if output_file:
            # Verify training data
            verify_training_data()
            
            print("✅ SUCCESS! ETH data has been updated.")
            print("   You can now run: python src/main_multi_coin_dqn.py")
        else:
            print("❌ Failed to process ETH data")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
