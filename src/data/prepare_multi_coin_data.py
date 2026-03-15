"""
Script để download và prepare multi-coin data
Download and prepare multi-coin training data
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.multi_coin_loader import MultiCoinLoader
from src.utils.indicators import TechnicalIndicators


def download_all_coins():
    """
    Download data cho tất cả 5 coins
    """
    print("="*70)
    print("📥 DOWNLOADING MULTI-COIN DATA")
    print("="*70)
    
    loader = MultiCoinLoader()
    
    # Download từng coin
    coins = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA']
    
    for coin in coins:
        try:
            loader.download_coin_data(coin, force_download=False)
        except Exception as e:
            print(f"⚠️  Error with {coin}: {e}")
            continue
    
    print("\n✅ Download phase complete!")


def prepare_training_data(timeframe: str = '1h'):
    """
    Prepare data cho training
    
    Args:
        timeframe: '1h', '4h', '1d', etc.
    """
    print("\n" + "="*70)
    print(f"📊 PREPARING TRAINING DATA ({timeframe})")
    print("="*70)
    
    loader = MultiCoinLoader()
    
    # Load all coins
    coins_data = loader.load_all_coins(
        coins=['BTC', 'ETH', 'BNB', 'SOL', 'ADA'],
        timeframe=timeframe,
        force_download=False
    )
    
    if not coins_data:
        print("❌ No data loaded!")
        return None
    
    # Add technical indicators cho từng coin
    print("\n📈 Adding technical indicators...")
    
    for coin, df in coins_data.items():
        print(f"   Processing {coin}...")
        
        # Add indicators
        df = TechnicalIndicators.add_all_indicators(df)
        coins_data[coin] = df
        
        print(f"   {coin}: {len(df)} rows with {len(df.columns)} features")
    
    # Combine all
    combined = loader.combine_all_coins(coins_data)
    
    # Save
    output_file = f'multi_coin_{timeframe}.csv'
    output_path = loader.save_combined_data(combined, output_file)
    
    print(f"\n✅ Training data ready: {output_path}")
    print(f"   Total samples: {len(combined):,}")
    print(f"   Features: {len(combined.columns)}")
    print(f"   Coins: {combined['coin'].nunique()}")
    
    # Show sample
    print("\n📋 Sample data:")
    print(combined.head())
    
    print("\n📊 Data statistics:")
    print(combined.describe())
    
    return output_path


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download and prepare multi-coin data')
    parser.add_argument('--timeframe', type=str, default='1h',
                       help='Timeframe: 1h, 4h, 1d (default: 1h)')
    parser.add_argument('--download-only', action='store_true',
                       help='Only download data, do not process')
    
    args = parser.parse_args()
    
    if args.download_only:
        download_all_coins()
    else:
        download_all_coins()
        prepare_training_data(args.timeframe)


if __name__ == "__main__":
    main()
