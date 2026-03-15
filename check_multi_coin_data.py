"""Quick script to check multi-coin data"""
import pandas as pd

# ✅ Use actual multi-coin data file
df = pd.read_csv('data/raw/multi_coin_1h.csv')

print('\n' + '='*70)
print('✅ MULTI-COIN DATA SUMMARY')
print('='*70)

print(f'\n📊 Total rows: {len(df):,}')
print(f'   Columns: {len(df.columns)}')

print('\n📈 Distribution by coin:')
print(df['coin'].value_counts())

if 'timestamp' in df.columns:
    print(f'\n📅 Date range:')
    print(f'   From: {df["timestamp"].min()}')
    print(f'   To:   {df["timestamp"].max()}')

print('\n🔍 Available features:')
print(f'   {list(df.columns)}')

print('\n📋 Sample data (first 5 rows):')
print(df.head())

print('\n📊 Data types:')
print(df.dtypes)

print('\n✅ Data is ready for training!')
print('   Next step: python src\\main_multi_coin.py --episodes 5000 --mode sequential')
print('='*70)
