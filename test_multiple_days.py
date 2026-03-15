"""
TEST NHIỀU NGÀY - Đánh giá tổng thể model
Không cần train lại, chỉ test trên nhiều ngày khác nhau
"""

import pandas as pd
import numpy as np
from predict_one_day_improved import ImprovedPredictor

def test_multiple_days():
    """Test trên nhiều ngày để có đánh giá tổng thể"""
    
    CHECKPOINT_PATH = 'src/checkpoints_dqn/checkpoint_best.pkl'
    DATA_PATH = 'data/raw/multi_coin_1h.csv'
    
    # Test suite: Mix của Bull, Bear, Sideways
    test_dates = [
        ('2020-04-06', 'Strong Bull', +6.76),
        ('2020-04-22', 'Bull', +4.30),
        ('2024-09-26', 'Bull', +3.54),
        ('2019-06-27', 'Strong Bear', -14.00),
        ('2021-05-19', 'Bear', -13.89),
        ('2024-09-30', 'Sideways', -3.26),
    ]
    
    print("="*80)
    print("TESTING ON MULTIPLE DAYS - NO RETRAINING")
    print("="*80)
    
    results = []
    predictor = ImprovedPredictor(CHECKPOINT_PATH, DATA_PATH)
    
    for date, market_type, expected_change in test_dates:
        print(f"\n{'='*80}")
        print(f"Testing: {date} ({market_type}, {expected_change:+.2f}%)")
        print(f"{'='*80}")
        
        try:
            pred_df, comparison = predictor.run(test_date=date)
            
            results.append({
                'date': date,
                'market_type': market_type,
                'expected_change': expected_change,
                'raw_return': comparison['raw_return'],
                'improved_return': comparison['improved_return'],
                'improvement': comparison['improvement']
            })
        except Exception as e:
            print(f"Error testing {date}: {e}")
            continue
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY - ALL TESTS")
    print("="*80)
    
    df_results = pd.DataFrame(results)
    
    print(f"\n{'Date':<12} {'Market':<14} {'Expected%':<10} {'Raw%':<10} {'Improved%':<10} {'Gain%':<10}")
    print("-"*80)
    
    for _, row in df_results.iterrows():
        print(f"{row['date']:<12} {row['market_type']:<14} "
              f"{row['expected_change']:>8.2f}  "
              f"{row['raw_return']:>8.2f}  "
              f"{row['improved_return']:>8.2f}  "
              f"{row['improvement']:>+8.2f}")
    
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    print(f"\nAverage Raw Return:      {df_results['raw_return'].mean():>8.2f}%")
    print(f"Average Improved Return: {df_results['improved_return'].mean():>8.2f}%")
    print(f"Average Improvement:     {df_results['improvement'].mean():>+8.2f}%")
    
    print(f"\nBest Day (Raw):      {df_results.loc[df_results['raw_return'].idxmax(), 'date']} "
          f"({df_results['raw_return'].max():.2f}%)")
    print(f"Best Day (Improved): {df_results.loc[df_results['improved_return'].idxmax(), 'date']} "
          f"({df_results['improved_return'].max():.2f}%)")
    
    print(f"\nWorst Day (Raw):      {df_results.loc[df_results['raw_return'].idxmin(), 'date']} "
          f"({df_results['raw_return'].min():.2f}%)")
    print(f"Worst Day (Improved): {df_results.loc[df_results['improved_return'].idxmin(), 'date']} "
          f"({df_results['improved_return'].min():.2f}%)")
    
    # Save results
    df_results.to_csv('results/reports/multi_day_test_results.csv', index=False)
    print(f"\nResults saved: results/reports/multi_day_test_results.csv")
    
    return df_results

if __name__ == '__main__':
    results = test_multiple_days()
