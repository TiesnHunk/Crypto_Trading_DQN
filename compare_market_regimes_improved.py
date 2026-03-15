"""
SO SÁNH RAW vs IMPROVED DQN TRÊN 3 MARKET REGIMES
Phân tích hiệu quả của Smart Rules trên Bull, Bear, Sideways
"""

import sys
sys.path.append('src')

from predict_one_day_improved import ImprovedPredictor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Configuration
CHECKPOINT_PATH = 'src/checkpoints_dqn/checkpoint_best.pkl'
DATA_PATH = 'data/raw/multi_coin_1h.csv'

# Test dates (từ find_best_test_day.py)
TEST_DATES = {
    'BULL': '2020-04-06',    # +6.76%, Strong Bull
    'BEAR': '2020-03-12',    # -39.34%, Strong Bear (COVID Crash)
    'SIDEWAYS': '2024-09-30' # -3.26%, Sideways
}

def run_all_tests():
    """Run both Raw and Improved predictions on all 3 market regimes"""
    results = {}
    
    predictor = ImprovedPredictor(CHECKPOINT_PATH, DATA_PATH)
    
    for market_type, test_date in TEST_DATES.items():
        print("\n" + "="*80)
        print(f"TESTING {market_type} MARKET - {test_date}")
        print("="*80)
        
        try:
            pred_df, comparison = predictor.run(test_date=test_date)
            
            # Load market info
            test_df = predictor.load_data(test_date)
            market_info = predictor.analyze_market(test_df)
            
            # Calculate metrics
            metrics = predictor.calculate_metrics(pred_df, test_df)
            
            # Store results
            results[market_type] = {
                'test_date': test_date,
                'pred_df': pred_df,
                'test_df': test_df,
                'comparison': comparison,
                'market_info': market_info,
                'metrics': metrics
            }
            
        except Exception as e:
            print(f"Error testing {market_type}: {e}")
            continue
    
    return results

def create_comparison_table(results):
    """Tạo bảng so sánh Raw vs Improved cho tất cả regimes"""
    print("\n" + "="*100)
    print("SO SÁNH RAW vs IMPROVED TRÊN 3 MARKET REGIMES")
    print("="*100)
    
    comparison_data = []
    
    for market_type in ['BULL', 'BEAR', 'SIDEWAYS']:
        if market_type not in results:
            continue
        
        data = results[market_type]
        pred_df = data['pred_df']
        market_info = data['market_info']
        comparison = data['comparison']
        metrics = data['metrics']
        
        # Raw action distribution
        raw_dist = pred_df['raw_action_name'].value_counts()
        raw_hold = raw_dist.get('Hold', 0) / len(pred_df) * 100
        raw_buy = raw_dist.get('Buy', 0) / len(pred_df) * 100
        raw_sell = raw_dist.get('Sell', 0) / len(pred_df) * 100
        
        # Improved action distribution
        imp_dist = pred_df['improved_action_name'].value_counts()
        imp_hold = imp_dist.get('Hold', 0) / len(pred_df) * 100
        imp_buy = imp_dist.get('Buy', 0) / len(pred_df) * 100
        imp_sell = imp_dist.get('Sell', 0) / len(pred_df) * 100
        
        comparison_data.append({
            'Market': market_type,
            'Date': data['test_date'],
            'Price_Change': market_info['price_change'],
            'Trend': market_info['avg_trend'],
            'RSI': market_info['avg_rsi'],
            
            'Raw_Return': comparison['raw_return'],
            'Raw_Hold': raw_hold,
            'Raw_Buy': raw_buy,
            'Raw_Sell': raw_sell,
            
            'Imp_Return': comparison['improved_return'],
            'Imp_Hold': imp_hold,
            'Imp_Buy': imp_buy,
            'Imp_Sell': imp_sell,
            
            'Improvement': comparison['improvement'],
            
            # Metrics
            'MAE': metrics['MAE'],
            'MSE': metrics['MSE'],
            'RMSE': metrics['RMSE'],
            'MAPE': metrics['MAPE'],
            'R2': metrics['R2'],
            'Accuracy': metrics['Accuracy']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Print comparison table
    print(f"\n{'Market':<10} {'Date':<12} {'Price%':<8} {'Raw Ret%':<10} {'Imp Ret%':<10} {'Gain%':<8}")
    print("-"*100)
    
    for _, row in df_comparison.iterrows():
        print(f"{row['Market']:<10} {row['Date']:<12} "
              f"{row['Price_Change']:>7.2f} "
              f"{row['Raw_Return']:>9.2f} "
              f"{row['Imp_Return']:>9.2f} "
              f"{row['Improvement']:>+7.2f}")
    
    # Action distribution comparison
    print("\n" + "="*100)
    print("ACTION DISTRIBUTION COMPARISON")
    print("="*100)
    
    print(f"\n{'Market':<10} {'Strategy':<10} {'Hold%':<10} {'Buy%':<10} {'Sell%':<10}")
    print("-"*100)
    
    for _, row in df_comparison.iterrows():
        print(f"{row['Market']:<10} {'Raw':<10} {row['Raw_Hold']:>8.1f} {row['Raw_Buy']:>8.1f} {row['Raw_Sell']:>8.1f}")
        print(f"{'':<10} {'Improved':<10} {row['Imp_Hold']:>8.1f} {row['Imp_Buy']:>8.1f} {row['Imp_Sell']:>8.1f}")
        print("-"*100)
    
    return df_comparison

def plot_comparison(results, df_comparison, save_path='results/charts/'):
    """Vẽ biểu đồ so sánh toàn diện"""
    os.makedirs(save_path, exist_ok=True)
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.4)
    
    markets = ['BULL', 'BEAR', 'SIDEWAYS']
    colors = {'BULL': 'green', 'BEAR': 'red', 'SIDEWAYS': 'blue'}
    
    # Row 1: Price charts for each market
    for idx, market in enumerate(markets):
        if market not in results:
            continue
        
        data = results[market]
        pred_df = data['pred_df']
        
        ax = fig.add_subplot(gs[0, idx])
        ax.plot(pred_df['timestamp'], pred_df['price'], 
                color=colors[market], linewidth=2, label='BTC Price')
        
        # Mark improved actions
        buy_mask = pred_df['improved_action'] == 1
        sell_mask = pred_df['improved_action'] == 2
        
        ax.scatter(pred_df[buy_mask]['timestamp'], pred_df[buy_mask]['price'],
                  c='darkgreen', marker='^', s=100, label='Buy', zorder=5)
        ax.scatter(pred_df[sell_mask]['timestamp'], pred_df[sell_mask]['price'],
                  c='darkred', marker='v', s=100, label='Sell', zorder=5)
        
        ax.set_title(f'{market} Market\n{data["test_date"]}', 
                    fontweight='bold', fontsize=12)
        ax.set_xlabel('Time')
        ax.set_ylabel('Price (USD)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    # Row 2, Col 1: Performance Comparison
    ax1 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(df_comparison))
    width = 0.35
    
    ax1.bar(x - width/2, df_comparison['Raw_Return'], width, 
           label='Raw', alpha=0.8, color='orange')
    ax1.bar(x + width/2, df_comparison['Imp_Return'], width,
           label='Improved', alpha=0.8, color='blue')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    ax1.set_xlabel('Market Type')
    ax1.set_ylabel('Return (%)')
    ax1.set_title('Performance: Raw vs Improved', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_comparison['Market'])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Row 2, Col 2: Improvement
    ax2 = fig.add_subplot(gs[1, 1])
    colors_bars = ['green' if x > 0 else 'red' for x in df_comparison['Improvement']]
    ax2.bar(df_comparison['Market'], df_comparison['Improvement'], 
           color=colors_bars, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Improvement by Market', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Row 2, Col 3-4: Action Distribution Stacked
    ax3 = fig.add_subplot(gs[1, 2:])
    
    markets_list = df_comparison['Market'].tolist()
    raw_hold = df_comparison['Raw_Hold'].tolist()
    raw_buy = df_comparison['Raw_Buy'].tolist()
    raw_sell = df_comparison['Raw_Sell'].tolist()
    
    imp_hold = df_comparison['Imp_Hold'].tolist()
    imp_buy = df_comparison['Imp_Buy'].tolist()
    imp_sell = df_comparison['Imp_Sell'].tolist()
    
    x = np.arange(len(markets_list)) * 2
    width = 0.8
    
    # Raw bars
    ax3.bar(x - width/2, raw_hold, width, label='Hold', color='blue', alpha=0.7)
    ax3.bar(x - width/2, raw_buy, width, bottom=raw_hold, 
           label='Buy', color='green', alpha=0.7)
    ax3.bar(x - width/2, raw_sell, width, 
           bottom=[h+b for h,b in zip(raw_hold, raw_buy)],
           label='Sell', color='red', alpha=0.7)
    
    # Improved bars
    ax3.bar(x + width/2, imp_hold, width, color='blue', alpha=0.7, hatch='//')
    ax3.bar(x + width/2, imp_buy, width, bottom=imp_hold,
           color='green', alpha=0.7, hatch='//')
    ax3.bar(x + width/2, imp_sell, width,
           bottom=[h+b for h,b in zip(imp_hold, imp_buy)],
           color='red', alpha=0.7, hatch='//')
    
    ax3.set_ylabel('Action Distribution (%)')
    ax3.set_title('Action Distribution: Raw (solid) vs Improved (hatched)', 
                 fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(markets_list)
    ax3.legend(loc='upper right')
    ax3.set_ylim([0, 100])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Row 3: Summary Table with Metrics
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    summary_data = [
        ['Market', 'Price%', 'Raw Ret%', 'Imp Ret%', 'Gain%', 'MAE', 'RMSE', 'R²', 'Acc%'],
    ]
    
    for _, row in df_comparison.iterrows():
        summary_data.append([
            row['Market'],
            f"{row['Price_Change']:.2f}%",
            f"{row['Raw_Return']:.2f}%",
            f"{row['Imp_Return']:.2f}%",
            f"{row['Improvement']:+.2f}%",
            f"{row['MAE']:.3f}",
            f"{row['RMSE']:.3f}",
            f"{row['R2']:.3f}",
            f"{row['Accuracy']:.1f}%",
        ])
    
    # Add averages
    summary_data.append([
        'AVERAGE',
        f"{df_comparison['Price_Change'].mean():.2f}%",
        f"{df_comparison['Raw_Return'].mean():.2f}%",
        f"{df_comparison['Imp_Return'].mean():.2f}%",
        f"{df_comparison['Improvement'].mean():+.2f}%",
        f"{df_comparison['MAE'].mean():.3f}",
        f"{df_comparison['RMSE'].mean():.3f}",
        f"{df_comparison['R2'].mean():.3f}",
        f"{df_comparison['Accuracy'].mean():.1f}%",
    ])
    
    table = ax4.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style average row
    for i in range(len(summary_data[0])):
        table[(len(summary_data)-1, i)].set_facecolor('#FFD700')
        table[(len(summary_data)-1, i)].set_text_props(weight='bold')
    
    # Overall title
    fig.suptitle('SO SÁNH RAW vs IMPROVED DQN TRÊN 3 MARKET REGIMES', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save
    save_file = os.path.join(save_path, 'market_regimes_comparison_improved.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison chart: {save_file}")
    
    plt.close()

def generate_summary_report(results, df_comparison, save_path='results/reports/'):
    """Tạo báo cáo tổng kết markdown"""
    os.makedirs(save_path, exist_ok=True)
    
    report = f"""# SO SÁNH RAW vs IMPROVED DQN TRÊN 3 MARKET REGIMES

## Tổng Quan

So sánh hiệu quả của **Smart Post-Processing Rules** trên 3 loại thị trường khác nhau:
- **Bull Market**: Thị trường tăng giá
- **Bear Market**: Thị trường giảm giá  
- **Sideways Market**: Thị trường đi ngang

## 1. Kết Quả Tổng Thể

| Market | Date | Price Change | Raw Return | Improved Return | Improvement |
|--------|------|--------------|------------|-----------------|-------------|
"""
    
    for _, row in df_comparison.iterrows():
        report += f"| {row['Market']} | {row['Date']} | {row['Price_Change']:.2f}% | "
        report += f"{row['Raw_Return']:.2f}% | {row['Imp_Return']:.2f}% | "
        report += f"**{row['Improvement']:+.2f}%** |\n"
    
    avg_improvement = df_comparison['Improvement'].mean()
    report += f"\n**Trung bình Improvement: {avg_improvement:+.2f}%**\n"
    
    report += f"""

## 2. Phân Tích Chi Tiết

### 2.1. Bull Market ({TEST_DATES['BULL']})

"""
    
    if 'BULL' in results:
        bull_data = results['BULL']
        bull_comp = df_comparison[df_comparison['Market'] == 'BULL'].iloc[0]
        
        report += f"""- **Giá thị trường**: {bull_data['market_info']['price_change']:.2f}% (Tăng)
- **Raw Strategy**: {bull_comp['Raw_Return']:.2f}% return
  - Buy: {bull_comp['Raw_Buy']:.1f}% | Hold: {bull_comp['Raw_Hold']:.1f}% | Sell: {bull_comp['Raw_Sell']:.1f}%
- **Improved Strategy**: {bull_comp['Imp_Return']:.2f}% return
  - Buy: {bull_comp['Imp_Buy']:.1f}% | Hold: {bull_comp['Imp_Hold']:.1f}% | Sell: {bull_comp['Imp_Sell']:.1f}%
- **Improvement**: {bull_comp['Improvement']:+.2f}%

**Nhận xét**:
"""
        
        if bull_comp['Improvement'] > 0:
            report += "✅ Rules hoạt động tốt: Tối ưu hóa entry/exit points\n"
        elif bull_comp['Improvement'] < -1:
            report += "❌ Rules quá conservative: Bỏ lỡ cơ hội profit trong uptrend\n"
            report += "  - Cần điều chỉnh: Cho phép Buy nhiều hơn khi trend > 0.3 và RSI < 70\n"
        else:
            report += "⚠️ Rules có ảnh hưởng trung lập\n"
    
    report += f"""

### 2.2. Bear Market ({TEST_DATES['BEAR']})

"""
    
    if 'BEAR' in results:
        bear_data = results['BEAR']
        bear_comp = df_comparison[df_comparison['Market'] == 'BEAR'].iloc[0]
        
        report += f"""- **Giá thị trường**: {bear_data['market_info']['price_change']:.2f}% (Giảm mạnh)
- **Raw Strategy**: {bear_comp['Raw_Return']:.2f}% return
  - Buy: {bear_comp['Raw_Buy']:.1f}% | Hold: {bear_comp['Raw_Hold']:.1f}% | Sell: {bear_comp['Raw_Sell']:.1f}%
- **Improved Strategy**: {bear_comp['Imp_Return']:.2f}% return
  - Buy: {bear_comp['Imp_Buy']:.1f}% | Hold: {bear_comp['Imp_Hold']:.1f}% | Sell: {bear_comp['Imp_Sell']:.1f}%
- **Improvement**: {bear_comp['Improvement']:+.2f}%

**Nhận xét**:
"""
        
        if bear_comp['Improvement'] > 5:
            report += "✅✅ Rules hoạt động XUẤT SẮC: Bảo vệ vốn hiệu quả trong crash\n"
            report += "  - Chặn được Buy sai timing (falling knife)\n"
            report += "  - Hold cash để tránh lỗ\n"
        elif bear_comp['Improvement'] > 0:
            report += "✅ Rules hoạt động tốt: Giảm thiểu loss\n"
        else:
            report += "⚠️ Cần kiểm tra lại rules\n"
    
    report += f"""

### 2.3. Sideways Market ({TEST_DATES['SIDEWAYS']})

"""
    
    if 'SIDEWAYS' in results:
        side_data = results['SIDEWAYS']
        side_comp = df_comparison[df_comparison['Market'] == 'SIDEWAYS'].iloc[0]
        
        report += f"""- **Giá thị trường**: {side_data['market_info']['price_change']:.2f}% (Đi ngang)
- **Raw Strategy**: {side_comp['Raw_Return']:.2f}% return
  - Buy: {side_comp['Raw_Buy']:.1f}% | Hold: {side_comp['Raw_Hold']:.1f}% | Sell: {side_comp['Raw_Sell']:.1f}%
- **Improved Strategy**: {side_comp['Imp_Return']:.2f}% return
  - Buy: {side_comp['Imp_Buy']:.1f}% | Hold: {side_comp['Imp_Hold']:.1f}% | Sell: {side_comp['Imp_Sell']:.1f}%
- **Improvement**: {side_comp['Improvement']:+.2f}%

**Nhận xét**:
"""
        
        if side_comp['Improvement'] > 0:
            report += "✅ Rules giúp tránh over-trading\n"
        else:
            report += "⚠️ Rules có thể hạn chế cơ hội range trading\n"
    
    report += """

## 3. Kết Luận

### 3.1. Ưu điểm của Improved Strategy

"""
    
    best_market = df_comparison.loc[df_comparison['Improvement'].idxmax(), 'Market']
    best_improvement = df_comparison['Improvement'].max()
    
    report += f"""1. **Hiệu quả nhất trong {best_market} market** với improvement +{best_improvement:.2f}%
2. **Bảo vệ vốn tốt** trong bear market (chặn Buy sai timing)
3. **Giảm over-trading** với rules thông minh

### 3.2. Hạn chế

"""
    
    worst_market = df_comparison.loc[df_comparison['Improvement'].idxmin(), 'Market']
    worst_improvement = df_comparison['Improvement'].min()
    
    if worst_improvement < 0:
        report += f"""1. **Kém hiệu quả trong {worst_market} market** với improvement {worst_improvement:.2f}%
2. Rules quá conservative có thể bỏ lỡ cơ hội profit
3. Cần **market-adaptive rules** thay vì static rules

### 3.3. Đề Xuất Cải Thiện

#### Cách 1: Market-Adaptive Rules

```python
# Detect market regime first
if market_regime == 'BULL':
    # Relax Buy constraints
    if trend > 0.3 and rsi < 75:
        allow_buy = True
elif market_regime == 'BEAR':
    # Strict Buy constraints (current rules)
    if trend < -0.5 or (rsi < 30 and trend < 0):
        block_buy = True
else:  # SIDEWAYS
    # Moderate constraints
    if abs(trend) < 0.2 and 40 < rsi < 60:
        prefer_hold = True
```

#### Cách 2: Dynamic Thresholds

```python
# Adjust thresholds based on volatility
if volatility > 0.05:  # High volatility
    rsi_oversold = 25  # More strict
    trend_threshold = -0.7
else:  # Low volatility  
    rsi_oversold = 35  # More relaxed
    trend_threshold = -0.3
```

#### Cách 3: Hybrid Approach

- Dùng Improved trong Bear market (bảo vệ vốn)
- Dùng Raw trong Bull market (maximize profit)
- Auto-switch dựa trên market detection
"""
    else:
        report += "Không có hạn chế rõ ràng - Rules hoạt động tốt trên tất cả regimes!\n"
    
    report += f"""

## 4. Metrics Summary

| Metric | Bull | Bear | Sideways | Average |
|--------|------|------|----------|---------|
| Price Change | {df_comparison[df_comparison['Market']=='BULL']['Price_Change'].values[0] if 'BULL' in results else 'N/A':.2f}% | {df_comparison[df_comparison['Market']=='BEAR']['Price_Change'].values[0] if 'BEAR' in results else 'N/A':.2f}% | {df_comparison[df_comparison['Market']=='SIDEWAYS']['Price_Change'].values[0] if 'SIDEWAYS' in results else 'N/A':.2f}% | {df_comparison['Price_Change'].mean():.2f}% |
| Raw Return | {df_comparison[df_comparison['Market']=='BULL']['Raw_Return'].values[0] if 'BULL' in results else 'N/A':.2f}% | {df_comparison[df_comparison['Market']=='BEAR']['Raw_Return'].values[0] if 'BEAR' in results else 'N/A':.2f}% | {df_comparison[df_comparison['Market']=='SIDEWAYS']['Raw_Return'].values[0] if 'SIDEWAYS' in results else 'N/A':.2f}% | {df_comparison['Raw_Return'].mean():.2f}% |
| Improved Return | {df_comparison[df_comparison['Market']=='BULL']['Imp_Return'].values[0] if 'BULL' in results else 'N/A':.2f}% | {df_comparison[df_comparison['Market']=='BEAR']['Imp_Return'].values[0] if 'BEAR' in results else 'N/A':.2f}% | {df_comparison[df_comparison['Market']=='SIDEWAYS']['Imp_Return'].values[0] if 'SIDEWAYS' in results else 'N/A':.2f}% | {df_comparison['Imp_Return'].mean():.2f}% |
| **Improvement** | **{df_comparison[df_comparison['Market']=='BULL']['Improvement'].values[0] if 'BULL' in results else 'N/A':+.2f}%** | **{df_comparison[df_comparison['Market']=='BEAR']['Improvement'].values[0] if 'BEAR' in results else 'N/A':+.2f}%** | **{df_comparison[df_comparison['Market']=='SIDEWAYS']['Improvement'].values[0] if 'SIDEWAYS' in results else 'N/A':+.2f}%** | **{avg_improvement:+.2f}%** |

---

**Ngày tạo**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Charts**: `results/charts/market_regimes_comparison_improved.png`
"""
    
    # Save report
    report_file = os.path.join(save_path, 'market_regimes_improved_report.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Saved report: {report_file}")

if __name__ == '__main__':
    print("="*80)
    print("SO SÁNH RAW vs IMPROVED DQN TRÊN 3 MARKET REGIMES")
    print("="*80)
    
    # Run tests
    results = run_all_tests()
    
    # Create comparison table
    df_comparison = create_comparison_table(results)
    
    # Plot comparison
    plot_comparison(results, df_comparison)
    
    # Generate summary report
    generate_summary_report(results, df_comparison)
    
    # Save CSV
    df_comparison.to_csv('results/reports/market_regimes_improved_comparison.csv', index=False)
    print("\nSaved CSV: results/reports/market_regimes_improved_comparison.csv")
    
    print("\n" + "="*80)
    print("HOÀN THÀNH!")
    print("="*80)
    print(f"Average Improvement: {df_comparison['Improvement'].mean():+.2f}%")
    print("Chart: results/charts/market_regimes_comparison_improved.png")
    print("Report: results/reports/market_regimes_improved_report.md")
