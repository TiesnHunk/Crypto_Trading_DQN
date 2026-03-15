"""
SO SÁNH DQN TRÊN 3 MARKET REGIMES: BULL, BEAR, SIDEWAYS
"""

import sys
sys.path.append('src')

from predict_one_day_dqn import OneDayPredictor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
CHECKPOINT_PATH = 'src/checkpoints_dqn/checkpoint_best.pkl'
DATA_PATH = 'data/raw/multi_coin_1h.csv'

# Test dates (từ find_market_regimes.py)
TEST_DATES = {
    'BULL': '2021-02-08',    # +19.75%, Trend=0.42, RSI=70
    'BEAR': '2020-03-12',    # -39.34%, Trend=-1.00, RSI=22
    'SIDEWAYS': '2025-02-07' # -0.00%, Trend=-0.08, RSI=52
}

def run_all_tests():
    """Run DQN prediction on all 3 market regimes"""
    results = {}
    
    for market_type, test_date in TEST_DATES.items():
        print("\n" + "="*80)
        print(f"TESTING {market_type} MARKET - {test_date}")
        print("="*80)
        
        predictor = OneDayPredictor(CHECKPOINT_PATH, DATA_PATH)
        
        # Load data
        df_full = predictor.load_data()
        
        # Select test day
        test_df, start_idx = predictor.select_test_day(df_full, test_date)
        
        # Load DQN model
        agent, env = predictor.load_dqn_model(test_df)
        
        # Predict actions
        pred_df, final_env = predictor.predict_actions(agent, test_df)
        
        # Calculate metrics
        metrics, ideal_actions = predictor.calculate_metrics(pred_df, test_df)
        
        # Analyze market environment
        market_info = predictor.analyze_market_environment(test_df, pred_df)
        
        # Store results
        results[market_type] = {
            'test_date': test_date,
            'pred_df': pred_df,
            'test_df': test_df,
            'metrics': metrics,
            'market_info': market_info,
            'ideal_actions': ideal_actions
        }
    
    return results

def compare_results(results):
    """So sánh kết quả giữa 3 market regimes"""
    print("\n" + "="*80)
    print("SO SÁNH KẾT QUẢ GIỮA 3 MARKET REGIMES")
    print("="*80)
    
    # Create comparison table
    comparison = []
    
    for market_type in ['BULL', 'BEAR', 'SIDEWAYS']:
        if market_type not in results:
            continue
            
        data = results[market_type]
        metrics = data['metrics']
        market_info = data['market_info']
        pred_df = data['pred_df']
        
        # Action distribution
        action_dist = pred_df['action_name'].value_counts()
        
        comparison.append({
            'Market': market_type,
            'Date': data['test_date'],
            'Price Change': f"{market_info['price_change']:.2f}%",
            'Trend Avg': f"{market_info['avg_trend']:.3f}",
            'RSI Avg': f"{market_info['avg_rsi']:.2f}",
            'Total Return': f"{metrics['Total_Return']:.2f}%",
            'Final Balance': f"${metrics['Final_Balance']:.2f}",
            'Num Trades': metrics['Num_Trades'],
            'Hold %': f"{action_dist.get('Hold', 0)/len(pred_df)*100:.1f}%",
            'Buy %': f"{action_dist.get('Buy', 0)/len(pred_df)*100:.1f}%",
            'Sell %': f"{action_dist.get('Sell', 0)/len(pred_df)*100:.1f}%",
            'MAE': f"{metrics['MAE']:.4f}",
            'Accuracy': f"{metrics['Accuracy']:.2f}%",
            'R²': f"{metrics['R2']:.4f}"
        })
    
    df_comparison = pd.DataFrame(comparison)
    
    print("\n" + df_comparison.to_string(index=False))
    
    # Save to file
    os.makedirs('results/reports', exist_ok=True)
    df_comparison.to_csv('results/reports/market_regime_comparison.csv', index=False)
    print("\n✓ Saved comparison to: results/reports/market_regime_comparison.csv")
    
    return df_comparison

def plot_comparison(results):
    """Vẽ biểu đồ so sánh 3 market regimes"""
    print("\nGenerating comparison visualizations...")
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    markets = ['BULL', 'BEAR', 'SIDEWAYS']
    colors = {'BULL': 'green', 'BEAR': 'red', 'SIDEWAYS': 'blue'}
    
    for idx, market_type in enumerate(markets):
        if market_type not in results:
            continue
            
        data = results[market_type]
        pred_df = data['pred_df']
        test_df = data['test_df']
        metrics = data['metrics']
        market_info = data['market_info']
        
        # Row for each market
        row = idx
        
        # Panel 1: Price + Actions
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.plot(pred_df['timestamp'], pred_df['price'], 'k-', linewidth=2, alpha=0.7)
        
        buy_mask = pred_df['action'] == 1
        sell_mask = pred_df['action'] == 2
        
        ax1.scatter(pred_df[buy_mask]['timestamp'], pred_df[buy_mask]['price'], 
                   c='green', marker='^', s=100, label='Buy', zorder=5)
        ax1.scatter(pred_df[sell_mask]['timestamp'], pred_df[sell_mask]['price'], 
                   c='red', marker='v', s=100, label='Sell', zorder=5)
        
        ax1.set_title(f'{market_type} - Price & Actions\n{data["test_date"]}', 
                     fontweight='bold', color=colors[market_type])
        ax1.set_ylabel('Price (USD)')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45, labelsize=8)
        
        # Panel 2: Portfolio Value
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.plot(pred_df['timestamp'], pred_df['portfolio_value'], 
                color=colors[market_type], linewidth=2)
        ax2.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Initial')
        ax2.fill_between(pred_df['timestamp'], 10000, pred_df['portfolio_value'], 
                         where=pred_df['portfolio_value']>=10000, alpha=0.3, color='green')
        ax2.fill_between(pred_df['timestamp'], 10000, pred_df['portfolio_value'], 
                         where=pred_df['portfolio_value']<10000, alpha=0.3, color='red')
        
        ax2.set_title(f'Portfolio Value\nReturn: {metrics["Total_Return"]:.2f}%', 
                     fontweight='bold')
        ax2.set_ylabel('Value (USD)')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45, labelsize=8)
        
        # Panel 3: Action Distribution
        ax3 = fig.add_subplot(gs[row, 2])
        action_counts = pred_df['action_name'].value_counts()
        colors_pie = {'Hold': 'blue', 'Buy': 'green', 'Sell': 'red'}
        colors_list = [colors_pie.get(action, 'gray') for action in action_counts.index]
        
        ax3.pie(action_counts.values, labels=action_counts.index, autopct='%1.1f%%',
               colors=colors_list, startangle=90)
        ax3.set_title('Action Distribution', fontweight='bold')
        
        # Panel 4: Metrics Table
        ax4 = fig.add_subplot(gs[row, 3])
        ax4.axis('off')
        
        metrics_data = [
            ['Metric', 'Value'],
            ['Price Change', f"{market_info['price_change']:.2f}%"],
            ['Trend Avg', f"{market_info['avg_trend']:.3f}"],
            ['RSI Avg', f"{market_info['avg_rsi']:.2f}"],
            ['', ''],
            ['Total Return', f"{metrics['Total_Return']:.2f}%"],
            ['Final Balance', f"${metrics['Final_Balance']:.2f}"],
            ['Num Trades', f"{metrics['Num_Trades']}"],
            ['', ''],
            ['MAE', f"{metrics['MAE']:.4f}"],
            ['Accuracy', f"{metrics['Accuracy']:.2f}%"],
            ['R²', f"{metrics['R2']:.4f}"],
        ]
        
        table = ax4.table(cellText=metrics_data, cellLoc='left', loc='center',
                         colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Style header
        table[(0, 0)].set_facecolor(colors[market_type])
        table[(0, 1)].set_facecolor(colors[market_type])
        table[(0, 0)].set_text_props(weight='bold', color='white')
        table[(0, 1)].set_text_props(weight='bold', color='white')
        
        ax4.set_title(f'{market_type} Metrics', fontweight='bold')
    
    fig.suptitle('SO SÁNH DQN TRÊN 3 MARKET REGIMES', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save
    os.makedirs('results/charts', exist_ok=True)
    save_file = 'results/charts/market_regime_comparison.png'
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization to: {save_file}")
    
    plt.close()

def analyze_alignment(results):
    """Phân tích alignment của DQN với từng market regime"""
    print("\n" + "="*80)
    print("PHÂN TÍCH ALIGNMENT CỦA DQN VỚI MARKET REGIME")
    print("="*80)
    
    for market_type in ['BULL', 'BEAR', 'SIDEWAYS']:
        if market_type not in results:
            continue
            
        data = results[market_type]
        pred_df = data['pred_df']
        market_info = data['market_info']
        
        action_dist = pred_df['action_name'].value_counts()
        total = len(pred_df)
        
        print(f"\n{market_type} MARKET ({data['test_date']}):")
        print(f"  Price change: {market_info['price_change']:.2f}%")
        print(f"  Trend avg: {market_info['avg_trend']:.3f}")
        print(f"  RSI avg: {market_info['avg_rsi']:.2f}")
        
        print(f"\n  DQN Actions:")
        print(f"    Hold: {action_dist.get('Hold', 0)}/{total} ({action_dist.get('Hold', 0)/total*100:.1f}%)")
        print(f"    Buy:  {action_dist.get('Buy', 0)}/{total} ({action_dist.get('Buy', 0)/total*100:.1f}%)")
        print(f"    Sell: {action_dist.get('Sell', 0)}/{total} ({action_dist.get('Sell', 0)/total*100:.1f}%)")
        
        # Evaluate alignment
        if market_type == 'BULL':
            buy_pct = action_dist.get('Buy', 0) / total * 100
            if buy_pct > 30:
                alignment = "✓ Phù hợp: DQN mua nhiều trong bull market"
            elif buy_pct > 10:
                alignment = "~ Trung bình: DQN mua ít trong bull market"
            else:
                alignment = "✗ Không tối ưu: DQN tránh mua trong bull market"
        elif market_type == 'BEAR':
            sell_pct = action_dist.get('Sell', 0) / total * 100
            if sell_pct > 50:
                alignment = "✓ Phù hợp: DQN bán nhiều trong bear market"
            elif sell_pct > 30:
                alignment = "~ Trung bình: DQN bán vừa phải trong bear market"
            else:
                alignment = "✗ Không tối ưu: DQN tránh bán trong bear market"
        else:  # SIDEWAYS
            hold_pct = action_dist.get('Hold', 0) / total * 100
            trade_freq = (action_dist.get('Buy', 0) + action_dist.get('Sell', 0)) / total * 100
            if trade_freq > 40 and abs(action_dist.get('Buy', 0) - action_dist.get('Sell', 0)) < 5:
                alignment = "✓ Phù hợp: Trading cân bằng trong sideways market"
            elif trade_freq > 20:
                alignment = "~ Trung bình: Có trading nhưng chưa cân bằng"
            else:
                alignment = "✗ Không tối ưu: Quá ít trading trong sideways market"
        
        print(f"\n  Đánh giá: {alignment}")

def generate_summary_report(results, df_comparison):
    """Tạo báo cáo tổng kết"""
    print("\nGenerating summary report...")
    
    report = f"""# SO SÁNH DQN TRÊN 3 MARKET REGIMES

## Tóm Tắt

Nghiên cứu này đánh giá hiệu suất của Deep Q-Network (DQN) trên 3 loại thị trường khác nhau:
- **Bull Market**: Thị trường tăng giá mạnh
- **Bear Market**: Thị trường giảm giá mạnh  
- **Sideways Market**: Thị trường đi ngang

## 1. Các Ngày Test

| Market | Date | Price Change | Trend Avg | RSI Avg |
|--------|------|--------------|-----------|---------|
"""
    
    for market_type in ['BULL', 'BEAR', 'SIDEWAYS']:
        if market_type in results:
            data = results[market_type]
            market_info = data['market_info']
            report += f"| {market_type} | {data['test_date']} | {market_info['price_change']:.2f}% | {market_info['avg_trend']:.3f} | {market_info['avg_rsi']:.2f} |\n"
    
    report += f"""
## 2. Kết Quả Chi Tiết

### 2.1. Bảng So Sánh Tổng Quan

{df_comparison.to_markdown(index=False)}

### 2.2. Trading Performance

"""
    
    for market_type in ['BULL', 'BEAR', 'SIDEWAYS']:
        if market_type not in results:
            continue
            
        data = results[market_type]
        metrics = data['metrics']
        pred_df = data['pred_df']
        action_dist = pred_df['action_name'].value_counts()
        
        report += f"""
#### {market_type} Market ({data['test_date']})

**Market Conditions:**
- Price change: {data['market_info']['price_change']:.2f}%
- Trend average: {data['market_info']['avg_trend']:.3f}
- RSI average: {data['market_info']['avg_rsi']:.2f}

**DQN Performance:**
- Total Return: {metrics['Total_Return']:.2f}%
- Final Balance: ${metrics['Final_Balance']:.2f}
- Number of Trades: {metrics['Num_Trades']}

**Action Distribution:**
- Hold: {action_dist.get('Hold', 0)} ({action_dist.get('Hold', 0)/len(pred_df)*100:.1f}%)
- Buy: {action_dist.get('Buy', 0)} ({action_dist.get('Buy', 0)/len(pred_df)*100:.1f}%)
- Sell: {action_dist.get('Sell', 0)} ({action_dist.get('Sell', 0)/len(pred_df)*100:.1f}%)

**Metrics:**
- MAE: {metrics['MAE']:.4f}
- Accuracy: {metrics['Accuracy']:.2f}%
- R² Score: {metrics['R2']:.4f}

"""
    
    report += """
## 3. Phân Tích và Nhận Xét

### 3.1. Bull Market Performance

"""
    
    if 'BULL' in results:
        bull_metrics = results['BULL']['metrics']
        bull_actions = results['BULL']['pred_df']['action_name'].value_counts()
        buy_pct = bull_actions.get('Buy', 0) / len(results['BULL']['pred_df']) * 100
        
        report += f"""
**Kết quả:**
- DQN return: {bull_metrics['Total_Return']:.2f}%
- Market return: {results['BULL']['market_info']['price_change']:.2f}%
- Buy actions: {buy_pct:.1f}%

**Nhận xét:**
"""
        if buy_pct > 30:
            report += "✓ DQN nhận diện đúng bull market và thực hiện nhiều lệnh Buy\n"
        elif buy_pct > 10:
            report += "~ DQN có nhận diện bull market nhưng còn thận trọng\n"
        else:
            report += "✗ DQN không tận dụng tốt bull market, quá ít lệnh Buy\n"

    report += """
### 3.2. Bear Market Performance

"""
    
    if 'BEAR' in results:
        bear_metrics = results['BEAR']['metrics']
        bear_actions = results['BEAR']['pred_df']['action_name'].value_counts()
        sell_pct = bear_actions.get('Sell', 0) / len(results['BEAR']['pred_df']) * 100
        
        report += f"""
**Kết quả:**
- DQN return: {bear_metrics['Total_Return']:.2f}%
- Market return: {results['BEAR']['market_info']['price_change']:.2f}%
- Sell actions: {sell_pct:.1f}%

**Nhận xét:**
"""
        if sell_pct > 50:
            report += "✓ DQN nhận diện đúng bear market và bảo vệ vốn tốt\n"
        elif sell_pct > 30:
            report += "~ DQN có nhận diện bear market nhưng phản ứng chậm\n"
        else:
            report += "✗ DQN không bảo vệ vốn tốt trong bear market\n"

    report += """
### 3.3. Sideways Market Performance

"""
    
    if 'SIDEWAYS' in results:
        sideways_metrics = results['SIDEWAYS']['metrics']
        sideways_actions = results['SIDEWAYS']['pred_df']['action_name'].value_counts()
        trade_freq = (sideways_actions.get('Buy', 0) + sideways_actions.get('Sell', 0)) / len(results['SIDEWAYS']['pred_df']) * 100
        
        report += f"""
**Kết quả:**
- DQN return: {sideways_metrics['Total_Return']:.2f}%
- Market return: {results['SIDEWAYS']['market_info']['price_change']:.2f}%
- Trading frequency: {trade_freq:.1f}%

**Nhận xét:**
"""
        if trade_freq > 40:
            report += "✓ DQN thực hiện range trading tốt trong sideways market\n"
        elif trade_freq > 20:
            report += "~ DQN có trading nhưng chưa tối ưu\n"
        else:
            report += "✗ DQN quá thận trọng, bỏ lỡ cơ hội trong sideways market\n"

    report += """
## 4. Kết Luận Chung

### 4.1. Market-Adaptive Behavior

DQN cho thấy khả năng thích nghi với các market regime:
"""
    
    # Summary of which markets DQN performs best
    returns = []
    for market_type in ['BULL', 'BEAR', 'SIDEWAYS']:
        if market_type in results:
            returns.append((market_type, results[market_type]['metrics']['Total_Return']))
    
    returns.sort(key=lambda x: x[1], reverse=True)
    
    report += f"""
**Performance ranking:**
1. {returns[0][0]}: {returns[0][1]:.2f}% return
"""
    if len(returns) > 1:
        report += f"2. {returns[1][0]}: {returns[1][1]:.2f}% return\n"
    if len(returns) > 2:
        report += f"3. {returns[2][0]}: {returns[2][1]:.2f}% return\n"

    report += """
### 4.2. Limitations

1. **Training Bias**: DQN có thể bị bias theo market regime chủ đạo trong training data
2. **Short-term Testing**: Chỉ test 24h/regime, cần test dài hạn hơn
3. **Reward Function**: Có thể cần điều chỉnh reward cho từng market type

### 4.3. Khuyến Nghị

1. **Adaptive Reward**: Điều chỉnh reward function theo market regime
2. **Ensemble Models**: Kết hợp nhiều models cho từng market type
3. **Market Detection**: Thêm module phát hiện market regime tự động
4. **Long-term Testing**: Test trên periods dài hơn (7-30 days)

---

**Ngày tạo**: 2025-12-15  
**Mô hình**: DQN (Episode 1831, Best Profit $3.8M)  
**Checkpoint**: src/checkpoints_dqn/checkpoint_best.pkl
"""
    
    # Save report
    os.makedirs('results/reports', exist_ok=True)
    save_file = 'results/reports/market_regime_comparison_report.md'
    with open(save_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ Saved report to: {save_file}")

if __name__ == '__main__':
    print("="*80)
    print("SO SÁNH DQN TRÊN 3 MARKET REGIMES")
    print("="*80)
    
    # Run tests on all markets
    results = run_all_tests()
    
    # Compare results
    df_comparison = compare_results(results)
    
    # Plot comparison
    plot_comparison(results)
    
    # Analyze alignment
    analyze_alignment(results)
    
    # Generate summary report
    generate_summary_report(results, df_comparison)
    
    print("\n" + "="*80)
    print("HOÀN THÀNH!")
    print("="*80)
    print("Files created:")
    print("  - results/reports/market_regime_comparison.csv")
    print("  - results/reports/market_regime_comparison_report.md")
    print("  - results/charts/market_regime_comparison.png")
