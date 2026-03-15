"""
Script để so sánh validation performance giữa Episode 1831 và Episode 2504
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Data từ validation results
coins = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL']

# Episode 1831 (checkpoint_best.pkl)
ep1831_profit = [46.59, 20.68, 58.34, -2.03, 3.08]
ep1831_return = [117.25, 54.55, 141.68, -5.11, 7.73]
ep1831_mdd = [21.86, 29.27, 19.67, 41.92, 37.63]
ep1831_sharpe = [1.85, 0.64, 2.48, -0.04, 0.07]

# Episode 2504 (checkpoint_latest.pkl)
ep2504_profit = [27.33, 17.23, 26.30, -33.48, 30.44]
ep2504_return = [70.22, 45.56, 67.68, -73.69, 78.11]
ep2504_mdd = [28.28, 31.85, 30.24, 48.20, 32.00]
ep2504_sharpe = [0.86, 0.49, 0.77, -0.53, 0.84]

# Average metrics
avg_metrics = {
    'Episode 1831': {
        'Avg Profit': 25.33,
        'Avg Return': 63.22,
        'Avg MDD': 30.07,
        'Avg Sharpe': 1.00
    },
    'Episode 2504': {
        'Avg Profit': 13.56,
        'Avg Return': 37.58,
        'Avg MDD': 34.11,
        'Avg Sharpe': 0.49
    }
}

# Training metrics
training_metrics = {
    'Episode 1831': {
        'Training Profit': 573.33,
        'Validation Profit': 25.33,
        'Gap': 22.6
    },
    'Episode 2504': {
        'Training Profit': 1143.62,
        'Validation Profit': 13.56,
        'Gap': 84.3
    }
}

# Create figure with subplots
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Profit comparison per coin
ax1 = fig.add_subplot(gs[0, :2])
x = np.arange(len(coins))
width = 0.35
bars1 = ax1.bar(x - width/2, ep1831_profit, width, label='Episode 1831', color='#2ecc71', alpha=0.8)
bars2 = ax1.bar(x + width/2, ep2504_profit, width, label='Episode 2504', color='#e74c3c', alpha=0.8)

ax1.set_xlabel('Coin', fontsize=12, fontweight='bold')
ax1.set_ylabel('Profit (%)', fontsize=12, fontweight='bold')
ax1.set_title('Validation Profit Comparison by Coin', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(coins)
ax1.legend()
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9, fontweight='bold')

# 2. Average metrics comparison
ax2 = fig.add_subplot(gs[0, 2])
metrics_names = ['Avg\nProfit', 'Avg\nReturn', 'Avg\nSharpe']
ep1831_vals = [25.33, 63.22, 1.00]
ep2504_vals = [13.56, 37.58, 0.49]

x = np.arange(len(metrics_names))
bars1 = ax2.bar(x - width/2, ep1831_vals, width, label='Ep 1831', color='#2ecc71', alpha=0.8)
bars2 = ax2.bar(x + width/2, ep2504_vals, width, label='Ep 2504', color='#e74c3c', alpha=0.8)

ax2.set_ylabel('Value', fontsize=12, fontweight='bold')
ax2.set_title('Average Metrics', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics_names, fontsize=10)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 3. Annualized Return comparison
ax3 = fig.add_subplot(gs[1, 0])
x_coins = np.arange(len(coins))  # Re-define x for coin-based plots
bars1 = ax3.bar(x_coins - width/2, ep1831_return, width, label='Episode 1831', color='#3498db', alpha=0.8)
bars2 = ax3.bar(x_coins + width/2, ep2504_return, width, label='Episode 2504', color='#9b59b6', alpha=0.8)

ax3.set_xlabel('Coin', fontsize=12, fontweight='bold')
ax3.set_ylabel('Annualized Return (%)', fontsize=12, fontweight='bold')
ax3.set_title('Annualized Return Comparison', fontsize=14, fontweight='bold')
ax3.set_xticks(x_coins)
ax3.set_xticklabels(coins)
ax3.legend()
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.grid(axis='y', alpha=0.3)

# 4. Maximum Drawdown comparison (lower is better)
ax4 = fig.add_subplot(gs[1, 1])
bars1 = ax4.bar(x_coins - width/2, ep1831_mdd, width, label='Episode 1831', color='#e67e22', alpha=0.8)
bars2 = ax4.bar(x_coins + width/2, ep2504_mdd, width, label='Episode 2504', color='#c0392b', alpha=0.8)

ax4.set_xlabel('Coin', fontsize=12, fontweight='bold')
ax4.set_ylabel('Maximum Drawdown (%)', fontsize=12, fontweight='bold')
ax4.set_title('Maximum Drawdown (Lower is Better)', fontsize=14, fontweight='bold')
ax4.set_xticks(x_coins)
ax4.set_xticklabels(coins)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)
ax4.invert_yaxis()  # Invert vì MDD thấp hơn là tốt hơn

# 5. Sharpe Ratio comparison
ax5 = fig.add_subplot(gs[1, 2])
bars1 = ax5.bar(x_coins - width/2, ep1831_sharpe, width, label='Episode 1831', color='#1abc9c', alpha=0.8)
bars2 = ax5.bar(x_coins + width/2, ep2504_sharpe, width, label='Episode 2504', color='#34495e', alpha=0.8)

ax5.set_xlabel('Coin', fontsize=12, fontweight='bold')
ax5.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
ax5.set_title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
ax5.set_xticks(x_coins)
ax5.set_xticklabels(coins)
ax5.legend()
ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax5.grid(axis='y', alpha=0.3)

# 6. Training vs Validation Gap
ax6 = fig.add_subplot(gs[2, 0])
episodes = ['Episode\n1831', 'Episode\n2504']
training = [573.33, 1143.62]
validation = [25.33, 13.56]

x_pos = np.arange(len(episodes))
width = 0.35

bars1 = ax6.bar(x_pos - width/2, training, width, label='Training Profit', color='#f39c12', alpha=0.8)
bars2 = ax6.bar(x_pos + width/2, validation, width, label='Validation Profit', color='#27ae60', alpha=0.8)

ax6.set_ylabel('Profit (%)', fontsize=12, fontweight='bold')
ax6.set_title('Training vs Validation Gap', fontsize=14, fontweight='bold')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(episodes)
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

# Add gap ratio labels
for i, (t, v) in enumerate(zip(training, validation)):
    gap_ratio = t / v
    ax6.text(i, max(t, v) + 50, f'Gap: {gap_ratio:.1f}x', 
            ha='center', fontsize=10, fontweight='bold', color='red')

# 7. Overfitting visualization
ax7 = fig.add_subplot(gs[2, 1])
gap_ratios = [22.6, 84.3]
colors = ['#2ecc71', '#e74c3c']

bars = ax7.bar(x_pos, gap_ratios, color=colors, alpha=0.8)
ax7.set_ylabel('Training/Validation Gap Ratio', fontsize=12, fontweight='bold')
ax7.set_title('Overfitting Severity (Lower is Better)', fontsize=14, fontweight='bold')
ax7.set_xticks(x_pos)
ax7.set_xticklabels(episodes)
ax7.axhline(y=30, color='orange', linestyle='--', linewidth=2, label='Warning threshold')
ax7.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Critical threshold')
ax7.legend()
ax7.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}x',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# 8. Win statistics
ax8 = fig.add_subplot(gs[2, 2])
categories = ['Profitable\nCoins', 'Positive\nReturn', 'Sharpe>0']
ep1831_wins = [4, 4, 4]  # 4/5 profitable, 4/5 positive return, 4/5 sharpe>0
ep2504_wins = [4, 4, 3]  # 4/5 profitable, 4/5 positive return, 3/5 sharpe>0

x_pos = np.arange(len(categories))
bars1 = ax8.bar(x_pos - width/2, ep1831_wins, width, label='Episode 1831', color='#2ecc71', alpha=0.8)
bars2 = ax8.bar(x_pos + width/2, ep2504_wins, width, label='Episode 2504', color='#e74c3c', alpha=0.8)

ax8.set_ylabel('Count (out of 5)', fontsize=12, fontweight='bold')
ax8.set_title('Success Metrics', fontsize=14, fontweight='bold')
ax8.set_xticks(x_pos)
ax8.set_xticklabels(categories)
ax8.set_ylim([0, 5])
ax8.legend()
ax8.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}/5',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# Main title
fig.suptitle('Validation Performance: Episode 1831 vs Episode 2504\nBest Checkpoint (Episode 1831) vs Latest Checkpoint (Episode 2504)', 
             fontsize=16, fontweight='bold', y=0.995)

# Add summary text box
summary_text = f"""
SUMMARY:
Episode 1831 (Best):
• Avg Profit: +25.33%
• Avg Return: +63.22%
• Avg MDD: 30.07%
• Training Gap: 22.6x
• Profitable: 4/5 coins

Episode 2504 (Latest):
• Avg Profit: +13.56%
• Avg Return: +37.58%
• Avg MDD: 34.11%
• Training Gap: 84.3x
• Profitable: 4/5 coins

VERDICT: Episode 1831 is 87% better!
Episode 2504 shows severe overfitting.
"""

# Add text box at bottom
fig.text(0.02, 0.02, summary_text, fontsize=10, family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save figure
output_path = 'results/charts/checkpoint_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved comparison chart to: {output_path}")

plt.show()

print("\n" + "="*80)
print("DETAILED COMPARISON")
print("="*80)

print("\n📊 PROFIT COMPARISON:")
print(f"{'Coin':<8} {'Episode 1831':<15} {'Episode 2504':<15} {'Difference':<15} {'Winner'}")
print("-" * 70)
for i, coin in enumerate(coins):
    diff = ep1831_profit[i] - ep2504_profit[i]
    winner = "✅ Ep1831" if diff > 0 else "❌ Ep2504"
    print(f"{coin:<8} {ep1831_profit[i]:>12.2f}%   {ep2504_profit[i]:>12.2f}%   {diff:>12.2f}%   {winner}")

print("\n📈 ANNUALIZED RETURN COMPARISON:")
print(f"{'Coin':<8} {'Episode 1831':<15} {'Episode 2504':<15} {'Difference':<15} {'Winner'}")
print("-" * 70)
for i, coin in enumerate(coins):
    diff = ep1831_return[i] - ep2504_return[i]
    winner = "✅ Ep1831" if diff > 0 else "❌ Ep2504"
    print(f"{coin:<8} {ep1831_return[i]:>12.2f}%   {ep2504_return[i]:>12.2f}%   {diff:>12.2f}%   {winner}")

print("\n📉 MAXIMUM DRAWDOWN COMPARISON (Lower is Better):")
print(f"{'Coin':<8} {'Episode 1831':<15} {'Episode 2504':<15} {'Difference':<15} {'Winner'}")
print("-" * 70)
for i, coin in enumerate(coins):
    diff = ep1831_mdd[i] - ep2504_mdd[i]
    winner = "✅ Ep1831" if diff < 0 else "❌ Ep2504"  # Lower MDD wins
    print(f"{coin:<8} {ep1831_mdd[i]:>12.2f}%   {ep2504_mdd[i]:>12.2f}%   {diff:>12.2f}%   {winner}")

print("\n💎 SHARPE RATIO COMPARISON:")
print(f"{'Coin':<8} {'Episode 1831':<15} {'Episode 2504':<15} {'Difference':<15} {'Winner'}")
print("-" * 70)
for i, coin in enumerate(coins):
    diff = ep1831_sharpe[i] - ep2504_sharpe[i]
    winner = "✅ Ep1831" if diff > 0 else "❌ Ep2504"
    print(f"{coin:<8} {ep1831_sharpe[i]:>12.2f}    {ep2504_sharpe[i]:>12.2f}    {diff:>12.2f}    {winner}")

print("\n" + "="*80)
print("🏆 FINAL VERDICT")
print("="*80)
print(f"Episode 1831 Avg Profit: {avg_metrics['Episode 1831']['Avg Profit']:.2f}%")
print(f"Episode 2504 Avg Profit: {avg_metrics['Episode 2504']['Avg Profit']:.2f}%")
improvement = ((avg_metrics['Episode 1831']['Avg Profit'] - avg_metrics['Episode 2504']['Avg Profit']) / 
               avg_metrics['Episode 2504']['Avg Profit'] * 100)
print(f"\n✅ Episode 1831 is {improvement:.1f}% BETTER than Episode 2504!")
print(f"🚨 Episode 2504 shows severe overfitting (84.3x gap vs 22.6x gap)")
print(f"💡 RECOMMENDATION: Use checkpoint_best.pkl (Episode 1831) for deployment")
print("="*80)
