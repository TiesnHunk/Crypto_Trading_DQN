"""
CHẨN ĐOÁN 4 VẤN ĐỀ CHÍNH CỦA DQN

1. Training data bias
2. Reward function quá conservative  
3. State space thiếu market regime detection
4. Hyperparameters chưa optimal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import sys
import os

sys.path.append('src')

from models.dqn_agent import DQNAgent
from models.mdp_trading import TradingMDP

print("="*80)
print("CHẨN ĐOÁN DQN - 4 VẤN ĐỀ CHÍNH")
print("="*80)

# ============================================================================
# 1. KIỂM TRA TRAINING DATA BIAS
# ============================================================================

def check_training_data_bias():
    """Phân tích phân bố market regime trong training data"""
    print("\n" + "="*80)
    print("1. KIỂM TRA TRAINING DATA BIAS")
    print("="*80)
    
    # Load full data
    df = pd.read_csv('data/raw/multi_coin_1h.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_btc = df[df['coin'] == 'BTC'].copy()
    df_btc = df_btc.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\nTotal BTC data: {len(df_btc)} rows")
    print(f"Date range: {df_btc['timestamp'].min()} to {df_btc['timestamp'].max()}")
    
    # Classify each hour into market regime
    def classify_regime(row):
        trend = row['trend']
        rsi = row['rsi']
        
        if trend > 0.5 and rsi > 55:
            return 'BULL'
        elif trend < -0.5 and rsi < 45:
            return 'BEAR'
        elif abs(trend) < 0.3 and 45 <= rsi <= 55:
            return 'SIDEWAYS'
        else:
            return 'MIXED'
    
    df_btc['regime'] = df_btc.apply(classify_regime, axis=1)
    
    # Count regimes
    regime_counts = df_btc['regime'].value_counts()
    regime_pcts = regime_counts / len(df_btc) * 100
    
    print("\n📊 Market Regime Distribution in Full Dataset:")
    print(f"{'Regime':<12} {'Count':<10} {'Percentage':<12}")
    print("-" * 40)
    for regime in ['BULL', 'BEAR', 'SIDEWAYS', 'MIXED']:
        count = regime_counts.get(regime, 0)
        pct = regime_pcts.get(regime, 0)
        print(f"{regime:<12} {count:<10} {pct:>6.2f}%")
    
    # Analyze training period (assuming 80% train, 20% test)
    train_size = int(len(df_btc) * 0.8)
    df_train = df_btc.iloc[:train_size]
    df_test = df_btc.iloc[train_size:]
    
    train_regime_counts = df_train['regime'].value_counts()
    train_regime_pcts = train_regime_counts / len(df_train) * 100
    
    print("\n📊 Training Data (80%) Regime Distribution:")
    print(f"{'Regime':<12} {'Count':<10} {'Percentage':<12}")
    print("-" * 40)
    for regime in ['BULL', 'BEAR', 'SIDEWAYS', 'MIXED']:
        count = train_regime_counts.get(regime, 0)
        pct = train_regime_pcts.get(regime, 0)
        print(f"{regime:<12} {count:<10} {pct:>6.2f}%")
    
    # Analyze by year
    df_btc['year'] = df_btc['timestamp'].dt.year
    yearly_regimes = df_btc.groupby(['year', 'regime']).size().unstack(fill_value=0)
    yearly_regimes_pct = yearly_regimes.div(yearly_regimes.sum(axis=1), axis=0) * 100
    
    print("\n📊 Yearly Regime Distribution (%):")
    print(yearly_regimes_pct.round(2))
    
    # Diagnosis
    print("\n🔍 CHẨN ĐOÁN:")
    
    bull_pct = regime_pcts.get('BULL', 0)
    bear_pct = regime_pcts.get('BEAR', 0)
    
    if abs(bull_pct - bear_pct) > 10:
        print(f"⚠️ BIAS DETECTED: {abs(bull_pct - bear_pct):.1f}% difference between Bull and Bear")
        if bull_pct > bear_pct:
            print(f"   → More BULL data ({bull_pct:.1f}%) than BEAR ({bear_pct:.1f}%)")
            print(f"   → But DQN behaves like trained on BEAR market!")
            print(f"   → CONTRADICTORY: Likely reward function issue, not data bias")
        else:
            print(f"   → More BEAR data ({bear_pct:.1f}%) than BULL ({bull_pct:.1f}%)")
            print(f"   → DQN behavior aligns with training bias")
    else:
        print(f"✓ Balanced: Bull {bull_pct:.1f}% vs Bear {bear_pct:.1f}%")
    
    # Plot distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Overall distribution
    regime_counts.plot(kind='bar', ax=axes[0], color=['green', 'red', 'blue', 'gray'])
    axes[0].set_title('Overall Market Regime Distribution', fontweight='bold')
    axes[0].set_xlabel('Market Regime')
    axes[0].set_ylabel('Count (hours)')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # Yearly distribution
    yearly_regimes_pct.plot(kind='bar', stacked=True, ax=axes[1], 
                            color={'BULL': 'green', 'BEAR': 'red', 
                                   'SIDEWAYS': 'blue', 'MIXED': 'gray'})
    axes[1].set_title('Yearly Market Regime Distribution (%)', fontweight='bold')
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend(title='Regime')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    os.makedirs('results/charts', exist_ok=True)
    plt.savefig('results/charts/training_data_bias_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved chart to: results/charts/training_data_bias_analysis.png")
    plt.close()
    
    return regime_pcts

# ============================================================================
# 2. KIỂM TRA REWARD FUNCTION
# ============================================================================

def check_reward_function():
    """Phân tích reward function parameters"""
    print("\n" + "="*80)
    print("2. KIỂM TRA REWARD FUNCTION")
    print("="*80)
    
    # Load sample data
    df = pd.read_csv('data/raw/multi_coin_1h.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_btc = df[df['coin'] == 'BTC'].copy()
    df_btc = df_btc.sort_values('timestamp').reset_index(drop=True)
    
    # Sample 100 hours
    sample_df = df_btc.sample(100, random_state=42).reset_index(drop=True)
    
    # Create environment
    env = TradingMDP(sample_df, initial_balance=10000.0)
    
    print("\n📋 Current Reward Function Parameters:")
    print(f"  Transaction cost: {env.transaction_cost} ({env.transaction_cost*100:.2f}%)")
    print(f"  Hold penalty: {env.hold_penalty} ({env.hold_penalty*100:.2f}%)")
    print(f"  Stop loss enabled: {env.enable_risk_management}")
    if env.enable_risk_management:
        print(f"  Stop loss %: {env.stop_loss_pct*100:.1f}%")
        print(f"  Max position %: {env.max_position_pct*100:.1f}%")
    
    # Simulate different actions
    print("\n🧪 Reward Simulation (on sample state):")
    
    state = env.reset()
    
    # Buy action
    state_buy = env.reset()
    _, reward_buy, _, _ = env.step(1)  # Buy
    
    # Sell action  
    state_sell = env.reset()
    env.step(1)  # Buy first
    _, reward_sell, _, _ = env.step(2)  # Then sell
    
    # Hold action
    state_hold = env.reset()
    _, reward_hold, _, _ = env.step(0)  # Hold
    
    print(f"  Buy reward: {reward_buy:.6f}")
    print(f"  Sell reward: {reward_sell:.6f}")
    print(f"  Hold reward: {reward_hold:.6f}")
    
    # Check if hold penalty discourages trading
    print("\n🔍 CHẨN ĐOÁN:")
    
    if abs(reward_hold) > abs(reward_buy) and abs(reward_hold) > abs(reward_sell):
        print("⚠️ CONSERVATIVE DETECTED:")
        print("   → Hold penalty makes trading less attractive")
        print("   → DQN learns to avoid trading (over-conservative)")
    
    if env.transaction_cost > 0.001:  # 0.1%
        print("⚠️ HIGH TRANSACTION COST:")
        print(f"   → {env.transaction_cost*100:.2f}% cost discourages frequent trading")
    
    if env.enable_risk_management and env.stop_loss_pct < 0.15:
        print("⚠️ TIGHT STOP LOSS:")
        print(f"   → {env.stop_loss_pct*100:.1f}% stop loss triggers too early")
        print("   → Forces sell even in temporary dips")
    
    # Recommend optimal parameters
    print("\n💡 RECOMMENDED PARAMETERS:")
    print("  Transaction cost: 0.01% (current: {:.2f}%)".format(env.transaction_cost*100))
    print("  Hold penalty: 0.001% (current: {:.2f}%)".format(env.hold_penalty*100))
    print("  Stop loss: 15-20% (current: {:.1f}%)".format(env.stop_loss_pct*100 if env.enable_risk_management else 0))

# ============================================================================
# 3. KIỂM TRA STATE SPACE
# ============================================================================

def check_state_space():
    """Phân tích state space và missing features"""
    print("\n" + "="*80)
    print("3. KIỂM TRA STATE SPACE")
    print("="*80)
    
    # Load sample data
    df = pd.read_csv('data/raw/multi_coin_1h.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_btc = df[df['coin'] == 'BTC'].copy()
    df_btc = df_btc.sort_values('timestamp').reset_index(drop=True)
    
    sample_df = df_btc.sample(100, random_state=42).reset_index(drop=True)
    
    # Create environment
    env = TradingMDP(sample_df, initial_balance=10000.0)
    state = env.reset()
    
    print("\n📋 Current State Space (8 dimensions):")
    state_names = [
        'position',
        'rsi', 
        'macd_hist',
        'trend',
        'bb_position',
        'volatility',
        'price_change',
        'profit_pct'
    ]
    
    for i, name in enumerate(state_names):
        print(f"  [{i}] {name:<15}: {state[i]:.6f}")
    
    print(f"\n  Total dimensions: {len(state)}")
    
    # Check missing features
    print("\n🔍 MISSING FEATURES FOR MARKET REGIME DETECTION:")
    
    missing_features = [
        "market_regime (explicit bull/bear/sideways label)",
        "trend_strength (magnitude of trend, not just direction)",
        "volume_trend (increasing/decreasing volume)",
        "price_momentum (rate of change acceleration)",
        "market_phase (accumulation/distribution/markup/markdown)",
        "multi_timeframe_trend (trend on 4h, 1d timeframes)",
        "support_resistance_distance",
        "market_sentiment_score"
    ]
    
    print("\n⚠️ CRITICAL MISSING:")
    for i, feature in enumerate(missing_features[:4], 1):
        print(f"   {i}. {feature}")
    
    print("\n💡 RECOMMENDED STATE SPACE ENHANCEMENTS:")
    print("   Option 1: Add market_regime as explicit feature")
    print("   Option 2: Add trend_strength = abs(trend) × RSI_momentum")
    print("   Option 3: Add multi_timeframe_trend (4h, 1d)")
    print("   Option 4: Add all 8 features → 16-dim state space")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    current_features = state_names
    enhanced_features = state_names + [
        'market_regime',
        'trend_strength', 
        'volume_trend',
        'price_momentum'
    ]
    
    y_pos = np.arange(len(enhanced_features))
    colors = ['green' if i < len(current_features) else 'orange' 
              for i in range(len(enhanced_features))]
    labels = ['Current' if i < len(current_features) else 'Missing' 
              for i in range(len(enhanced_features))]
    
    for i, (feature, color, label) in enumerate(zip(enhanced_features, colors, labels)):
        ax.barh(i, 1, color=color, alpha=0.7, label=label if i == 0 or i == len(current_features) else "")
        ax.text(0.5, i, feature, ha='center', va='center', fontweight='bold', color='white')
    
    ax.set_yticks([])
    ax.set_xlim([0, 1])
    ax.set_xlabel('State Space Features')
    ax.set_title('Current vs Enhanced State Space', fontweight='bold', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('results/charts/state_space_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved chart to: results/charts/state_space_analysis.png")
    plt.close()

# ============================================================================
# 4. KIỂM TRA HYPERPARAMETERS
# ============================================================================

def check_hyperparameters():
    """Phân tích hyperparameters của DQN"""
    print("\n" + "="*80)
    print("4. KIỂM TRA HYPERPARAMETERS")
    print("="*80)
    
    # Load checkpoint to check actual hyperparameters used
    checkpoint_path = 'src/checkpoints_dqn/checkpoint_best.pkl'
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Checkpoint not found: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("\n📋 Current Hyperparameters (from checkpoint):")
    
    # Extract hyperparameters from checkpoint
    hyperparams = checkpoint.get('hyperparameters', {})
    
    if hyperparams:
        for key, value in hyperparams.items():
            print(f"  {key:<25}: {value}")
    else:
        print("  (Hyperparameters not saved in checkpoint)")
        print("\n  Using default values from DQNAgent:")
        print(f"  {'learning_rate':<25}: 0.001")
        print(f"  {'gamma':<25}: 0.99")
        print(f"  {'epsilon_start':<25}: 1.0")
        print(f"  {'epsilon_end':<25}: 0.01")
        print(f"  {'epsilon_decay':<25}: 0.995")
        print(f"  {'buffer_capacity':<25}: 100000")
        print(f"  {'batch_size':<25}: 64")
        print(f"  {'target_update_freq':<25}: 1000")
    
    print(f"\n  Episode trained: {checkpoint.get('episode', 'N/A')}")
    print(f"  Best profit: ${checkpoint.get('best_profit', 0):,.2f}")
    
    # Analysis
    print("\n🔍 CHẨN ĐOÁN:")
    
    # Learning rate
    lr = hyperparams.get('learning_rate', 0.001)
    if lr > 0.01:
        print(f"⚠️ HIGH LEARNING RATE: {lr}")
        print("   → May cause instability, overshooting optimal policy")
    elif lr < 0.0001:
        print(f"⚠️ LOW LEARNING RATE: {lr}")
        print("   → Slow convergence, may not reach optimal policy")
    else:
        print(f"✓ Learning rate OK: {lr}")
    
    # Gamma (discount factor)
    gamma = hyperparams.get('gamma', 0.99)
    if gamma > 0.99:
        print(f"⚠️ VERY HIGH GAMMA: {gamma}")
        print("   → Overly focused on long-term rewards")
        print("   → May ignore short-term opportunities")
    elif gamma < 0.9:
        print(f"⚠️ LOW GAMMA: {gamma}")
        print("   → Too myopic, ignores future rewards")
    else:
        print(f"✓ Gamma OK: {gamma}")
    
    # Epsilon decay
    epsilon_decay = hyperparams.get('epsilon_decay', 0.995)
    if epsilon_decay > 0.999:
        print(f"⚠️ SLOW EPSILON DECAY: {epsilon_decay}")
        print("   → Too much exploration, slow convergence")
    elif epsilon_decay < 0.99:
        print(f"⚠️ FAST EPSILON DECAY: {epsilon_decay}")
        print("   → Too little exploration, may miss optimal actions")
    else:
        print(f"✓ Epsilon decay OK: {epsilon_decay}")
    
    # Buffer size
    buffer_size = hyperparams.get('buffer_capacity', 100000)
    if buffer_size < 10000:
        print(f"⚠️ SMALL BUFFER: {buffer_size}")
        print("   → Limited experience diversity")
    else:
        print(f"✓ Buffer size OK: {buffer_size}")
    
    # Recommendations
    print("\n💡 RECOMMENDED HYPERPARAMETERS:")
    print("\n  Standard DQN:")
    print("    learning_rate: 0.001 (current)")
    print("    gamma: 0.99 (current)")
    print("    epsilon_decay: 0.995 (current)")
    
    print("\n  For faster convergence:")
    print("    learning_rate: 0.003")
    print("    epsilon_decay: 0.99")
    
    print("\n  For better exploration:")
    print("    epsilon_start: 1.0")
    print("    epsilon_end: 0.05 (higher than 0.01)")
    print("    epsilon_decay: 0.997 (slower)")
    
    print("\n  For market regime adaptation:")
    print("    gamma: 0.95 (less long-term focused)")
    print("    target_update_freq: 500 (faster adaptation)")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    # 1. Check training data bias
    regime_pcts = check_training_data_bias()
    
    # 2. Check reward function
    check_reward_function()
    
    # 3. Check state space
    check_state_space()
    
    # 4. Check hyperparameters
    check_hyperparameters()
    
    # Summary
    print("\n" + "="*80)
    print("TỔNG KẾT CHẨN ĐOÁN")
    print("="*80)
    
    print("\n1️⃣ TRAINING DATA BIAS:")
    bull_pct = regime_pcts.get('BULL', 0)
    bear_pct = regime_pcts.get('BEAR', 0)
    if abs(bull_pct - bear_pct) > 10:
        if bull_pct > bear_pct:
            print(f"   ⚠️ More BULL ({bull_pct:.1f}%) but DQN acts like BEAR-trained")
            print(f"   → Not data bias, likely REWARD FUNCTION issue")
        else:
            print(f"   ⚠️ More BEAR data ({bear_pct:.1f}%)")
            print(f"   → Training bias exists")
    else:
        print(f"   ✓ Balanced data")
    
    print("\n2️⃣ REWARD FUNCTION:")
    print("   ⚠️ Likely TOO CONSERVATIVE")
    print("   → Transaction cost + hold penalty discourage trading")
    print("   → DQN learns to avoid Buy in bull market")
    
    print("\n3️⃣ STATE SPACE:")
    print("   ⚠️ MISSING market regime detection features")
    print("   → Cannot distinguish bull/bear/sideways")
    print("   → Treats all markets the same")
    
    print("\n4️⃣ HYPERPARAMETERS:")
    print("   ~ Generally OK for standard DQN")
    print("   → But may need tuning for market regime adaptation")
    
    print("\n" + "="*80)
    print("KHUYẾN NGHỊ CHÍNH")
    print("="*80)
    
    print("\n🎯 Priority 1: ENHANCE STATE SPACE")
    print("   → Add market_regime feature")
    print("   → Add trend_strength, volume_trend")
    
    print("\n🎯 Priority 2: ADJUST REWARD FUNCTION")
    print("   → Reduce hold penalty (0.01% → 0.001%)")
    print("   → Add regime-adaptive rewards")
    
    print("\n🎯 Priority 3: TUNE HYPERPARAMETERS")
    print("   → Gamma: 0.99 → 0.95 (less long-term bias)")
    print("   → Epsilon end: 0.01 → 0.05 (more exploration)")
    
    print("\n🎯 Priority 4: RETRAIN WITH BALANCED DATA")
    print("   → Ensure equal bull/bear/sideways samples")
    print("   → Or use regime-specific models")
    
    print("\n✓ Analysis complete!")
    print("  Charts saved to: results/charts/")
