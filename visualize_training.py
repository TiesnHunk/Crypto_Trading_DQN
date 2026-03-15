"""
Visualize Training Results - Plot charts after training completes
Run this after training to generate charts
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.dqn_agent import DQNAgent
from models.mdp_trading import TradingMDP


def plot_training_progress(checkpoint_path):
    """
    Plot training progress từ checkpoint
    
    Charts:
    1. Profit over episodes
    2. Loss over episodes  
    3. Epsilon decay
    4. MDD over episodes
    5. Action distribution
    """
    print("="*70)
    print("📊 VISUALIZING TRAINING PROGRESS")
    print("="*70)
    
    # Load checkpoint
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"\n📦 Loading: {checkpoint_path.name}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract data
    metadata = checkpoint.get('metadata', {})
    all_profits = metadata.get('all_profits', [])
    all_losses = metadata.get('all_losses', [])
    all_episodes = metadata.get('all_episodes', list(range(len(all_profits))))
    
    if not all_profits:
        print("❌ No training data found in checkpoint")
        return
    
    print(f"✅ Loaded {len(all_profits)} episodes")
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Profit over episodes
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(all_episodes, all_profits, alpha=0.6, linewidth=0.5, color='blue', label='Episode Profit')
    
    # Moving average
    if len(all_profits) > 50:
        window = 50
        ma_profits = pd.Series(all_profits).rolling(window=window).mean()
        ax1.plot(all_episodes, ma_profits, linewidth=2, color='red', label=f'MA{window}')
    
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Profit (%)', fontsize=12)
    ax1.set_title('Training Profit Over Episodes', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Loss over episodes
    ax2 = fig.add_subplot(gs[1, 0])
    if all_losses:
        episodes_with_loss = [ep for ep, loss in zip(all_episodes, all_losses) if loss > 0]
        losses_filtered = [loss for loss in all_losses if loss > 0]
        
        ax2.plot(episodes_with_loss, losses_filtered, alpha=0.6, linewidth=0.5, color='orange', label='TD Loss')
        
        # Moving average
        if len(losses_filtered) > 50:
            ma_losses = pd.Series(losses_filtered).rolling(window=50).mean()
            ax2.plot(episodes_with_loss[:len(ma_losses)], ma_losses, 
                    linewidth=2, color='red', label='MA50')
    
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training Loss Over Episodes', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale for better visibility
    
    # 3. Epsilon decay
    ax3 = fig.add_subplot(gs[1, 1])
    epsilon = checkpoint.get('epsilon', 0.01)
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    
    # Calculate epsilon trajectory
    epsilons = [max(epsilon_end, epsilon_start * (epsilon_decay ** ep)) 
                for ep in range(len(all_profits))]
    
    ax3.plot(all_episodes, epsilons, linewidth=2, color='green')
    ax3.axhline(y=epsilon_end, color='red', linestyle='--', linewidth=1, 
                alpha=0.5, label=f'Min ε = {epsilon_end}')
    ax3.set_xlabel('Episode', fontsize=12)
    ax3.set_ylabel('Epsilon (ε)', fontsize=12)
    ax3.set_title('Exploration Rate Decay', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Profit distribution
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(all_profits, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax4.axvline(x=np.mean(all_profits), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(all_profits):.2f}%')
    ax4.axvline(x=np.median(all_profits), color='green', linestyle='--', 
                linewidth=2, label=f'Median: {np.median(all_profits):.2f}%')
    ax4.set_xlabel('Profit (%)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Profit Distribution', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Statistics
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    stats_text = f"""
    TRAINING STATISTICS
    
    Episodes:        {len(all_profits):,}
    Current Epsilon: {epsilon:.6f}
    
    Profit:
      Mean:          {np.mean(all_profits):+.2f}%
      Median:        {np.median(all_profits):+.2f}%
      Std:           ±{np.std(all_profits):.2f}%
      Best:          {np.max(all_profits):+.2f}%
      Worst:         {np.min(all_profits):+.2f}%
    
    Positive episodes: {sum(1 for p in all_profits if p > 0)} / {len(all_profits)}
    Win rate:          {sum(1 for p in all_profits if p > 0) / len(all_profits) * 100:.1f}%
    """
    
    if all_losses:
        losses_filtered = [l for l in all_losses if l > 0]
        stats_text += f"""
    Loss:
      Mean:          {np.mean(losses_filtered):.2f}
      Median:        {np.median(losses_filtered):.2f}
      Std:           ±{np.std(losses_filtered):.2f}
        """
    
    ax5.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center')
    
    # Main title
    fig.suptitle('DQN Multi-Coin Trading - Training Progress', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save
    output_path = Path('results/charts/training_progress.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {output_path}")
    
    # Show
    plt.show()
    
    print("\n" + "="*70)


def plot_validation_results(validation_results):
    """
    Plot validation results cho từng coin
    
    Args:
        validation_results: List of dict với keys: coin, profit, mdd, ann_return, actions
    """
    print("\n📊 PLOTTING VALIDATION RESULTS...")
    
    if not validation_results:
        print("❌ No validation results to plot")
        return
    
    # Extract data
    coins = [r['coin'] for r in validation_results]
    profits = [r['profit'] for r in validation_results]
    mdds = [r['mdd'] for r in validation_results]
    ann_returns = [r['ann_return'] for r in validation_results]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Profit by coin
    ax1 = axes[0, 0]
    colors = ['green' if p > 0 else 'red' for p in profits]
    bars1 = ax1.bar(coins, profits, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_ylabel('Profit (%)', fontsize=12)
    ax1.set_title('Profit by Coin', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, profit in zip(bars1, profits):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{profit:+.1f}%', ha='center', va='bottom' if height > 0 else 'top')
    
    # 2. MDD by coin
    ax2 = axes[0, 1]
    bars2 = ax2.bar(coins, mdds, color='orange', alpha=0.7, edgecolor='black')
    ax2.set_ylabel('MDD (%)', fontsize=12)
    ax2.set_title('Maximum Drawdown by Coin', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, mdd in zip(bars2, mdds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{mdd:.1f}%', ha='center', va='bottom')
    
    # 3. Annualized Return by coin
    ax3 = axes[1, 0]
    colors3 = ['green' if r > 0 else 'red' for r in ann_returns]
    bars3 = ax3.bar(coins, ann_returns, color=colors3, alpha=0.7, edgecolor='black')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_ylabel('Annualized Return (%)', fontsize=12)
    ax3.set_title('Annualized Return by Coin', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, ann_ret in zip(bars3, ann_returns):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{ann_ret:+.1f}%', ha='center', va='bottom' if height > 0 else 'top')
    
    # 4. Action distribution
    ax4 = axes[1, 1]
    action_types = ['HOLD', 'BUY', 'SELL']
    avg_actions = {
        'HOLD': np.mean([r['actions']['hold'] for r in validation_results]),
        'BUY': np.mean([r['actions']['buy'] for r in validation_results]),
        'SELL': np.mean([r['actions']['sell'] for r in validation_results])
    }
    
    colors4 = ['gray', 'green', 'red']
    bars4 = ax4.bar(action_types, list(avg_actions.values()), 
                    color=colors4, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Percentage (%)', fontsize=12)
    ax4.set_title('Avg Action Distribution', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, pct in zip(bars4, avg_actions.values()):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom')
    
    # Main title
    fig.suptitle('Validation Results - Multi-Coin Performance', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_path = Path('results/charts/validation_results.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize training results')
    parser.add_argument('--checkpoint', type=str, 
                       default='src/checkpoints_dqn/checkpoint_latest.pkl',
                       help='Path to checkpoint')
    parser.add_argument('--type', type=str, choices=['training', 'validation', 'both'],
                       default='training',
                       help='Type of visualization')
    
    args = parser.parse_args()
    
    if args.type in ['training', 'both']:
        plot_training_progress(args.checkpoint)
    
    if args.type in ['validation', 'both']:
        print("\n💡 To plot validation results, run validate_model.py first")
        print("   Then modify this script to accept validation results")
    
    print("\n✅ Visualization complete!")
