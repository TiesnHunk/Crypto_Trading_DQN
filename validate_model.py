"""
Validation Script - Test model on unseen validation data
Run periodically to detect overfitting
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.dqn_agent import DQNAgent
from models.mdp_trading import TradingMDP

class ValidationTester:
    def __init__(self, checkpoint_path, data_path="data/raw/multi_coin_1h.csv"):
        self.checkpoint_path = Path(checkpoint_path)
        self.data_path = Path(data_path)
        
    def load_validation_data(self):
        """Load validation data (2024-01-01 to 2024-06-30)"""
        print("📥 Loading validation data...")
        df = pd.read_csv(self.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter validation period
        val_df = df[(df['timestamp'] >= '2024-01-01') & 
                    (df['timestamp'] <= '2024-06-30')]
        
        print(f"✅ Loaded {len(val_df):,} rows")
        
        # Split by coin
        coins = {}
        for coin in val_df['coin'].unique():
            coin_df = val_df[val_df['coin'] == coin].copy()
            coin_df = coin_df.reset_index(drop=True)
            coins[coin] = coin_df
            print(f"   {coin}: {len(coin_df):,} samples")
        
        return coins
    
    def load_agent(self):
        """Load trained agent from checkpoint"""
        print(f"\n📦 Loading checkpoint: {self.checkpoint_path.name}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        
        episode = checkpoint.get('episode', 0)
        epsilon = checkpoint.get('epsilon', 0.01)
        
        print(f"   Episode: {episode}")
        print(f"   Epsilon: {epsilon:.6f}")
        
        # Create agent
        hyperparams = checkpoint.get('hyperparameters', {})
        state_dim = hyperparams.get('state_dim', 7)
        action_dim = hyperparams.get('action_dim', 3)
        
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device='cpu'
        )
        
        # Load weights
        agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        agent.epsilon = 0.01  # Use minimal exploration for validation
        
        print("✅ Agent loaded")
        
        return agent, episode
    
    def validate_on_coin(self, agent, coin_name, coin_df):
        """Run validation episode on one coin"""
        print(f"\n🔬 Validating on {coin_name}...")
        
        # Create environment
        mdp = TradingMDP(
            coin_df,
            initial_balance=10000.0,
            transaction_cost=0.0001,
            hold_penalty=0.001,
            interval='1h',
            enable_risk_management=False
        )
        
        # Run episode WITHOUT updating model
        state = mdp.reset()
        done = False
        total_reward = 0
        steps = 0
        
        actions_taken = {'hold': 0, 'buy': 0, 'sell': 0}
        
        while not done and steps < len(coin_df) - 1:
            # Select action (epsilon = 0.01 for minimal exploration)
            action = agent.select_action(state)
            
            # Count actions
            if action == 0:
                actions_taken['buy'] += 1
            elif action == 1:
                actions_taken['sell'] += 1
            else:
                actions_taken['hold'] += 1
            
            # Step environment
            next_state, reward, done, info = mdp.step(action)
            
            total_reward += reward
            state = next_state
            steps += 1
        
        # Get final metrics
        final_portfolio = mdp.get_portfolio_value()
        profit = ((final_portfolio - 10000.0) / 10000.0) * 100
        max_mdd = info.get('max_mdd', 0.0)
        
        # Calculate annualized return manually
        total_return = profit / 100.0
        episode_hours = steps
        episode_years = episode_hours / (24 * 365.25)
        
        if episode_years > 0 and total_return > -1:
            ann_return = (((1 + total_return) ** (1 / episode_years)) - 1) * 100
        else:
            ann_return = 0.0
        
        # Action distribution
        total_actions = sum(actions_taken.values())
        action_pct = {k: (v/total_actions*100) if total_actions > 0 else 0 
                      for k, v in actions_taken.items()}
        
        return {
            'coin': coin_name,
            'reward': total_reward,
            'steps': steps,
            'portfolio': final_portfolio,
            'profit': profit,
            'mdd': max_mdd,
            'ann_return': ann_return,
            'actions': action_pct
        }
    
    def run_validation(self):
        """Run full validation"""
        print("\n" + "="*70)
        print("🔬 VALIDATION TEST".center(70))
        print("="*70)
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load data and agent
        coins_data = self.load_validation_data()
        agent, train_episode = self.load_agent()
        
        # Validate on each coin
        results = []
        
        for coin_name, coin_df in coins_data.items():
            result = self.validate_on_coin(agent, coin_name, coin_df)
            results.append(result)
            
            # Print individual result
            print(f"\n  Results:")
            print(f"    Portfolio: ${result['portfolio']:,.2f}")
            print(f"    Profit:    {result['profit']:+.2f}%")
            print(f"    MDD:       {result['mdd']:.2f}%")
            print(f"    Ann.Ret:   {result['ann_return']:.2f}%")
            print(f"    Actions:   HOLD {result['actions']['hold']:.1f}%, "
                  f"BUY {result['actions']['buy']:.1f}%, "
                  f"SELL {result['actions']['sell']:.1f}%")
        
        # Summary
        print("\n" + "="*70)
        print("📊 VALIDATION SUMMARY")
        print("="*70)
        
        avg_profit = np.mean([r['profit'] for r in results])
        avg_mdd = np.mean([r['mdd'] for r in results])
        avg_ann_ret = np.mean([r['ann_return'] for r in results])
        
        print(f"  Training Episode: {train_episode:,}")
        print(f"  Validation Coins: {len(results)}")
        print(f"  Avg Profit:       {avg_profit:+.2f}%")
        print(f"  Avg MDD:          {avg_mdd:.2f}%")
        print(f"  Avg Ann.Return:   {avg_ann_ret:.2f}%")
        
        # Overfitting check
        print("\n🔍 OVERFITTING CHECK:")
        
        positive_coins = sum(1 for r in results if r['profit'] > 0)
        negative_coins = len(results) - positive_coins
        
        print(f"  Profitable coins: {positive_coins}/{len(results)}")
        print(f"  Negative coins:   {negative_coins}/{len(results)}")
        
        if positive_coins >= len(results) * 0.6:
            print("  ✅ Good generalization - model performs well on unseen data")
        elif positive_coins >= len(results) * 0.4:
            print("  ⚠️  Mixed results - some overfitting possible")
        else:
            print("  🚨 Poor generalization - likely overfitting!")
        
        # Action balance check
        avg_hold = np.mean([r['actions']['hold'] for r in results])
        avg_buy = np.mean([r['actions']['buy'] for r in results])
        avg_sell = np.mean([r['actions']['sell'] for r in results])
        
        print(f"\n  Avg Action Distribution:")
        print(f"    HOLD: {avg_hold:.1f}%")
        print(f"    BUY:  {avg_buy:.1f}%")
        print(f"    SELL: {avg_sell:.1f}%")
        
        if avg_hold > 80:
            print("  ⚠️  Too much HOLD - agent may be too conservative")
        elif avg_hold < 20:
            print("  ⚠️  Too little HOLD - agent may be overtrading")
        else:
            print("  ✅ Balanced action distribution")
        
        print("\n" + "="*70)
        
        return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate DQN model")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='src/checkpoints_dqn/checkpoint_latest.pkl',
        help='Path to checkpoint file'
    )
    
    args = parser.parse_args()
    
    tester = ValidationTester(args.checkpoint)
    results = tester.run_validation()
    
    print("\n✅ Validation complete!")
    print("💾 Results saved in memory")
    print("🔄 Run again later to compare progress\n")
