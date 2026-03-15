"""
Đánh giá mô hình DQN trên nhiều loại coin và tạo biểu đồ so sánh
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

# Import các model cần thiết
import sys
sys.path.append('src')
from models.dqn_agent import DQNAgent
import config.config as config

def load_checkpoint(checkpoint_dir):
    """Load checkpoint mới nhất"""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        return None
    
    # Tìm checkpoint best trước
    best_checkpoint = checkpoint_path / 'checkpoint_best.pkl'
    if best_checkpoint.exists():
        return best_checkpoint
    
    # Tìm checkpoint mới nhất
    checkpoints = list(checkpoint_path.glob('*.pkl'))
    if not checkpoints:
        return None
    
    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return latest_checkpoint

def prepare_state(data, index, position, entry_price, current_price, trade_entry_step, current_step, max_hold_steps=168, state_dim=8):
    """Chuẩn bị state từ dữ liệu - KHỚP VỚI MDP.get_state()
    
    State format theo MDP (8 dimensions):
    [position, rsi_norm, macd_hist_norm, trend_norm, bb_position, volatility_norm, profit_pct_norm, time_since_trade_norm]
    """
    if index < 1:
        return None
    
    row = data.iloc[index]
    
    # 1. Position (0=cash, 1=holding)
    position_val = float(position)
    
    # 2. RSI normalized [0, 1]
    rsi = row.get('rsi', 50.0)
    rsi_norm = np.clip(rsi / 100.0, 0, 1)
    
    # 3. MACD histogram normalized [-1, 1]
    macd_hist = row.get('macd_hist', 0.0)
    if pd.isna(macd_hist):
        macd_hist = row.get('macd', 0.0)
    macd_norm = np.tanh(macd_hist / 100.0)  # Tanh normalization
    
    # 4. Trend normalized [-1, 1]
    trend = row.get('trend', 0.0)
    if pd.isna(trend):
        # Calculate from price change
        if index >= 5:
            price_5_ago = data.iloc[index-5]['close']
            trend = 1 if current_price > price_5_ago else (-1 if current_price < price_5_ago else 0)
        else:
            trend = 0
    trend_norm = np.clip(trend, -1, 1)
    
    # 5. BB position [0, 1] - vị trí giá so với Bollinger Bands
    bb_upper = row.get('bb_upper', current_price * 1.02)
    bb_lower = row.get('bb_lower', current_price * 0.98)
    if bb_upper > bb_lower:
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
    else:
        bb_position = 0.5
    bb_position = np.clip(bb_position, 0, 1)
    
    # 6. Volatility normalized [0, 1]
    volatility = row.get('volatility', 0.01)
    volatility_norm = np.clip(volatility * 10, 0, 1)  # Scale up for visibility
    
    # 7. Current profit/loss if holding, else neutral
    if position == 1 and entry_price > 0:
        profit_pct = (current_price - entry_price) / entry_price
        profit_norm = np.clip((profit_pct + 0.5) / 1.0, 0, 1)  # -50% to +50% -> [0, 1]
    else:
        profit_norm = 0.5  # Neutral
    
    # 8. Time since trade entry (normalized)
    if position == 1 and trade_entry_step > 0:
        steps_since_trade = current_step - trade_entry_step
        time_norm = min(steps_since_trade / max_hold_steps, 1.0)
    else:
        time_norm = 0.0  # No trade active
    
    state = np.array([
        position_val,
        rsi_norm,
        macd_norm,
        trend_norm,
        bb_position,
        volatility_norm,
        profit_norm,
        time_norm
    ], dtype=np.float32)
    
    # Check for NaN
    if np.any(np.isnan(state)):
        state = np.nan_to_num(state, nan=0.5, posinf=1.0, neginf=0.0)
    
    return state

def evaluate_coin(coin, data, agent, state_dim, start_date='2024-10-01', end_date='2024-12-31'):
    """Đánh giá agent trên 1 coin trong giai đoạn cụ thể"""
    print(f"\n{'='*60}")
    print(f"   Đánh giá coin: {coin}")
    print(f"{'='*60}")
    
    # Lọc dữ liệu theo giai đoạn
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    test_data = data[(data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)].reset_index(drop=True)
    
    if len(test_data) < 100:
        print(f"   ⚠️ Không đủ dữ liệu test cho {coin} trong giai đoạn {start_date} - {end_date}")
        print(f"   Có {len(test_data)} rows")
        return None
    
    # Khởi tạo portfolio
    balance = 10000.0
    holdings = 0.0
    initial_balance = balance
    position = 0  # 0=cash, 1=holding
    entry_price = 0.0  # Track buy price
    trade_entry_step = 0  # Step khi vào lệnh
    max_hold_steps = 168  # 1 week for 1h data
    
    trades = []
    portfolio_values = []
    actions_history = []
    
    # Test qua từng timestep
    for i in range(10, len(test_data)):
        current_price = test_data.loc[i, 'close']
        
        # Chuẩn bị state - KHỚP VỚI MDP
        state = prepare_state(test_data, i, position, entry_price, current_price, 
                            trade_entry_step, i, max_hold_steps, state_dim=state_dim)
        if state is None:
            continue
        
        # Agent chọn action
        action = agent.select_action(state, epsilon=0)  # Greedy
        actions_history.append(action)
        
        # Execute action
        # Action mapping theo MDP: 0=Buy, 1=Sell, 2=Hold
        if action == 0:  # Buy
            if balance > 0 and position == 0:  # Only buy if in cash position
                amount = balance * 0.95  # 95% of balance
                can_buy = amount / current_price
                holdings += can_buy
                balance -= amount
                position = 1  # Now holding
                entry_price = current_price  # Track entry price
                trade_entry_step = i  # Track entry step
                trades.append({
                    'time': test_data.loc[i, 'timestamp'],
                    'action': 'BUY',
                    'price': current_price,
                    'amount': can_buy,
                    'value': amount
                })
        
        elif action == 1:  # Sell
            if holdings > 0 and position == 1:  # Only sell if holding
                amount = holdings * 0.95  # 95% of holdings
                balance += amount * current_price
                holdings -= amount
                position = 0  # Back to cash
                entry_price = 0.0  # Reset entry price
                trade_entry_step = 0  # Reset entry step
                trades.append({
                    'time': test_data.loc[i, 'timestamp'],
                    'action': 'SELL',
                    'price': current_price,
                    'amount': amount,
                    'value': amount * current_price
                })
        # action == 2 is Hold - do nothing
        
        # Calculate portfolio value
        portfolio_value = balance + holdings * current_price
        portfolio_values.append(portfolio_value)
    
    # Tính toán metrics
    final_value = balance + holdings * test_data.iloc[-1]['close']
    total_return = ((final_value - initial_balance) / initial_balance) * 100
    
    # Price change
    initial_price = test_data.iloc[10]['close']
    final_price = test_data.iloc[-1]['close']
    market_return = ((final_price - initial_price) / initial_price) * 100
    
    # Calculate Maximum Drawdown (MDD)
    peak = initial_balance
    max_drawdown = 0
    for pv in portfolio_values:
        if pv > peak:
            peak = pv
        drawdown = ((peak - pv) / peak) * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    # Calculate Sharpe Ratio
    if len(portfolio_values) > 1:
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = (np.mean(returns) * np.sqrt(252 * 24)) / np.std(returns)  # Annualized for hourly data
        else:
            sharpe_ratio = 0
    else:
        sharpe_ratio = 0
    
    # Win rate
    winning_trades = 0
    losing_trades = 0
    for i, trade in enumerate(trades):
        if trade['action'] == 'SELL' and i > 0:
            # Find corresponding buy
            for j in range(i-1, -1, -1):
                if trades[j]['action'] == 'BUY':
                    if trade['price'] > trades[j]['price']:
                        winning_trades += 1
                    else:
                        losing_trades += 1
                    break
    
    total_trade_pairs = winning_trades + losing_trades
    win_rate = (winning_trades / total_trade_pairs * 100) if total_trade_pairs > 0 else 0
    
    # Action distribution (Action mapping: 0=Buy, 1=Sell, 2=Hold)
    action_counts = pd.Series(actions_history).value_counts()
    
    results = {
        'coin': coin,
        'initial_balance': initial_balance,
        'final_value': final_value,
        'total_return': total_return,
        'market_return': market_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'num_trades': len(trades),
        'num_buy': len([t for t in trades if t['action'] == 'BUY']),
        'num_sell': len([t for t in trades if t['action'] == 'SELL']),
        'hold_pct': (action_counts.get(2, 0) / len(actions_history)) * 100 if actions_history else 0,  # Fixed: action 2 is Hold
        'initial_price': initial_price,
        'final_price': final_price,
        'portfolio_values': portfolio_values,
        'actions': actions_history
    }
    
    print(f"\n   📊 Kết quả:")
    print(f"      • Return: {total_return:.2f}%")
    print(f"      • Market Return: {market_return:.2f}%")
    print(f"      • Max Drawdown: {max_drawdown:.2f}%")
    print(f"      • Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"      • Win Rate: {win_rate:.1f}%")
    print(f"      • Trades: {len(trades)}")
    print(f"      • Final Value: ${final_value:.2f}")
    
    return results

def create_comparison_chart(all_results, output_path, test_period='Oct-Dec 2024'):
    """Tạo biểu đồ so sánh các coin"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Đánh Giá Mô Hình DQN Trên Các Loại Coin ({test_period})', 
                 fontsize=20, fontweight='bold', y=0.995)
    
    coins = [r['coin'] for r in all_results]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    
    # 1. Total Returns
    ax1 = axes[0, 0]
    returns = [r['total_return'] for r in all_results]
    market_returns = [r['market_return'] for r in all_results]
    
    x = np.arange(len(coins))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, returns, width, label='DQN Returns', color=colors, alpha=0.8)
    bars2 = ax1.bar(x + width/2, market_returns, width, label='Market Returns', 
                    color='gray', alpha=0.5)
    
    ax1.set_xlabel('Coin', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Return (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Tổng Lợi Nhuận: DQN vs Market', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(coins)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9, fontweight='bold')
    
    # 2. Final Portfolio Value
    ax2 = axes[0, 1]
    final_values = [r['final_value'] for r in all_results]
    bars = ax2.bar(coins, final_values, color=colors, alpha=0.8)
    ax2.set_xlabel('Coin', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Final Value ($)', fontsize=12, fontweight='bold')
    ax2.set_title('Giá Trị Portfolio Cuối Cùng', fontsize=14, fontweight='bold')
    ax2.axhline(y=10000, color='red', linestyle='--', linewidth=2, label='Initial ($10,000)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.0f}',
                ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    
    # 3. Trading Activity
    ax3 = axes[0, 2]
    num_trades = [r['num_trades'] for r in all_results]
    num_buy = [r['num_buy'] for r in all_results]
    num_sell = [r['num_sell'] for r in all_results]
    
    x = np.arange(len(coins))
    width = 0.25
    
    ax3.bar(x - width, num_buy, width, label='Buy', color='#2ecc71', alpha=0.8)
    ax3.bar(x, num_sell, width, label='Sell', color='#e74c3c', alpha=0.8)
    ax3.bar(x + width, num_trades, width, label='Total', color='#3498db', alpha=0.8)
    
    ax3.set_xlabel('Coin', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Số Lượng Giao Dịch', fontsize=12, fontweight='bold')
    ax3.set_title('Hoạt Động Giao Dịch', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(coins)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Maximum Drawdown
    ax4 = axes[1, 0]
    mdd_values = [r['max_drawdown'] for r in all_results]
    bars = ax4.bar(coins, mdd_values, color=colors, alpha=0.8)
    ax4.set_xlabel('Coin', fontsize=12, fontweight='bold')
    ax4.set_ylabel('MDD (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Maximum Drawdown (MDD)', fontsize=14, fontweight='bold')
    ax4.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Target (<50%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    
    # 5. Sharpe Ratio
    ax5 = axes[1, 1]
    sharpe_values = [r['sharpe_ratio'] for r in all_results]
    bars = ax5.bar(coins, sharpe_values, color=colors, alpha=0.8)
    ax5.set_xlabel('Coin', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax5.set_title('Sharpe Ratio (Risk-Adjusted Return)', fontsize=14, fontweight='bold')
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax5.axhline(y=1, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Good (>1.0)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=9, fontweight='bold')
    
    # 6. Win Rate
    ax6 = axes[1, 2]
    win_rates = [r['win_rate'] for r in all_results]
    bars = ax6.bar(coins, win_rates, color=colors, alpha=0.8)
    ax6.set_xlabel('Coin', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
    ax6.set_title('Tỷ Lệ Thắng', fontsize=14, fontweight='bold')
    ax6.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Break-even (50%)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, 100)
    
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Đã lưu biểu đồ: {output_path}")
    
    return fig

def main():
    print("\n" + "="*70)
    print("   ĐÁNH GIÁ MÔ HÌNH DQN TRÊN NHIỀU LOẠI COIN")
    print("="*70)
    
    # Load data
    print("\n📊 Loading multi-coin data...")
    data = pd.read_csv('data/raw/multi_coin_1h.csv')
    
    # Load DQN model
    print("\n🤖 Loading DQN model...")
    
    # Tìm checkpoint
    checkpoint_dirs = [
        'checkpoints/dqn_pso_lstm_gpu',
        'checkpoints/dqn_pso_lstm',
        'src/checkpoints_dqn'
    ]
    
    checkpoint_file = None
    for dir_path in checkpoint_dirs:
        ckpt = load_checkpoint(dir_path)
        if ckpt:
            checkpoint_file = ckpt
            print(f"   ✅ Found checkpoint: {checkpoint_file}")
            break
    
    if not checkpoint_file:
        print("   ❌ Không tìm thấy checkpoint!")
        return
    
    # Load checkpoint để lấy metadata
    import torch
    checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
    
    # Get state_dim từ checkpoint
    if 'hyperparameters' in checkpoint:
        state_dim = checkpoint['hyperparameters'].get('state_dim', 8)  # MDP default: 8
    else:
        # Infer from model weights
        state_dim = checkpoint['policy_net_state_dict']['fc1.weight'].shape[1]
    
    print(f"   State dimension: {state_dim}")
    
    # Initialize agent với đúng state_dim
    action_dim = 3  # Hold, Buy, Sell
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim
    )
    
    # Load weights
    try:
        agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        print(f"   ✅ Model loaded successfully!")
    except Exception as e:
        print(f"   ❌ Error loading checkpoint: {e}")
        return
    
    # Evaluate each coin
    # Chỉ đánh giá các coins có đủ dữ liệu
    target_coins = ['ADA', 'BNB', 'BTC', 'ETH', 'SOL']
    available_coins = [c for c in target_coins if c in data['coin'].unique()]
    
    print(f"\n📈 Evaluating {len(available_coins)} coins: {', '.join(available_coins)}")
    print(f"   Test Period: Oct 1, 2024 - Dec 31, 2024 (3 months)")
    
    all_results = []
    for coin in available_coins:
        coin_data = data[data['coin'] == coin].reset_index(drop=True)
        result = evaluate_coin(coin, coin_data, agent, state_dim, 
                              start_date='2024-10-01', end_date='2024-12-31')
        if result:
            all_results.append(result)
    
    # Create comparison chart
    if all_results:
        print(f"\n📊 Tạo biểu đồ so sánh...")
        output_path = 'results/charts/dqn_multi_coin_evaluation_oct_dec_2024.png'
        Path('results/charts').mkdir(parents=True, exist_ok=True)
        create_comparison_chart(all_results, output_path, test_period='Oct-Dec 2024')
        
        # Save detailed results
        results_df = pd.DataFrame([{
            'Coin': r['coin'],
            'Return (%)': f"{r['total_return']:.2f}",
            'MDD (%)': f"{r['max_drawdown']:.2f}",
            'Sharpe': f"{r['sharpe_ratio']:.2f}",
            'Win Rate (%)': f"{r['win_rate']:.1f}",
            'Trades': r['num_trades'],
            'Final Value ($)': f"{r['final_value']:.2f}",
            'Market Return (%)': f"{r['market_return']:.2f}",
            'Alpha (%)': f"{r['total_return'] - r['market_return']:.2f}"
        } for r in all_results])
        
        csv_path = 'results/reports/dqn_multi_coin_evaluation_oct_dec_2024.csv'
        results_df.to_csv(csv_path, index=False)
        print(f"   ✅ Đã lưu báo cáo: {csv_path}")
        
        # Print summary
        print(f"\n" + "="*70)
        print("   TỔNG KẾT KẾT QUẢ")
        print("="*70)
        print(results_df.to_string(index=False))
        
        # Average performance
        avg_return = np.mean([r['total_return'] for r in all_results])
        avg_market = np.mean([r['market_return'] for r in all_results])
        avg_alpha = avg_return - avg_market
        avg_mdd = np.mean([r['max_drawdown'] for r in all_results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in all_results])
        avg_win_rate = np.mean([r['win_rate'] for r in all_results])
        
        print(f"\n📊 TRUNG BÌNH (Oct-Dec 2024):")
        print(f"   • DQN Return: {avg_return:.2f}%")
        print(f"   • Market Return: {avg_market:.2f}%")
        print(f"   • Alpha: {avg_alpha:.2f}%")
        print(f"   • Average MDD: {avg_mdd:.2f}%")
        print(f"   • Average Sharpe: {avg_sharpe:.2f}")
        print(f"   • Average Win Rate: {avg_win_rate:.1f}%")
        
        print(f"\n✅ Hoàn thành!")
        print(f"   📊 Biểu đồ: {output_path}")
        print(f"   📋 Báo cáo: {csv_path}")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    main()
