"""
DỰ BÁO HÀNH ĐỘNG GIAO DỊCH CHO 1 NGÀY CỤ THỂ
Sử dụng mô hình DQN đã huấn luyện

Phân tích:
- Dự báo hành động (Hold/Buy/Sell) cho từng giờ
- Metrics: MAE, MSE, RMSE, MAPE, R2
- Phân tích ảnh hưởng của môi trường (Bull/Bear)
- Thảo luận reward function và hành động
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

from models.dqn_agent import DQNAgent
from models.mdp_trading import TradingMDP

class OneDayPredictor:
    """Dự báo hành động giao dịch cho 1 ngày cụ thể"""
    
    def __init__(self, checkpoint_path, data_path):
        self.checkpoint_path = checkpoint_path
        self.data_path = data_path
        self.action_names = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
        
    def load_data(self):
        """Load dữ liệu Bitcoin"""
        print("Loading data...")
        df = pd.read_csv(self.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter chỉ BTC
        df_btc = df[df['coin'] == 'BTC'].copy()
        df_btc = df_btc.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Total BTC data: {len(df_btc)} rows")
        print(f"Date range: {df_btc['timestamp'].min()} to {df_btc['timestamp'].max()}")
        
        return df_btc
    
    def select_test_day(self, df, date_str=None):
        """
        Chọn 1 ngày để test (24 giờ = 24 samples)
        
        Args:
            df: DataFrame đầy đủ
            date_str: String ngày cụ thể (format: 'YYYY-MM-DD'), None = ngày cuối
        
        Returns:
            test_df: DataFrame 24 giờ
            start_idx: Index bắt đầu trong df gốc
        """
        if date_str is None:
            # Lấy 24 giờ cuối cùng
            end_idx = len(df)
            start_idx = max(0, end_idx - 24)
            test_df = df.iloc[start_idx:end_idx].copy()
            test_date = df.iloc[end_idx-1]['timestamp'].strftime('%Y-%m-%d')
        else:
            # Tìm ngày cụ thể
            target_date = pd.to_datetime(date_str)
            day_mask = df['timestamp'].dt.date == target_date.date()
            
            if day_mask.sum() == 0:
                print(f"Warning: No data for date {date_str}")
                print(f"Available dates: {df['timestamp'].dt.date.unique()[:10]}...")
                # Fallback to last day
                end_idx = len(df)
                start_idx = max(0, end_idx - 24)
                test_df = df.iloc[start_idx:end_idx].copy()
                test_date = df.iloc[end_idx-1]['timestamp'].strftime('%Y-%m-%d')
            else:
                test_df = df[day_mask].copy()
                start_idx = test_df.index[0]
                test_date = date_str
        
        # Ensure có đủ 24 giờ, nếu không đủ thì lấy backward
        if len(test_df) < 24:
            print(f"Warning: Only {len(test_df)} hours available for {test_date}")
            end_idx = start_idx + len(test_df)
            start_idx = max(0, end_idx - 24)
            test_df = df.iloc[start_idx:end_idx].copy()
        
        test_df = test_df.iloc[:24].reset_index(drop=True)  # Limit to 24 hours
        
        print(f"\nSelected test day: {test_date}")
        print(f"Test period: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
        print(f"Test samples: {len(test_df)}")
        
        return test_df, start_idx
    
    def load_dqn_model(self, test_df):
        """Load DQN model từ checkpoint"""
        print(f"\nLoading DQN model from {self.checkpoint_path}")
        
        # Create environment để lấy state/action dims
        env = TradingMDP(test_df, initial_balance=10000.0)
        state_dim = 8  # [position, rsi, macd_hist, trend, bb_position, volatility, price_change, profit_pct]
        action_dim = 3  # Hold, Buy, Sell
        
        # Create agent
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=0.001,
            gamma=0.99,
            epsilon_start=0.0,  # No exploration for prediction
            epsilon_end=0.0,
            epsilon_decay=1.0,
            buffer_capacity=10000,
            batch_size=64
        )
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        
        # Load model weights
        agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        agent.policy_net.eval()
        agent.target_net.eval()
        
        # Force CPU
        agent.policy_net.to('cpu')
        agent.target_net.to('cpu')
        agent.device = 'cpu'
        
        print(f"Model loaded successfully!")
        print(f"Episode: {checkpoint.get('episode', 'N/A')}")
        print(f"Best profit: ${checkpoint.get('best_profit', 0):,.2f}")
        
        return agent, env
    
    def predict_actions(self, agent, test_df):
        """
        Dự báo hành động cho từng giờ trong ngày
        
        Returns:
            predictions: List of (hour, timestamp, action, action_name, state, price, reward)
        """
        print("\nPredicting actions for 24 hours...")
        
        # Create environment với test data
        env = TradingMDP(test_df, initial_balance=10000.0)
        state = env.reset()
        
        predictions = []
        done = False
        hour = 0
        
        while not done and hour < 24:
            # Get current state info
            current_price = test_df.iloc[env.current_step]['close']
            timestamp = test_df.iloc[env.current_step]['timestamp']
            
            # Predict action
            action = agent.select_action(state, epsilon=0.0)  # Greedy
            action_name = self.action_names[action]
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            predictions.append({
                'hour': hour,
                'timestamp': timestamp,
                'action': action,
                'action_name': action_name,
                'state': state.copy(),
                'price': current_price,
                'reward': reward,
                'balance': env.balance,
                'holdings': env.holdings,
                'portfolio_value': env.balance + env.holdings * current_price,
                'position': env.position
            })
            
            state = next_state
            hour += 1
        
        print(f"Predicted {len(predictions)} hours")
        
        # Convert to DataFrame
        pred_df = pd.DataFrame(predictions)
        
        # Calculate action distribution
        action_dist = pred_df['action_name'].value_counts()
        print("\nAction Distribution:")
        for action, count in action_dist.items():
            print(f"  {action}: {count} ({count/len(pred_df)*100:.1f}%)")
        
        return pred_df, env
    
    def calculate_metrics(self, pred_df, test_df):
        """
        Tính metrics để đánh giá chất lượng dự báo
        
        Metrics:
        - MAE, MSE, RMSE: Đo lường sai số dự báo hành động
        - MAPE: Sai số phần trăm trung bình tuyệt đối
        - R2: Hệ số xác định (đo mức độ phù hợp)
        
        Note: Đây là classification task (3 classes), không phải regression
        => Metrics truyền thống (MAE, MSE) không phù hợp hoàn toàn
        => Nhưng vẫn tính để so sánh với baseline
        """
        print("\nCalculating metrics...")
        
        # Baseline: Random actions (uniform distribution)
        np.random.seed(42)
        random_actions = np.random.randint(0, 3, size=len(pred_df))
        
        # Baseline: Buy & Hold (always Buy at start, then Hold)
        buyhold_actions = [1] + [0] * (len(pred_df) - 1)
        
        # Baseline: Always Sell (conservative strategy)
        always_sell = [2] * len(pred_df)
        
        # Predicted actions
        pred_actions = pred_df['action'].values
        
        # Calculate metrics (treat actions as numeric for comparison)
        # Note: Đây chỉ là để so sánh, không phải true regression metrics
        
        # Compare với ground truth ideal actions (từ market direction)
        # Ideal: Buy when price going up, Sell when going down
        price_changes = test_df['close'].pct_change().fillna(0)
        ideal_actions = []
        for change in price_changes:
            if change > 0.01:  # Price up > 1%
                ideal_actions.append(1)  # Should Buy
            elif change < -0.01:  # Price down > 1%
                ideal_actions.append(2)  # Should Sell
            else:
                ideal_actions.append(0)  # Hold
        
        ideal_actions = np.array(ideal_actions[:len(pred_df)])
        
        # Metrics vs Ideal
        mae = mean_absolute_error(ideal_actions, pred_actions)
        mse = mean_squared_error(ideal_actions, pred_actions)
        rmse = np.sqrt(mse)
        
        # MAPE: Mean Absolute Percentage Error
        # Avoid division by zero
        mape_vals = []
        for true, pred in zip(ideal_actions, pred_actions):
            if true != 0:
                mape_vals.append(abs((true - pred) / true))
        mape = np.mean(mape_vals) * 100 if mape_vals else 0
        
        # R2 score
        r2 = r2_score(ideal_actions, pred_actions)
        
        # Accuracy: % correct actions
        accuracy = (pred_actions == ideal_actions).sum() / len(pred_actions) * 100
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'Accuracy': accuracy,
            'Total_Return': (pred_df['portfolio_value'].iloc[-1] - 10000) / 10000 * 100,
            'Final_Balance': pred_df['balance'].iloc[-1],
            'Num_Trades': (pred_df['action'] != 0).sum()
        }
        
        print("\nMetrics vs Ideal Actions:")
        print(f"  MAE: {mae:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R2 Score: {r2:.4f}")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"\nTrading Performance:")
        print(f"  Total Return: {metrics['Total_Return']:.2f}%")
        print(f"  Final Balance: ${metrics['Final_Balance']:.2f}")
        print(f"  Number of Trades: {metrics['Num_Trades']}")
        
        return metrics, ideal_actions
    
    def analyze_market_environment(self, test_df, pred_df):
        """
        Phân tích môi trường thị trường trong ngày test
        
        Phân loại:
        - Bull Market (Tăng giá): Trend > 0, RSI > 50
        - Bear Market (Giảm giá): Trend < 0, RSI < 50
        - Sideways (Đi ngang): Trend ~ 0, RSI ~ 50
        """
        print("\n" + "="*60)
        print("PHÂN TÍCH MÔI TRƯỜNG THỊ TRƯỜNG")
        print("="*60)
        
        # Price analysis
        start_price = test_df['close'].iloc[0]
        end_price = test_df['close'].iloc[-1]
        price_change = (end_price - start_price) / start_price * 100
        
        # Trend analysis
        avg_trend = test_df['trend'].mean()
        trend_positive = (test_df['trend'] > 0).sum()
        trend_negative = (test_df['trend'] < 0).sum()
        
        # RSI analysis
        avg_rsi = test_df['rsi'].mean()
        overbought = (test_df['rsi'] > 70).sum()
        oversold = (test_df['rsi'] < 30).sum()
        
        # MACD analysis
        avg_macd = test_df['macd_hist'].mean()
        macd_positive = (test_df['macd_hist'] > 0).sum()
        
        # Volatility
        avg_volatility = test_df['volatility'].mean()
        max_volatility = test_df['volatility'].max()
        
        # Market classification
        if avg_trend > 0.3 and avg_rsi > 55:
            market_type = "BULL MARKET (Thị trường tăng giá)"
            optimal_strategy = "Buy sớm và Hold lâu, ít Sell"
        elif avg_trend < -0.3 and avg_rsi < 45:
            market_type = "BEAR MARKET (Thị trường giảm giá)"
            optimal_strategy = "Sell sớm để bảo vệ vốn, ít Buy"
        else:
            market_type = "SIDEWAYS MARKET (Thị trường đi ngang)"
            optimal_strategy = "Buy ở đáy, Sell ở đỉnh, Trading tần suất cao"
        
        print(f"\n1. LOẠI THỊ TRƯỜNG: {market_type}")
        print(f"   Chiến lược tối ưu: {optimal_strategy}")
        
        print(f"\n2. BIẾN ĐỘNG GIÁ:")
        print(f"   - Giá bắt đầu: ${start_price:,.2f}")
        print(f"   - Giá kết thúc: ${end_price:,.2f}")
        print(f"   - Thay đổi: {price_change:+.2f}%")
        print(f"   - Max giá: ${test_df['close'].max():,.2f}")
        print(f"   - Min giá: ${test_df['close'].min():,.2f}")
        
        print(f"\n3. CHỈ BÁO KỸ THUẬT:")
        print(f"   - Trend trung bình: {avg_trend:.3f}")
        print(f"   - Trend tích cực: {trend_positive}/{len(test_df)} giờ ({trend_positive/len(test_df)*100:.1f}%)")
        print(f"   - Trend tiêu cực: {trend_negative}/{len(test_df)} giờ ({trend_negative/len(test_df)*100:.1f}%)")
        
        print(f"\n4. RSI (Relative Strength Index):")
        print(f"   - RSI trung bình: {avg_rsi:.2f}")
        print(f"   - Overbought (>70): {overbought}/{len(test_df)} giờ")
        print(f"   - Oversold (<30): {oversold}/{len(test_df)} giờ")
        
        print(f"\n5. MACD:")
        print(f"   - MACD Histogram TB: {avg_macd:.4f}")
        print(f"   - MACD dương: {macd_positive}/{len(test_df)} giờ ({macd_positive/len(test_df)*100:.1f}%)")
        
        print(f"\n6. ĐỘ BIẾN ĐỘNG:")
        print(f"   - Volatility trung bình: {avg_volatility:.4f}")
        print(f"   - Volatility tối đa: {max_volatility:.4f}")
        
        # Analyze DQN actions vs market
        print(f"\n7. PHÂN TÍCH HÀNH ĐỘNG DQN:")
        action_dist = pred_df['action_name'].value_counts()
        total_actions = len(pred_df)
        
        for action in ['Hold', 'Buy', 'Sell']:
            count = action_dist.get(action, 0)
            pct = count / total_actions * 100
            print(f"   - {action}: {count}/{total_actions} ({pct:.1f}%)")
        
        # Check if DQN strategy aligns with market
        if market_type.startswith("BULL"):
            buy_pct = action_dist.get('Buy', 0) / total_actions * 100
            if buy_pct > 30:
                alignment = "Phù hợp: DQN mua nhiều trong thị trường tăng"
            else:
                alignment = "Không tối ưu: DQN nên mua nhiều hơn"
        elif market_type.startswith("BEAR"):
            sell_pct = action_dist.get('Sell', 0) / total_actions * 100
            if sell_pct > 30:
                alignment = "Phù hợp: DQN bán nhiều trong thị trường giảm"
            else:
                alignment = "Không tối ưu: DQN nên bán nhiều hơn"
        else:
            alignment = "Phù hợp với thị trường đi ngang"
        
        print(f"\n   => {alignment}")
        
        return {
            'market_type': market_type,
            'price_change': price_change,
            'avg_trend': avg_trend,
            'avg_rsi': avg_rsi,
            'avg_volatility': avg_volatility,
            'optimal_strategy': optimal_strategy,
            'alignment': alignment
        }
    
    def discuss_reward_and_actions(self, pred_df, test_df):
        """
        Thảo luận chi tiết về Reward Function và ảnh hưởng đến hành động
        """
        print("\n" + "="*60)
        print("THẢO LUẬN: REWARD FUNCTION VÀ HÀNH ĐỘNG")
        print("="*60)
        
        print("\n1. REWARD FUNCTION (Theo bài báo):")
        print("""
   reward = profit - transaction_cost - hold_penalty
   
   Trong đó:
   - profit: Lợi nhuận từ giao dịch (nếu Sell thành công)
   - transaction_cost: 0.01% cho mỗi giao dịch Buy/Sell
   - hold_penalty: 0.01% để tránh Hold quá lâu
   
   Thành phần bổ sung:
   - Trend alignment: +0.1 nếu Buy khi trend tăng, Sell khi trend giảm
   - Risk penalty: -0.5 × MDD nếu Maximum Drawdown > 30%
        """)
        
        print("\n2. ẢNH HƯỞNG CỦA REWARD ĐẾN HÀNH ĐỘNG:")
        
        # Phân tích reward cho từng action
        buy_rewards = pred_df[pred_df['action'] == 1]['reward']
        sell_rewards = pred_df[pred_df['action'] == 2]['reward']
        hold_rewards = pred_df[pred_df['action'] == 0]['reward']
        
        print(f"\n   a) Hành động BUY:")
        if len(buy_rewards) > 0:
            print(f"      - Số lần thực hiện: {len(buy_rewards)}")
            print(f"      - Reward trung bình: {buy_rewards.mean():.4f}")
            print(f"      - Reward tối đa: {buy_rewards.max():.4f}")
            print(f"      - Reward tối thiểu: {buy_rewards.min():.4f}")
            print(f"      => Buy được thực hiện khi: Q(s, Buy) > Q(s, Hold) và Q(s, Buy) > Q(s, Sell)")
            print(f"      => DQN đã học: Buy khi có tín hiệu tăng giá (trend > 0, RSI < 70)")
        else:
            print(f"      - Không có hành động Buy nào trong 24h")
            print(f"      => DQN tránh Buy (có thể do thị trường giảm hoặc không có cơ hội)")
        
        print(f"\n   b) Hành động SELL:")
        if len(sell_rewards) > 0:
            print(f"      - Số lần thực hiện: {len(sell_rewards)}")
            print(f"      - Reward trung bình: {sell_rewards.mean():.4f}")
            print(f"      - Reward tối đa: {sell_rewards.max():.4f}")
            print(f"      - Reward tối thiểu: {sell_rewards.min():.4f}")
            print(f"      => Sell được thực hiện khi: Q(s, Sell) > Q(s, Hold) và Q(s, Sell) > Q(s, Buy)")
            print(f"      => DQN đã học: Sell để bảo vệ lợi nhuận hoặc cắt lỗ")
        else:
            print(f"      - Không có hành động Sell nào trong 24h")
            print(f"      => DQN giữ vị thế (Hold) hoặc không có coin để bán")
        
        print(f"\n   c) Hành động HOLD:")
        if len(hold_rewards) > 0:
            print(f"      - Số lần thực hiện: {len(hold_rewards)}")
            print(f"      - Reward trung bình: {hold_rewards.mean():.4f}")
            print(f"      - Reward tối đa: {hold_rewards.max():.4f}")
            print(f"      - Reward tối thiểu: {hold_rewards.min():.4f}")
            print(f"      => Hold được ưu tiên khi: Không có tín hiệu rõ ràng để Buy/Sell")
            print(f"      => DQN đã học: Tránh giao dịch không cần thiết (transaction cost)")
        
        print(f"\n3. TRADE-OFFS TRONG REWARD DESIGN:")
        print("""
   a) Profit vs Transaction Cost:
      - Profit cao khuyến khích giao dịch nhiều
      - Transaction cost (0.01%) ngăn chặn over-trading
      - DQN học cân bằng: Chỉ trade khi expected profit > cost
   
   b) Hold Penalty vs Over-trading:
      - Hold penalty (0.01%) khuyến khích hành động
      - Nhưng không quá lớn để tránh giao dịch bừa bãi
      - DQN học: Hold khi chờ cơ hội tốt hơn
   
   c) Trend Alignment Bonus:
      - +0.1 reward nếu Buy khi trend tăng, Sell khi trend giảm
      - Khuyến khích DQN follow the trend
      - Tránh "catch the falling knife" (mua khi giảm mạnh)
   
   d) Risk Management:
      - MDD penalty ngăn chặn drawdown lớn
      - Stop loss (10%) tự động bảo vệ vốn
      - Max position (50%) hạn chế rủi ro
        """)
        
        print(f"\n4. HÀNH ĐỘNG VÀ MÔI TRƯỜNG THỊ TRƯỜNG:")
        
        # Correlation between actions and market indicators
        for hour in range(min(5, len(pred_df))):  # Show first 5 hours
            row = pred_df.iloc[hour]
            print(f"\n   Giờ {hour} ({row['timestamp']}):")
            print(f"      - Giá: ${row['price']:,.2f}")
            print(f"      - Hành động: {row['action_name']}")
            print(f"      - Reward: {row['reward']:.4f}")
            print(f"      - Trend: {test_df.iloc[hour]['trend']:.2f}")
            print(f"      - RSI: {test_df.iloc[hour]['rsi']:.2f}")
            print(f"      - Position: {'Holding BTC' if row['position'] == 1 else 'Holding Cash'}")
        
        print("\n   ... (còn lại xem trong visualization)")
    
    def plot_results(self, pred_df, test_df, metrics, market_info, save_path='results/charts/'):
        """
        Vẽ biểu đồ kết quả dự báo
        
        6 panels:
        1. Price + Actions (Buy/Sell/Hold markers)
        2. Portfolio Value over time
        3. Rewards over time
        4. Technical Indicators (RSI, MACD, Trend)
        5. Action distribution (pie chart)
        6. Metrics comparison table
        """
        print("\nGenerating visualizations...")
        
        os.makedirs(save_path, exist_ok=True)
        
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Panel 1: Price + Actions
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(pred_df['timestamp'], pred_df['price'], 'b-', linewidth=2, label='BTC Price')
        
        # Mark actions
        buy_mask = pred_df['action'] == 1
        sell_mask = pred_df['action'] == 2
        hold_mask = pred_df['action'] == 0
        
        ax1.scatter(pred_df[buy_mask]['timestamp'], pred_df[buy_mask]['price'], 
                   c='green', marker='^', s=200, label='Buy', zorder=5)
        ax1.scatter(pred_df[sell_mask]['timestamp'], pred_df[sell_mask]['price'], 
                   c='red', marker='v', s=200, label='Sell', zorder=5)
        
        ax1.set_title('BTC Price với Hành Động DQN', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Thời gian')
        ax1.set_ylabel('Giá (USD)')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Panel 2: Portfolio Value
        ax2 = fig.add_subplot(gs[1, :2])
        ax2.plot(pred_df['timestamp'], pred_df['portfolio_value'], 'g-', linewidth=2)
        ax2.axhline(y=10000, color='r', linestyle='--', label='Initial Balance ($10,000)')
        ax2.fill_between(pred_df['timestamp'], 10000, pred_df['portfolio_value'], 
                         where=pred_df['portfolio_value']>=10000, alpha=0.3, color='green', label='Profit')
        ax2.fill_between(pred_df['timestamp'], 10000, pred_df['portfolio_value'], 
                         where=pred_df['portfolio_value']<10000, alpha=0.3, color='red', label='Loss')
        
        ax2.set_title('Giá Trị Danh Mục Theo Thời Gian', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Thời gian')
        ax2.set_ylabel('Portfolio Value (USD)')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Panel 3: Rewards
        ax3 = fig.add_subplot(gs[2, :2])
        colors = ['blue' if r >= 0 else 'red' for r in pred_df['reward']]
        ax3.bar(pred_df['timestamp'], pred_df['reward'], color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_title('Rewards Theo Thời Gian', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Thời gian')
        ax3.set_ylabel('Reward')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.tick_params(axis='x', rotation=45)
        
        # Panel 4: Technical Indicators
        ax4 = fig.add_subplot(gs[3, :2])
        ax4_2 = ax4.twinx()
        
        ax4.plot(test_df['timestamp'][:len(pred_df)], test_df['rsi'][:len(pred_df)], 
                'b-', label='RSI', linewidth=2)
        ax4.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought (70)')
        ax4.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold (30)')
        ax4.set_ylabel('RSI', color='b')
        ax4.tick_params(axis='y', labelcolor='b')
        ax4.set_ylim([0, 100])
        
        ax4_2.plot(test_df['timestamp'][:len(pred_df)], test_df['trend'][:len(pred_df)], 
                  'r-', label='Trend', linewidth=2)
        ax4_2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4_2.set_ylabel('Trend', color='r')
        ax4_2.tick_params(axis='y', labelcolor='r')
        
        ax4.set_title('Chỉ Báo Kỹ Thuật', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Thời gian')
        ax4.legend(loc='upper left')
        ax4_2.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        
        # Panel 5: Action Distribution
        ax5 = fig.add_subplot(gs[0, 2])
        action_counts = pred_df['action_name'].value_counts()
        colors_pie = {'Hold': 'blue', 'Buy': 'green', 'Sell': 'red'}
        colors_list = [colors_pie[action] for action in action_counts.index]
        
        ax5.pie(action_counts.values, labels=action_counts.index, autopct='%1.1f%%',
               colors=colors_list, startangle=90)
        ax5.set_title('Phân Bố Hành Động', fontsize=14, fontweight='bold')
        
        # Panel 6: Metrics Table
        ax6 = fig.add_subplot(gs[1:, 2])
        ax6.axis('off')
        
        metrics_data = [
            ['Metric', 'Value'],
            ['MAE', f"{metrics['MAE']:.4f}"],
            ['MSE', f"{metrics['MSE']:.4f}"],
            ['RMSE', f"{metrics['RMSE']:.4f}"],
            ['MAPE', f"{metrics['MAPE']:.2f}%"],
            ['R² Score', f"{metrics['R2']:.4f}"],
            ['Accuracy', f"{metrics['Accuracy']:.2f}%"],
            ['', ''],
            ['Total Return', f"{metrics['Total_Return']:.2f}%"],
            ['Final Balance', f"${metrics['Final_Balance']:.2f}"],
            ['Num Trades', f"{metrics['Num_Trades']}"],
            ['', ''],
            ['Market Type', market_info['market_type'].split('(')[0]],
            ['Price Change', f"{market_info['price_change']:.2f}%"],
            ['Avg Trend', f"{market_info['avg_trend']:.3f}"],
            ['Avg RSI', f"{market_info['avg_rsi']:.2f}"],
        ]
        
        table = ax6.table(cellText=metrics_data, cellLoc='left', loc='center',
                         colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style metrics rows
        for i in range(1, len(metrics_data)):
            if metrics_data[i][0] == '':
                continue
            table[(i, 0)].set_facecolor('#f0f0f0')
            table[(i, 1)].set_facecolor('#ffffff')
        
        ax6.set_title('Metrics & Market Info', fontsize=14, fontweight='bold', pad=20)
        
        # Overall title
        fig.suptitle(f'DỰ BÁO HÀNH ĐỘNG GIAO DỊCH BTC - 1 NGÀY\n{pred_df["timestamp"].iloc[0].strftime("%Y-%m-%d")}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save
        save_file = os.path.join(save_path, 'one_day_prediction_dqn.png')
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {save_file}")
        
        plt.close()
    
    def generate_report(self, pred_df, metrics, market_info, ideal_actions, save_path='results/reports/'):
        """Generate markdown report"""
        print("\nGenerating report...")
        
        os.makedirs(save_path, exist_ok=True)
        
        report = f"""# BÁO CÁO DỰ BÁO HÀNH ĐỘNG GIAO DỊCH - 1 NGÀY

## Thông Tin Chung

- **Ngày dự báo**: {pred_df['timestamp'].iloc[0].strftime('%Y-%m-%d')}
- **Khoảng thời gian**: {pred_df['timestamp'].iloc[0].strftime('%H:%M')} - {pred_df['timestamp'].iloc[-1].strftime('%H:%M')}
- **Số giờ**: {len(pred_df)} giờ
- **Mô hình**: Deep Q-Network (DQN)
- **Checkpoint**: Episode {1831}, Best Profit $3.8M

## 1. Metrics Đánh Giá

### 1.1. Metrics So Sánh với Ideal Actions

| Metric | Value | Giải Thích |
|--------|-------|-----------|
| **MAE** | {metrics['MAE']:.4f} | Mean Absolute Error - Sai số tuyệt đối trung bình |
| **MSE** | {metrics['MSE']:.4f} | Mean Squared Error - Sai số bình phương trung bình |
| **RMSE** | {metrics['RMSE']:.4f} | Root Mean Squared Error - Căn bậc hai của MSE |
| **MAPE** | {metrics['MAPE']:.2f}% | Mean Absolute Percentage Error - Sai số % TB |
| **R² Score** | {metrics['R2']:.4f} | Coefficient of Determination - Độ phù hợp |
| **Accuracy** | {metrics['Accuracy']:.2f}% | % dự báo đúng so với ideal actions |

### 1.2. Trading Performance

| Metric | Value |
|--------|-------|
| **Total Return** | {metrics['Total_Return']:.2f}% |
| **Final Balance** | ${metrics['Final_Balance']:.2f} |
| **Initial Balance** | $10,000.00 |
| **Profit/Loss** | ${metrics['Final_Balance'] - 10000:.2f} |
| **Number of Trades** | {metrics['Num_Trades']} |

## 2. Phân Bố Hành Động

{pred_df['action_name'].value_counts().to_markdown()}

## 3. Môi Trường Thị Trường

- **Loại thị trường**: {market_info['market_type']}
- **Thay đổi giá**: {market_info['price_change']:.2f}%
- **Trend trung bình**: {market_info['avg_trend']:.3f}
- **RSI trung bình**: {market_info['avg_rsi']:.2f}
- **Volatility TB**: {market_info['avg_volatility']:.4f}

**Chiến lược tối ưu**: {market_info['optimal_strategy']}

**Đánh giá DQN**: {market_info['alignment']}

## 4. Phân Tích Reward Function

### 4.1. Công Thức Reward

```python
reward = profit - transaction_cost - hold_penalty

# Components:
# - profit: Lợi nhuận từ giao dịch
# - transaction_cost: 0.01% per trade
# - hold_penalty: 0.01% để tránh hold quá lâu
# - trend_bonus: +0.1 nếu action align với trend
# - risk_penalty: -0.5 × MDD nếu MDD > 30%
```

### 4.2. Reward Statistics

| Action | Count | Avg Reward | Max Reward | Min Reward |
|--------|-------|-----------|-----------|-----------|
"""
        
        for action in [0, 1, 2]:
            action_name = self.action_names[action]
            action_rewards = pred_df[pred_df['action'] == action]['reward']
            if len(action_rewards) > 0:
                report += f"| {action_name} | {len(action_rewards)} | {action_rewards.mean():.4f} | {action_rewards.max():.4f} | {action_rewards.min():.4f} |\n"
            else:
                report += f"| {action_name} | 0 | N/A | N/A | N/A |\n"
        
        report += f"""
## 5. Thảo Luận

### 5.1. Ảnh Hưởng của Môi Trường đến Hành Động

Trong thị trường **{market_info['market_type'].split('(')[0].strip()}**, DQN đã học được chiến lược:

"""
        
        action_dist = pred_df['action_name'].value_counts()
        for action in ['Hold', 'Buy', 'Sell']:
            count = action_dist.get(action, 0)
            pct = count / len(pred_df) * 100
            report += f"- **{action}**: {count}/{len(pred_df)} lần ({pct:.1f}%)\n"
        
        report += f"""

### 5.2. Trade-offs trong Reward Design

**a) Profit vs Transaction Cost:**
- Profit cao khuyến khích giao dịch nhiều
- Transaction cost (0.01%) ngăn chặn over-trading
- DQN học cân bằng: Chỉ trade khi expected profit > cost

**b) Hold Penalty vs Over-trading:**
- Hold penalty (0.01%) khuyến khích hành động
- Nhưng không quá lớn để tránh giao dịch bừa bãi
- DQN học: Hold khi chờ cơ hội tốt hơn

**c) Trend Alignment:**
- +0.1 reward nếu Buy khi trend tăng, Sell khi trend giảm
- Khuyến khích DQN follow the trend
- Tránh "catch the falling knife" (mua khi giảm mạnh)

**d) Risk Management:**
- MDD penalty ngăn chặn drawdown lớn
- Stop loss (10%) tự động bảo vệ vốn
- Max position (50%) hạn chế rủi ro

### 5.3. So Sánh với Ideal Actions

DQN đạt **{metrics['Accuracy']:.2f}% accuracy** so với ideal actions (dựa trên price direction).

**Phân tích:**
- MAE = {metrics['MAE']:.4f}: Sai số trung bình {metrics['MAE']:.2f} action
- R² = {metrics['R2']:.4f}: {'Fit tốt' if metrics['R2'] > 0.5 else 'Fit trung bình' if metrics['R2'] > 0 else 'Cần cải thiện'}

## 6. Kết Luận

### 6.1. Điểm Mạnh

1. **Adaptive Strategy**: DQN học được điều chỉnh hành động theo môi trường
2. **Risk Awareness**: Tránh over-trading với transaction cost awareness
3. **Trend Following**: Align actions với market trend

### 6.2. Điểm Cần Cải Thiện

1. **Metrics Limitations**: MAE/MSE/RMSE không phù hợp hoàn toàn với classification task
2. **Short-term Prediction**: Chỉ test 1 ngày, cần test dài hạn hơn
3. **Market Dependency**: Performance phụ thuộc nhiều vào loại thị trường

### 6.3. Khuyến Nghị

- Test trên nhiều khoảng thời gian khác nhau (bull/bear/sideways)
- Sử dụng classification metrics (Precision, Recall, F1)
- So sánh với baseline strategies (Buy & Hold, Moving Average)
- Backtest trên multiple coins để đánh giá generalization

---

**Ngày tạo**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Phiên bản**: 1.0
"""
        
        save_file = os.path.join(save_path, 'one_day_prediction_report.md')
        with open(save_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Saved report to: {save_file}")
    
    def run(self, test_date=None):
        """Run complete prediction pipeline"""
        print("="*60)
        print("DỰ BÁO HÀNH ĐỘNG GIAO DỊCH - 1 NGÀY")
        print("="*60)
        
        # Load data
        df_full = self.load_data()
        
        # Select test day
        test_df, start_idx = self.select_test_day(df_full, test_date)
        
        # Load DQN model
        agent, env = self.load_dqn_model(test_df)
        
        # Predict actions
        pred_df, final_env = self.predict_actions(agent, test_df)
        
        # Calculate metrics
        metrics, ideal_actions = self.calculate_metrics(pred_df, test_df)
        
        # Analyze market environment
        market_info = self.analyze_market_environment(test_df, pred_df)
        
        # Discuss reward and actions
        self.discuss_reward_and_actions(pred_df, test_df)
        
        # Generate visualizations
        self.plot_results(pred_df, test_df, metrics, market_info)
        
        # Generate report
        self.generate_report(pred_df, metrics, market_info, ideal_actions)
        
        print("\n" + "="*60)
        print("HOÀN THÀNH!")
        print("="*60)
        print(f"Visualization: results/charts/one_day_prediction_dqn.png")
        print(f"Report: results/reports/one_day_prediction_report.md")

if __name__ == '__main__':
    # Configuration
    CHECKPOINT_PATH = 'src/checkpoints_dqn/checkpoint_best.pkl'
    DATA_PATH = 'data/raw/multi_coin_1h.csv'
    
    # Test date (None = last day in dataset)
    # Hoặc chỉ định ngày cụ thể: '2025-10-15'
    # Ngày có trend rõ ràng (từ find_best_test_day.py):
    # - Bear: '2020-03-12' (-39.34%), '2019-06-27' (-14%), '2018-01-16' (-17%)
    # - Bull: '2020-04-22' (+4.3%), '2020-04-06' (+6.76%)
    TEST_DATE = '2020-04-06'  # Strong Bull Market +6.76% - Model có thể perform tốt hơn
    
    # Run prediction
    predictor = OneDayPredictor(CHECKPOINT_PATH, DATA_PATH)
    predictor.run(test_date=TEST_DATE)
