"""
Module định nghĩa MDP (Markov Decision Process) cho trading
MDP definition module for trading
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict


class TradingMDP:
    """
    Class định nghĩa MDP cho hệ thống giao dịch
    
    Theo bài báo:
    - Action 0: Mua (Buy)
    - Action 1: Bán (Sell)  
    - Action 2: Giữ (Hold)
    
    State Space: [position, rsi, macd_hist, trend, bb_position, volatility]
    Action Space: {0: Buy, 1: Sell, 2: Hold}
    Reward: Dựa trên lợi nhuận và xu hướng dữ liệu
    """
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000.0, 
                 transaction_cost: float = 0.0001, hold_penalty: float = 0.00001,
                 interval: str = '1h', stop_loss_pct: float = 0.15,
                 max_position_pct: float = 0.5, max_loss_pct: float = 0.20,
                 trailing_stop_pct: float = 0.05, enable_risk_management: bool = True,
                 trade_cooldown: int = 24):
        """
        Khởi tạo MDP - V7 ENHANCED: Improved reward & state space
        
        Args:
            data: DataFrame chứa dữ liệu giá và indicators
            initial_balance: Số tiền ban đầu
            transaction_cost: Phí giao dịch (0.01% = 0.0001 - realistic for major exchanges)
            hold_penalty: Penalty cho mỗi step hold (0.001% mặc định - REDUCED from 0.01%)
            interval: Timeframe ('1h', '1d', '15m', etc.)
            stop_loss_pct: Stop loss percentage (15% mặc định - INCREASED from 10%)
            max_position_pct: Maximum position size (50% balance mặc định)
            max_loss_pct: Maximum loss per episode before early stop (20% mặc định)
            trailing_stop_pct: Trailing stop loss percentage (5% mặc định)
            enable_risk_management: Enable risk management features (True mặc định)
            trade_cooldown: Steps between trades (24 = once per day for 1h data)
        """
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.holdings = 0  # Số lượng coin đang giữ
        self.position = 0  # 0: cash, 1: holding
        self.current_step = 0
        self.transaction_cost = transaction_cost  # ✅ Phí giao dịch
        self.hold_penalty = hold_penalty  # ✅ Penalty cho hold
        self.consecutive_holds = 0  # ✅ Đếm số lần hold liên tiếp
        self.last_action = 2  # ✅ Action trước đó
        
        # 🚀 V4 FIX: COOLDOWN to prevent over-trading
        self.trade_cooldown = trade_cooldown  # Steps between trades
        self.last_trade_step = -trade_cooldown  # Allow first trade immediately
        
        # ✅ V2 Improvements
        self.interval = interval  # Timeframe
        self.stop_loss_pct = stop_loss_pct  # Stop loss
        self.entry_price = 0  # Track buy price
        self.trade_entry_step = 0  # ✅ NEW APPROACH: Track step when trade entered
        self.max_hold_steps = self._calculate_hold_threshold()  # Dynamic threshold
        
        # ✅ V3 Risk Management
        self.max_position_pct = max_position_pct  # Position sizing: max 50% balance
        self.max_loss_pct = max_loss_pct  # Max loss: 20% before early stop
        self.trailing_stop_pct = trailing_stop_pct  # Trailing stop: 5%
        self.enable_risk_management = enable_risk_management
        self.highest_price = 0  # Track highest price for trailing stop
        self.episode_ended_early = False  # Track if episode ended due to max loss
        
        # 🆕 V5: MDD & Annualized Return Tracking (Paper Methodology)
        self.portfolio_history = []  # Track portfolio value over time
        self.peak_portfolio = initial_balance  # Track peak for MDD calculation
        self.current_mdd = 0.0  # Current Maximum Drawdown
        self.max_mdd = 0.0  # Worst MDD in episode
        self.episode_start_time = 0  # Track episode start for annualized return
        
        # 🆕 V5.1: Epsilon-aware emergency stop (disable during exploration)
        self.current_epsilon = 1.0  # Track current epsilon for adaptive risk management
        self.epsilon_threshold = 0.3  # Only enable emergency stop when epsilon < 0.3
        
        # Xử lý missing values
        # ✅ FIX: Fill NaN với neutral values thay vì 0
        self.data['rsi'] = self.data['rsi'].fillna(50.0)  # Neutral RSI
        self.data['macd_hist'] = self.data['macd_hist'].fillna(0.0)  # Neutral MACD
        self.data['volatility'] = self.data['volatility'].fillna(0.0)  # No volatility
        self.data = self.data.ffill().fillna(0)  # Forward fill còn lại
        
        # Chuẩn hóa dữ liệu
        self._normalize_features()
    
    def _calculate_hold_threshold(self) -> int:
        """Tính hold threshold dựa trên timeframe"""
        thresholds = {
            '1m': 60 * 24 * 7,    # 1 tuần = 10,080 phút
            '5m': 12 * 24 * 7,    # 1 tuần = 2,016 × 5 phút
            '15m': 4 * 24 * 7,    # 1 tuần = 672 × 15 phút
            '1h': 24 * 7,         # 1 tuần = 168 giờ
            '4h': 6 * 7,          # 1 tuần = 42 × 4 giờ
            '1d': 14,             # 2 tuần
        }
        return thresholds.get(self.interval, 24)  # Default: 24 steps
    
    def _normalize_features(self):
        """
        Chuẩn hóa các features để làm state
        """
        # RSI: giá trị từ 0-100
        if 'rsi' in self.data.columns:
            self.data['rsi_norm'] = self.data['rsi'] / 100.0
        
        # MACD Histogram: chuẩn hóa
        if 'macd_hist' in self.data.columns:
            max_macd = self.data['macd_hist'].abs().max()
            if max_macd > 0:
                self.data['macd_hist_norm'] = self.data['macd_hist'] / max_macd
        
        # Trend: -1, 0, 1 -> 0, 0.5, 1
        if 'trend' in self.data.columns:
            self.data['trend_norm'] = (self.data['trend'] + 1) / 2.0
        
        # Bollinger Band position: vị trí giá trong BB
        if all(col in self.data.columns for col in ['bb_upper', 'bb_lower', 'close']):
            bb_range = self.data['bb_upper'] - self.data['bb_lower']
            self.data['bb_position'] = np.where(
                bb_range > 0,
                (self.data['close'] - self.data['bb_lower']) / bb_range,
                0.5
            )
        
        # Volatility: chuẩn hóa
        if 'volatility' in self.data.columns:
            max_vol = self.data['volatility'].abs().max()
            if max_vol > 0:
                self.data['volatility_norm'] = self.data['volatility'] / max_vol
    
    def get_state(self, step: int = None) -> np.ndarray:
        """
        Lấy state hiện tại
        
        Args:
            step: Bước thời gian (nếu None thì lấy current_step)
        
        Returns:
            State vector
        """
        if step is None:
            step = self.current_step
        
        if step >= len(self.data):
            step = len(self.data) - 1
        
        # Lấy các features
        features = []
        
        # Position (0 hoặc 1)
        features.append(self.position)
        
        # Technical indicators
        if 'rsi_norm' in self.data.columns:
            features.append(self.data.iloc[step]['rsi_norm'])
        else:
            features.append(0.5)
        
        if 'macd_hist_norm' in self.data.columns:
            features.append(self.data.iloc[step]['macd_hist_norm'])
        else:
            features.append(0.0)
        
        if 'trend_norm' in self.data.columns:
            features.append(self.data.iloc[step]['trend_norm'])
        else:
            features.append(0.5)
        
        if 'bb_position' in self.data.columns:
            features.append(self.data.iloc[step]['bb_position'])
        else:
            features.append(0.5)
        
        if 'volatility_norm' in self.data.columns:
            features.append(self.data.iloc[step]['volatility_norm'])
        else:
            features.append(0.5)
        
        # ✅ NEW APPROACH: Thêm features quan trọng
        # Current profit/loss (nếu holding)
        if self.position == 1 and self.entry_price > 0:
            current_price = self.data.iloc[step]['close']
            current_profit_pct = (current_price - self.entry_price) / self.entry_price
            # Normalize về [0, 1]: -0.5 to 0.5 -> 0 to 1
            features.append(np.clip((current_profit_pct + 0.5) / 1.0, 0, 1))
        else:
            features.append(0.5)  # Neutral value
        
        # Time since trade entry (normalized)
        if self.position == 1 and self.trade_entry_step > 0:
            steps_since_trade = self.current_step - self.trade_entry_step
            time_norm = min(steps_since_trade / self.max_hold_steps, 1.0)
            features.append(time_norm)
        else:
            features.append(0.0)  # No trade active
        
        return np.array(features, dtype=np.float32)
    
    def reset(self) -> np.ndarray:
        """
        Reset môi trường về trạng thái ban đầu
        
        Returns:
            Initial state
        """
        self.balance = self.initial_balance
        self.holdings = 0
        self.position = 0
        self.consecutive_holds = 0  # ✅ Reset hold counter
        self.last_action = 2  # ✅ Reset last action
        self.entry_price = 0  # ✅ V2: Reset entry price
        self.trade_entry_step = 0  # ✅ NEW APPROACH: Reset trade entry step
        self.current_step = 0
        
        # 🚀 V4 FIX: Reset cooldown
        self.last_trade_step = -self.trade_cooldown  # Allow first trade immediately
        
        # ✅ V3: Reset risk management vars
        self.highest_price = 0
        self.episode_ended_early = False
        
        # 🆕 V5: Reset MDD tracking
        self.portfolio_history = [self.initial_balance]
        self.peak_portfolio = self.initial_balance
        self.current_mdd = 0.0
        self.max_mdd = 0.0
        self.episode_start_time = self.current_step
        
        return self._get_state()
    
    def set_epsilon(self, epsilon: float):
        """
        🆕 V5.1: Update current epsilon for epsilon-aware risk management
        Call this at the start of each episode to enable adaptive emergency stop
        
        Args:
            epsilon: Current exploration rate (0.0 to 1.0)
        """
        self.current_epsilon = epsilon
    
    def _get_state(self) -> np.ndarray:
        """
        Internal method to get current state
        Wrapper around get_state() for backward compatibility
        
        Returns:
            State vector at current step
        """
        return self.get_state(self.current_step)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Thực hiện một bước trong môi trường
        
        Theo bài báo + IMPROVEMENTS:
        - action 0: Mua (Buy)
        - action 1: Bán (Sell)
        - action 2: Giữ (Hold)
        - ✅ Transaction cost cho mỗi giao dịch
        - ✅ Hold penalty để tránh hold quá lâu
        - ✅ Reward cân bằng để khuyến khích trading
        
        Args:
            action: 0 (Buy), 1 (Sell), 2 (Hold)
        
        Returns:
            (next_state, reward, done, info)
        """
        if self.current_step >= len(self.data) - 1:
            done = True
        else:
            done = False
        
        current_price = self.data.iloc[self.current_step]['close']
        next_price = self.data.iloc[min(self.current_step + 1, len(self.data) - 1)]['close']
        
        reward = 0.0
        info = {}
        transaction_executed = False  # ✅ Track nếu có giao dịch
        
        # 🚀 V4 FIX: COOLDOWN CHECK - Prevent over-trading
        steps_since_last_trade = self.current_step - self.last_trade_step
        cooldown_active = steps_since_last_trade < self.trade_cooldown
        
        if cooldown_active and action in [0, 1]:  # Buy or Sell attempted during cooldown
            # Force Hold if cooldown not passed
            action = 2  # Override to Hold
            info['cooldown_active'] = True
            info['steps_until_trade'] = self.trade_cooldown - steps_since_last_trade
        
        # Thực hiện action (theo mapping của bài báo)
        if action == 0:  # Mua (Buy)
            if self.position == 0 and self.balance > 0:  # Có tiền và không đang hold
                # ✅ V3: POSITION SIZING - Chỉ mua tối đa max_position_pct% balance
                if self.enable_risk_management:
                    max_investment = self.balance * self.max_position_pct
                    investment_amount = min(self.balance, max_investment)
                else:
                    investment_amount = self.balance
                
                # ✅ Mua với position sizing - TRANSACTION COST
                effective_balance = investment_amount * (1 - self.transaction_cost)
                self.holdings = effective_balance / current_price
                self.balance = self.balance - investment_amount  # Giữ lại tiền chưa dùng
                self.position = 1
                self.entry_price = current_price  # ✅ Track entry price
                self.trade_entry_step = self.current_step  # ✅ NEW APPROACH: Track entry step
                self.highest_price = current_price  # ✅ Track highest price for trailing stop
                transaction_executed = True
                self.consecutive_holds = 0  # Reset hold counter
                self.last_trade_step = self.current_step  # 🚀 V4 FIX: Update last trade step
        elif action == 1:  # Bán (Sell)
            if self.position == 1 and self.holdings > 0:  # Có coin và đang hold
                # ✅ Bán toàn bộ holdings - TRANSACTION COST
                sell_value = self.holdings * current_price
                self.balance = sell_value * (1 - self.transaction_cost)
                self.holdings = 0
                self.position = 0
                self.entry_price = 0  # ✅ V2: Reset entry price
                self.trade_entry_step = 0  # ✅ NEW APPROACH: Reset trade entry step
                transaction_executed = True
                self.consecutive_holds = 0  # Reset hold counter
                self.last_trade_step = self.current_step  # 🚀 V4 FIX: Update last trade step
        
        # 🆕 V6.1: REMOVED ALL STOP LOSS MECHANISMS (Not in paper!)
        # Paper methodology: Let agent learn without forced sells
        # - No stop loss (10% threshold)
        # - No trailing stop (5% threshold)
        # - Agent controls all trades through Q-Learning
        
        # Only update highest price for tracking (not for stops)
        if self.position == 1 and current_price > self.highest_price:
            self.highest_price = current_price
        
        if action == 2:  # Giữ (Hold)
            self.consecutive_holds += 1  # ✅ Đếm số lần hold liên tiếp
        
        # ✅ THEO BÀI BÁO: Reward dựa trên xu hướng VÀ lợi nhuận
        portfolio_value = self.balance + self.holdings * current_price
        next_portfolio_value = self.balance + self.holdings * next_price
        
        # Lấy xu hướng từ data (theo bài báo: dựa trên SMA hoặc giá tương lai)
        current_trend = self.data.iloc[self.current_step].get('trend', 0)  # -1: giảm, 0: sideways, 1: tăng
        
        # Tính lợi nhuận (nếu có trade)
        trade_profit_pct = 0.0
        if transaction_executed and self.entry_price > 0 and action == 1:  # Sell executed
            trade_profit_pct = (current_price - self.entry_price) / self.entry_price
        elif self.holdings > 0 and self.entry_price > 0:  # Holding position
            trade_profit_pct = (current_price - self.entry_price) / self.entry_price
        
        # ✅ THEO BÀI BÁO: Kiểm tra hành động có đúng xu hướng không
        action_correct_trend = False
        
        if action == 0:  # Buy
            # ✅ Đúng xu hướng: Xu hướng tăng (trend > 0) → Mua là đúng
            action_correct_trend = (current_trend > 0)
        elif action == 1:  # Sell
            # ✅ Đúng xu hướng: Xu hướng giảm (trend < 0) → Bán là đúng
            action_correct_trend = (current_trend < 0)
        elif action == 2:  # Hold
            # ✅ Giữ: Luôn được xem là OK (theo Hình 4)
            action_correct_trend = True
        
        # ✅ THEO BÀI BÁO: Reward dựa trên xu hướng
        # "Phần thưởng nếu hành động thực hiện đúng chiều xu hướng sẽ được cho là luôn dương"
        # "Ngược lại, nếu không đúng theo xu hướng thì hàm phần thưởng sẽ âm"
        if action_correct_trend:
            # Đúng xu hướng → Reward dương (base reward)
            reward += 0.1  # Base reward cho đúng xu hướng
        else:
            # Sai xu hướng → Reward âm (penalty)
            reward -= 0.1  # Penalty cho sai xu hướng
        
        # ✅ THEO BÀI BÁO: Reward dựa trên lợi nhuận
        # "Nếu hành động sinh lợi nhuận, thì hàm phần thưởng sẽ có giá trị dương"
        # "khi sinh lỗ sẽ có giá trị âm"
        if trade_profit_pct > 0:
            # Có lợi nhuận → Reward dương
            reward += trade_profit_pct * 10.0  # Scale lợi nhuận
        elif trade_profit_pct < 0:
            # Lỗ → Reward âm
            reward += trade_profit_pct * 10.0  # Scale loss (sẽ âm)
        
        # ✅ THEO BÀI BÁO: Portfolio change reward
        if portfolio_value > 0:
            portfolio_change = (next_portfolio_value - portfolio_value) / self.initial_balance
            reward += portfolio_change * 5.0  # Moderate scaling
        else:
            # Bankruptcy penalty
            reward = -100.0
        
        # 🆕 V6: MDD TRACKING ONLY (Paper Methodology - NO PENALTY, NO EMERGENCY STOP)
        # Update portfolio history
        self.portfolio_history.append(portfolio_value)
        
        # Update peak portfolio
        if portfolio_value > self.peak_portfolio:
            self.peak_portfolio = portfolio_value
        
        # Calculate current drawdown (FOR REPORTING ONLY)
        if self.peak_portfolio > 0:
            self.current_mdd = (self.peak_portfolio - portfolio_value) / self.peak_portfolio
            
            # Track max MDD
            if self.current_mdd > self.max_mdd:
                self.max_mdd = self.current_mdd
        
        # 📊 MDD is tracked for evaluation ONLY - no penalties, no emergency stops
        # Paper accepts MDD 75-99% as normal behavior
        # Focus: Maximize profit, accept high MDD
        
        early_stopped = False
        early_stop_reason = None
        
        self.current_step += 1
        self.last_action = action  # ✅ Lưu action
        
        if done or self.current_step >= len(self.data):
            done = True
        
        next_state = self.get_state() if not done else self.get_state(len(self.data) - 1)
        
        # 🚀 V4 FIX: Final reward adjustment - BIG bonus for profitable episodes
        if done:
            final_profit_pct = (portfolio_value - self.initial_balance) / self.initial_balance
            
            if final_profit_pct > 0:
                # HUGE bonus for profitable episodes (100x scaling!)
                reward += final_profit_pct * 500.0  # 10% profit → +50 final bonus!
            # No penalty for losses - already reflected in trade rewards
        
        # ✅ V3: Build info dict
        info = {
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'holdings': self.holdings,
            'current_price': current_price,
            'action': action,  # ✅ Log action
            'transaction_executed': transaction_executed,  # ✅ Log transaction
            'consecutive_holds': self.consecutive_holds,  # ✅ Log holds
            'episode_ended_early': early_stopped or getattr(self, 'episode_ended_early', False),
            'early_stopped': early_stopped,  # ✅ Explicit flag
            'early_stop_reason': early_stop_reason,  # ✅ Reason nếu có
            'final_profit_pct': (portfolio_value - self.initial_balance) / self.initial_balance if done else None,
            # 🆕 V5: MDD & Annualized Return (Paper Metrics)
            'current_mdd': self.current_mdd * 100,  # Percentage
            'max_mdd': self.max_mdd * 100,  # Worst MDD in episode
            'peak_portfolio': self.peak_portfolio
        }
        
        # 🆕 V5: Calculate Annualized Return (if episode done)
        if done:
            total_return = (portfolio_value - self.initial_balance) / self.initial_balance
            episode_steps = self.current_step - self.episode_start_time
            # Assuming 1h candles: steps to years
            episode_hours = episode_steps
            episode_years = episode_hours / (24 * 365.25)
            
            if episode_years > 0 and total_return > -1:  # Avoid division by zero and invalid returns
                annualized_return = ((1 + total_return) ** (1 / episode_years)) - 1
                info['annualized_return'] = annualized_return * 100  # Percentage
            else:
                info['annualized_return'] = 0.0
        
        return next_state, reward, done, info
    
    def get_portfolio_value(self, price: float = None) -> float:
        """
        Tính giá trị portfolio hiện tại
        
        Args:
            price: Giá để tính (nếu None thì dùng giá hiện tại)
        
        Returns:
            Giá trị portfolio
        """
        if price is None:
            if self.current_step < len(self.data):
                price = self.data.iloc[self.current_step]['close']
            else:
                price = self.data.iloc[-1]['close']
        
        return self.balance + self.holdings * price
    
    def __len__(self):
        """Số lượng steps trong môi trường"""
        return len(self.data)


if __name__ == "__main__":
    # Test MDP
    print("=== Test Trading MDP ===\n")
    
    # Tạo dữ liệu mẫu
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    prices = 100 + np.cumsum(np.random.randn(100) * 2)
    
    df = pd.DataFrame({
        'open': prices + np.random.randn(100) * 0.5,
        'high': prices + np.abs(np.random.randn(100) * 0.5),
        'low': prices - np.abs(np.random.randn(100) * 0.5),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Thêm indicators
    from indicators import TechnicalIndicators
    df = TechnicalIndicators.add_all_indicators(df)
    
    # Tạo MDP
    mdp = TradingMDP(df, initial_balance=10000)
    
    # Test một episode
    state = mdp.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"Initial state: {state}")
    
    # Chạy một vài steps
    for i in range(5):
        action = np.random.randint(0, 3)  # Random action
        next_state, reward, done, info = mdp.step(action)
        print(f"\nStep {i+1}:")
        print(f"  Action: {action} ({['Hold', 'Buy', 'Sell'][action]})")
        print(f"  Reward: {reward:.4f}")
        print(f"  Portfolio Value: ${info['portfolio_value']:.2f}")
        print(f"  Done: {done}")
        
        if done:
            break

