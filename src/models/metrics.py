"""
Module tính toán các metrics hiệu suất trading
Trading performance metrics module
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


class TradingMetrics:
    """
    Class tính toán các metrics hiệu suất trading
    Bao gồm: Maximum Drawdown (MDD), Annualized Return, Sharpe Ratio, v.v.
    """
    
    @staticmethod
    def calculate_maximum_drawdown(portfolio_values: pd.Series) -> Tuple[float, pd.DataFrame]:
        """
        Tính Maximum Drawdown (MDD)
        
        MDD = max[(Peak - Trough) / Peak]
        
        Args:
            portfolio_values: Series giá trị portfolio theo thời gian
        
        Returns:
            (mdd, drawdown_df)
        """
        # Tính running maximum (peak)
        running_max = portfolio_values.expanding().max()
        
        # Tính drawdown
        drawdown = (portfolio_values - running_max) / running_max
        
        # Maximum drawdown
        mdd = drawdown.min()
        
        # Lấy thông tin về drawdown
        dd_df = pd.DataFrame({
            'portfolio_value': portfolio_values,
            'running_max': running_max,
            'drawdown': drawdown
        })
        
        return abs(mdd), dd_df
    
    @staticmethod
    def calculate_annualized_return(portfolio_values: pd.Series, periods_per_year: int = 252) -> float:
        """
        Tính Annualized Return
        
        Args:
            portfolio_values: Series giá trị portfolio
            periods_per_year: Số period trong 1 năm (252 cho daily, 365 cho hourly nếu 24/7)
        
        Returns:
            Annualized return
        """
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1.0
        num_periods = len(portfolio_values)
        
        if num_periods == 0:
            return 0.0
        
        years = num_periods / periods_per_year
        
        if years <= 0:
            return 0.0
        
        annualized_return = (1 + total_return) ** (1 / years) - 1
        
        return annualized_return
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, periods_per_year: int = 252, risk_free_rate: float = 0.02) -> float:
        """
        Tính Sharpe Ratio
        
        Sharpe Ratio = (Rp - Rf) / σp
        
        Args:
            returns: Series returns
            periods_per_year: Số period trong 1 năm
            risk_free_rate: Risk-free rate (annualized)
        
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0:
            return 0.0
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return 0.0
        
        # Annualize
        mean_annual = mean_return * periods_per_year
        std_annual = std_return * np.sqrt(periods_per_year)
        
        risk_free_per_period = risk_free_rate / periods_per_year
        
        sharpe = (mean_annual - risk_free_rate) / std_annual
        
        return sharpe
    
    @staticmethod
    def calculate_max_drawdown_duration(portfolio_values: pd.Series) -> Tuple[int, pd.Series]:
        """
        Tính thời gian từ peak đến recovery
        
        Args:
            portfolio_values: Series giá trị portfolio
        
        Returns:
            (max_duration, duration_series)
        """
        # Tính running maximum
        running_max = portfolio_values.expanding().max()
        
        # Tìm các vị trí đạt peak mới
        new_peaks = running_max != running_max.shift(1)
        
        # Tính duration từ mỗi peak
        duration = pd.Series(0, index=portfolio_values.index, dtype=int)
        peak_idx = None
        
        for i in range(len(portfolio_values)):
            if new_peaks.iloc[i]:
                peak_idx = i
            
            if peak_idx is not None:
                duration.iloc[i] = i - peak_idx
        
        max_duration = duration.max()
        
        return max_duration, duration
    
    @staticmethod
    def calculate_win_rate(trading_history: pd.DataFrame) -> float:
        """
        Tính win rate (tỷ lệ trade thắng)
        
        Args:
            trading_history: DataFrame chứa lịch sử giao dịch
        
        Returns:
            Win rate (0-1)
        """
        if 'action' not in trading_history.columns:
            return 0.0
        
        # Chỉ tính cho các trade (Buy hoặc Sell)
        trades = trading_history[trading_history['action'] != 0]
        
        if len(trades) < 2:
            return 0.0
        
        # Tính profit/loss cho mỗi trade
        profits = []
        
        for i in range(len(trades) - 1):
            buy_idx = None
            sell_idx = None
            
            if trades.iloc[i]['action'] == 1:  # Buy
                buy_idx = i
                # Tìm Sell tiếp theo
                for j in range(i + 1, len(trades)):
                    if trades.iloc[j]['action'] == 2:  # Sell
                        sell_idx = j
                        break
            
            if buy_idx is not None and sell_idx is not None:
                profit = trades.iloc[sell_idx]['portfolio_value'] - trades.iloc[buy_idx]['portfolio_value']
                profits.append(profit)
        
        if len(profits) == 0:
            return 0.0
        
        wins = sum(p > 0 for p in profits)
        win_rate = wins / len(profits)
        
        return win_rate
    
    @staticmethod
    def calculate_total_trades(trading_history: pd.DataFrame) -> Dict:
        """
        Tính tổng số trades
        
        Args:
            trading_history: DataFrame chứa lịch sử giao dịch
        
        Returns:
            Dict với thông tin trades
        """
        if 'action' not in trading_history.columns:
            return {'total': 0, 'buy': 0, 'sell': 0}
        
        trades = trading_history[trading_history['action'] != 0]
        
        return {
            'total': len(trades),
            'buy': len(trades[trades['action'] == 1]),
            'sell': len(trades[trades['action'] == 2])
        }
    
    @staticmethod
    def calculate_all_metrics(trading_history: pd.DataFrame, portfolio_values: pd.Series,
                            periods_per_year: int = 252) -> Dict:
        """
        Tính tất cả các metrics
        
        Args:
            trading_history: DataFrame lịch sử giao dịch
            portfolio_values: Series giá trị portfolio
            periods_per_year: Số period trong 1 năm
        
        Returns:
            Dict chứa tất cả metrics
        """
        metrics = {}
        
        # Basic returns
        initial_value = portfolio_values.iloc[0]
        final_value = portfolio_values.iloc[-1]
        total_return = (final_value / initial_value) - 1.0
        
        metrics['initial_value'] = initial_value
        metrics['final_value'] = final_value
        metrics['total_return'] = total_return
        metrics['total_return_pct'] = total_return * 100
        
        # Annualized return
        annualized_return = TradingMetrics.calculate_annualized_return(portfolio_values, periods_per_year)
        metrics['annualized_return'] = annualized_return
        metrics['annualized_return_pct'] = annualized_return * 100
        
        # Maximum Drawdown
        mdd, dd_df = TradingMetrics.calculate_maximum_drawdown(portfolio_values)
        metrics['max_drawdown'] = mdd
        metrics['max_drawdown_pct'] = mdd * 100
        
        # Max drawdown duration
        max_dd_duration, dd_duration = TradingMetrics.calculate_max_drawdown_duration(portfolio_values)
        metrics['max_drawdown_duration'] = max_dd_duration
        
        # Returns series
        returns = portfolio_values.pct_change().dropna()
        
        if len(returns) > 0:
            # Sharpe ratio
            sharpe = TradingMetrics.calculate_sharpe_ratio(returns, periods_per_year)
            metrics['sharpe_ratio'] = sharpe
            
            # Volatility
            volatility = returns.std() * np.sqrt(periods_per_year)
            metrics['annualized_volatility'] = volatility
            metrics['annualized_volatility_pct'] = volatility * 100
        
        # Trading statistics
        metrics['total_trades'] = TradingMetrics.calculate_total_trades(trading_history)
        metrics['win_rate'] = TradingMetrics.calculate_win_rate(trading_history)
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict):
        """
        In các metrics ra màn hình
        
        Args:
            metrics: Dict chứa metrics
        """
        print("\n" + "="*60)
        print("TRAĐING PERFORMANCE METRICS")
        print("="*60)
        
        print(f"\nPortfolio Performance:")
        print(f"  Initial Value:      ${metrics['initial_value']:,.2f}")
        print(f"  Final Value:        ${metrics['final_value']:,.2f}")
        print(f"  Total Return:       {metrics['total_return_pct']:.2f}%")
        print(f"  Annualized Return:  {metrics['annualized_return_pct']:.2f}%")
        
        print(f"\nRisk Metrics:")
        print(f"  Max Drawdown:       {metrics['max_drawdown_pct']:.2f}%")
        print(f"  Max DD Duration:    {metrics['max_drawdown_duration']} periods")
        if 'annualized_volatility' in metrics:
            print(f"  Volatility:         {metrics['annualized_volatility_pct']:.2f}%")
        
        print(f"\nRisk-Adjusted Metrics:")
        if 'sharpe_ratio' in metrics:
            print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:.2f}")
        
        print(f"\nTrading Statistics:")
        print(f"  Total Trades:       {metrics['total_trades']['total']}")
        print(f"    - Buy Orders:     {metrics['total_trades']['buy']}")
        print(f"    - Sell Orders:    {metrics['total_trades']['sell']}")
        print(f"  Win Rate:           {metrics['win_rate']*100:.2f}%")
        
        print("="*60 + "\n")


if __name__ == "__main__":
    # Test metrics
    print("=== Test Trading Metrics ===\n")
    
    # Tạo dữ liệu mẫu
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Tạo portfolio values mô phỏng
    returns = np.random.randn(100) * 0.02
    portfolio_values = 10000 * (1 + pd.Series(returns)).cumprod()
    
    # Tạo trading history mẫu
    trading_history = pd.DataFrame({
        'step': range(100),
        'action': np.random.choice([0, 1, 2], 100, p=[0.7, 0.15, 0.15]),
        'portfolio_value': portfolio_values
    })
    
    # Tính metrics
    metrics = TradingMetrics.calculate_all_metrics(trading_history, portfolio_values, periods_per_year=252)
    
    # In kết quả
    TradingMetrics.print_metrics(metrics)

