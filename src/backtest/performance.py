import numpy as np
import pandas as pd

class PerformanceMetrics:
    def __init__(self, trades, portfolio_history, risk_free_rate=0.03):
        self.trades = trades if not isinstance(trades, type(None)) else pd.DataFrame()
        self.portfolio_history = portfolio_history if portfolio_history is not None else [100000]
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = 252

    def win_rate(self):
        if self.trades.empty or 'profit' not in self.trades.columns:
            return 0
        return (self.trades['profit'] > 0).mean()

    def profit_factor(self):
        if self.trades.empty or 'profit' not in self.trades.columns:
            return 0
        gross_profit = self.trades[self.trades['profit'] > 0]['profit'].sum()
        gross_loss = abs(self.trades[self.trades['profit'] < 0]['profit'].sum())
        return gross_profit / gross_loss if gross_loss != 0 else float('inf')

    def sharpe_ratio(self):
        if len(self.portfolio_history) <= 1:
            return 0
            
        daily_returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
        
        if len(daily_returns) == 0 or np.std(daily_returns) == 0:
            return 0
            
        annual_std = np.sqrt(self.trading_days_per_year) * np.std(daily_returns)
        annual_return = self.trading_days_per_year * np.mean(daily_returns) - self.risk_free_rate
        
        return annual_return / annual_std if annual_std != 0 else 0

    def maximum_drawdown(self):
        if len(self.portfolio_history) <= 1:
            return 0
            
        peak = np.maximum.accumulate(self.portfolio_history)
        drawdown = (self.portfolio_history - peak) / peak
        return np.min(drawdown)

    def average_trade_return(self):
        if self.trades.empty or 'profit' not in self.trades.columns:
            return 0
        return self.trades['profit'].mean()

    def total_return(self):
        if len(self.portfolio_history) <= 1:
            return 0
        return (self.portfolio_history[-1] - self.portfolio_history[0]) / self.portfolio_history[0]

    def average_win(self):
        if self.trades.empty or 'profit' not in self.trades.columns:
            return 0
        winning_trades = self.trades[self.trades['profit'] > 0]
        if len(winning_trades) == 0:
            return 0
        return winning_trades['profit'].mean()

    def average_loss(self):
        if self.trades.empty or 'profit' not in self.trades.columns:
            return 0
        losing_trades = self.trades[self.trades['profit'] < 0]
        if len(losing_trades) == 0:
            return 0
        return losing_trades['profit'].mean()

    def expectancy(self):
        if self.trades.empty or 'profit' not in self.trades.columns:
            return 0
        win_rate = self.win_rate()
        avg_win = self.average_win()
        avg_loss = self.average_loss()
        
        if avg_win == 0 and avg_loss == 0:
            return 0
            
        return win_rate * avg_win + (1 - win_rate) * avg_loss

    def generate_report(self):
        return {
            'Win Rate': self.win_rate(),
            'Profit Factor': self.profit_factor(),
            'Sharpe Ratio': self.sharpe_ratio(),
            'Maximum Drawdown': self.maximum_drawdown(),
            'Average Trade Return': self.average_trade_return(),
            'Total Return': self.total_return(),
            'Total Trades': len(self.trades) if not self.trades.empty else 0,
            'Average Win': self.average_win(),
            'Average Loss': self.average_loss(),
            'Expectancy': self.expectancy()
        }