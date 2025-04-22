import numpy as np
import pandas as pd

class RiskManager:
    def __init__(self, max_daily_loss=2, max_drawdown=0.05, risk_per_trade=0.005):
        self.max_daily_loss = float(max_daily_loss)
        self.max_drawdown = float(max_drawdown)
        self.risk_per_trade = float(risk_per_trade)

    def calculate_position_size(self, account_balance, atr, entry_price):
        account_balance = float(account_balance)
        atr = float(atr)
        entry_price = float(entry_price)

        if atr == 0 or np.isnan(atr):
            atr = max(entry_price * 0.01, 1)
        
        risk_amount = account_balance * self.risk_per_trade
        stop_loss_distance = atr
        
        position_size = max(1, int(risk_amount / stop_loss_distance))
        return position_size

    def check_daily_loss_limit(self, current_daily_loss):
        return float(current_daily_loss) <= self.max_daily_loss

    def check_drawdown_limit(self, current_drawdown):
        return float(current_drawdown) <= self.max_drawdown

    def calculate_portfolio_drawdown(self, equity_curve):
        equity_curve = [float(x) for x in equity_curve]
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return drawdown

    def validate_trade(self, trade, portfolio_state):
        trade_risk = abs(float(trade['entry_price']) - float(trade.get('stop_loss', trade['entry_price'])))
        current_balance = float(portfolio_state.get('current_balance', 100000))
        trade_risk_percentage = trade_risk / current_balance
        return trade_risk_percentage <= self.risk_per_trade