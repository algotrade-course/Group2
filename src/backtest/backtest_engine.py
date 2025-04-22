import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from decimal import Decimal

class BacktestEngine:
    def __init__(self, initial_balance=100000, commission=0.001):
        self.initial_balance = float(initial_balance)
        self.commission = float(commission)

    def run_backtest(self, signals_df, risk_manager=None):
        signals_df = signals_df.copy()
        
        for col in ['open', 'high', 'low', 'close', 'atr']:
            if col in signals_df.columns:
                signals_df[col] = signals_df[col].astype(float)
        
        if 'buy_signal' not in signals_df.columns:
            signals_df['buy_signal'] = 0
        if 'sell_signal' not in signals_df.columns:
            signals_df['sell_signal'] = 0
        
        df = signals_df.copy()
        
        portfolio_history = []
        current_balance = self.initial_balance
        portfolio_history.append(current_balance)
        
        df['position'] = 0
        df['entry_price'] = np.nan
        df['exit_price'] = np.nan
        df['profit'] = 0.0
        df['current_balance'] = current_balance
        
        current_position = 0
        entry_price = 0.0
        entry_time = None
        entry_index = None
        entry_atr = 0.0
        
        trades_list = []
        trade_id = 0
        
        daily_positions = 0
        current_date = None
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            row_date = row['datetime'].date()
            row_time = row['datetime'].time()
            
            if current_date is None or row_date != current_date:
                current_date = row_date
                daily_positions = 0
            
            if current_position == 0:
                if row['buy_signal'] == 1 and daily_positions < 3 and row_time < time(14, 30):
                    current_position = 1
                    entry_price = float(row['close'])
                    entry_time = row['datetime']
                    entry_index = i
                    entry_atr = float(row['atr'])
                    trade_id += 1
                    daily_positions += 1
                    
                    df.at[i, 'position'] = 1
                    df.at[i, 'entry_price'] = entry_price
                    
                elif row['sell_signal'] == 1 and daily_positions < 3 and row_time < time(14, 30):
                    current_position = -1
                    entry_price = float(row['close'])
                    entry_time = row['datetime']
                    entry_index = i
                    entry_atr = float(row['atr'])
                    trade_id += 1
                    daily_positions += 1
                    
                    df.at[i, 'position'] = -1
                    df.at[i, 'entry_price'] = entry_price
            
            elif current_position != 0:
                df.at[i, 'position'] = current_position
                
                exit_signal = False
                exit_reason = ""
                exit_price = float(row['close'])
                
                if current_position == 1:
                    take_profit = entry_price + (entry_atr * 2)
                    stop_loss = entry_price - (entry_atr * 1)
                    
                    if float(row['high']) >= take_profit:
                        exit_signal = True
                        exit_reason = "take_profit"
                        exit_price = take_profit
                    elif float(row['low']) <= stop_loss:
                        exit_signal = True
                        exit_reason = "stop_loss"
                        exit_price = stop_loss
                    elif (row['datetime'] - entry_time) >= timedelta(hours=4):
                        exit_signal = True
                        exit_reason = "time_exit"
                    elif row_time >= time(14, 45):
                        exit_signal = True
                        exit_reason = "market_close"
                
                elif current_position == -1:
                    take_profit = entry_price - (entry_atr * 2)
                    stop_loss = entry_price + (entry_atr * 1)
                    
                    if float(row['low']) <= take_profit:
                        exit_signal = True
                        exit_reason = "take_profit"
                        exit_price = take_profit
                    elif float(row['high']) >= stop_loss:
                        exit_signal = True
                        exit_reason = "stop_loss"
                        exit_price = stop_loss
                    elif (row['datetime'] - entry_time) >= timedelta(hours=4):
                        exit_signal = True
                        exit_reason = "time_exit"
                    elif row_time >= time(14, 45):
                        exit_signal = True
                        exit_reason = "market_close"
                
                if exit_signal:
                    position_size = 1
                    if risk_manager:
                        try:
                            position_size = risk_manager.calculate_position_size(
                                current_balance, entry_atr, entry_price
                            )
                        except:
                            position_size = 1
                    
                    if current_position == 1:
                        profit = (exit_price - entry_price) * position_size
                    else:
                        profit = (entry_price - exit_price) * position_size
                    
                    commission_cost = abs(profit) * self.commission
                    net_profit = profit - commission_cost
                    
                    current_balance += net_profit
                    
                    df.at[i, 'exit_price'] = exit_price
                    df.at[i, 'profit'] = net_profit
                    
                    trades_list.append({
                        'trade_id': trade_id,
                        'type': 'buy' if current_position == 1 else 'sell',
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'exit_time': row['datetime'],
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'position_size': position_size,
                        'profit': net_profit,
                        'atr_at_entry': entry_atr
                    })
                    
                    current_position = 0
            
            df.at[i, 'current_balance'] = current_balance
            portfolio_history.append(current_balance)
        
        trades_df = pd.DataFrame(trades_list)
        
        backtest_results = {
            'trades': trades_df,
            'portfolio_history': portfolio_history,
            'final_balance': current_balance,
            'total_return': (current_balance - self.initial_balance) / self.initial_balance,
            'backtest_df': df
        }
        
        return backtest_results