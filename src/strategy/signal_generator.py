import pandas as pd
import numpy as np
from datetime import time

class SignalGenerator:

    def __init__(self, rsi_lower=30, rsi_upper=70, trading_start=time(9, 15), trading_end=time(14, 30)):
        self.rsi_lower = rsi_lower
        self.rsi_upper = rsi_upper
        
        if isinstance(trading_start, str):
            hours, minutes = map(int, trading_start.split(':'))
            self.trading_start = time(hours, minutes)
        else:
            self.trading_start = trading_start
            
        if isinstance(trading_end, str):
            hours, minutes = map(int, trading_end.split(':'))
            self.trading_end = time(hours, minutes)
        else:
            self.trading_end = trading_end
    
    def generate_signals(self, df):
        result_df = df.copy()
        
        result_df['buy_signal'] = 0
        result_df['sell_signal'] = 0
        
        result_df['price_below_lower_bb'] = result_df['close'] < result_df['lower_band']
        result_df['price_above_upper_bb'] = result_df['close'] > result_df['upper_band']
        
        result_df['cross_above_lower_bb'] = (
            (result_df['close'] >= result_df['lower_band']) & 
            (result_df['close'].shift(1) < result_df['lower_band'].shift(1))
        )
        
        result_df['cross_below_upper_bb'] = (
            (result_df['close'] <= result_df['upper_band']) & 
            (result_df['close'].shift(1) > result_df['upper_band'].shift(1))
        )
        
        result_df['buy_signal'] = (
            (result_df['cross_above_lower_bb']) & 
            (result_df['rsi'] < self.rsi_lower) &
            self._is_within_trading_hours(result_df['datetime'])
        )
        
        result_df['sell_signal'] = (
            (result_df['cross_below_upper_bb']) & 
            (result_df['rsi'] > self.rsi_upper) &
            self._is_within_trading_hours(result_df['datetime'])
        )
        
        result_df['buy_signal'] = result_df['buy_signal'].astype(int)
        result_df['sell_signal'] = result_df['sell_signal'].astype(int)
        
        return result_df
    
    def _is_within_trading_hours(self, datetimes):
        return (datetimes.dt.time >= self.trading_start) & (datetimes.dt.time <= self.trading_end)