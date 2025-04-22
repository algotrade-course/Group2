import pandas as pd
import numpy as np

def calculate_rsi(prices, period=13):
    if isinstance(prices, pd.DataFrame) and 'close' in prices.columns:
        prices = prices['close']
    
    delta = prices.diff()
    
    gain = delta.copy()
    gain[gain < 0] = 0
    
    loss = -delta.copy()
    loss[loss < 0] = 0
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def add_rsi(df, price_col='close', period=13):
    df = df.copy()
    df['rsi'] = calculate_rsi(df[price_col], period)
    return df

def is_rsi_oversold(df, threshold=30):
    return df['rsi'] < threshold

def is_rsi_overbought(df, threshold=70):
    return df['rsi'] > threshold
