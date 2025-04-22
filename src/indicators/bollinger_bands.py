import numpy as np
import pandas as pd

def calculate_bollinger_bands(prices, window=20, num_std=1.8):
    if isinstance(prices, pd.DataFrame) and 'close' in prices.columns:
        prices = prices['close']
    
    middle_band = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    return upper_band, middle_band, lower_band


def add_bollinger_bands(df, price_col='close', window=20, num_std=1.8):
    df = df.copy()
    
    upper, middle, lower = calculate_bollinger_bands(
        df[price_col], window, num_std
    )
    
    df['upper_band'] = upper
    df['middle_band'] = middle
    df['lower_band'] = lower
    
    return df


def is_price_above_lower_band(df):
    return df['close'] > df['lower_band']


def is_price_below_upper_band(df):
    return df['close'] < df['upper_band']


def detect_band_crossovers(df):
    df = df.copy()
    
    df['above_lower_band'] = df['close'] > df['lower_band']
    df['below_upper_band'] = df['close'] < df['upper_band']
    
    df['prev_above_lower_band'] = df['above_lower_band'].shift(1)
    df['prev_below_upper_band'] = df['below_upper_band'].shift(1)
    
    df['cross_above_lower'] = (df['above_lower_band']) & (~df['prev_above_lower_band'])
    df['cross_below_upper'] = (df['below_upper_band']) & (~df['prev_below_upper_band'])
    
    return df
