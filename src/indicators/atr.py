import pandas as pd
import numpy as np

def calculate_atr(df, period=14):
    df = df.copy()
    
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    
    return atr

def add_atr(df, period=14):
    df = df.copy()
    df['atr'] = calculate_atr(df, period)
    return df

def calculate_atr_bands(df, multiplier=2):
    df = df.copy()
    
    if 'atr' not in df.columns:
        df = add_atr(df)
    
    df['atr_upper'] = df['close'] + (df['atr'] * multiplier)
    df['atr_lower'] = df['close'] - (df['atr'] * multiplier)
    
    return df

def calculate_chandelier_exit(df, period=14, multiplier=3):
    df = df.copy()
    
    if 'atr' not in df.columns:
        df = add_atr(df, period)
    
    df['chandelier_long'] = df['high'].rolling(window=period).max() - (df['atr'] * multiplier)
    df['chandelier_short'] = df['low'].rolling(window=period).min() + (df['atr'] * multiplier)
    
    return df