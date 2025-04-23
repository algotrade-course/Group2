import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import logging
import os
from src.utils.caching import CacheManager

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, trading_start=time(9, 0), trading_end=time(15, 0), cache_dir=None):
        self.trading_start = trading_start
        self.trading_end = trading_end
        
        if cache_dir is None:
            # Use project root-based path by default
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
            self.cache_dir = os.path.join(project_root, 'data_cache', 'ohlcv')
        else:
            self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.cache_manager = CacheManager(self.cache_dir)
    
    def resample_to_ohlcv(self, df, timeframe='15min'):
        if df.empty:
            logger.warning("Empty DataFrame provided for resampling")
            return pd.DataFrame()
        
        data_hash = self.cache_manager.generate_data_hash(df)
        if data_hash:
            if 'datetime' in df.columns:
                start_date = df['datetime'].min().strftime('%Y%m%d')
                end_date = df['datetime'].max().strftime('%Y%m%d')
            else:
                start_date = df.index.min().strftime('%Y%m%d') 
                end_date = df.index.max().strftime('%Y%m%d')
                
            ticker = ""
            if 'tickersymbol' in df.columns:
                ticker = "_" + df['tickersymbol'].iloc[0] if not df.empty else ""
                
            cache_path = self.cache_manager.get_cache_path("ohlcv", f"{start_date}_{end_date}{ticker}_{timeframe}", data_hash)
            cached_data = self.cache_manager.load_from_cache(cache_path)
            
            if cached_data is not None:
                return cached_data
        
        # Process data if not in cache
        if 'datetime' in df.columns:
            df = df.copy()
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        
        trading_hours_mask = self.filter_trading_hours(df.index)
        df_trading = df[trading_hours_mask]
        
        if df_trading.empty:
            logger.warning("No data within trading hours")
            return pd.DataFrame()
        
        has_quantity = 'quantity' in df.columns
        
        ohlcv = pd.DataFrame()
        ohlcv['open'] = df_trading['price'].resample(timeframe).first()
        ohlcv['high'] = df_trading['price'].resample(timeframe).max()
        ohlcv['low'] = df_trading['price'].resample(timeframe).min()
        ohlcv['close'] = df_trading['price'].resample(timeframe).last()
        
        if has_quantity:
            ohlcv['volume'] = df_trading['quantity'].resample(timeframe).sum()
        else:
            ohlcv['volume'] = df_trading['price'].resample(timeframe).count()
        
        ohlcv = ohlcv.dropna(subset=['open', 'high', 'low', 'close'])
        
        if 'tickersymbol' in df.columns:
            ticker_groups = df_trading['tickersymbol'].resample(timeframe).apply(
                lambda x: x.mode()[0] if not x.empty else np.nan
            )
            ohlcv['tickersymbol'] = ticker_groups
        
        logger.info(f"Resampled to {len(ohlcv)} {timeframe} candles")
        
        # Save to cache if we have a valid cache path
        if data_hash and cache_path:
            self.cache_manager.save_to_cache(ohlcv, cache_path)
        
        return ohlcv
    
    def filter_trading_hours(self, datetimes):
        times = pd.Series([dt.time() for dt in datetimes], index=datetimes)
        return (times >= self.trading_start) & (times <= self.trading_end)
    
    def prepare_data_for_backtest(self, ohlcv_df, add_indicators=False):
        if ohlcv_df.empty:
            return ohlcv_df
        
        df = ohlcv_df.copy()
        
        if isinstance(df.index, pd.DatetimeIndex):
            df.reset_index(inplace=True)
            
        if 'datetime' not in df.columns and 'date' in df.columns:
            df.rename(columns={'date': 'datetime'}, inplace=True)
        
        df['date'] = df['datetime'].dt.date
        df['time'] = df['datetime'].dt.time
        
        df['trading_session'] = df['time'].apply(
            lambda t: (t >= self.trading_start) and (t <= self.trading_end)
        )
        
        if add_indicators:
            logger.info("Indicator calculation not implemented yet")
        
        return df
    
    def split_train_test(self, df, train_end_date, test_start_date=None):
        if df.empty:
            return df.copy(), df.copy()
            
        if isinstance(df.index, pd.DatetimeIndex) and 'datetime' not in df.columns:
            df = df.reset_index()
        
        train_end = pd.to_datetime(train_end_date)
        
        if test_start_date is None:
            test_start = train_end + pd.Timedelta(days=1)
        else:
            test_start = pd.to_datetime(test_start_date)
        
        train_df = df[df['datetime'] <= train_end].copy()
        test_df = df[df['datetime'] >= test_start].copy()
        
        logger.info(f"Split data into {len(train_df)} training samples and {len(test_df)} testing samples")
        
        return train_df, test_df