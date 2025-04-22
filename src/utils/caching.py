import os
import pickle
import hashlib
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, cache_dir="data_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def get_cache_path(self, data_type, identifier, suffix=None):
        filename = f"{data_type}_{identifier}"
        if suffix:
            filename = f"{filename}_{suffix}"
        return os.path.join(self.cache_dir, f"{filename}.pkl")
        
    def save_to_cache(self, data, cache_path):
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Data saved to cache: {cache_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving data to cache: {e}")
            return False
            
    def load_from_cache(self, cache_path):
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"Data loaded from cache: {cache_path}")
                return data
            return None
        except Exception as e:
            logger.error(f"Error loading data from cache: {e}")
            return None
            
    def generate_data_hash(self, df):
        if df.empty:
            return None
            
        if 'datetime' in df.columns:
            df_hash = hashlib.md5(pd.util.hash_pandas_object(df[['datetime', 'price']]).values).hexdigest()
        else:
            df_hash = hashlib.md5(pd.util.hash_pandas_object(df.index.tolist() + df['price'].tolist()).values).hexdigest()
            
        return df_hash