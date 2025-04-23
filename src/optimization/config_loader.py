import os
import json
import logging
from datetime import datetime, time

class ConfigLoader:
    def __init__(self, config_path=None):
        if config_path is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
            possible_paths = [
                os.path.join(project_root, 'config/strategy_config.json'),
                os.path.join(os.path.dirname(project_root), 'config/strategy_config.json'),
                'config/strategy_config.json',
                '../config/strategy_config.json',
                '../../config/strategy_config.json',
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
        
        self.config_path = config_path
        self.logger = logging.getLogger("config_loader")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.config = self._load_config()
    
    def _load_config(self):
        try:
            if self.config_path and os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                self.logger.info(f"Loaded strategy configuration from {self.config_path}")
                return config
            else:
                self.logger.warning("No configuration file found, using default configuration")
                return self._get_default_config()
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            self.logger.info("Using default configuration instead")
            return self._get_default_config()
    
    def _get_default_config(self):
        return {
            "strategy": {
                "name": "Intraday BB-RSI Strategy",
                "description": "VN30F intraday strategy using Bollinger Bands and RSI for mean-reversion signals"
            },
            "data": {
                "timeframe": "15min",
                "trading_hours": {
                    "start": "09:15",
                    "end": "14:30"
                },
                "market_close": "14:45"
            },
            "position_management": {
                "max_positions_per_day": 3,
                "position_size": "fixed",
                "risk_per_trade": 0.005
            },
            "default_parameters": {
                "bb_window": 20,
                "bb_std": 1.8,
                "rsi_period": 13,
                "rsi_lower": 30,
                "rsi_upper": 70,
                "atr_period": 14,
                "take_profit_mult": 2.0,
                "stop_loss_mult": 1.0,
                "trailing_trigger": 1.5,
                "trailing_atr": 0.5
            },
            "backtest": {
                "initial_capital": 100000,
                "commission": 0.001,
                "slippage": 0.0005
            }
        }
    
    def get_config(self):
        return self.config
    
    def get_strategy_name(self):
        return self.config.get("strategy", {}).get("name", "Intraday BB-RSI Strategy")
    
    def get_timeframe(self):
        return self.config.get("data", {}).get("timeframe", "15min")
    
    def get_trading_hours(self):
        hours = self.config.get("data", {}).get("trading_hours", {})
        start_str = hours.get("start", "09:15")
        end_str = hours.get("end", "14:30")
        
        start_hour, start_min = map(int, start_str.split(":"))
        end_hour, end_min = map(int, end_str.split(":"))
        
        return time(start_hour, start_min), time(end_hour, end_min)
    
    def get_market_close_time(self):
        close_str = self.config.get("data", {}).get("market_close", "14:45")
        close_hour, close_min = map(int, close_str.split(":"))
        return time(close_hour, close_min)
    
    def get_default_parameters(self):
        return self.config.get("default_parameters", {})
    
    def get_backtest_settings(self):
        return self.config.get("backtest", {})
    
    def get_optimization_settings(self):
        return self.config.get("optimization", {})
    
    def get_position_management(self):
        return self.config.get("position_management", {})
    
    def save_config(self, config):
        if self.config_path is None:
            self.config_path = 'config/strategy_config.json'
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            self.logger.info(f"Configuration saved to {self.config_path}")
            self.config = config
            return True
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return False

if __name__ == "__main__":
    loader = ConfigLoader()
    config = loader.get_config()
    print("Strategy Configuration:")
    print(f"Strategy Name: {loader.get_strategy_name()}")
    print(f"Timeframe: {loader.get_timeframe()}")
    print(f"Trading Hours: {loader.get_trading_hours()}")
    print(f"Default Parameters: {loader.get_default_parameters()}")