from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import re
import json
from typing import Dict, List, Tuple, Optional
import optuna
import logging
import os
from src.visualization.optimization import (
    plot_optimization_progress,
    plot_parameter_importance,
    plot_timeframe_analysis,
    plot_parameter_space
)

logger = logging.getLogger(__name__)

class OptimizationAnalyzer:
    
    def __init__(self, log_file: str = None, study: optuna.Study = None):
        self.log_file = log_file
        self.study = study
        self.trials_df = None
        
        if study is not None:
            self.trials_df = study.trials_dataframe()
        elif log_file is not None:
            self.trials_df = self.parse_optimization_logs(log_file)
    
    def parse_optimization_logs(self, log_file: str) -> pd.DataFrame:
        log_data = []
        pattern = r"Trial (\d+), Timeframe: ([\w]+), BB\[(\d+), ([\d\.]+)\], RSI\[(\d+), (\d+), (\d+)\], ATR\[(\d+)\], TP/SL\[([\d\.]+)/([\d\.]+)\], Trades: (\d+), Win Rate: ([\d\.]+), Sharpe: ([\d\.\-]+), MDD: ([\d\.]+), Score: ([\d\.\-]+)"
        
        with open(log_file, "r") as f:
            for line in f:
                if "Trial" in line:
                    match = re.search(pattern, line)
                    if match:
                        log_data.append({
                            'trial': int(match.group(1)),
                            'timeframe': match.group(2),
                            'bb_window': int(match.group(3)),
                            'bb_std': float(match.group(4)),
                            'rsi_period': int(match.group(5)),
                            'rsi_lower': int(match.group(6)),
                            'rsi_upper': int(match.group(7)),
                            'atr_period': int(match.group(8)),
                            'take_profit_mult': float(match.group(9)),
                            'stop_loss_mult': float(match.group(10)),
                            'total_trades': int(match.group(11)),
                            'win_rate': float(match.group(12)),
                            'sharpe': float(match.group(13)),
                            'max_drawdown': float(match.group(14)),
                            'score': float(match.group(15))
                        })
        
        return pd.DataFrame(log_data)
    
    def plot_optimization_progress(self, save_path: str = None) -> plt.Figure:
        if self.trials_df is None:
            raise ValueError("No trials data available")
        
        return plot_optimization_progress(self.trials_df, save_path)
    
    def plot_parameter_importance(self, save_path: str = None) -> plt.Figure:
        if self.trials_df is None:
            raise ValueError("No trials data available")
        
        return plot_parameter_importance(self.trials_df, save_path)
    
    def plot_timeframe_analysis(self, save_dir: str = None) -> Tuple[plt.Figure, pd.DataFrame]:
        if self.trials_df is None:
            raise ValueError("No trials data available")
        
        save_path = os.path.join(save_dir, 'timeframe_comparison.png') if save_dir else None
        fig, timeframe_analysis = plot_timeframe_analysis(self.trials_df, save_path)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            timeframe_analysis.to_csv(os.path.join(save_dir, 'timeframe_analysis.csv'))
            logger.info(f"Timeframe analysis saved to {save_dir}")
        
        return fig, timeframe_analysis
    
    def plot_parameter_space(self, x_param: str, y_param: str, save_path: str = None) -> plt.Figure:
        if self.trials_df is None:
            raise ValueError("No trials data available")
        
        return plot_parameter_space(self.trials_df, x_param, y_param, save_path)
    
    def get_top_parameters(self, n: int = 5) -> pd.DataFrame:
        if self.trials_df is None:
            raise ValueError("No trials data available")
        
        param_columns = ['timeframe', 'bb_window', 'bb_std', 'rsi_period', 
                        'rsi_lower', 'rsi_upper', 'atr_period', 
                        'take_profit_mult', 'stop_loss_mult']
        
        metric_columns = ['sharpe', 'total_trades', 'win_rate', 'score']
        
        top_params = self.trials_df.nlargest(n, 'score')[param_columns + metric_columns]
        
        return top_params
    
    def generate_summary_report(self, save_dir: str = None) -> Dict:
        if self.trials_df is None:
            raise ValueError("No trials data available")
        
        best_idx = self.trials_df['score'].idxmax()
        best_trial = self.trials_df.iloc[best_idx]
        
        summary = {
            'optimization_summary': {
                'total_trials': len(self.trials_df),
                'best_score': float(best_trial['score']),
                'best_trial_number': int(best_trial['trial'])
            },
            'best_parameters': {
                'timeframe': best_trial['timeframe'],
                'bb_window': int(best_trial['bb_window']),
                'bb_std': float(best_trial['bb_std']),
                'rsi_period': int(best_trial['rsi_period']),
                'rsi_lower': int(best_trial['rsi_lower']),
                'rsi_upper': int(best_trial['rsi_upper']),
                'atr_period': int(best_trial['atr_period']),
                'take_profit_mult': float(best_trial['take_profit_mult']),
                'stop_loss_mult': float(best_trial['stop_loss_mult'])
            },
            'best_performance': {
                'total_trades': int(best_trial['total_trades']),
                'win_rate': float(best_trial['win_rate']),
                'sharpe_ratio': float(best_trial['sharpe']),
                'max_drawdown': float(best_trial['max_drawdown'])
            }
        }
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, 'optimization_summary.json'), 'w') as f:
                json.dump(summary, f, indent=4)
            logger.info(f"Optimization summary saved to {save_dir}")
        
        return summary