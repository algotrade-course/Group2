import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

os.makedirs('results', exist_ok=True)
os.makedirs('data_cache', exist_ok=True)

sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)

from optimization.config_loader import ConfigLoader
from pipeline import TradingPipeline
from src.visualization.backtest import plot_performance_metrics, plot_equity_curve, plot_trade_analysis

def run_backtest(start_date='2024-01-01', end_date='2024-06-01', save_plots=True):
    config_loader = ConfigLoader()
    config = config_loader.get_config()
    params = config_loader.get_default_parameters()
    
    if 'parameters' not in config:
        config['parameters'] = {}
    config['parameters'].update({
        'bb_window': params.get("bb_window", 20),
        'bb_std': params.get("bb_std", 1.8),
        'rsi_period': params.get("rsi_period", 13),
        'rsi_lower': params.get("rsi_lower", 30),
        'rsi_upper': params.get("rsi_upper", 70),
        'atr_period': params.get("atr_period", 14),
        'take_profit_mult': params.get("take_profit_mult", 4.0),
        'stop_loss_mult': params.get("stop_loss_mult", 1.0),
        'default_timeframe': params.get("default_timeframe", "15min")
    })
    
    pipeline = TradingPipeline(config)
    
    timeframe = config['parameters'].get('default_timeframe', '15min')
    results, signals_df = pipeline.run_backtest(start_date, end_date, timeframe)
    
    metrics = {}
    buy_signals = 0
    sell_signals = 0
    
    if signals_df is not None:
        buy_signals = signals_df['buy_signal'].sum()
        sell_signals = signals_df['sell_signal'].sum()
    
    if results is not None and 'trades' in results and not results['trades'].empty:
        if save_plots:
            if 'portfolio_history' in results and len(results['portfolio_history']) > 1:
                fig = plot_equity_curve(results['portfolio_history'], save_path='results/equity_curve.png')
                plt.close(fig)
            
            if 'metrics' in results:
                metrics = results['metrics']
            else:
                from backtest.performance import PerformanceMetrics
                performance = PerformanceMetrics(
                    results['trades'], 
                    results['portfolio_history']
                )
                metrics = performance.generate_report()
            
            fig = plot_performance_metrics(metrics)
            plt.savefig('results/performance_metrics.png')
            plt.close(fig)
            
            trades_df = results['trades']
            fig_trades, fig_exit, profit_by_exit = plot_trade_analysis(trades_df)
            
            if fig_trades:
                plt.figure(fig_trades.number)
                plt.savefig('results/trade_distribution.png')
                plt.close(fig_trades)
            
            if fig_exit:
                plt.figure(fig_exit.number)
                plt.savefig('results/exit_analysis.png')
                plt.close(fig_exit)
        
        trades_df = results['trades']
        trades_df.to_csv('results/trades.csv', index=False)
        print(f"Saved {len(trades_df)} trades to results/trades.csv")
        
        if metrics:
            metrics_dict = {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in metrics.items()}
            
            result_data = {
                "parameters": params,
                "test_period": {
                    "start_date": start_date,
                    "end_date": end_date
                },
                "metrics": metrics_dict,
                "summary": {
                    "initial_balance": 100000,
                    "final_balance": float(results['final_balance']),
                    "total_trades": len(trades_df),
                    "buy_signals": int(buy_signals),
                    "sell_signals": int(sell_signals),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            
            with open('results/backtest_results.json', 'w') as f:
                json.dump(result_data, f, indent=4)
            
            print("Saved performance metrics to results/backtest_results.json")
    else:
        print("No results saved - no trades were executed")
    
    return results, metrics

if __name__ == "__main__":
    backtest_results, metrics = run_backtest()