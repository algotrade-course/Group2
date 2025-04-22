from typing import Dict, List, Tuple
import logging
from matplotlib import pyplot as plt
import pandas as pd
from .base import save_figure, setup_plot_style

logger = logging.getLogger(__name__)

def plot_parameter_comparison(default_metrics: Dict, optimized_metrics: Dict, 
                            save_path: str = None) -> Tuple[plt.Figure, pd.DataFrame]:
    setup_plot_style()
    
    comparison_metrics = ['Win Rate', 'Profit Factor', 'Sharpe Ratio', 
                         'Maximum Drawdown', 'Total Return', 'Total Trades']
    
    comparison_df = pd.DataFrame({
        'Default Parameters': [default_metrics[m] for m in comparison_metrics],
        'Optimized Parameters': [optimized_metrics[m] for m in comparison_metrics]
    }, index=comparison_metrics)
    
    comparison_df['Improvement'] = comparison_df['Optimized Parameters'] - comparison_df['Default Parameters']
    comparison_df['Improvement %'] = comparison_df['Improvement'] / comparison_df['Default Parameters'].abs() * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    plot_metrics = ['Win Rate', 'Sharpe Ratio', 'Total Return']
    plot_df = comparison_df.loc[plot_metrics, ['Default Parameters', 'Optimized Parameters']]
    
    plot_df.plot(kind='bar', rot=0, ax=ax1)
    ax1.set_title('Performance Metrics Comparison', fontsize=16)
    ax1.set_ylabel('Value')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.4f')
    
    improvement_df = comparison_df.loc[:, ['Improvement %']]
    improvement_df.plot(kind='bar', ax=ax2, color='skyblue')
    ax2.set_title('Performance Improvement (%)', fontsize=16)
    ax2.set_ylabel('Improvement %')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.1f%%')
    
    plt.tight_layout()
    
    save_figure(fig, save_path)
    return fig, comparison_df

def plot_equity_curves_comparison(default_portfolio_history: List[float], 
                                optimized_portfolio_history: List[float],
                                initial_capital: float = 100000,
                                save_path: str = None) -> plt.Figure:
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(default_portfolio_history, label='Default Parameters', alpha=0.7, linewidth=2)
    ax.plot(optimized_portfolio_history, label='Optimized Parameters', alpha=0.7, linewidth=2)
    ax.axhline(y=initial_capital, color='r', linestyle='--', label='Initial Capital')
    
    ax.set_title('Equity Curves: Default vs Optimized Parameters', fontsize=16)
    ax.set_xlabel('Trade Number')
    ax.set_ylabel('Portfolio Value')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    final_default = default_portfolio_history[-1]
    final_optimized = optimized_portfolio_history[-1]
    
    ax.text(len(default_portfolio_history) - 1, final_default, 
            f'${final_default:.0f}', ha='right', va='bottom')
    ax.text(len(optimized_portfolio_history) - 1, final_optimized, 
            f'${final_optimized:.0f}', ha='right', va='bottom')
    
    save_figure(fig, save_path)
    return fig