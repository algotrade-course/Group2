import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from .base import save_figure, setup_plot_style

logger = logging.getLogger(__name__)

def plot_optimization_progress(trials_df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(trials_df['trial'], trials_df['score'], 'o-', alpha=0.6)
    
    best_score = trials_df['score'].max()
    ax.axhline(y=best_score, color='green', linestyle='--', 
               label=f'Best score: {best_score:.4f}')
    
    ax.set_title('Optimization Progress', fontsize=16)
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Objective Score')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    save_figure(fig, save_path)
    return fig

def plot_parameter_importance(trials_df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    setup_plot_style()
    
    timeframe_map = {tf: i for i, tf in enumerate(trials_df['timeframe'].unique())}
    df_corr = trials_df.copy()
    df_corr['timeframe_num'] = df_corr['timeframe'].map(timeframe_map)
    
    param_columns = ['timeframe_num', 'bb_window', 'bb_std', 'rsi_period', 
                    'rsi_lower', 'rsi_upper', 'atr_period', 
                    'take_profit_mult', 'stop_loss_mult']
    
    corr_matrix = df_corr[param_columns + ['score']].corr()
    param_score_corr = corr_matrix.loc[param_columns, ['score']].sort_values(by='score', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(param_score_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax)
    ax.set_title('Parameter Impact on Optimization Score', fontsize=16)
    
    save_figure(fig, save_path)
    return fig

def plot_timeframe_analysis(trials_df: pd.DataFrame, save_path: str = None) -> Tuple[plt.Figure, pd.DataFrame]:
    setup_plot_style()
    
    timeframe_analysis = trials_df.groupby('timeframe').agg({
        'total_trades': 'mean',
        'win_rate': 'mean',
        'sharpe': 'mean',
        'max_drawdown': 'mean',
        'score': ['mean', 'max']
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    timeframe_analysis[('score', 'max')].plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Max Score by Timeframe')
    axes[0, 0].set_ylabel('Max Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    timeframe_analysis[('sharpe', 'mean')].plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('Average Sharpe Ratio by Timeframe')
    axes[0, 1].set_ylabel('Avg Sharpe')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    timeframe_analysis[('win_rate', 'mean')].plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title('Average Win Rate by Timeframe')
    axes[1, 0].set_ylabel('Avg Win Rate')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    timeframe_analysis[('total_trades', 'mean')].plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_title('Average Number of Trades by Timeframe')
    axes[1, 1].set_ylabel('Avg Trades')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    save_figure(fig, save_path)
    return fig, timeframe_analysis

def plot_parameter_space(trials_df: pd.DataFrame, x_param: str, y_param: str, 
                        save_path: str = None) -> plt.Figure:
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(trials_df[x_param], trials_df[y_param], 
                        c=trials_df['score'], cmap='viridis', alpha=0.6, s=100)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Score', rotation=270, labelpad=15)
    
    ax.set_xlabel(x_param.replace('_', ' ').title())
    ax.set_ylabel(y_param.replace('_', ' ').title())
    ax.set_title(f'Parameter Space: {x_param} vs {y_param}')
    ax.grid(True, alpha=0.3)
    
    save_figure(fig, save_path)
    return fig