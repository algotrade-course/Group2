import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

logger = logging.getLogger(__name__)

def setup_plot_style():
    sns.set_style("darkgrid")
    plt.rcParams['figure.figsize'] = (14, 8)
    
def save_figure(fig, save_path=None):
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path)
            logger.info(f"Figure saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving figure: {e}")