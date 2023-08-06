import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import probplot

def plot_q_q_plot(data: pd.DataFrame, col_name: str):
    if data.shape[0] == 0:
        raise Exception("There is no data -- Please check")
    if col_name not in data.columns:
        raise Exception("The Column is not in the data")
    probplot(data[col_name], plot=plt)
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel(f"Ordered Values {col_name}")
    plt.title(f"Q-Q Plot: {col_name}")
    plt.show()