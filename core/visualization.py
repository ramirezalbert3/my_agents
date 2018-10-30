import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import List
from math import sqrt, ceil

def rolling_mean(history: List[pd.Series], window: int = 100) -> None:
    cols = max(ceil(sqrt(len(history))), 1)
    rows = max(ceil(len(history) / cols), 1)
    fig, axis = plt.subplots(nrows=rows, ncols=cols)
    
    for series, ax in zip(history, axis):
        rolling_mean = series.rolling(window=window).mean()
        sns.lineplot(data=rolling_mean, ax=ax)
        ax.set_xlabel('episodes')
        ax.set_ylabel(series.name)
        ax.grid(b=True)
    
    plt.show()
