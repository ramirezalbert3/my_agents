import platform
if platform.system() == 'Darwin':
    import matplotlib
    matplotlib.use('TkAgg')
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import List
from math import sqrt, ceil


def rolling_mean(history: List[pd.Series], window: int = 100, label = None, axis = None, show: bool = True):
    cols = max(ceil(sqrt(len(history))), 1)
    rows = max(ceil(len(history) / cols), 1)
    if axis is None:
        fig, axis = plt.subplots(nrows=rows, ncols=cols)
    assert len(axis) == len(history)
    
    for series, ax in zip(history, axis):
        rolling_mean = series.rolling(window=window).mean()
        sns.lineplot(data=rolling_mean, ax=ax)
        ax.set_xlabel('episodes')
        ax.set_ylabel(series.name)
        ax.grid(b=True)
        if label is not None:
            ax.legend([label])
    
    if show:
        plt.show()
    return axis # TODO: maybe return fig too
