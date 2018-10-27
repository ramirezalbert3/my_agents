import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def rolling_mean(history: list, window: int = 20):
    rolling_mean = history['reward'].rolling(window=window).mean()
    ax = sns.lineplot(data=rolling_mean)
    plt.show()
