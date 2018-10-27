import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def rolling_mean(history: list, window: int = 20):
    compiled_rewards = []
    for time, epsilon, num_episodes, rewards, steps, aborted_episodes in history:
        compiled_rewards += (rewards)
    s = pd.Dataframe(compiled_rewards)
    rolling_mean = s.rolling(window=window).mean()
    ax = sns.lineplot(data=rolling_mean)
    plt.show()
