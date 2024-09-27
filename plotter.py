import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## visualise the samples
def plot_r(sampled_rewards, ax, title=None, cbar=False):
    sns.heatmap(sampled_rewards, ax=ax, cbar=cbar, square=True, cmap='viridis', vmin=0, vmax=1,  cbar_kws={'ticks': [0, 1], 'label': '$', 'shrink': 0.7})
    # ax.imshow(sampled_rewards, extent=(0, self.N, 0, self.N), origin = 'upper')
    # ax.set_xticks(np.arange(0, self.N+1, 5))
    # ax.set_yticks(np.arange(0, self.N+1, 5))
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    ax.set_title(title)
    # ax.set_title('Sampled Reward Distribution,\nkernel: {}'.format(self.kernel_type))
    if cbar:
        ax.collections[0].colorbar.set_label('$')
    return ax

## visualise training points
def plot_obs(obs, ax, text=False):
    for i, (_, x, y, r) in enumerate(obs):
        if text:
            ax.text(y,x , round(r, 2), ha='center', va='center', color='red') # note the x,y are flipped because they are matrix indices
        else:
            ax.scatter(y, x, color='red')

## plot kernel
def plot_kernel(K, ax, title=None):
    sns.heatmap(K, ax=ax, cbar=False, square=True, cmap='viridis')
    ax.set_title(title)
    return ax

## plot RPE
def plot_RPE(RPE, ax, title=None, cbar = False):
    sns.heatmap(RPE, ax=ax, cbar=cbar, square=True, cmap='coolwarm_r', vmin=-1, vmax=1, cbar_kws={'ticks': [-1, 0, 1], 'label': 'RPE', 'shrink': 0.7})
    # if cbar:
    #     ax.collections[0].colorbar.set_label('RPE')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    return ax

