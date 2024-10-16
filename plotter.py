import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## visualise the samples
def plot_r(sampled_rewards, ax, title=None, cbar=False):
    # vmin = np.min(sampled_rewards)
    # vmax = np.max(sampled_rewards)
    vmin = 0
    vmax = 1
    sns.heatmap(sampled_rewards, ax=ax, cbar=cbar, square=True, cmap='viridis', vmin=vmin, vmax=vmax,  cbar_kws={'ticks': [0, 1], 'label': '$', 'shrink': 0.7})
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
        ax.collections[0].colorbar.set_label('Route cost')
    return ax

def plot_state(current, goal, ax, title=None):
    ax.scatter(current[1]+0.5, current[0]+0.5, color='red', s=200)
    ax.scatter(goal[1]+0.5, goal[0]+0.5, color='green', s=200)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    return ax

## visualise training points
def plot_obs(obs, ax, text=False):
    for i, (_, x, y, r) in enumerate(obs):
        if text:
            ax.text(y,x , round(r, 2), ha='center', va='center', color='red') # note the x,y are flipped because they are matrix indices
        else:
            ax.scatter(y+0.5, x+0.5, color='red', marker='x', s=100)

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

## plot path between two points
def plot_traj(trajs, ax, title=None):
    
    ## plot direct and optimal trajectories using different markers
    markers = ['*','x','x']
    colours = ['white','red','red']

    # if len(trajs[0]) ==2:
    #     trajs = [trajs]    

    for ti, traj in enumerate(trajs):

        ## plot start and goal points in red and green
        # ax.scatter(traj[0][1]+0.5, traj[0][0]+0.5, color='red', s=100)
        # ax.scatter(traj[-1][1]+0.5, traj[-1][0]+0.5, color='green', s=100))
        # print(traj[1:-1])

        ## plot path(s)
        # for t in traj[1:-1]:
        for t in traj:
            ax.plot(t[1]+0.5, t[0]+0.5, markers[ti], color=colours[ti], markersize=10, linewidth=10)
    
    # ax.legend(['Start', 'Goal','Direct', 'Optimal'], loc='upper right')
    ## legend for start and goal, and the two paths
     