import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## visualise the samples
def plot_r(sampled_rewards, ax, title=None, cbar=False):
    # vmin = np.min(sampled_rewards)
    # vmax = np.max(sampled_rewards)
    vmin = 0
    vmax = 1
    sns.heatmap(sampled_rewards, ax=ax, cbar=cbar, square=True, cmap='viridis', vmin=vmin, vmax=vmax,  cbar_kws={'ticks': [0, 1], 'label': 'Altitude', 'shrink': 0.7})
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
        ax.collections[0].colorbar.set_label('Altitude')
        # ax.collections[0].colorbar.set_label('Route cost')
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

## plot kernel weights
def plot_k_weights(k_weights, ax, title=None):

    sns.barplot(x=np.arange(len(k_weights)), y=k_weights, ax=ax)
    ax.set_title('Kernel weights')
    x_labels = ['Mountain', 'N-S Valleys', 'E-W Valleys', 'Periodic Valleys']
    ax.set_xticks(np.arange(len(k_weights)))
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_ylim(0,1)
    ax.set_aspect(aspect=1.4)


## plot action tree
def plot_action_tree(tree_q, start, goal):
    n_rows, n_cols, _ = tree_q.shape

    # Extract the action values for each direction directly from tree_q
    top = tree_q[:, :, 2]
    right = tree_q[:, :, 1]
    bottom = tree_q[:, :, 0] #NB top and bottom are swapped to match the plot
    left = tree_q[:, :, 3]

    # Generate positions dynamically based on grid size
    top_value_positions = [(j + 0.38, i + 0.25) for i in range(n_rows) for j in range(n_cols)]
    right_value_positions = [(j + 0.65, i + 0.5) for i in range(n_rows) for j in range(n_cols)]
    bottom_value_positions = [(j + 0.38, i + 0.8) for i in range(n_rows) for j in range(n_cols)]
    left_value_positions = [(j + 0.05, i + 0.5) for i in range(n_rows) for j in range(n_cols)]

    fig, ax = plt.subplots(figsize=(n_rows*4, n_rows*1.2))
    ax.set_ylim(n_rows, 0)
    tripcolor = quatromatrix(left, top, right, bottom, grid_size=(n_rows, n_cols), ax=ax,
                             triplotkw={"color": "k", "lw": 1},
                             tripcolorkw={"cmap": "coolwarm"})

    ax.margins(0)
    ax.set_aspect("equal")
    fig.colorbar(tripcolor)

    # Function to round to 2 significant figures
    round_to_sigfigs = lambda x: f"{x:.2g}"

    # Plot values on the grid with 2 significant figures
    for i, (xi, yi) in enumerate(top_value_positions):
        plt.text(xi, yi, round_to_sigfigs(top.flatten()[i]), size=9, color="w")
    for i, (xi, yi) in enumerate(right_value_positions):
        plt.text(xi, yi, round_to_sigfigs(right.flatten()[i]), size=9, color="w")
    for i, (xi, yi) in enumerate(left_value_positions):
        plt.text(xi, yi, round_to_sigfigs(left.flatten()[i]), size=9, color="w")
    for i, (xi, yi) in enumerate(bottom_value_positions):
        plt.text(xi, yi, round_to_sigfigs(bottom.flatten()[i]), size=9, color="w")

    ## set ticks as numbers 0-N
    ticks = np.linspace(0.5, n_rows-0.5, n_rows)
    labels = np.arange(n_rows)
    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels)

    # Mark the start and goal positions with circles
    start_x, start_y = start
    goal_x, goal_y = goal
    ax.plot(start_y + 0.5, start_x + 0.5, 'ro', markersize=10, label="Start")
    ax.plot(goal_y + 0.5, goal_x + 0.5, 'go', markersize=10, label="Goal")
    # ax.legend(loc="upper right")


    return ax
def quatromatrix(left, bottom, right, top, grid_size, ax=None, triplotkw={}, tripcolorkw={}):
    if not ax: 
        ax = plt.gca()

    n_rows, n_cols = grid_size

    # Define unit cell positions
    cell = np.array([[0, 0], [0, 1], [0.5, 0.5], [1, 0], [1, 1]])
    triangles = np.array([[0, 1, 2], [0, 2, 3], [2, 3, 4], [1, 2, 4]])

    # Prepare coordinates for entire grid
    A = np.zeros((n_rows * n_cols * 5, 2))
    Tr = np.zeros((n_rows * n_cols * 4, 3), dtype=int)

    for i in range(n_rows):
        for j in range(n_cols):
            k = i * n_cols + j
            A[k * 5:(k + 1) * 5, :] = np.c_[cell[:, 0] + j, cell[:, 1] + i]
            Tr[k * 4:(k + 1) * 4, :] = triangles + k * 5

    # Combine action values into a single array
    C = np.c_[left.flatten(), bottom.flatten(), right.flatten(), top.flatten()].flatten()

    # Plotting using triplot and tripcolor
    triplot = ax.triplot(A[:, 0], A[:, 1], Tr, **triplotkw)
    tripcolor = ax.tripcolor(A[:, 0], A[:, 1], Tr, facecolors=C, **tripcolorkw, vmin=-1, vmax=0)
    return tripcolor