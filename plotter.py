import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## visualise the samples
def plot_r(sampled_rewards, ax, title=None, cbar=False):
    # vmin = np.min(sampled_rewards)
    # vmax = np.max(sampled_rewards)
    vmin = -1
    vmax = 0
    sns.heatmap(sampled_rewards, ax=ax, cbar=cbar, square=True, cmap='viridis_r', vmin=vmin, vmax=vmax,  cbar_kws={'ticks': [0, 1], 'label': 'Altitude', 'shrink': 0.7})
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
def plot_action_tree(tree_q, start, goal, action_arrow = True, ax=None):
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

    ## determine the best action at each state 
    best_actions = np.argmax(tree_q, axis=2)

    ## switch 0 and 2 to match the plot
    # best_actions = np.where(best_actions == 0, -1, best_actions)
    # best_actions = np.where(best_actions == 2, 0, best_actions)
    # best_actions = np.where(best_actions == -1, 2, best_actions)

    if ax is None:
        fig, ax = plt.subplots(figsize=(n_rows*4, n_rows*1.2))
    else:
        fig = ax.figure

    ax.set_ylim(n_rows, 0)
    tripcolor = quatromatrix(left, top, right, bottom, grid_size=(n_rows, n_cols), ax=ax,
                             triplotkw={"color": "k", "lw": 1},
                             tripcolorkw={"cmap": "coolwarm"})

    ax.margins(0)
    ax.set_aspect("equal")
    # fig.colorbar(tripcolor, ax=ax)

    # Function to round to 2 significant figures
    round_to_sigfigs = lambda x: f"{x:.2f}"

    # Plot values on the grid with 2 significant figures, with the best action in bold and italics
    size = 9
    for i, (xi, yi) in enumerate(top_value_positions):
        ax.text(xi, yi, round_to_sigfigs(top.flatten()[i]), size=size, color="w", weight="bold" if best_actions.flatten()[i] == 2 else "normal", style="italic" if best_actions.flatten()[i] == 0 else "normal")
    for i, (xi, yi) in enumerate(right_value_positions):
        ax.text(xi, yi, round_to_sigfigs(right.flatten()[i]), size=size, color="w", weight="bold" if best_actions.flatten()[i] == 1 else "normal", style="italic" if best_actions.flatten()[i] == 1 else "normal")
    for i, (xi, yi) in enumerate(left_value_positions):
        ax.text(xi, yi, round_to_sigfigs(left.flatten()[i]), size=size, color="w", weight="bold" if best_actions.flatten()[i] == 3 else "normal", style="italic" if best_actions.flatten()[i] == 3 else "normal")
    for i, (xi, yi) in enumerate(bottom_value_positions):
        ax.text(xi, yi, round_to_sigfigs(bottom.flatten()[i]), size=size, color="w", weight="bold" if best_actions.flatten()[i] == 0 else "normal", style="italic" if best_actions.flatten()[i] == 2 else "normal")

    ## Plot arrows for the best action at each state
    if action_arrow:

        ## calculate the max(Q) route from start to goal
        current = start
        path = [current]
        stuck = False
        while np.array_equal(current, goal) == False and stuck == False:
            i, j = current
            action = best_actions[i, j]
            if action == 0:
                current = np.clip((i + 1, j), 0, n_rows-1)
            elif action == 1:
                current = np.clip((i, j + 1), 0, n_cols-1)
            elif action == 2:
                current = np.clip((i - 1, j), 0, n_rows-1)
            elif action == 3:
                current = np.clip((i, j - 1), 0, n_cols-1)
            path.append(current)

            ## check if the current state is already in the path
            for p in path[:-1]:
                if np.array_equal(p, current):
                    stuck = True
                    print('Stuck in a loop')


        ## plot the max arrow for all states
        for i in range(n_rows):
            for j in range(n_cols):
                if best_actions[i, j] == 2:
                    ax.arrow(j + 0.5, i + 0.5, 0, -0.3, head_width=0.1, head_length=0.1, fc="k", ec="k")
                elif best_actions[i, j] == 1:
                    ax.arrow(j + 0.5, i + 0.5, 0.3, 0, head_width=0.1, head_length=0.1, fc="k", ec="k")
                elif best_actions[i, j] == 0:
                    ax.arrow(j + 0.5, i + 0.5, 0, 0.3, head_width=0.1, head_length=0.1, fc="k", ec="k")
                elif best_actions[i, j] == 3:
                    ax.arrow(j + 0.5, i + 0.5, -0.3, 0, head_width=0.1, head_length=0.1, fc="k", ec="k")

    ## set ticks as numbers 0-N
    # ticks = np.linspace(0.5, n_rows-0.5, n_rows)
    # labels = np.arange(n_rows)
    # ax.set_xticks(ticks, labels)
    # ax.set_yticks(ticks, labels)
    ax.set_xticks([])
    ax.set_yticks([])

    # Mark the start and goal positions with circles
    start_x, start_y = start
    goal_x, goal_y = goal
    ax.plot(start_y + 0.5, start_x + 0.5, 'ro', markersize=10, label="Start")
    ax.plot(goal_y + 0.5, goal_x + 0.5, 'go', markersize=10, label="Goal")

    ## plot the max path
    for i in range(len(path)-1):
        ax.plot([path[i][1]+0.5, path[i+1][1]+0.5], [path[i][0]+0.5, path[i+1][0]+0.5], 'k-', linewidth=3, color='green')

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
    tripcolor = ax.tripcolor(A[:, 0], A[:, 1], Tr, facecolors=C, **tripcolorkw)#, vmin=-1, vmax=0)
    return tripcolor