from enum import Enum
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register, registry
import numpy as np
from plotter import *
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
import scipy
from scipy.spatial.distance import cdist
from scipy.special import softmax
import warnings
import heapq
from collections import defaultdict
from IPython.display import display, clear_output
import uuid
import random
from collections import deque
import ast
from scipy.spatial import cKDTree as KDTree
import cProfile
import pstats
import subprocess
import time
from numba import jit, njit
import pickle
import pandas as pd
import json
import os
from tqdm.auto import tqdm
import copy
# from agents import Farmer



## create a grid environment
def make_env(N, n_trials, expt_info, beta_params, metric, seed=None):

    ## register env
    
    # Unregister the environment if it's already registered
    env_id = "grids/GridEnv-v0"
    if env_id in registry:
        del registry[env_id]

    # Re-register the updated environment
    register(
        id=env_id,
        entry_point='grids.envs:GridEnv',
        max_episode_steps=100,
        kwargs={"size": N},
    )    
    env = gym.make("grids/GridEnv-v0", N=N, n_trials=n_trials, expt_info=expt_info, beta_params=beta_params, metric=metric, seed=seed)
    
    return env




## Node class
class Node:

    # __slots__ = ['state', 'n_state_visits', 'cost', 'terminated', 'node_id', 'parent_node_ids', 'N', 'untried_actions', 'action_leaves']

    def __init__(self, state, cost, node_id, goal, terminated, trial, n_afc, N):
        
        ## state info
        self.state = np.append(state, cost) ## in the AFC case, this amounts to current state + costs that have just been observed on prior simulated trial
        self.n_state_visits = 0
        self.cost = cost
        self.trial = trial
        self.terminated = terminated
        self.goal = goal
        # self.node_id = tuple(self.state)
        self.node_id = node_id
        self.parent_node_ids = []
        self.N = N


        ## define valid actions
        self.untried_actions = list(range(n_afc))
        if n_afc == 4: # i.e. free choice, meaning we want to restrict wall movements
            row, col,_ = self.state
            if row == self.N-1:
                self.untried_actions.remove(0)
            if row == 0:
                self.untried_actions.remove(2)
            if col == self.N-1:
                self.untried_actions.remove(1)
            if col == 0:
                self.untried_actions.remove(3)

        ## action leaves
        self.action_leaves = {a: None for a in self.untried_actions}


    def __str__(self):
        action_leaves_msg = {action: np.round(leaf.performance,3) if leaf is not None else None for action, leaf in self.action_leaves.items()}
        return "state {}: (trial={}, visits={}, terminated={})\n{})".format(
                                                  self.state,
                                                    self.trial,
                                                  self.n_state_visits,
                                                #   self.cost,
                                                  self.terminated,
                                                  action_leaves_msg
                                                  )

    ## select a random untried action
    def untried_action(self):
        action = random.choice(self.untried_actions)
        self.untried_actions.remove(action)
        return action
    
class Action_Node:

    def __init__(self, prev_state, action, next_state, terminated, trial, parent_id):
        self.prev_state = prev_state
        self.action = action ## in AFC, this specifies the path ID (i.e. 0 or 1)
        self.total_simulation_cost = 0
        self.performance = None
        self.n_action_visits = 0
        self.next_state = next_state
        self.terminated = terminated
        self.trial = trial
        self.node_id = (self.prev_state, self.action) #+ str(self.next_state)
        self.parent_id = parent_id
        self.children={}
        self.children_ids = []

    def __str__(self):
        # return "prev_state{}: (action={}, next_state={}, children={}, visits={}, performance={:0.4f})".format(
        return "prev_state{}: (action={}, next_state={}, n_children={}, visits={}, performance={:0.3f})".format(
                                                  self.prev_state,
                                                  self.action,
                                                self.next_state,
                                                  len(self.children_ids),
                                                  self.n_action_visits,
                                                  self.performance,
                                                  )
    
## Tree class
class Tree:

    def __init__(self,N):
        # self.nodes = {}
        self.root = None
        self.N = N

    ## check if node is expandable
    def is_expandable(self, node):
        return not node.terminated and len(node.untried_actions) > 0

    ## attach action leaf to child state
    def add_state_node(self, state, cost, node_id, goal, terminated, trial, n_afc, parent=None):

        # ## check for existing state node
        # node_id = str(history)
        # if node_id in self.nodes:
        #     # print(state,"already exists")
        #     return self.nodes[node_id]

        
        ## create a new state node
        node = Node(state=state, cost=cost, node_id=node_id, goal=goal, terminated=terminated, trial = trial, n_afc=n_afc, N=self.N)
        
        ## store parent-child relationships
        if parent is None:
            self.root = node
            # self.nodes[str(state)].parent = None
        else:
            node.parent_node_ids.append(parent.node_id)
            
            ## add this state node to the children of the previous action leaf
            parent.children_ids.append(node.node_id)
            # child_key = tuple(np.append(state, cost))
            parent.children[node.node_id] = node
            # parent.children[str(np.append(state, cost))] = node

        return node

    def get_children(self, node, dummy=False):
        children = []
        for a, leaf in node.action_leaves.items():
            if leaf is not None:
                for child_key in leaf.children.keys():
                    child = leaf.children[child_key]
                    children.append(tuple((a, leaf, child_key, child)))

                ## if there are no children (i.e. the S-A leaf has been made, but doesn't have any S nodes), add a dummy child
                if dummy:
                    if len(leaf.children) == 0:
                        children.append(tuple((a, leaf, None, None)))
        return children

    def parent(self, node):
        parent_node_id = self.nodes[node.node_id].parent_node_id
        if parent_node_id is None:
            return None #i.e. root reached, bc it has no parent
        else:
            return self.nodes[parent_node_id]

    ## calculate value of each S-A node
    def action_tree(self):

        self.tree_q = np.zeros((self.N,self.N,4)) + np.nan
        for sstate in self.nodes.keys():
            state = self.nodes[sstate].state
            for a in self.nodes[sstate].action_leaves.keys():
                try:
                    self.tree_q[state[0], state[1], a] = self.nodes[sstate].action_leaves[a].performance
                except:
                    pass


    def print_tree(self, node, indent="", is_last=True, dummy=False, depth=0, max_depth=None):
        """
        Recursively print the tree structure with markers, visit counts, and values.

        Args:
        - node_id: The ID of the current node.
        - indent: The current indentation string for formatting.
        - is_last: Whether this node is the last child of its parent.
        - dummy: Whether to print display action leaves that don't have any children.
        - depth: The current depth of the recursion.
        - max_depth: The maximum depth to print (None for no limit).
        """
        # Stop printing if max depth is reached
        if max_depth is not None and depth > max_depth:
            return

        # Get the current node
        # node = self.nodes[node_id]
        if dummy:
            if node is None:
                return
            else:
                node_label = f"{node.state}"
        else:
            node_label = f"{node.state}"
            # node_label = f"{node.node_id}"
        trial_label = f"{node.trial}"

        # Add branch marker
        branch = "└── " if is_last else "├── "
        print(f"{indent}{branch}Node: {node_label}, Episode: {trial_label}, Visits: {node.n_state_visits}")

        # Update indentation for children
        child_indent = indent + ("    " if is_last else "│   ")

        # Group children by action
        children_by_action = {}
        for action, leaf, child_id, child_node in self.get_children(node, dummy):
            if action not in children_by_action:
                children_by_action[action] = []
            children_by_action[action].append((leaf, child_id, child_node))

        # Find the best action based on performance
        best_action = max(
            children_by_action.items(),
            key=lambda item: item[1][0][0].performance,  # Access the performance of the first leaf
            default=(None, [])
        )[0]

        # Iterate through actions and their corresponding children
        num_actions = len(children_by_action)
        for i, (action, children) in enumerate(children_by_action.items()):
            # Check if this is the last action
            is_action_last = i == num_actions - 1

            # Print the action label (only once per action)
            leaf = children[0][0]  # Assume all children of the same action share the same leaf
            action_label = f"Action {action}, (n_v: {leaf.n_action_visits}, prev_state: {leaf.prev_state}, next_state: {leaf.next_state}, branch factor: {len(children)}, perf: {leaf.performance:.2f})"

            # Highlight the best action in bold (use ANSI escape codes for bold text)
            if action == best_action:
                action_label = f"\033[1m{action_label}\033[0m"

            action_branch = "└── " if is_action_last else "├── "
            print(f"{child_indent}{action_branch}{action_label}")

            # Update child indentation
            sub_child_indent = child_indent + ("    " if is_action_last else "│   ")

            # Print each child for this action
            for j, (leaf, child_id, child_node) in enumerate(children):
                # Check if this is the last child of this action
                is_child_last = j == len(children) - 1

                # Recursively print the child node with increased depth
                self.print_tree(
                    child_node,
                    indent=sub_child_indent,
                    is_last=is_child_last,
                    dummy=dummy,
                    depth=depth + 1,
                    max_depth=max_depth
                )


    def max_depth(self, node):
        """
        Recursively calculate the maximum depth of the tree starting from the given node.

        Args:
        - node: The current node (root of the subtree being evaluated).

        Returns:
        - int: The maximum depth of the tree.
        """
        # Base case: If the node has no children, its depth is 1
        if not self.get_children(node):
            return 1

        # Recursive case: Compute the depth for each child
        child_depths = []
        for _, _, _, child_node in self.get_children(node):
            child_depths.append(self.max_depth(child_node))

        # The depth of this node is 1 + max depth of its children
        return 1 + max(child_depths)
    

    ## function for getting the max and min Q-values at a given depth of the tree
    def min_max_Q(self, node, depth, current_depth=0):
        """
        Recursively calculate the maximum and minimum Q-values at a given depth of the tree starting from the given node.

        Args:
        - node: The current node (root of the subtree being evaluated).
        - depth: The target depth to calculate the Q-values.
        - current_depth: The current depth of the recursion.

        Returns:
        - (float, float): The maximum and minimum Q-values at the target depth.
        """
        # Base case: If the target depth is reached, return the Q-value of this node
        if current_depth == depth:
            Qs = []
            for a in node.action_leaves.keys():
                if node.action_leaves[a] is not None:
                    Qs.append(node.action_leaves[a].performance)
            if len(Qs) == 0:
                return np.inf, -np.inf
            return min(Qs), max(Qs)

        # Recursive case: Compute the maximum and minimum Q-values for each child
        max_Q = -np.inf
        min_Q = np.inf
        for _, _, _, child_node in self.get_children(node):
            child_min_Q, child_max_Q = self.min_max_Q(child_node, depth, current_depth + 1)
            max_Q = max(max_Q, child_max_Q)
            min_Q = min(min_Q, child_min_Q)

        return min_Q, max_Q



    ## prune, i.e. after taking a step, keep only that subtree
    # def prune(self):

    #     ## identify the root's children, i.e. the four adjacent states
    #     # keep_nodes = [str(self.root.state)]
    #     keep_nodes = self.root.node_id
    #     for leaf in self.root.action_leaves.values():
    #         if leaf is not None:
    #             keep_nodes.append(str(leaf.next_state))

    #     for sstate in list(self.nodes.keys()):
    #         if str(self.nodes[sstate].state) not in keep_nodes:
    #             del self.nodes[sstate]

    def prune(self, action, next_state):
        
        ## delete actions not taken
        actions_to_delete = [a for a in self.root.action_leaves.keys() if (a != action) and (self.root.action_leaves[a] is not None)]
        for a in actions_to_delete:
            del self.root.action_leaves[a]

        ## delete subtree for the other state children reachable from the root-action pair
        self.root.action_leaves[action].children = {tuple(next_state): self.root.action_leaves[action].children[tuple(next_state)]}

        ## update the root
        self.root = self.root.action_leaves[action].children[tuple(next_state)]
        


    
    ## calculate the best trajectory for any two points, given the tree
    def best_traj(self, start, goal):

        ## get the best action at each state
        best_actions = nanargmax(self.tree_q, axis=2)

        ## get the best trajectory from start to goal
        current = start
        traj_states = [current]
        traj_actions = []
        stuck = False
        while not np.array_equal(current, goal) and not stuck:
            i, j = current
            action = best_actions[i,j]
            action = int(action)
            traj_actions.append(action)
            if action==0:
                current = np.clip((i + 1, j), 0, self.N-1)
            elif action == 1:
                current = np.clip((i, j + 1), 0, self.N-1)
            elif action == 2:
                current = np.clip((i - 1, j), 0, self.N-1)
            elif action == 3:
                current = np.clip((i, j - 1), 0, self.N-1)
            traj_states.append(current)

            ## check if the current state is already in the path
            for s in traj_states[:-1]:
                if np.array_equal(s, current):
                    stuck = True
                    
        
        return traj_states, traj_actions
                    

    
    

### misc utils

## value iteration
@njit
def value_iteration(dp_costs, goal, max_iters=1000, theta=0.0001, discount=0.99):
    N = len(dp_costs)
    n_actions = 4

    # Action directions
    action_directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])

    # Initialize tables
    V = np.zeros((N, N))
    A = np.zeros((N, N), dtype=np.int32)
    Q = np.full((N, N, n_actions), np.nan)

    # Set cost of the goal to 0
    goal_x, goal_y = goal

    for i in range(max_iters):
        delta = 0

        for x in range(N):
            for y in range(N):
                if x == goal_x and y == goal_y:
                    continue

                v = V[x, y]

                # Compute Q-values for all valid actions in one loop
                for a in range(n_actions):
                    next_x = x + action_directions[a][0]
                    next_y = y + action_directions[a][1]

                    if 0 <= next_x < N and 0 <= next_y < N:
                        Q[x, y, a] = dp_costs[next_x, next_y] + discount * V[next_x, next_y]
                    else:
                        Q[x, y, a] = np.nan

                # Update value and action tables
                max_q = np.nanmax(Q[x, y])
                V[x, y] = max_q
                best_actions = np.where(Q[x, y] == max_q)[0]
                A[x, y] = np.random.choice(best_actions)

                # Update convergence threshold
                delta = max(delta, abs(v - max_q))

        if delta < theta:
            break

    return V, Q, A


## random choice between multiple minima/maxima
def argm(x, extreme_val):
    indices = np.where(x == extreme_val)[0]
    return np.random.choice(indices)

## calculate the angle between two nodes
def node_angle(a,b):
    rad = np.arctan2(b[1]-a[1], b[0]-a[0])
    ang = np.abs(np.degrees(rad))
    ang%=90
    return ang


## parse strings to lists
def parse_lists(df):
    cols = df.columns[2:]
    for key in cols:
        try:
            df[key] = df[key].apply(lambda x: np.array(ast.literal_eval(x)))
        except:
            pass
    return df

def parse_np(df):
    cols = df.columns[2:]
    for key in cols:
        try:
            df[key] = df[key].apply(lambda x: eval(x, {"array": np.array}))
        except:
            pass
    return df


## KL divergence between prior and posterior samples, where samples as assumed to be multivariate Gaussians
def KL_divergence(x, y):
    '''x is the prior, y is the posterior'''

    ## calculate gaussian terms
    cov_x = np.cov(x)
    cov_y = np.cov(y)
    mu_x = np.mean(x, axis=1)
    mu_y = np.mean(y, axis=1)
    d = len(mu_x)
    assert d == cov_x.shape[0], "Mean and covariance dimensions do not match"

    ## trace term
    inv_cov_y = np.linalg.inv(cov_y)
    trace_term = np.trace(inv_cov_y @ cov_x)

    ## log determinant term
    log_det_x = np.linalg.slogdet(cov_x)[1]
    log_det_y = np.linalg.slogdet(cov_y)[1]
    LD_term = log_det_y - log_det_x

    ## mu term
    mean_diff = mu_y - mu_x
    mean_term = mean_diff.T @ inv_cov_y @ mean_diff

    ## combine
    KL = 0.5* (trace_term - d + LD_term + mean_term) 

    return KL


## parallel function for uncertainty tests
def KL_sim(obs_set, t, farmer, n_samples, plotting = False):

    ## get expt + sampler info
    N = farmer.N
    n_trials = 2 ## arbitrary
    expt = farmer.expt
    expt_info = {
        'type': expt,
        'same_SGs': True,
    }
    n_iter = 10
    lazy = False
    CE = False

    ## get prior samples
    beta_params = {
        'alpha_row': farmer.alpha_row,
        'beta_row': farmer.beta_row,
        'alpha_col': farmer.alpha_col,
        'beta_col': farmer.beta_col
        }
    prior_p_samples = farmer.all_posterior_ps
    prior_q_samples = farmer.all_posterior_qs
    prior_samples = np.vstack([prior_p_samples.T, prior_q_samples.T])

    ## save the posterior means
    all_posterior_mean_p_costs = []

    ## reset env
    env = make_env(N, n_trials,expt_info, beta_params, 'cityblock')
    env.reset()

    ## loop through obs sets
    KLs = []
    n_seqs = len(obs_set)
    for a, obs in enumerate(obs_set):
        if plotting and a==n_seqs-1:
            fig, axs = plt.subplots(1,5, figsize = (20, 5))
        costs = []
        for oi, o in enumerate(obs):
            o_tmp = o[:2].astype(int)
            # costs.append(env.get_cost(o_tmp))
            # costs.append(arbitrary_costs[oi])
            costs.append(env.costs[o_tmp[0], o_tmp[1]])
        obs[:,2] = costs
        env.set_obs(obs)

        ## farmer generates new set of root samples, given the obs
        farmer = Farmer(N)
        farmer.get_env_info(env)
        farmer.root_samples(farmer.obs, n_samples,n_iter, lazy=lazy,CE=False)
        posterior_p_samples = farmer.all_posterior_ps
        posterior_q_samples = farmer.all_posterior_qs
        posterior_samples = np.vstack([posterior_p_samples.T, posterior_q_samples.T])
        all_posterior_mean_p_costs.append(farmer.posterior_mean_p_cost)
        
        ## plot posterior
        if plotting and a==n_seqs-1:
            # plot_r(farmer.posterior_mean_p_cost, axs[a,1], title = 'Posterior reward distribution\nmean root sample\npost obs')
            plot_r(farmer.posterior_mean_p_cost, axs[0], title = 'Posterior reward distribution\nmean root sample\npost obs')
            plot_obs(env.obs, ax = axs[0], text=True)
            # plot_obs(env.obs, ax = axs[a,1], text=True)
            
            ## plot the prior and posterior KDEs of row and column parameters
            for n in range(N):
                sns.kdeplot(prior_p_samples[:,n], ax=axs[1], fill=True)
                sns.kdeplot(posterior_p_samples[:,n], ax=axs[2], fill=True)
                sns.kdeplot(prior_q_samples[:,n], ax=axs[3], fill=True)
                sns.kdeplot(posterior_q_samples[:,n], ax=axs[4], fill=True)
            axs[1].set_title('prior p')
            axs[2].set_title('posterior p')
            axs[3].set_title('prior q')
            axs[4].set_title('posterior q')

        ## KL divergence
        # KL = KL_divergence(prior_samples, posterior_samples)
        KL = KL_divergence(posterior_samples, prior_samples)
        KLs.append(KL)
        # KLs[t,a] = KL
        # print('KL after obs along ',axes[a],':',KL)

        ## plot formatting
        if plotting and a==n_seqs-1:
            plt.suptitle('KL = '+str(np.round(KL,2)), fontsize = 20)
            plt.tight_layout()
            plt.show()
    all_posterior_mean_p_costs = np.array(all_posterior_mean_p_costs)

    return KLs, obs_set, all_posterior_mean_p_costs, t


## profiling
def profile_func(func, *args, **kwargs):

    ## check first of all if a profiler is active, in which case disable it
    if cProfile.Profile().disable() is not None:
        cProfile.Profile().disable()

    ## profile the function
    profiler = cProfile.Profile()
    profiler.enable()
    func(*args, **kwargs)
    profiler.disable()

    ## save profiling report
    func_name = func.__name__
    profile_file = f'{func_name}_profile.pstats'
    profiler.dump_stats(profile_file)
    with open(f'{func_name}_profile.txt', 'w') as f:
        p = pstats.Stats(profiler, stream=f)
        p.sort_stats('cumulative').print_stats(50)

    ## convert to dot file
    dot_file = f'{func_name}_profile.dot'
    subprocess.run(['gprof2dot', '-f', 'pstats', profile_file, '-o', dot_file], check=True)

    ## generate PNG visualization
    png_file = f'{func_name}_profile.png'
    subprocess.run(['dot', '-Tpng', dot_file, '-o', png_file], check=True)

    print(f"Profiling complete. Visualization saved as {png_file}")


## cached function for moving to the next state
def get_next_state(current, direction, N):
    next_state = np.clip(
        current + direction,
        0,
        N - 1
    )
    return next_state


## rotate grids
def rotate_grid_world(grid_data, rotation_direction="clockwise", grid_size=8):
    """
    Rotate all grid world data in a JSON file or gym env either clockwise or counter-clockwise.
    
    Args:
        grid_data (dict or str): Either the parsed JSON data as a dict, or a file path to the JSON file, or the original env object
        rotation_direction (str): Either "clockwise" or "counter_clockwise"
        grid_size (int): Size of the grid (assumes square grid)
    
    Returns:
        dict: Copy of the input data with rotated coordinates and cost surfaces,
        or the original env object with rotated coordinates and cost surfaces.
    """

    # Load if a file path is provided
    if isinstance(grid_data, str):
        with open(grid_data, 'r') as f:
            data = json.load(f)
    else:
        data = grid_data
    
    # Create a deep copy to avoid modifying the original
    rotated_data = copy.deepcopy(data)
    
    # Function to rotate a single coordinate
    def rotate_coord(coord):
        x, y = coord
        if rotation_direction == "clockwise":
            # 90° clockwise: (x,y) -> (y, grid_size-1-x)
            return [y, grid_size - 1 - x]
        else:
            # 90° counter-clockwise: (x,y) -> (grid_size-1-y, x)
            return [grid_size - 1 - y, x]
    
    # Function to rotate a single cost grid
    def rotate_cost_grid(grid):
        n = len(grid)
        rotated = [[0 for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if rotation_direction == "clockwise":
                    # 90° clockwise: (i,j) -> (j, n-1-i)
                    rotated[j][n - 1 - i] = grid[i][j]
                else:
                    # 90° counter-clockwise: (i,j) -> (n-1-j, i)
                    rotated[n - 1 - j][i] = grid[i][j]
                    
        return rotated
    
    # Rotate trial information, depending on whether a json or env object is provided
    if 'sequence' in rotated_data:
        if 'trial_info' in rotated_data['sequence']:
            for idx, trial in enumerate(rotated_data['sequence']['trial_info']):
                # Rotate start and goal coordinates
                if 'start_A' in trial:
                    rotated_data['sequence']['trial_info'][idx]['start_A'] = rotate_coord(trial['start_A'])
                if 'start_B' in trial:
                    rotated_data['sequence']['trial_info'][idx]['start_B'] = rotate_coord(trial['start_B'])
                if 'goal_A' in trial:
                    rotated_data['sequence']['trial_info'][idx]['goal_A'] = rotate_coord(trial['goal_A'])
                if 'goal_B' in trial:
                    rotated_data['sequence']['trial_info'][idx]['goal_B'] = rotate_coord(trial['goal_B'])
                
                # Rotate path coordinates
                if 'path_A' in trial:
                    rotated_data['sequence']['trial_info'][idx]['path_A'] = [rotate_coord(coord) for coord in trial['path_A']]
                if 'path_B' in trial:
                    rotated_data['sequence']['trial_info'][idx]['path_B'] = [rotate_coord(coord) for coord in trial['path_B']]
                
                # Swap axis information
                if 'dominant_axis_A' in trial:
                    rotated_data['sequence']['trial_info'][idx]['dominant_axis_A'] = ('horizontal' 
                                                                    if trial['dominant_axis_A'] == 'vertical' 
                                                                    else 'vertical')
                if 'dominant_axis_B' in trial:
                    rotated_data['sequence']['trial_info'][idx]['dominant_axis_B'] = ('horizontal' 
                                                                    if trial['dominant_axis_B'] == 'vertical' 
                                                                    else 'vertical')
                if 'better_axis' in trial:
                    rotated_data['sequence']['trial_info'][idx]['better_axis'] = ('horizontal' 
                                                                if trial['better_axis'] == 'vertical' 
                                                                else 'vertical')
                    
                # reverse abstract sequences - e.g. if abstract_sequence_A is [5,0], it should now be [0,5]
                if 'abstract_sequence_A' in trial:
                    # rotated_data['sequence']['trial_info'][idx]['abstract_sequence_A'] = [coord[::-1] for coord in trial['abstract_sequence_A']]
                    rotated_data['sequence']['trial_info'][idx]['abstract_sequence_A'] = trial['abstract_sequence_A'][::-1]
                if 'abstract_sequence_B' in trial:
                    # rotated_data['sequence']['trial_info'][idx]['abstract_sequence_B'] = [coord[::-1] for coord in trial['abstract_sequence_B']]
                    rotated_data['sequence']['trial_info'][idx]['abstract_sequence_B'] = trial['abstract_sequence_B'][::-1]
                
                
                # Swap context if it's 'row' or 'column'
                if 'context' in trial:
                    if trial['context'] == 'row':
                        rotated_data['sequence']['trial_info'][idx]['context'] = 'column'
                    elif trial['context'] == 'column':
                        rotated_data['sequence']['trial_info'][idx]['context'] = 'row'
        
        # Rotate environment cost surfaces
        if 'env_costs' in rotated_data['sequence']:
            for key in rotated_data['sequence']['env_costs']:
                if key.startswith('city_') and '_grid_' in key:
                    rotated_data['sequence']['env_costs'][key] = rotate_cost_grid(data['sequence']['env_costs'][key])
    
    else:
        
        ##hacky init
        n_cities = 8
        n_days = 5
        n_trials = 4

        ## loop through grid envs
        for key in rotated_data.keys():
            for t in range(n_trials):
            
                ## rotate start and goal coordinates
                start_A = rotated_data[key][0].starts[t][0]
                start_B = rotated_data[key][0].starts[t][1]
                goal_A = rotated_data[key][0].goals[t][0]
                goal_B = rotated_data[key][0].goals[t][1]
                rotated_data[key][0].starts[t][0] = np.array(rotate_coord(start_A))
                rotated_data[key][0].starts[t][1] = np.array(rotate_coord(start_B))
                rotated_data[key][0].goals[t][0] = np.array(rotate_coord(goal_A))
                rotated_data[key][0].goals[t][1] = np.array(rotate_coord(goal_B))

                ## rotate actions - i.e. remap 0,1,2,3 to 1,2,3,0. e.g. [0,0,0,1] becomes [3,0,0,0]
                # rotate_action = lambda action: (action + 1) % 4 if rotation_direction == "clockwise" else (action - 1) % 4
                rotate_action = lambda action: (action - 1) % 4 if rotation_direction == "clockwise" else (action + 1) % 4
                rotated_data[key][0].path_actions[t][0] = [rotate_action(action) for action in rotated_data[key][0].path_actions[t][0]]
                rotated_data[key][0].path_actions[t][1] = [rotate_action(action) for action in rotated_data[key][0].path_actions[t][1]]
                
                ## paths
                path_A = rotated_data[key][0].path_states[t][0]
                path_B = rotated_data[key][0].path_states[t][1]
                rotated_data[key][0].path_states[t][0] = np.array([rotate_coord(coord) for coord in path_A])
                rotated_data[key][0].path_states[t][1] = np.array([rotate_coord(coord) for coord in path_B])

                ## axis info
                rotated_data[key][0].dominant_axis_A[t] = ('horizontal'
                                                            if rotated_data[key][0].dominant_axis_A[t] == 'vertical'
                                                            else 'vertical')
                rotated_data[key][0].dominant_axis_B[t] = ('horizontal'
                                                            if rotated_data[key][0].dominant_axis_B[t] == 'vertical'
                                                            else 'vertical')
                
                ## abstract sequences
                rotated_data[key][0].sampled_abstract_sequences[t][0] = rotated_data[key][0].sampled_abstract_sequences[t][0][::-1]
                rotated_data[key][0].sampled_abstract_sequences[t][1] = rotated_data[key][0].sampled_abstract_sequences[t][1][::-1]

            ## context
            if rotated_data[key][0] == 'row':
                rotated_data[key][0] = 'column'
            elif rotated_data[key][0] == 'column':
                rotated_data[key][0] = 'row'

            ## rotate entire grid
            rotated_data[key][0].p_costs = np.array(rotate_cost_grid(rotated_data[key][0].p_costs))
            for t in range(n_trials):
                rotated_data[key][0].costss[t] = np.array(rotate_cost_grid(rotated_data[key][0].costss[t]))
                
    
    return rotated_data

def save_rotated_data(rotated_data, output_file):
    """
    Save rotated data to a JSON file or .pkl
    
    Args:
        rotated_data (dict): The rotated data to save
        output_file (str): Path to the output file
    """
    if ('sequence' in rotated_data) or ('trial_info' in rotated_data):
        with open(output_file, 'w') as f:
            json.dump(rotated_data, f, indent=2)
        print(f"Rotated data saved to {output_file}")
    else:
        with open(output_file, 'wb') as f:
            pickle.dump(rotated_data, f)

def load_data(path):
    from agents import Farmer ## for later simulation
    fieldnames = [
        "pid", 
        "trial", "city", "path_chosen", "button_pressed", "reaction_time_ms", 
        "context", "grid", "path_A_expected_cost", "path_B_expected_cost", 
        "path_A_actual_cost", "path_B_actual_cost", "path_A_future_overlap", 
        "path_B_future_overlap", "abstract_sequence_A", "abstract_sequence_B", 
        "dominant_axis_A", "dominant_axis_B", "better_path", "chose_better_path",
        "bonusAchieved",
        'expt_info_filename'
    ]
    df_all = pd.DataFrame(columns=fieldnames)

    # Load id mapping (for later simulation)
    with open('expt/assets/trial_sequences/id_mapping.pkl', 'rb') as f:
        id_mapping = pickle.load(f)
    all_expt_info_ids = []

    # Initialize questionnaire dictionary
    questionnaire = {
        "pid": [],
    }
    for q in range(1, 18+1):
        questionnaire['NFC'+str(q)] = []
    questionnaire['screener'] = []
    
    for file in os.listdir(path):
        if not file.endswith('.json'):
            continue
        filename = os.path.join(path, file)
        pid = file[:-5]

        # Load and decode JSON (double decoding)
        with open(filename, 'r', encoding='utf-8') as f:
            try:
                raw = f.read()
                first_pass = json.loads(raw)
                data = json.loads(first_pass)
            except Exception as e:
                print(f"Decoding error in file {file}: {e}")
                continue

        ## sanity check: print 0i30zpvykjyk5btzylrbcfjk
        # if pid == '0i30zpvykjyk5btzylrbcfjk':
        #     print('Sanity check for participant 0i30zpvykjyk5btzylrbcfjk')

        ## bonus?
        # bonus = [trial.get('bonusAchieved', 0) for trial in data if 'bonusAchieved' in trial]
        # if bonus:
        #     print('Bonus for participant', pid, ':', bonus[0])


        # Filter for relevant trials
        trial_data = [
            trial for trial in data
            if trial.get('trial_type') == 'html-keyboard-response' and trial.get('choice')
        ]

        if not trial_data:
            continue

        ## save questionnaire data
        questionnaire['pid'].append(pid)
        nfc_data = [entry for entry in data if entry.get("task") == "NFC"]
        nfc_data_flat = [
            item for sublist in nfc_data for item in sublist['response'].items()
        ]
        nfc_data_flat = dict(nfc_data_flat)
        keep_keys = []
        for key in list(nfc_data_flat.keys()):  # Iterate over a copy of the keys
            if key[-1] == '.':
                nfc_data_flat[key[:-1]] = nfc_data_flat[key]
                keep_keys.append(key[:-1])
            else:
                keep_keys.append(key)

        for key in questionnaire.keys():
            if key in nfc_data_flat.keys():
                questionnaire[key].append(nfc_data_flat[key])
            elif key != 'pid':
                questionnaire[key].append(np.nan)

        df_tmp = pd.DataFrame([{key: trial.get(key, '') for key in fieldnames} for trial in trial_data])

        # Check for completeness (8 cities)
        n_cities = df_tmp['city'].nunique()
        if n_cities < 8:
            print('Incomplete dataset for participant:', file,'. Found only', n_cities, 'cities.')
            continue

        # Skip empty lines
        df_tmp = df_tmp[df_tmp['trial'] != ''].reset_index(drop=True)

        # skip practice trials, i.e. check how many trials have city==1
        n_city_1 = df_tmp[df_tmp['city'] == 1].shape[0]
        if n_city_1==28:
            df_tmp = df_tmp.iloc[8:].reset_index(drop=True)
        elif n_city_1==36:
            df_tmp = df_tmp.iloc[16:].reset_index(drop=True)
        else:
            print('city 1 trials:',n_city_1)
        assert (df_tmp.iloc[0]['city'] == 1) & (df_tmp.iloc[0]['trial'] == 1) \
            and (df_tmp.iloc[0]['grid'] == 1), 'First trial should be city 1, grid 1, trial 1. Instead got: ' \
            + str(df_tmp.iloc[0]['city']) + ', ' + str(df_tmp.iloc[0]['grid']) + ', ' + str(df_tmp.iloc[0]['trial'])
        
        ## check how many trials
        n_total_trials = len(df_tmp)
        if n_total_trials != 160:
            print('Expected 160 trials, but found:', n_total_trials, 'for participant:', pid)
            display(df_tmp.tail())

            ## hacky fix for 'e248nl43jdfwg8bisl7sjezc' who is missing the final day of the final city: add four more trials with nans
            if pid == 'e248nl43jdfwg8bisl7sjezc' and n_total_trials == 156:
                print('Applying hacky fix for participant', pid, 'by adding four empty trials for the missing final day of the final city')
                last_city = df_tmp['city'].max()
                last_trial = df_tmp[df_tmp['city'] == last_city]['trial'].max()
                last_day = df_tmp[df_tmp['city'] == last_city]['grid'].max()
                for i in range(1, 5):
                    new_row = {
                        'pid': pid,
                        'trial': i,
                        'city': 8,
                        'path_chosen': np.nan,
                        'button_pressed': np.nan,
                        'reaction_time_ms': np.nan,
                        'context': np.nan,
                        'grid': 5,
                        'path_A_expected_cost': np.nan,
                        'path_B_expected_cost': np.nan,
                        'path_A_actual_cost': np.nan,
                        'path_B_actual_cost': np.nan,
                        'path_A_future_overlap': np.nan,
                        'path_B_future_overlap': np.nan,
                        'abstract_sequence_A': np.nan,
                        'abstract_sequence_B': np.nan,
                        'dominant_axis_A': np.nan,
                        'dominant_axis_B': np.nan,
                        'better_path': np.nan,
                        'chose_better_path': np.nan,
                        'bonusAchieved': np.nan,
                        # 'expt_info_filename': id_mapping.get(pid, '')
                        'expt_info_filename': np.nan,
                    }
                    df_tmp = pd.concat([df_tmp, pd.DataFrame([new_row])], ignore_index=True)
                print('New total trials after fix:', len(df_tmp))
            # continue ## skip


        # rename a few cols, e.g. 'grid' to 'day', 'reaction_time_ms' to 'RT'
        df_tmp['pid'] = pid
        df_tmp.rename(columns={'grid': 'day'
                                 , 'reaction_time_ms': 'RT',
                               }, inplace=True)
        
        ## check for duplicated trial sequences
        try:
            id = id_mapping[pid][10:]
        except KeyError:
            raise KeyError(f'No id mapping for participant {pid}')
        if id in all_expt_info_ids:
            print('Warning: id already in list:', id)

            ## remove
            # df_all = df_all[df_all['pid'] != pid]

            ## tweak id
            # id = id + '_dup'

            ## do nothing
            continue
        
        try:
            df_tmp.loc[df_tmp['pid'] == pid, 'expt_info_filename'] = str(int(id))
        except:
            df_tmp.loc[df_tmp['pid'] == pid, 'expt_info_filename'] = str(id)
        all_expt_info_ids.append(id)
        df_all = pd.concat([df_all, df_tmp], ignore_index=True)

    # Cleaning
    df_all = df_all.replace('', np.nan)
    df_all = df_all.replace('nan', np.nan)
    df_all = df_all.replace('NaN', np.nan)
    df_all = df_all.replace('none', np.nan)
    df_all = df_all.replace('None', np.nan)
    for col in fieldnames:
        try:
            df_all[col] = df_all[col].astype(float)
        except ValueError:
            pass

    # count number of nan trials per participant - i.e. nan in path_chosen
    df_all['path_chosen'] = df_all['path_chosen'].replace('nan', np.nan)
    df_all['path_chosen'] = df_all['path_chosen'].replace('none', np.nan)
    df_all.loc[df_all['path_chosen'].isna(), 'chose_better_path'] = np.nan
    for p in df_all['pid'].unique():
        n_nan = df_all.loc[df_all['pid'] == p, 'path_chosen'].isna().sum()
        # if n_nan > 0:
        #     print('n_nan for participant', p, ':', n_nan)

    # Label path IDs and aligned path info
    df_all['path_chosen'] = df_all['path_chosen'].map({'blue': 'a', 'green': 'b'})
    df_all = df_all[df_all['trial'].notna()]
    df_all['chose_aligned'] = np.nan
    df_all['aligned_path'] = np.nan
    df_all['chose_vertical'] = np.nan # so we have a consistent reference frame

    ## remove all non-choices?
    # df_all = df_all[df_all['path_chosen'].notna()]

    df_all.loc[(df_all['context'] == 'column') & (df_all['dominant_axis_A'] == 'vertical'), 'aligned_path'] = 'a'
    df_all.loc[(df_all['context'] == 'column') & (df_all['dominant_axis_B'] == 'vertical'), 'aligned_path'] = 'b'
    df_all.loc[(df_all['context'] == 'row') & (df_all['dominant_axis_A'] == 'horizontal'), 'aligned_path'] = 'a'
    df_all.loc[(df_all['context'] == 'row') & (df_all['dominant_axis_B'] == 'horizontal'), 'aligned_path'] = 'b'

    df_all['chose_aligned'] = (df_all['path_chosen'] == df_all['aligned_path']).astype(bool)
    df_all['chose_orthogonal'] = (df_all['path_chosen'] != df_all['aligned_path']).astype(bool)
    df_all.loc[(df_all['context'] == 'column') & (df_all['chose_aligned'] == True), 'chose_vertical'] = True
    df_all.loc[(df_all['context'] == 'column') & (df_all['chose_aligned'] == False), 'chose_vertical'] = False
    df_all.loc[(df_all['context'] == 'row') & (df_all['chose_aligned'] == True), 'chose_vertical'] = False
    df_all.loc[(df_all['context'] == 'row') & (df_all['chose_aligned'] == False), 'chose_vertical'] = True


    ### get some additional data
    df_all['CE_action'] = np.nan
    df_all['CE_chose_vertical'] = np.nan
    df_all['CE_chose_aligned'] = np.nan
    df_all['CE_chose_orthogonal'] = np.nan
    df_all['first_path'] = np.nan
    df_all['second_path'] = np.nan
    df_all['third_path'] = np.nan
    df_all['first_path_orthogonal'] = np.nan
    df_all['second_path_orthogonal'] = np.nan
    df_all['third_path_orthogonal'] = np.nan
    df_all['prev_chose_orthogonal'] = np.nan
    df_all['prev_chose_aligned'] = np.nan
    df_all['prev_chose_vertical'] = np.nan
    df_all['prev_day_chose_aligned'] = np.nan ## i.e. for the same trial of the previous day
    df_all['prev_day_chose_orthogonal'] = np.nan
    df_all['prev_day_chose_vertical'] = np.nan
    df_all['path_A_past_overlaps'] = np.nan
    df_all['path_B_past_overlaps'] = np.nan
    df_all['path_A_past_observed_costs'] = np.nan
    df_all['path_B_past_observed_costs'] = np.nan
    df_all['path_A_past_observed_no_costs'] = np.nan
    df_all['path_B_past_observed_no_costs'] = np.nan
    df_all['total_past_overlaps'] = np.nan
    df_all['observed_cost'] = np.nan
    df_all['prev_observed_cost'] = np.nan
    df_all['path_quality'] = np.nan
    df_all['prev_path_quality'] = np.nan
    df_all['day_cost'] = np.nan
    df_all['path_len'] = np.nan
    df_all['switched_axis'] = np.nan
    
    ## diffs, where this is always defined as vertical - horizontal
    df_all['past_overlaps_diff'] = np.nan 
    df_all['observed_costs_diff'] = np.nan 
    df_all['observed_no_costs_diff'] = np.nan
    
    ## accuracy as a function of first-trial choice - i.e. what is the trial-wise accuracy, conditional on having chosen path a or b first
    for p in tqdm(range(len(df_all['pid'].unique())), total=len(df_all['pid'].unique()), desc='Extracting participant trial info'):
        pid = df_all['pid'].unique()[p]
        for city in df_all['city'].unique():
            prev_day_chose_aligned = np.nan
            prev_day_chose_orthogonal = np.nan
            prev_day_chose_vertical = np.nan
            for day in df_all['day'].unique():
                try:
                    df_all.loc[(df_all['city'] == city) & (df_all['day'] == day) & (df_all['pid'] == pid), 'first_path'] = df_all.loc[(df_all['city'] == city) & (df_all['day'] == day) & (df_all['pid'] == pid), 'path_chosen'].iloc[0]
                    df_all.loc[(df_all['city'] == city) & (df_all['day'] == day) & (df_all['pid'] == pid), 'first_path_orthogonal'] = df_all.loc[(df_all['city'] == city) & (df_all['day'] == day) & (df_all['pid'] == pid), 'chose_orthogonal'].iloc[0]
                    df_all.loc[(df_all['city'] == city) & (df_all['day'] == day) & (df_all['pid'] == pid), 'second_path'] = df_all.loc[(df_all['city'] == city) & (df_all['day'] == day) & (df_all['pid'] == pid), 'path_chosen'].iloc[1]
                    df_all.loc[(df_all['city'] == city) & (df_all['day'] == day) & (df_all['pid'] == pid), 'second_path_orthogonal'] = df_all.loc[(df_all['city'] == city) & (df_all['day'] == day) & (df_all['pid'] == pid), 'chose_orthogonal'].iloc[1]
                    df_all.loc[(df_all['city'] == city) & (df_all['day'] == day) & (df_all['pid'] == pid), 'third_path'] = df_all.loc[(df_all['city'] == city) & (df_all['day'] == day) & (df_all['pid'] == pid), 'path_chosen'].iloc[2]
                    df_all.loc[(df_all['city'] == city) & (df_all['day'] == day) & (df_all['pid'] == pid), 'third_path_orthogonal'] = df_all.loc[(df_all['city'] == city) & (df_all['day'] == day) & (df_all['pid'] == pid), 'chose_orthogonal'].iloc[2]

                    ## save previous day choice (only interested in t=1)
                    df_all.loc[(df_all['city'] == city) & (df_all['day'] == day) & (df_all['trial']==1) & (df_all['pid'] == pid), 'prev_day_chose_aligned'] = prev_day_chose_aligned
                    df_all.loc[(df_all['city'] == city) & (df_all['day'] == day) & (df_all['trial']==1) & (df_all['pid'] == pid), 'prev_day_chose_orthogonal'] = prev_day_chose_orthogonal
                    df_all.loc[(df_all['city'] == city) & (df_all['day'] == day) & (df_all['trial']==1) & (df_all['pid'] == pid), 'prev_day_chose_vertical'] = prev_day_chose_vertical

                    ## save day choice for subsequent day...
                    prev_day_chose_aligned = df_all.loc[(df_all['city'] == city) & (df_all['day'] == day) & (df_all['trial']==1) & (df_all['pid'] == pid), 'chose_aligned'].iloc[0]
                    prev_day_chose_orthogonal = df_all.loc[(df_all['city'] == city) & (df_all['day'] == day) & (df_all['trial']==1) & (df_all['pid'] == pid), 'chose_orthogonal'].iloc[0]
                    prev_day_chose_vertical = df_all.loc[(df_all['city'] == city) & (df_all['day'] == day) & (df_all['trial']==1) & (df_all['pid'] == pid), 'chose_vertical'].iloc[0]

                except:
                    pass
        
        ## iterate through each participant's dataset and set the previous choice
        prev_choice = None
        prev_choice_aligned = None
        prev_choice_orthogonal = None
        prev_choice_vertical = None
        for i, row in df_all.loc[df_all['pid'] == pid].iterrows():
            if pd.isna(row['path_chosen']): 
                continue
            if prev_choice is None: 
                df_all.at[i, 'prev_chose_aligned'] = np.nan
                df_all.at[i, 'prev_chose_orthogonal'] = np.nan
                df_all.at[i, 'prev_chose_vertical'] = np.nan
            else:
                df_all.at[i, 'prev_chose_vertical'] = prev_choice_vertical
                df_all.at[i, 'prev_chose_aligned'] = prev_choice_aligned
                df_all.at[i, 'prev_chose_orthogonal'] = prev_choice_orthogonal
                df_all.at[i, 'switched_axis'] = (prev_choice_aligned != row['chose_aligned']) or (prev_choice_orthogonal != row['chose_orthogonal'])
            prev_choice = row['path_chosen']
            prev_choice_aligned = row['chose_aligned']
            prev_choice_orthogonal = row['chose_orthogonal']
            prev_choice_vertical = row['chose_vertical']

        
        ### simulate each participant's choices to extract the missing trial info

        ## get envs
        try:
            id = id_mapping[pid][10:]
        except KeyError:
            raise KeyError(f'No id mapping for participant {pid}')
        try:
            try:
                with open('expt/assets/trial_sequences/env_objects/env_objects_{}.pkl'.format(id), 'rb') as f:
                    envs = pickle.load(f)
            except:
                with open('expt/assets/trial_sequences/rotated_env_objects/env_objects_{}.pkl'.format(id), 'rb') as f:
                    envs = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f'No env objects found for participant {pid} with id {id}')
        
        ## simulate
        agent = Farmer(N=8, context_prior=0.5)
        agent.run(params = None, hyperparams=None, agent = 'human', df_trials= df_all.loc[df_all['pid'] == pid],envs=envs, fit=False)

        ## extract cost info
        df_all.loc[df_all['pid'] == pid, 'observed_cost'] = agent.total_costs.flatten()
        df_all.loc[df_all['pid'] == pid, 'prev_observed_cost'] = df_all.loc[df_all['pid'] == pid, 'observed_cost'].shift(1)
        df_all.loc[df_all['pid'] == pid, 'day_cost'] = agent.day_costs.flatten()
        df_all.loc[df_all['pid'] == pid, 'path_len'] = agent.path_len.flatten()
        df_all.loc[df_all['pid'] == pid, 'path_quality'] = agent.path_quality.flatten()
        df_all.loc[df_all['pid'] == pid, 'prev_path_quality'] = df_all.loc[df_all['pid'] == pid, 'path_quality'].shift(1)
        
        ## first trial of each day should have no previous trial info?
        df_all.loc[(df_all['pid'] == pid) & (df_all['trial'] == 1), 'prev_observed_cost'] = np.nan
        df_all.loc[(df_all['pid'] == pid) & (df_all['trial'] == 1), 'prev_path_quality'] = np.nan
        df_all.loc[(df_all['pid'] == pid) & (df_all['trial'] == 1), 'prev_chose_aligned'] = np.nan
        df_all.loc[(df_all['pid'] == pid) & (df_all['trial'] == 1), 'prev_chose_orthogonal'] = np.nan
        df_all.loc[(df_all['pid'] == pid) & (df_all['trial'] == 1), 'prev_chose_vertical'] = np.nan
        
        ## extract overlap info
        df_all.loc[df_all['pid'] == pid, 'path_A_past_overlaps'] = agent.path_past_overlaps[:,:,:,0].flatten()
        df_all.loc[df_all['pid'] == pid, 'path_B_past_overlaps'] = agent.path_past_overlaps[:,:,:,1].flatten()
        df_all.loc[df_all['pid'] == pid, 'total_past_overlaps'] = agent.path_past_overlaps[:,:,:,0].flatten() + agent.path_past_overlaps[:,:,:,1].flatten()
        df_all.loc[df_all['pid'] == pid, 'path_A_past_observed_costs'] = agent.path_past_observed_costs[:,:,:,0].flatten()
        df_all.loc[df_all['pid'] == pid, 'path_B_past_observed_costs'] = agent.path_past_observed_costs[:,:,:,1].flatten()
        df_all.loc[df_all['pid'] == pid, 'path_A_past_observed_no_costs'] = agent.path_past_observed_no_costs[:,:,:,0].flatten()
        df_all.loc[df_all['pid'] == pid, 'path_B_past_observed_no_costs'] = agent.path_past_observed_no_costs[:,:,:,1].flatten()

        ## diffs (vertical - horizontal)
        # df_all.loc[(df_all['pid'] == pid)
        #            & (df_all['dominant_axis_A'] == 'vertical')
        #            , 'past_overlaps_diff'] = df_all.loc[(df_all['pid'] == pid) & (df_all['dominant_axis_A'] == 'vertical'), 'path_A_past_overlaps'] - df_all.loc[(df_all['pid'] == pid) & (df_all['dominant_axis_A'] == 'vertical'), 'path_B_past_overlaps']
        # df_all.loc[(df_all['pid'] == pid)
        #              & (df_all['dominant_axis_A'] == 'horizontal')
        #              , 'past_overlaps_diff'] = df_all.loc[(df_all['pid'] == pid) & (df_all['dominant_axis_A'] == 'horizontal'), 'path_B_past_overlaps'] - df_all.loc[(df_all['pid'] == pid) & (df_all['dominant_axis_A'] == 'horizontal'), 'path_A_past_overlaps']
        # df_all.loc[(df_all['pid'] == pid)
        #              & (df_all['dominant_axis_A'] == 'vertical')
        #              , 'observed_costs_diff'] = df_all.loc[(df_all['pid'] == pid) & (df_all['dominant_axis_A'] == 'vertical'), 'path_A_past_observed_costs'] - df_all.loc[(df_all['pid'] == pid) & (df_all['dominant_axis_A'] == 'vertical'), 'path_B_past_observed_costs']
        # df_all.loc[(df_all['pid'] == pid)
        #                 & (df_all['dominant_axis_A'] == 'horizontal')
        #                 , 'observed_costs_diff'] = df_all.loc[(df_all['pid'] == pid) & (df_all['dominant_axis_A'] == 'horizontal'), 'path_B_past_observed_costs'] - df_all.loc[(df_all['pid'] == pid) & (df_all['dominant_axis_A'] == 'horizontal'), 'path_A_past_observed_costs']
        # df_all.loc[(df_all['pid'] == pid)
        #                 & (df_all['dominant_axis_A'] == 'vertical')
        #                 , 'observed_no_costs_diff'] = df_all.loc[(df_all['pid'] == pid) & (df_all['dominant_axis_A'] == 'vertical'), 'path_A_past_observed_no_costs'] - df_all.loc[(df_all['pid'] == pid) & (df_all['dominant_axis_A'] == 'vertical'), 'path_B_past_observed_no_costs']
        # df_all.loc[(df_all['pid'] == pid)
        #                 & (df_all['dominant_axis_A'] == 'horizontal')
        #                 , 'observed_no_costs_diff'] = df_all.loc[(df_all['pid'] == pid) & (df_all['dominant_axis_A'] == 'horizontal'), 'path_B_past_observed_no_costs'] - df_all.loc[(df_all['pid'] == pid) & (df_all['dominant_axis_A'] == 'horizontal'), 'path_A_past_observed_no_costs']
        
        ## or, diffs (orthogonal - aligned)
        df_all.loc[(df_all['pid'] == pid)
                   & (df_all['aligned_path'] == 'b')
                   , 'past_overlaps_diff'] = df_all.loc[(df_all['pid'] == pid) & (df_all['aligned_path'] == 'b'), 'path_A_past_overlaps'] - df_all.loc[(df_all['pid'] == pid) & (df_all['aligned_path'] == 'b'), 'path_B_past_overlaps']
        df_all.loc[(df_all['pid'] == pid)
                     & (df_all['aligned_path'] == 'a')
                     , 'past_overlaps_diff'] = df_all.loc[(df_all['pid'] == pid) & (df_all['aligned_path'] == 'a'), 'path_B_past_overlaps'] - df_all.loc[(df_all['pid'] == pid) & (df_all['aligned_path'] == 'a'), 'path_A_past_overlaps']
        df_all.loc[(df_all['pid'] == pid)
                     & (df_all['aligned_path'] == 'b')
                     , 'observed_costs_diff'] = df_all.loc[(df_all['pid'] == pid) & (df_all['aligned_path'] == 'b'), 'path_A_past_observed_costs'] - df_all.loc[(df_all['pid'] == pid) & (df_all['aligned_path'] == 'b'), 'path_B_past_observed_costs']
        df_all.loc[(df_all['pid'] == pid)
                        & (df_all['aligned_path'] == 'a')
                        , 'observed_costs_diff'] = df_all.loc[(df_all['pid'] == pid) & (df_all['aligned_path'] == 'a'), 'path_B_past_observed_costs'] - df_all.loc[(df_all['pid'] == pid) & (df_all['aligned_path'] == 'a'), 'path_A_past_observed_costs']
        df_all.loc[(df_all['pid'] == pid)
                        & (df_all['aligned_path'] == 'b')
                        , 'observed_no_costs_diff'] = df_all.loc[(df_all['pid'] == pid) & (df_all['aligned_path'] == 'b'), 'path_A_past_observed_no_costs'] - df_all.loc[(df_all['pid'] == pid) & (df_all['aligned_path'] == 'b'), 'path_B_past_observed_no_costs']
        df_all.loc[(df_all['pid'] == pid)
                        & (df_all['aligned_path'] == 'a')
                        , 'observed_no_costs_diff'] = df_all.loc[(df_all['pid'] == pid) & (df_all['aligned_path'] == 'a'), 'path_B_past_observed_no_costs'] - df_all.loc[(df_all['pid'] == pid) & (df_all['aligned_path'] == 'a'), 'path_A_past_observed_no_costs']

        
        ## sanity check: total past overlaps should be the sum of path A and B past overlaps
        assert np.all(df_all.loc[df_all['pid'] == pid, 'total_past_overlaps'] == df_all.loc[df_all['pid'] == pid, 'path_A_past_overlaps'] + df_all.loc[df_all['pid'] == pid, 'path_B_past_overlaps']), \
            'Total past overlaps should be the sum of path A and B past overlaps for participant ' + pid
        
        ## extract CE choices
        df_all.loc[df_all['pid'] == pid, 'CE_action'] = agent.CE_actions.flatten()
    df_all['CE_action'] = df_all['CE_action'].replace({1: 'b', 0: 'a'})
    df_all['CE_action'] = df_all['CE_action'].astype(str)
    df_all['CE_chose_aligned'] = (df_all['CE_action'] == df_all['aligned_path']).astype(bool)
    df_all['CE_chose_orthogonal'] = (df_all['CE_action'] != df_all['aligned_path']).astype(bool)
    df_all.loc[(df_all['context'] == 'column') & (df_all['CE_chose_aligned'] == True), 'CE_chose_vertical'] = True
    df_all.loc[(df_all['context'] == 'column') & (df_all['CE_chose_aligned'] == False), 'CE_chose_vertical'] = False
    df_all.loc[(df_all['context'] == 'row') & (df_all['CE_chose_aligned'] == True), 'CE_chose_vertical'] = False
    df_all.loc[(df_all['context'] == 'row') & (df_all['CE_chose_aligned'] == False), 'CE_chose_vertical'] = True
    df_all['CE_human_consistent'] = (df_all['CE_action'] == df_all['path_chosen']).astype(bool)
        

    # last bit of cleaning of questionnaire data
    df_q = pd.DataFrame.from_dict(questionnaire)
    answers = ["extremely uncharacteristic of me", "somewhat uncharacteristic of me", "uncertain", "somewhat characteristic of me", "extremely characteristic of me"];
    for q in range(1, 18+1):
        df_q['NFC'+str(q)] = df_q['NFC'+str(q)].replace(answers, [1, 2, 3, 4, 5])
    df_q = df_q.replace('', np.nan)
    df_q = df_q.replace('nan', np.nan)
    df_q = df_q.replace('NaN', np.nan)
    df_q = df_q.replace('none', np.nan)
    df_q = df_q.replace('None', np.nan)
    reverse_items = [3, 4, 5, 7, 8, 9, 12, 16, 17]  # 1-indexed
    for i in reverse_items:
        col = f"NFC{i}"
        df_q[col] = 6 - df_q[col]  # reverse-score
    df_q['NFC_total'] = df_q[[f"NFC{i}" for i in range(1, 19)]].sum(axis=1)
    return df_all, df_q


## check counterbalancing - does each unrotated id have a rotated counterpart?
def check_counterbalance(df):
    unrotated_ids = []
    rotated_ids = []
    # for id in sorted(df['expt_info_filename'].unique()):
    for id in df['expt_info_filename'].unique():
        if isinstance(id, float):
            id = str(int(id))
        if id[-7:] == 'rotated':
            rotated_ids.append(id)
        else:
            unrotated_ids.append(id)
        if id[-7:] != 'rotated':
            if id+'_rotated' not in df['expt_info_filename'].unique():
                print('No rotated counterpart for id:', id)
    print('n unrotated ids:', len(unrotated_ids))
    print('n rotated ids:', len(rotated_ids))




## data-saving/dict stuff
data_keys = [
    'agent',
    'day',
    'grid',
    'trial',
    'start',
    'goal',
    'costs',
    'path_A',
    'path_B',
    'path_A_expected_cost',
    'path_B_expected_cost',
    'path_A_actual_cost',
    'path_B_actual_cost',
    'abstract_sequence_A',
    'abstract_sequence_B',
    'path_A_future_overlap',
    'path_B_future_overlap',
    'context_prior',
    'context_posterior',
    'optimal_costs',
    'actions',
    'CE_actions',
    'optimal_actions',
    'total_cost',
    'total_optimal_cost',
    'action_score',
    'cost_ratio',
    'n_steps',
    'actual_trajectory',
    'optimal_trajectory',
    'observations',
    'action_tree',
    'discounted_costs',
    'total_discounted_cost',
    'discounted_optimal_costs',
    'total_discounted_optimal_cost',
    'expected_LD',
    'expected_KL',
    'Q_values',
    'choice_probs',
    'leaf_visits',
    'CE_Q_values',

    ## GP-specific
    # 'true_k',
    # 'RPE',
    # 'posterior_mean',
    # 'theta_MLE',
]

## misc grid keys
grid_keys = [
    # 'grid',
    # 'env',
    'p_costs',
    'path_states',
    'path_actions',
    'starts',
    'goals',
    'context',
    'p0_overlaps'
]