from enum import Enum
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register, registry
import pygame
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
from minimax_tilting_sampler import TruncatedMVN
import ast
from scipy.spatial import cKDTree as KDTree
import cProfile
import pstats
import subprocess
import time
from numba import jit, njit




## create a mountain environment
def make_env(N, n_episodes, expt, beta_params, metric, seed=None):

    ## register env
    
    # Unregister the environment if it's already registered
    env_id = "mountains/MountainEnv-v0"
    if env_id in registry:
        del registry[env_id]

    # Re-register the updated environment
    register(
        id=env_id,
        entry_point='mountains.envs:MountainEnv',
        max_episode_steps=100,
        kwargs={"size": N},
    )
    
    env = gym.make("mountains/MountainEnv-v0", N=N, n_episodes=n_episodes, expt=expt, beta_params=beta_params, metric=metric, seed=seed)
    return env




## Node class
class Node:

    # __slots__ = ['state', 'n_state_visits', 'cost', 'terminated', 'node_id', 'parent_node_ids', 'N', 'untried_actions', 'action_leaves']

    def __init__(self, state, cost, node_id, goal, terminated, episode, n_afc, N):
        
        ## state info
        self.state = np.append(state, cost) ## in the 2AFC case, this amounts to current state + costs that have just been observed on prior simulated episode
        self.n_state_visits = 0
        self.cost = cost
        self.episode = episode
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
        return "state {}: (episode={}, visits={}, terminated={})\n{})".format(
                                                  self.state,
                                                    self.episode,
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

    def __init__(self, prev_state, action, next_state, terminated, episode, parent_id):
        self.prev_state = prev_state
        self.action = action ## in 2AFC, this specifies the path ID (i.e. 0 or 1)
        self.total_simulation_cost = 0
        self.performance = None
        self.n_action_visits = 0
        self.next_state = next_state
        self.terminated = terminated
        self.episode = episode
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
    def add_state_node(self, state, cost, node_id, goal, terminated, episode, n_afc, parent=None):

        # ## check for existing state node
        # node_id = str(history)
        # if node_id in self.nodes:
        #     # print(state,"already exists")
        #     return self.nodes[node_id]

        
        ## create a new state node
        node = Node(state=state, cost=cost, node_id=node_id, goal=goal, terminated=terminated, episode = episode, n_afc=n_afc, N=self.N)
        
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


    def print_tree(self, node, indent="", is_last=True, dummy=False):
        """
        Recursively print the tree structure with markers, visit counts, and values.

        Args:
        - node_id: The ID of the current node.
        - indent: The current indentation string for formatting.
        - is_last: Whether this node is the last child of its parent.
        - dummy: Whether to print display action leaves that don't have any children.
        """
        # Get the current node
        # node = self.nodes[node_id]
        if dummy:
            if node is None:
                return
            else:
                node_label = f"{node.state}"
        else:
            node_label = f"{node.state}"
        episode_label = f"{node.episode}"

        # Add branch marker
        branch = "└── " if is_last else "├── "
        print(f"{indent}{branch}Node: {node_label}, Episode: {episode_label}")

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
            action_label = f"Action {action}, (n_v: {leaf.n_action_visits}, branch factor: {len(children)}, perf: {leaf.performance:.2f})"

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

                # Recursively print the child node
                self.print_tree(
                    child_node,
                    indent=sub_child_indent,
                    is_last=is_child_last,
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


## sample from the GP
def sample(mean, K, sigma=0.01, high_cost=-0.9, low_cost=-0.1):
    if sigma is None:
        sigma = 0.01 #i.e. just to add to the diagonal of the kernel matrix

    N = int(np.sqrt(len(mean)))

    ## check kernel is valid
    k_check(K)

    # sample
    # if mean is None:
    #     mean = np.zeros(self.N**2)
    # mean = np.zeros(self.N**2)
    # samples = np.random.multivariate_normal(mean, K).reshape(self.N, self.N)

    #normalise
    # high_cost = self.high_cost
    # low_cost = self.low_cost
    # samples = high_cost + (low_cost - high_cost) * (samples - np.min(samples)) / (np.max(samples) - np.min(samples))


    ## or truncated
    lb = np.zeros(N**2) + high_cost
    ub = np.zeros(N**2) + low_cost
    K_tmp = K + sigma**2 * np.eye(N**2)
    tmvn = TruncatedMVN(mean, K_tmp, lb, ub)
    samples = tmvn.sample(1)
    samples = samples.reshape(N, N)

    return samples

## check that kernel is PSD and symmetric
def k_check(K):
    symm = np.allclose(K,K.T)
    if not symm:
        warnings.warn("Kernel matrix is not symmetric.", UserWarning)
    
    eigenvalues = np.linalg.eigvals(K)
    psd = np.all(eigenvalues >= -1e-10)
    if not psd:
        warnings.warn("Kernel matrix is not positive semi-definite.", UserWarning)

    return np.any([not symm, not psd])


## parse strings to lists
def parse_lists(df):
    cols = df.columns[2:]
    for key in cols:
        try:
            df[key] = df[key].apply(lambda x: np.array(ast.literal_eval(x)))
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



## data-saving/dict stuff
data_keys = [
    'agent',
    'mountain',
    'episode',
    'start',
    'goal',
    'costs',
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
    'search_attempts',
    'action_tree',
    'discounted_costs',
    'total_discounted_cost',
    'discounted_optimal_costs',
    'total_discounted_optimal_cost',
    'expected_LD',
    'expected_KL',

    ## GP-specific
    # 'true_k',
    # 'RPE',
    # 'posterior_mean',
    # 'theta_MLE',
]