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
# from MCTS import MonteCarloTreeSearch



## create a mountain environment
def make_env(N, true_k, kernel_params, metric):

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
    
    env = gym.make("mountains/MountainEnv-v0", N=N, true_k=true_k, kernel_params=kernel_params, metric=metric)
    return env




## Node class
class Node:

    # __slots__ = ['state', 'n_state_visits', 'cost', 'terminated', 'node_id', 'parent_node_ids', 'N', 'untried_actions', 'action_leaves']

    def __init__(self, state, cost, history, terminated, action_space, N):
        
        ## state info
        self.state = np.append(state, cost)
        self.n_state_visits = 0
        self.cost = cost
        self.terminated = terminated
        self.history = history
        self.node_id = str(np.append(self.history, self.state))
        self.state_id = str(state)
        self.parent_node_ids = []
        # self.children_node_ids = []
        self.N = N


        ## define valid actions
        self.untried_actions = list(range(action_space))
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
        action_leaves_msg = {action: leaf.performance if leaf is not None else None for action, leaf in self.action_leaves.items()}
        return "state {}: (visits={}, cost={:0.4f}, terminated={})\n{})".format(
                                                  self.state,
                                                  self.n_state_visits,
                                                  self.cost,
                                                  self.terminated,
                                                  action_leaves_msg
                                                  )

    ## select a random untried action
    def untried_action(self):
        action = random.choice(self.untried_actions)
        self.untried_actions.remove(action)
        return action
    
class Action_Node:

    def __init__(self, prev_state, action, next_state, terminated):
        self.prev_state = prev_state
        self.action = action
        self.total_simulation_cost = 0
        self.performance = None
        self.n_action_visits = 0
        self.next_state = next_state
        self.terminated = terminated
        self.node_id = str(self.prev_state) + str(self.action) #+ str(self.next_state)
        self.children={}
        self.children_ids = []

    def __str__(self):
        return "prev_state{}: (action={}, next_state={}, children={}, visits={}, performance={:0.4f})".format(
                                                  self.prev_state,
                                                  self.action,
                                                self.next_state,
                                                  self.children_ids,
                                                  self.n_action_visits,
                                                  self.performance,
                                                  )


## Tree class
class Tree:

    def __init__(self,N):
        # self.nodes = {}
        self.root = None
        self.N = N
        self.n_state_visits = np.zeros((N,N))

    ## check if node is expandable
    def is_expandable(self, node):
        return not node.terminated and len(node.untried_actions) > 0

    ## attach action leaf to child state
    def add_state_node(self, state, cost, history, terminated, action_space, parent=None):

        # ## check for existing state node
        # node_id = str(history)
        # if node_id in self.nodes:
        #     # print(state,"already exists")
        #     return self.nodes[node_id]

        
        ## create a new state node
        node_id = str(history)
        node = Node(state=state, cost=cost, history=history, terminated=terminated, action_space=action_space, N=self.N)
        # self.nodes.update({node_id: node})
        
        ## store parent-child relationships
        if parent is None:
            self.root = node
            # self.nodes[str(state)].parent = None
        else:
            node.parent_node_ids.append(parent.node_id)
            
            ## add this state node to the children of the previous action leaf
            parent.children_ids.append(node.node_id)
            parent.children[str(np.append(state, cost))] = node
            # parent.children[node.state_id] = node

        return node



    def get_children(self, node):
        children = []
        for a, leaf in node.action_leaves.items():
            if leaf is not None:
                # for node_id in leaf.children_ids:
                for child_key in leaf.children.keys():
                    child = leaf.children[child_key]
                    children.append(tuple((a, leaf, child_key, child)))
                    # children.append(tuple((a, self.nodes[node_id].state, self.nodes[node_id])))
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


    def print_tree(self, node, indent="", is_last=True):
        """
        Recursively print the tree structure with markers, visit counts, and values.

        Args:
        - node_id: The ID of the current node.
        - indent: The current indentation string for formatting.
        - is_last: Whether this node is the last child of its parent.
        """
        # Get the current node
        # node = self.nodes[node_id]
        node_label = f"{node.state}"

        # Add branch marker
        branch = "└── " if is_last else "├── "
        print(f"{indent}{branch}Node: {node_label}")

        # Update indentation for children
        child_indent = indent + ("    " if is_last else "│   ")

        # Group children by action
        children_by_action = {}
        for action, leaf, child_id, child_node in self.get_children(node):
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
            action_label = f"Action {action}, (n_v: {leaf.n_action_visits}, perf: {leaf.performance:.2f})"

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

        ## delete subtree for the other state reachable from the root-action pair
        self.root.action_leaves[action].children = {str(next_state): self.root.action_leaves[action].children[str(next_state)]}

        ## update the root
        self.root = self.root.action_leaves[action].children[str(next_state)]

        


            




        


    
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

## value iteration
def value_iteration(dp_costs, goal, max_iters = 1000, theta = 0.0001, discount = 0.99):

    N = len(dp_costs)
    n_actions = 4

    ## init actions
    action_directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    
    ## init tables
    V = np.zeros((N, N))
    A = np.zeros((N, N))
    Q = np.zeros((N, N, n_actions))

    ## determine whether to use true costs or inferred costs
    # if self.known_costs:
    #     dp_costs = self.costs.copy()
    # else:
    #     dp_costs = self.posterior_mean.reshape(self.N, self.N).copy()

    ## set cost of goal to 0
    dp_costs[goal[0], goal[1]] = 0

    assert np.all(dp_costs <= 0), 'costs are not all negative: {}'.format(dp_costs)

    ## loop through states
    for i in range(max_iters):
        delta = 0
        for x in range(N):
            for y in range(N):
                
                ## (make sure the goal state has value 0)
                if (x, y) == tuple(goal):
                    # V[x, y] = 0
                    continue

                v = V[x, y]
                q = np.zeros(n_actions)

                ## loop through actions and get the discounted value of each of the next states
                for a in range(n_actions):

                    ## allow wall moves
                    # next_state = np.clip([x, y] + self._action_to_direction[a], 0, self.N-1)
                    # q[a] = dp_costs[next_state[0], next_state[1]] + discount*V[next_state[0], next_state[1]]

                    ## or, don't allow wall moves
                    next_state = [x, y] + action_directions[a]
                    if (next_state[0] >= 0) and (next_state[0] < N) and (next_state[1] >= 0) and (next_state[1] < N):
                        q[a] = dp_costs[next_state[0], next_state[1]] + discount*V[next_state[0], next_state[1]]
                    else:
                        q[a] = np.nan

                    ## update the Q-table
                    Q[x, y, a] = q[a]

                ## use the best action to update the value of the current state
                V[x, y] = np.nanmax(q)

                # A[x, y] = np.argmax(q)
                A[x, y] = argm(q, np.nanmax(q))

                ## check if converged
                delta = max(delta, np.abs(v - V[x, y]))
        
        if delta < theta:
            # print('DP converged after {} iterations'.format(i))
            break

        if i == max_iters-1:
            print('DP did not converge after {} iterations'.format(i))

    ## need to check if this has lead to a valid policy

    return V, Q, A

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


## data-saving/dict stuff
data_keys = [
    'agent',
    'mountain',
    'episode',
    'start',
    'goal',
    'actual_cost',
    'optimal_cost',
    'action_score',
    'cost_ratio',
    'n_steps',
    'actual_trajectory',
    'optimal_trajectory',
    'observations',
    'search_attempts',
    'action_tree',

    ## GP-specific
    # 'true_k',
    # 'RPE',
    # 'posterior_mean',
    # 'theta_MLE',
]