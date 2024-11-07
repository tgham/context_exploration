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
# from MCTS import MonteCarloTreeSearch



## create a mountain environment
def make_env(N, params, metric, true_k, inf_k, render_mode, r_noise):

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
    
    env = gym.make("mountains/MountainEnv-v0", N=N, params=params, metric=metric, true_k=true_k, inf_k=inf_k, render_mode=render_mode, r_noise=r_noise)
    return env




## Node class
class Node:

    def __init__(self, state, cost, terminated, action_space, N):
        
        ## state info
        self.state = state
        self.n_state_visits = 0
        self.cost = cost
        self.terminated = terminated
        self.identifier = str(self.state)
        self.parent_identifiers = []
        # self.children_identifiers = []
        self.N = N

        ## action leaves
        self.action_leaves = {0: None, 1: None, 2: None, 3: None}
        # for leaf in self.action_leaves.keys():
        #     self.action_leaves[leaf] = Action_Node(next_state=None, cost=None, terminated=None) ## this is just a placeholder until proper expansion

        ## define valid actions
        self.untried_actions = list(range(action_space))


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

    def __init__(self, prev_state, action, next_state, next_cost, terminated):
        self.prev_state = prev_state
        self.action = action
        self.total_simulation_cost = 0
        self.performance = 0
        self.n_action_visits = 0
        self.next_state = next_state
        self.next_cost = next_cost
        self.terminated = terminated
        self.identifier = str(self.prev_state) + str(self.action) + str(self.next_state)

    def __str__(self):
        return "prev_state{}: (action={}, next_state={}, visits={}, performance={:0.4f})".format(
                                                  self.prev_state,
                                                  self.action,
                                                self.next_state,
                                                  self.n_action_visits,
                                                  self.performance)



## Tree class
class Tree:

    def __init__(self,N):
        self.nodes = {}
        self.root = None
        self.N = N
        self.n_state_visits = np.zeros((N,N))

    ## check if node is expandable
    def is_expandable(self, node):
        if node.terminated:
            return False
        if len(node.untried_actions) > 0:
            return True
        return False

    ## iterate through the tree
    def iter(self, identifier, depth, last_node_flags):
        if identifier is None:
            node = self.root
        else:
            node = self.nodes[identifier]

        if depth == 0:
            yield "", node
        else:
            yield vertical_lines(last_node_flags) + horizontal_line(last_node_flags), node

        children = [self.nodes[identifier] for identifier in node.children_identifiers]
        last_index = len(children) - 1

        depth += 1
        for index, child in enumerate(children):
            last_node_flags.append(index == last_index)
            for edge, node in self.iter(child.identifier, depth, last_node_flags):
                yield edge, node
            last_node_flags.pop()

    def add_node(self, node, parent=None):
        self.nodes.update({node.identifier: node})

        if parent is None:
            self.root = node
            self.nodes[node.identifier].parent = None
        else:
            self.nodes[parent.identifier].children_identifiers.append(node.identifier)
            self.nodes[node.identifier].parent_identifier=parent.identifier

    def add_state_node(self, state, cost, terminated, action_space, parent=None):

        ## check for existing state node
        if str(state) in self.nodes:
            # print(state,"already exists")
            return self.nodes[str(state)]
        
        ## else, create a new state node
        node = Node(state=state, cost=cost, terminated=terminated, action_space=action_space, N=self.N)
        self.nodes.update({str(state): node})
        
        ## store parent-child relationships
        if parent is None:
            self.root = node
            # self.nodes[str(state)].parent = None
        else:
            node.parent_identifiers.append(parent.identifier)
            ## don't need to do children, since these are already in the aciotn_leaves

        return node



    def children(self, node):
        children = []
        for identifier in self.nodes[node.identifier].children_identifiers:
            children.append(self.nodes[identifier])
        return children

    def parent(self, node):
        parent_identifier = self.nodes[node.identifier].parent_identifier
        if parent_identifier is None:
            return None #i.e. root reached, bc it has no parent
        else:
            return self.nodes[parent_identifier]

    def show(self):
        lines = ""
        for edge, node in self.iter(identifier=None, depth=0, last_node_flags=[]):
            lines += "{}{}\n".format(edge, node)
        print(lines)


    ## calculate value of each S-A node
    def action_tree(self):

        ## average over all nodes in the tree
        tree_q = np.zeros((self.N, self.N, 4))
        tree_q_counts = np.zeros((self.N, self.N, 4))
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            for child in self.children(node):
                tree_q[node.state[0], node.state[1], child.action] += child.performance
                tree_q_counts[node.state[0], node.state[1], child.action] += 1
                queue.append(child)
        tree_q /= tree_q_counts


        ## or just standard saving??
        # tree_q = np.zeros((N, N, 4)) + np.nan
        # tree_q_counts = np.zeros((N, N, 4))
        # queue = deque([tree.root])
        # while queue:
        #     node = queue.popleft()
        #     for child in tree.children(node):
        #         tree_q[node.state[0], node.state[1], child.action] = child.performance
        #         tree_q_counts[node.state[0], node.state[1], child.action] += 1
        #         queue.append(child)

        return tree_q
    
## some extra functions related to the tree
def vertical_lines(last_node_flags):
    vertical_lines = []
    vertical_line = '\u2502'
    for last_node_flag in last_node_flags[0:-1]:
        if last_node_flag == False:
            vertical_lines.append(vertical_line + ' ' * 3)
        else:
            # space between vertical lines
            vertical_lines.append(' ' * 4)
    return ''.join(vertical_lines)

def horizontal_line(last_node_flags):
    horizontal_line = '\u251c\u2500\u2500 '
    horizontal_line_end = '\u2514\u2500\u2500 '
    if last_node_flags[-1]:
        return horizontal_line_end
    else:
        return horizontal_line
    

### misc utils

## calculate the angle between two nodes
def node_angle(a,b):
    rad = np.arctan2(b[1]-a[1], b[0]-a[0])
    ang = np.abs(np.degrees(rad))
    return ang

## random choice between multiple minima/maxima
def argm(x, extreme_val):
    indices = np.where(x == extreme_val)[0]
    return np.random.choice(indices)