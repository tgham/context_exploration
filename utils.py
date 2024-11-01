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

    def __init__(self, state, action, action_space, reward, terminal, N, prev_state=None):
        # self.untried_actions = list(range(action_space))
        self.state = state
        self.total_simulation_reward  = 0
        self.num_visits = 0
        self.performance = 0
        self.action = action
        self.reward = reward
        self.terminal = terminal
        self.identifier = str(uuid.uuid1())
        # self.identifier = str(self.state) + ', '+str(self.action)
        self.parent_identifier = None
        self.children_identifiers = []
        self.N = N
        self.prev_state = prev_state

        ## define valid actions
        self.untried_actions = list(range(action_space))
        row, col = self.state
        if row == self.N-1:
            self.untried_actions.remove(0)
        if row == 0:
            self.untried_actions.remove(2)
        if col == self.N-1:
            self.untried_actions.remove(1)
        if col == 0:
            self.untried_actions.remove(3)
        
        ## remove the action that takes P back to prev state
        # if self.prev_state is not None:
        #     prev_row, prev_col = self.prev_state
        #     if row == prev_row:
        #         if col < prev_col:
        #             try:
        #                 self.untried_actions.remove(1)
        #             except:
        #                 pass
        #         else:
        #             try:
        #                 self.untried_actions.remove(3)
        #             except:
        #                 pass
        #     if col == prev_col:
        #         if row < prev_row:
        #             try:
        #                 self.untried_actions.remove(0)
        #             except:
        #                 pass
        #         else:
        #             try:
        #                 self.untried_actions.remove(2)
        #             except:
        #                 pass


    def __str__(self):
        return "{}: (action={}, visits={}, reward={:d}, ratio={:0.4f})".format(
                                                  self.state,
                                                  self.action,
                                                  self.num_visits,
                                                  int(self.total_simulation_reward),
                                                  self.performance)

    ## select a random untried action
    def untried_action(self):
        action = random.choice(self.untried_actions)
        self.untried_actions.remove(action)
        return action
    


## Tree class
class Tree:

    def __init__(self):
        self.nodes = {}
        self.root = None

    ## check if node is expandable
    def is_expandable(self, node):
        if node.terminal:
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