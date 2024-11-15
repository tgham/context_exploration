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
def make_env(N, params, metric, true_k, inf_k, known_costs, render_mode, r_noise):

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
    
    env = gym.make("mountains/MountainEnv-v0", N=N, params=params, metric=metric, true_k=true_k, inf_k=inf_k, known_costs=known_costs, render_mode=render_mode, r_noise=r_noise)
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


        ## define valid actions
        self.untried_actions = list(range(action_space))
        # row, col = self.state
        # if row == self.N-1:
        #     self.untried_actions.remove(0)
        # if row == 0:
        #     self.untried_actions.remove(2)
        # if col == self.N-1:
        #     self.untried_actions.remove(1)
        # if col == 0:
        #     self.untried_actions.remove(3)

        ## action leaves
        self.action_leaves = {a: None for a in self.untried_actions}
        # self.action_leaves = {0: None, 1: None, 2: None, 3: None}
        # for action in range(action_space):
        #     self.action_leaves[action] = Action_Node(prev_state=self.state, action=action, next_state=None, next_cost=None, terminated=None) 
        # for leaf in self.action_leaves.keys():
        #     self.action_leaves[leaf] = Action_Node(next_state=None, cost=None, terminated=None) ## this is just a placeholder until proper expansion


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
        self.performance = None
        self.n_action_visits = 0
        self.next_state = next_state
        self.next_cost = next_cost
        self.terminated = terminated
        self.identifier = str(self.prev_state) + str(self.action) + str(self.next_state)

    def __str__(self):
        return "prev_state{}: (action={}, next_state={}, next_cost={}, visits={}, performance={:0.4f})".format(
                                                  self.prev_state,
                                                  self.action,
                                                self.next_state,
                                                self.next_cost,
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
        return not node.terminated and len(node.untried_actions) > 0

    ## add node
    def add_state_node(self, state, cost, terminated, action_space, parent=None):

        ## check for existing state node
        state_str = str(state)
        if state_str in self.nodes:
            # print(state,"already exists")
            return self.nodes[state_str]
        
        ## else, create a new state node
        node = Node(state=state, cost=cost, terminated=terminated, action_space=action_space, N=self.N)
        self.nodes.update({state_str: node})
        
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

    ## calculate value of each S-A node
    def action_tree(self):

        tree_q = np.zeros((self.N,self.N,4)) + np.nan
        for sstate in self.nodes.keys():
            state = self.nodes[sstate].state
            for a in self.nodes[sstate].action_leaves.keys():
                try:
                    tree_q[state[0], state[1], a] = self.nodes[sstate].action_leaves[a].performance
                except:
                    pass

        return tree_q
    
    

### misc utils

## random choice between multiple minima/maxima
def argm(x, extreme_val):
    indices = np.where(x == extreme_val)[0]
    return np.random.choice(indices)