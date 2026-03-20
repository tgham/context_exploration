from abc import ABC, abstractmethod
import random
from math import sqrt, log
from bisect import bisect_left
from utils import Action_Node, Tree, make_env, argm, data_keys, KL_divergence, get_next_state
import copy
import numpy as np
from tqdm.auto import tqdm
import os
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from scipy import special
from scipy.special import softmax

from plotter import *
# from agents import Farmer

## base class 
class MonteCarloTreeSearch():

    def __init__(self, env, tree, exploration_constant=2, discount_factor=0.99, horizon=None):
        
        ## initialize tree and and MCTS params
        self.tree = tree
        self.discount_factor = discount_factor
        self.exploration_constant = exploration_constant
        if horizon is None:
            self.horizon = env.n_trials-1 #default to n_trials-1 (i.e. look to the end of the day, which is up to n_trials-1 from the root)
        else:
            self.horizon = horizon
        
        ## init root info
        self.refresh_env(env)
        self.n_afc = self.env.n_afc

        ## create id for root node
        node_id = self.init_node_id(self.env.obs, None)

        ## add state node to the tree
        self.tree.add_state_node(state = env.current, node_id=node_id, reward=None, terminated=False, trial = self.root_trial, n_afc=self.n_afc, parent=None, 
                                 )

    
    ### abstract methods

    ## create node id, which represents the agent's current state of knowledge
    @abstractmethod
    def init_node_id(self, obs=None, init_info_state=None):
        pass

    ## update MCTS object with misc info from the env
    @abstractmethod
    def refresh_env(self, env=None):
        pass
    
    ## rollout policy
    @abstractmethod
    def rollout_policy(self, action_leaf):
        pass
    
    ## debugging method for checking if node's belief state matches the env state
    @abstractmethod
    def check_node(self, node):
        pass
    

    
    ### general MCTS methods
    
    ## expand the action space of a node
    def expand(self, node):
        assert self.env.sim, 'env is not in sim mode'

        ### take action (or path) and get new state
        action = node.untried_action()
            
        ## create new action leaf and attach to node
        node.action_leaves[action] = Action_Node(action=action, trial=node.trial, parent_id=node.node_id)
        node.action_leaves[action].performance = 0
        
        return node.action_leaves[action]
    

    ## take an action according to the tree policy, i.e. take the best UCT child and see where it takes you
    def traverse_tree(self):

        ## initialise the tree
        node = self.tree.root
        node_trial = node.trial

        ## create a record of the nodes/leaves visited in the tree
        self.tree_rewards = [] ## i.e. the reward associated with each traversal of the tree *under the tree policy*. Hence, this does not include the reward of the current state, which is the starting point of the tree policy, nor does it include the reward of expansion.
        self.tree_actions = [] ## i.e. the states and actions visited in the tree. This *does* include the root, because it is from the root that we move to the next leaf (and then next node). 
        self.node_id_path = []
        
        ## loop until you reach a leaf node or terminal state
        assert self.env.sim, 'env is not in sim mode'
        terminated=False
        truncated = False
        while not terminated and not truncated:
            self.check_node(node)

            ## expansion step
            if self.tree.is_expandable(node):
                action_leaf = self.expand(node)
                self.tree_actions.append(action_leaf.action)
                self.node_id_path.append(node.node_id)

                return action_leaf
                
            ## selection step
            else:

                ## get the best child
                action_leaf = self.best_child(node)
                assert self.env.trial == action_leaf.trial, 'trial mismatch between env and tree\n env: {} \n tree: {}\n MCTS: {}'.format(self.env.trial, action_leaf.trial, self.root_trial)

                ## update the tree path
                self.tree_actions.append(action_leaf.action)
                self.node_id_path.append(node.node_id)

                ## move in env
                step_obs, rewards, terminated, truncated, _ = self.env.step(action_leaf.action)
                self.tree_rewards.append(sum(rewards))

                ## continue down the tree if not terminated (NB: WE MAY ACTUALLY WANT TO STILL CREATE THE NODE FOR THE TERMINAL STATE)
                if not terminated and not truncated:

                    ## get the next node id, i.e. the informational state after taking this path
                    next_node_id = self.init_node_id(step_obs, action_leaf.parent_id)
                    # print('tree actions and states: {}, {}, step_obs: {}, next node id: {}'.format(self.tree_actions, self.tree_rewards, step_obs, next_node_id))
                    node_trial += 1
                    assert node_trial == action_leaf.trial+1, 'trial mismatch between env and tree after step\n env: {} \n tree: {}'.format(node_trial, action_leaf.trial+1)

                    ## see if the next state node already exists as a child of this action leaf
                    if next_node_id in action_leaf.children:
                        node = action_leaf.children[next_node_id]
                    else:

                        ## create new node
                        node = self.tree.add_state_node(state = self.env.current, node_id=next_node_id, reward = rewards, terminated=terminated, trial = node_trial, n_afc=self.n_afc, parent=action_leaf,
                                                        )


        ## if terminal node, there are no more action leaves to choose from
        if terminated or truncated:
            action_leaf = None

        return action_leaf
    
    ## rollout policy
    def rollout(self, action_leaf):

        ## if no action leaf because tree policy has reached a terminal node, return None
        if action_leaf is None:
            return None

        ## first need to get the starting reward r, which is essentially the reward of choice that corresponds to the action leaf
        first_trial = action_leaf.trial
        _, rewards, terminated, truncated, _ = self.env.step(action_leaf.action)
        total_reward = sum(rewards)

        ## if final trial, just stop here
        if terminated or truncated:
            self.tree_rewards.append(total_reward)
            return total_reward
        
        ## else, loop through remaining trials
        depth = 0
        remaining_ro_rewards = []
        while not terminated and not truncated:
            depth+=1
            ro_action = self.rollout_policy()
            _, rewards, terminated, truncated, _ = self.env.step(ro_action)
            total_reward += sum(rewards) * self.discount_factor**depth
            remaining_ro_rewards.append(total_reward)

        self.tree_rewards.append(total_reward)
        assert len(remaining_ro_rewards)+first_trial+1 == self.horizon_trial + 1, 'remaining RO rewards do not match number of trials\n n remaining RO rewards: {}, n trials: {}'.format(len(remaining_ro_rewards), self.horizon_trial + 1)
        return total_reward 


    ## backup rewards until you reach the root
    def backup(self):
        tree_len = len(self.tree_rewards)
        assert tree_len == len(self.tree_actions), 'tree rewards and path lengths do not match\n n tree rewards: {} \n n tree path: {}\ntree rewards: {}\n tree path: {}'.format(len(self.tree_rewards), len(self.tree_actions), self.tree_rewards, self.tree_actions)

        ## efficiently precompute discounted returns via backward pass
        discounted_returns = [0.0] * tree_len
        discounted_returns[-1] = self.tree_rewards[-1]
        for i in range(tree_len - 2, -1, -1):
            discounted_returns[i] = self.tree_rewards[i] + self.discount_factor * discounted_returns[i + 1]

        ## Loop through the tree path
        node = self.tree.root
        for depth, action in enumerate(self.tree_actions):

            ## Get the corresponding action leaf
            action_leaf = node.action_leaves[action]

            ## Discounted reward from the current node to the terminal node
            discounted_reward = discounted_returns[depth]

            ## update visit counts and performance estimates
            action_leaf.n_action_visits += 1
            node.n_state_visits += 1

            ## Incremental average update for performance
            action_leaf.performance += (
                (discounted_reward - action_leaf.performance) / action_leaf.n_action_visits
            )

            ## save per-node max and min Q values to normalise Qs in UCT calculation
            if action_leaf.performance > node.max_Q:
                node.max_Q = action_leaf.performance
            if action_leaf.performance < node.min_Q:
                node.min_Q = action_leaf.performance

            
            ## Move to the next node in the path if not at the end
            if depth < tree_len - 1:
                next_node_id = self.node_id_path[depth+1]
                node = action_leaf.children[next_node_id]

            
            # ### some lists for debugging 

            # ## debugging: save updates applied to the first node
            # if depth == 0:
            #     to_append = [np.nan] * self.n_afc
            #     to_append[action] = discounted_reward
            #     self.first_node_updates.append(to_append)
            #     self.first_node_updates_by_depth[tree_len-1].append(to_append)
            
            # ## save rewards of each step in the tree - i.e. the reward of making each move in the tree
            # to_append = [np.nan] * self.n_afc
            # to_append[action] = self.tree_rewards[depth]
            # self.tree_reward_tracker[depth].append(to_append)

            # ## updates, conditional on first action
            # first_action = self.tree_actions[0]
            # to_append = [np.nan] * self.n_afc
            # to_append[first_action] = self.tree_rewards[depth]
            # self.conditional_tree_reward_tracker[action][depth].append(to_append)




    ## calculate E-E value
    def compute_UCT(self, node, action_leaf, min_Q=None, max_Q=None): 
        assert action_leaf.n_action_visits > 0 or action_leaf.terminated, 'action leaf has not been visited: {}'.format(action_leaf)
        
        ## standard case
        # exploitation_term = action_leaf.performance
        # exploration_term = self.exploration_constant * sqrt(log(node.n_state_visits) / action_leaf.n_action_visits)
        
        ### or, min-max normalisation based on min and max for that node
        norm_term = max_Q - min_Q + 1e-8 ## add small constant to avoid divide by zero
        exploitation_term = (action_leaf.performance - min_Q) / norm_term
        exploration_term = self.exploration_constant * sqrt(log(node.n_state_visits) / action_leaf.n_action_visits)
        
        return exploitation_term + exploration_term

    
    ## argmax based on UCT values?
    def best_child(self, node):

        min_Q, max_Q = node.min_Q, node.max_Q
        norm_term = max_Q - min_Q + 1e-8
        log_N = log(node.n_state_visits)
        c = self.exploration_constant

        best_leaf = None
        best_uct = -float('inf')
        n_ties = 0

        for leaf in node.action_leaves.values():
            uct = (leaf.performance - min_Q) / norm_term + c * sqrt(log_N / leaf.n_action_visits)
            if uct > best_uct:
                best_uct = uct
                best_leaf = leaf
                n_ties = 1
            elif uct == best_uct:
                n_ties += 1
                if np.random.randint(n_ties) == 0:
                    best_leaf = leaf

        return best_leaf


class MonteCarloTreeSearch_AFC(MonteCarloTreeSearch):

    def __init__(self, env, tree, exploration_constant=2, discount_factor=0.99, horizon=None):
        super().__init__(env, tree, exploration_constant, discount_factor, horizon)

    ## node_ids are defined by the informational state, i.e. the counts of low and high cost states in each cell
    def init_node_id(self, obs=None, parent_node_id=None):
        
        ## uses sparse representation: tuple of ((i, j), (low_count, high_count)) for observed cells only

        # Initialize counts from parent node_id (sparse tuple) if provided
        if parent_node_id is not None:
            counts = dict(parent_node_id)
        else:
            counts = {}
        
        # Add new observations - optimized loop
        high_cost = self.env.high_cost
        for i, j, c in obs:
            key = (int(i), int(j))
            prev = counts.get(key)
            if prev is None:
                counts[key] = (0, 1) if c == high_cost else (1, 0)
            elif c == high_cost:
                counts[key] = (prev[0], prev[1] + 1)
            else:
                counts[key] = (prev[0] + 1, prev[1])
        
        return tuple(sorted(counts.items()))
    
    ## in AFC, there is no meaningful state of the MDP, so belief state just contains the trial number
    def check_node(self, node):
        # assert np.array_equal(node.belief_state[:2*self.n_afc].reshape(self.n_afc,2), self.env.current), 'mismatch between node and env state\n node: {} \n env: {}'.format(node.belief_state[:2*self.n_afc].reshape(self.n_afc,2), self.env.current)
        # assert node.belief_state[0] == self.env.trial, 'mismatch between node and env trial\n node: {} \n env: {}'.format(node.belief_state[0], self.env.trial)
        assert node.trial == self.env.trial, 'mismatch between node and env trial\n node: {} \n env: {}'.format(node.trial, self.env.trial)


    ## when the external environment has changed, we need to update the MCTS object's version to reflect this
    def refresh_env(self, env=None):

        if env is not None:
            self.env = env.unwrapped

        ## i.e. the trial that the agent is current faced with in the real env
        self.root_trial = self.env.trial 

        ## set the horizon_trial - i.e. the trial at which search terminates
        self.horizon_trial = min(self.root_trial + self.horizon, self.env.n_trials-1)
        self.env.set_trunc_trial(self.horizon_trial)


    ## rollout policy (greedy or random)
    def rollout_policy(self):

        ### greedy:

        ## get the total reward of the paths
        # t = self.env.trial
        # path_rewards = []
        # for action in range(self.n_afc):
        #     path = self.env.path_states[t][action]
        #     path_weight_idx = self.env.path_weights[t][action]
        #     weighted_rewards = [float(self.env.costs[x, y]) * self.env.sim_weight_map[path_weight_idx[k]] for k, (x, y) in enumerate(path)]
        #     path_rewards.append(sum(weighted_rewards))
        
        # ## take greedy action
        # best_ro_reward = max(path_rewards)
        # ro_action = path_rewards.index(best_ro_reward)

        
        ### RANDOM: randomly choose between the paths
        ro_action = random.choice(range(self.n_afc))

        return ro_action
    

# 2. Bandit-specific MCTS
class MonteCarloTreeSearch_Bandit(MonteCarloTreeSearch):
    """MCTS subclass with bandit-appropriate node IDs and rollout policy."""

    def __init__(self, env, tree, exploration_constant=2,
                 discount_factor=0.99, horizon=None):
        super().__init__(env, tree, exploration_constant,
                         discount_factor, horizon)

    
    ## node_id is the sufficient statistic for the belief state in the bandit: the counts of successes and failures for each arm
    ## stored as a flat tuple of length n_arms*2: (s0, f0, s1, f1, ...)
    def init_node_id(self, obs=None, parent_node_id=None):
        """
        Node ID = sufficient statistic for Beta-Bernoulli bandit:
        per-arm (n_successes, n_failures) counts, stored as a flat tuple.
        """
        if parent_node_id is not None:
            counts = list(parent_node_id)
        else:
            counts = [0] * (self.n_afc * 2)

        if len(obs) == 0:
            return tuple(counts)

        ## fast path for single observation (common case during tree traversal)
        if obs.ndim == 2 and obs.shape[0] == 1:
            arm = int(obs[0, 0])
            if obs[0, 1] == 1:
                counts[arm * 2] += 1
            else:
                counts[arm * 2 + 1] += 1
            return tuple(counts)

        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        for action, reward in obs:
            arm = int(action)
            if reward == 1:
                counts[arm * 2] += 1
            else:
                counts[arm * 2 + 1] += 1

        return tuple(counts)


    def refresh_env(self, env=None):
        if env is not None:
            self.env = env.unwrapped if hasattr(env, 'unwrapped') else env

        self.root_trial = self.env.trial
        self.horizon_trial = min(self.root_trial + self.horizon,
                                 self.env.n_trials - 1)
        self.env.set_trunc_trial(self.horizon_trial)

    def rollout_policy(self):

        ## random
        ro_action = random.choice(range(self.n_afc))

        ## greedy wrt/ current MDP?
        # ro_action = np.argmax([self.env.p_dist[a] for a in range(self.n_afc)])

        return ro_action

    def check_node(self, node):
        assert node.trial == self.env.trial, \
            f'mismatch: node trial={node.trial}, env trial={self.env.trial}'
