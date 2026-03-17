from abc import ABC, abstractmethod
import random
from math import sqrt, log
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

    def __init__(self, env, tree, exploration_constant=2, discount_factor=0.99, horizon=None, real_future_paths=True, aligned_weight=1.0, orthogonal_weight=1.0):
        self.env = env.unwrapped
        self.expt = env.expt
        self.n_afc = self.env.n_afc
        self.aligned_weight = aligned_weight
        self.orthogonal_weight = orthogonal_weight
        self.tree = tree
        self.discount_factor = discount_factor
        self.exploration_constant = exploration_constant
        self.real_future_paths = real_future_paths

        ## hacky fix of alignment issue
        self.env.path_aligned_states, self.env.path_orthogonal_states = self.env.get_alignment(self.env.path_states)
        
        ## if horizon isn't specified, default to n_trials-1 (i.e. look to the end of the day, which is up to n_trials-1 from the root)
        if horizon is None:
            self.horizon = self.env.n_trials-1 
        else:
            self.horizon = horizon

        ## init root info
        self.update_trial()

        ## create id for root node
        node_id = self.init_node_id(self.env.obs, None)

        
        ### node needs to contain paths, actions for that trial so that these can be inherited by the action node
        path_actions = self.env.path_actions[self.root_trial].copy()
        path_states = self.env.path_states[self.root_trial].copy()

        ## add state node to the tree
        self.tree.add_state_node(node_id=node_id, cost=None, terminated=False, trial = self.root_trial, parent=None, 
                                path_actions=path_actions, path_states=path_states,
                                 )

    
    ### abstract methods

    ## create node id, which represents the agent's current state of knowledge, a flattened N*N*2 array representing which cells have a high or low cost
    @abstractmethod
    def init_node_id(self, obs=None, init_info_state=None):
        pass

    ## update MCTS with trial info
    @abstractmethod
    def update_trial(self):
        pass

    ## tree step
    @abstractmethod
    def tree_step(self, action_leaf):
        pass    
    
    ## rollout policy
    @abstractmethod
    def rollout_policy(self, action_leaf):
        pass
    
    
    ## debugging method for checking if node's belief state matches the env state
    @abstractmethod
    def check_node(self, node):
        pass
    
    ## revert env to root state and trial
    @abstractmethod
    def revert_env(self):
        pass

    
    ### general MCTS methods
    
    ## expand the action space of a node
    def expand(self, node):
        assert self.env.sim, 'env is not in sim mode'

        ### take action (or path) and get new state
        action = node.untried_action()
        # terminated = node.trial == self.env.n_trials-1 ## i.e. this action leaf corresponds to the action made in the final trial, so it leads to termination of the day

        ## or, we terminate depending on horizon
        terminated = node.trial == self.horizon_trial ## i.e. this action leaf corresponds to a trial that is at or beyond the horizon, so it leads to termination of the day
            
        ### update info for s-a leaf - i.e. the state-action pair


        node.action_leaves[action] = Action_Node(action=action, terminated=terminated, trial=node.trial, parent_id=node.node_id)
        node.action_leaves[action].performance = 0
        node.action_leaves[action].norm_performance = 0

        ### expansion of node attaches a (potentially sampled) pair of paths to the leaf
        node.action_leaves[action].path_actions = node.path_actions[action]
        node.action_leaves[action].path_states = node.path_states[action]
        
        ## one-armed weighting: store the aligned and orthogonal states
        if self.real_future_paths:
            node.action_leaves[action].aligned_states, node.action_leaves[action].orthogonal_states = self.env.path_aligned_states[node.trial][action], self.env.path_orthogonal_states[node.trial][action] ## full BAMCP
            # print('got alignment info: ', node.action_leaves[action].aligned_states, node.action_leaves[action].orthogonal_states)
        else:
            node.action_leaves[action].aligned_states, node.action_leaves[action].orthogonal_states = self.env.get_alignment([[node.path_states[action]]]) ## PA-BAMCP
        
        return node.action_leaves[action]
    

    ## take an action according to the tree policy, i.e. take the best UCT child and see where it takes you
    def tree_policy(self):

        ## initialise the tree
        node = self.tree.root
        node_trial = node.trial

        ## create a record of the nodes/leaves visited in the tree
        self.tree_costs = [] ## i.e. the cost associated with each traversal of the tree *under the tree policy*. Hence, this does not include the cost of the current state, which is the starting point of the tree policy, nor does it include the cost of expansion.
        self.tree_actions = [] ## i.e. the states and actions visited in the tree. This *does* include the root, because it is from the root that we move to the next leaf (and then next node). 
        self.node_id_path = []
        
        ## loop until you reach a leaf node or terminal state
        assert self.env.sim, 'env is not in sim mode'
        while not node.terminated:
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

                ## update the tree path
                self.tree_actions.append(action_leaf.action)
                self.node_id_path.append(node.node_id)

                ## move in env
                next_node_id, costs, terminated, node_trial  = self.tree_step(action_leaf)

                # see if the next state node already exists as a child of this action leaf
                if next_node_id in action_leaf.children:
                    node = action_leaf.children[next_node_id]
                else:

                    ## full BAMCP: trial info for next node is just inherited from the env
                    if self.real_future_paths:
                        if not terminated:
                            next_path_actions = self.env.path_actions[node_trial].copy()
                            next_path_states = self.env.path_states[node_trial].copy()
                        else:
                            next_path_actions= None
                            next_path_states= None

                    ## PA-BAMCP: trial info for the next node (i.e. the node to which the action leaf leads) is sampled
                    else:
                        if not terminated:
                            _, next_path_actions, next_path_states, _, _ = self.env.sample_paths_given_future_states(self.root_trial)
                        else:
                            next_path_actions= None
                            next_path_states= None

                    ## create new node
                    node = self.tree.add_state_node(node_id=next_node_id, cost = costs, terminated=terminated, trial = node_trial, parent=action_leaf,
                                        path_actions=next_path_actions, path_states=next_path_states,
                                                    )


        ## if terminal node, there are no mode action leaves to choose from
        if node.terminated:
            action_leaf = None

        return action_leaf


    ## backup costs until you reach the root
    def backup(self):
        tree_len = len(self.tree_costs)
        assert tree_len == len(self.tree_actions), 'tree costs and path lengths do not match\n n tree costs: {} \n n tree path: {}\ntree costs: {}\n tree path: {}'.format(len(self.tree_costs), len(self.tree_actions), self.tree_costs, self.tree_actions)

        ## Precompute discount factors
        discount_factors = [self.discount_factor ** d for d in range(tree_len)]

        ## Loop through the tree path
        node = self.tree.root
        for depth, action in enumerate(self.tree_actions):
            
            ## Get the corresponding action leaf
            action_leaf = node.action_leaves[action]

            ## Discounted cost from the current node to the terminal node
            discounted_cost = sum(
                c * d 
                for c, d in zip(self.tree_costs[depth:], discount_factors[:tree_len - depth])
            )

            ## update visit counts and performance estimates
            action_leaf.n_action_visits += 1
            node.n_state_visits += 1

            ## Incremental average update for performance
            action_leaf.performance += (
                (discounted_cost - action_leaf.performance) / action_leaf.n_action_visits
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
            #     to_append[action] = discounted_cost
            #     self.first_node_updates.append(to_append)
            #     self.first_node_updates_by_depth[tree_len-1].append(to_append)
            
            # ## save costs of each step in the tree - i.e. the cost of making each move in the tree
            # to_append = [np.nan] * self.n_afc
            # to_append[action] = self.tree_costs[depth]
            # self.tree_cost_tracker[depth].append(to_append)

            # ## updates, conditional on first action
            # first_action = self.tree_actions[0]
            # to_append = [np.nan] * self.n_afc
            # to_append[first_action] = self.tree_costs[depth]
            # self.conditional_tree_cost_tracker[action][depth].append(to_append)




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
        
        # print('depth:', action_leaf.trial, 'min Q:', min_Q, 'max Q:', max_Q, 'performance:', action_leaf.performance, 'Q = ', exploitation_term + exploration_term)
        # print('node depth:', action_leaf.trial, 'exploration term:', exploration_term, 'exploitation term:', exploitation_term)
        
        return exploitation_term + exploration_term

    
    ## argmax based on UCT values? 
    def best_child(self, node):
    
        ## get action children - iterate directly over values
        action_leaves = list(node.action_leaves.values())

        ## calculate UCT for each action leaf
        min_Q, max_Q = node.min_Q, node.max_Q
        UCTs = [self.compute_UCT(node, leaf, min_Q, max_Q) for leaf in action_leaves]
        max_UCT = max(UCTs)
        max_idx = argm(UCTs, max_UCT)
        best_child = action_leaves[max_idx]

        return best_child


class MonteCarloTreeSearch_AFC(MonteCarloTreeSearch):

    def __init__(self, env, tree, exploration_constant=2, discount_factor=0.99, horizon=None, real_future_paths=True, aligned_weight=1.0, orthogonal_weight=1.0):
        super().__init__(env, tree, exploration_constant, discount_factor, horizon, real_future_paths, aligned_weight, orthogonal_weight)

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


    
    def update_trial(self):
        self.root_trial = self.env.trial ## i.e. the trial that the agent is current faced with in the real env
        self.root_state = self.env.starts[self.root_trial].copy() ## i.e. the two possible start states for this trial

        ## set the horizon_trial - i.e. the trial at which search terminates
        self.horizon_trial = min(self.root_trial + self.horizon, self.env.n_trials-1)

    ## tree step
    def tree_step(self, action_leaf):
        
        ## get some initial info about the step
        step_trial = action_leaf.trial
        path_id = action_leaf.action
        assert self.env.trial == action_leaf.trial, 'trial mismatch between env and tree\n env: {} \n tree: {}\n MCTS: {}'.format(self.env.trial, action_leaf.trial, self.root_trial)

        ## full BAMCP: use actual upcoming paths
        if self.real_future_paths:
            action_sequence = self.env.path_actions[step_trial][path_id]
            # assert (start_tmp[0], start_tmp[1]) == (action_leaf.path_states[path_id][0][0], action_leaf.path_states[path_id][1]), 'mismatch between node and env start states\n node: {} \n env: {}'.format(action_leaf.path_states[path_id], start_tmp)

        ## PA-BAMCP: sample two paths 
        else:
            action_sequence = action_leaf.path_actions.copy()


        ## initialise costs and observations for this path
        simulated_obs = []

        ## take path
        states, costs, _, _, _ = self.env.step(path_id)

        ## add costs to states to create simulated obs for the tree
        simulated_obs = [(s[0], s[1], costs[i]) for i, s in enumerate(states)]

        ## one-arm bias
        if self.real_future_paths:
            aligned_states, orthogonal_states = self.env.path_aligned_states[step_trial][path_id], self.env.path_orthogonal_states[step_trial][path_id] ## full BAMCP
        else:
            aligned_states, orthogonal_states = action_leaf.aligned_states, action_leaf.orthogonal_states ## PA-BAMCP
        total_weighted_cost = self.env.arm_reweighting(self.env.costs, aligned_states, orthogonal_states, self.aligned_weight, self.orthogonal_weight)

        ## save costs
        self.tree_costs.append(total_weighted_cost)
        
        ## assertions (length checks commented out for performance)
        # n_weighted = len(aligned_states) + len(orthogonal_states)
        # assert len(simulated_obs) == n_weighted, 'sim obs and costs do not match\n sim obs: {}, weighted: {}'.format(len(simulated_obs), n_weighted)
        # assert len(costs) == n_weighted, 'costs and weighted costs do not match\n costs: {}, weighted: {}'.format(len(costs), n_weighted)
        assert len(simulated_obs) == len(action_sequence)+1, 'sim obs and action sequence do not match\n sim obs: {}, action seq: {}'.format(len(simulated_obs), len(action_sequence)+1)
        terminated = action_leaf.terminated

        ## get the next node id, i.e. the informational state after taking this path
        next_node_id = self.init_node_id(simulated_obs, action_leaf.parent_id)
        # n_total_obs = np.sum([len(self.env.path_states[trial][0]) for trial in range(action_leaf.trial+1)])
        # assert n_total_obs == np.sum(next_node_id), 'total obs and next node id do not match\n total obs: {}, next node id: {}'.format(n_total_obs, np.sum(next_node_id))

        ## debugging
        # if step_trial==0:
        #     print(self.env.context)
        #     print('costs:', costs)
        #     print('weighted costs:', weighted_costs)
        #     print('tree costs:', self.tree_costs)
        #     print('simulated obs:', simulated_obs)
        #     raise Exception

        ## since the agent has chosen a path to the goal, we need to move the environment to the next trial
        node_trial = step_trial+1

        ## some checks
        assert len(self.tree_costs)<=self.env.n_trials, 'tree costs exceed number of trials, tree len: {}, n_trials: {}'.format(len(self.tree_costs), self.env.n_trials)
        return next_node_id, costs, terminated, node_trial
    
    ## rollout policy
    def rollout_policy(self, action_leaf):

        ## if no action leaf because tree policy has reached a terminal node, return None
        if action_leaf is None:
            return None

        ### first need to get the starting cost r, which is essentially the cost of path choice that corresponds to the action leaf

        first_trial = action_leaf.trial
        path_id = action_leaf.action

        ## full BAMCP: use actual upcoming paths
        if self.real_future_paths:
            first_path_states = self.env.path_states[first_trial][path_id]

        ## PA-BAMCP: sample paths
        else:
            first_path_states = action_leaf.path_states

        ## unweighted
        # starting_costs = [self.env.costs[state[0], state[1]] for state in first_path_states]
        # total_cost = sum(starting_costs)

        ## weighted
        aligned_states, orthogonal_states = action_leaf.aligned_states, action_leaf.orthogonal_states
        total_cost = self.env.arm_reweighting(self.env.costs, aligned_states, orthogonal_states, self.aligned_weight, self.orthogonal_weight)

        ## if final trial, just stop here
        if action_leaf.terminated:
            self.tree_costs.append(total_cost)
            return total_cost
        
        ## loop through remaining trials
        depth = 0
        remaining_ro_costs = []
        for trial in range(first_trial+1, self.horizon_trial + 1): ## horizon-limited
            depth+=1

            ## PA-BAMCP: sample paths
            if not self.real_future_paths:
                _, _, path_states, _, _ = self.env.sample_paths_given_future_states(self.root_trial) ## PA-BAMCP
                aligned_states, orthogonal_states = self.env.get_alignment([path_states])
                assert len(self.env.path_states[trial][0]) == len(aligned_states[0]) + len(orthogonal_states[0]), 'path states and aligned/orthogonal states do not match\n path states: {}, aligned: {}, orthogonal: {}'.format(len(self.env.path_states[trial][0]), len(aligned_states[0]), len(orthogonal_states[0]))

            ## full BAMCP: use actual upcoming paths
            else:
                path_states = self.env.path_states[trial]
                aligned_states, orthogonal_states = self.env.path_aligned_states[trial], self.env.path_orthogonal_states[trial]
            assert len(path_states[0]) == len(aligned_states[0]) + len(orthogonal_states[0]), 'path states and aligned/orthogonal states do not match\n path states: {}, aligned: {}, orthogonal: {}'.format(len(path_states[0]), len(aligned_states[0]), len(orthogonal_states[0]))
                

            ### greedy: get the total cost of the paths and return the better one
            path_costs = []
            for path_id in range(self.n_afc):
                path_states_tmp = path_states[path_id]

                ## unweighted
                # costs = [self.env.costs[state[0], state[1]] for state in path_states]

                ## arm-weighted
                aligned_states_tmp, orthogonal_states_tmp = aligned_states[path_id], orthogonal_states[path_id]
                ro_cost = self.env.arm_reweighting(self.env.costs, aligned_states_tmp, orthogonal_states_tmp, self.aligned_weight, self.orthogonal_weight)
                
                path_costs.append(ro_cost)
            best_ro_cost = max(path_costs) 
            remaining_ro_costs.append(best_ro_cost)
            total_cost += best_ro_cost * self.discount_factor**depth

            
            ### RANDOM: randomly choose between the paths
            # path_id = np.random.choice(self.n_afc)

            # # full BAMCP - random choice between actual upcoming paths
            # # if self.real_future_paths:
            # #     path_id = np.random.choice(self.n_afc)
            # #     path_states = self.env.path_states[trial][path_id] ## full BAMCP

            # # ## PA-BAMCP - choose between randomly sampled upcoming paths
            # # else:
            # #     path_id = np.random.choice(self.n_afc)
            # #     path_states = sampled_path_states[path_id] ## PA-BAMCP
            # # path_id = np.random.choice(self.n_afc)
            # # path_states_tmp = path_states[path_id]

            # ## unweighted
            # # costs = [self.env.costs[state[0], state[1]] for state in path_states]

            # ## arm-weighted
            # aligned_states_tmp, orthogonal_states_tmp = aligned_states[path_id], orthogonal_states[path_id]
            # ro_cost = self.agent.arm_reweighting(self.env.costs, aligned_states_tmp, orthogonal_states_tmp)

            # remaining_ro_costs.append(ro_cost)
            # total_cost += ro_cost * self.discount_factor**depth
        # print(total_cost)
        # raise Exception

        self.tree_costs.append(total_cost)
        # assert len(remaining_ro_costs)+first_trial+1 == self.env.n_trials, 'remaining RO costs do not match number of trials\n n remaining RO costs: {}, n trials: {}'.format(len(remaining_ro_costs), self.env.n_trials)
        assert len(remaining_ro_costs)+first_trial+1 == self.horizon_trial + 1, 'remaining RO costs do not match number of trials\n n remaining RO costs: {}, n trials: {}'.format(len(remaining_ro_costs), self.horizon_trial + 1)
        return total_cost 

    