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

    def __init__(self, env, agent, tree, exploration_constant=2, discount_factor=0.99):
        self.env = env
        self.expt = env.expt
        self.n_afc = self.env.n_afc
        self.agent = agent
        self.low_cost = self.env.low_cost
        self.high_cost = self.env.high_cost
        self.update_trial()
        self.tree = tree
        self.N = self.env.N
        self.discount_factor = discount_factor
        self.exploration_constant = exploration_constant
        self.exploration_constants = [self.exploration_constant for t in range(self.env.n_trials)]

        ## or, scale exploration constant by the expected cost of the entire day
        # n_total_steps = sum([len(self.env.path_states[trial][0]) for trial in range(self.env.n_trials)])
        # expected_cost_per_t = abs(np.mean([self.low_cost, self.high_cost]))
        # self.exploration_constants = [exploration_constant * (expected_cost_per_t * n_total_steps) for t in range(self.env.n_trials)]
        # print(self.exploration_constants)

        ## or, multiple exploration constants, each scaled by the expected cost of the day from that trial onwards
        expected_cost = np.abs(np.mean([self.low_cost, self.high_cost]))
        self.exploration_constants = []
        for t in range(self.env.n_trials):
            n_steps = 0
            ec = exploration_constant
            for subseq_t in range(t, self.env.n_trials):
                n_steps += len(self.env.path_states[subseq_t][0])
                ec += (expected_cost * n_steps)
            self.exploration_constants.append(ec)
        # print(self.exploration_constants)

        ## create id for root node
        node_id = self.init_node_id(self.env.obs, None, self.actual_trial)

        ## some debugging metrics
        self.exploratory_steps = 0
        self.exploitative_steps = 0
        self.max_Q = np.zeros(self.env.n_trials) - np.inf
        self.min_Q = np.zeros(self.env.n_trials) 

        ## add state node to the tree
        self.tree.add_state_node(node_id=node_id, cost=None, terminated=False, trial = self.actual_trial, n_afc = self.n_afc, parent=None)

    ## create node id, which represents the agent's current state of knowledge, a flattened N*N*2 array representing which cells have a high or low cost
    def init_node_id(self, obs=None, init_info_state=None, trial = None):
        raise NotImplementedError('init_node_id not implemented in subclass')

    ## update MCTS with trial info
    def update_trial(self):
        raise NotImplementedError('trial update not implemented in subclass')

    ## tree step
    def tree_step(self, action_leaf):
        raise NotImplementedError('tree step not implemented in subclass')
    
    ## rollout policy
    def rollout_policy(self, action_leaf):
        raise NotImplementedError('rollout policy not implemented in subclass')
    
    ## myopic rollout
    def myopic_rollout(self, myopic_trial):
        raise NotImplementedError('myopic rollout not implemented in subclass')
    
    ## debugging method for checking if node's belief state matches the env state
    def check_node(self, node):
        raise NotImplementedError('check_node not implemented in subclass')

    
    ## expand the action space of a node
    def expand(self, node):
        raise NotImplementedError('expand not implemented in subclass')
    

    ## take an action according to the tree policy, i.e. take the best UCT child and see where it takes you
    def tree_policy(self):

        ## initialise the tree
        node = self.tree.root
        t = 0
        node_trial = self.env.trial
        assert node_trial == node.trial, 'trial mismatch between env and tree\n env: {} \n tree: {}\n MCTS: {}'.format(node_trial, node.trial, self.actual_trial)
        self.check_node(node)

        ## create a record of the nodes/leaves visited in the tree
        self.tree_costs = [] ## i.e. the cost associated with each traversal of the tree *under the tree policy*. Hence, this does not include the cost of the current state, which is the starting point of the tree policy, nor does it include the cost of expansion.
        self.tree_actions = [] ## i.e. the states and actions visited in the tree. This *does* include the root, because it is from the root that we move to the next leaf (and then next node). 
        self.node_id_path = []

        ## create a copy of the env
        # env_copy = copy.deepcopy(self.env)
        # assert env_copy.sim, 'env copy is not in sim mode'
        # if self.expt == 'AFC':
        #     # env_tmp = copy.deepcopy(self.env)
        #     env_tmp = self.env
        # elif self.expt == 'free':
        #     env_tmp = self.env
        
        ## loop until you reach a leaf node or terminal state
        assert self.env.sim, 'env is not in sim mode'
        while not node.terminated:
            t+=1
            self.check_node(node)

            ## myopia - i.e. initiate rollout of all subsequent trials
            # if t == 2:

            #     ## revert env
            #     self.env.set_state(self.actual_state)
            #     self.env.set_goal(self.actual_goal)
            #     self.env.set_trial(self.actual_trial)
            #     assert np.array_equal(self.env.current, self.actual_state), 'env state not reverted properly'
            #     assert self.env.trial == self.actual_trial, 'env trial not reverted properly'
            #     return False

            ## expansion step
            if self.tree.is_expandable(node):
                action_leaf = self.expand(node)
                self.tree_actions.append(action_leaf.action)
                self.node_id_path.append(node.node_id)

                ## update counts already?
                # action_leaf.n_action_visits += 1
                # node.n_state_visits += 1

                ## revert env
                self.env.set_state(self.actual_state)
                self.env.set_goal(self.actual_goal)
                self.env.set_trial(self.actual_trial)
                assert np.array_equal(self.env.current, self.actual_state), 'env state not reverted properly'
                assert self.env.trial == self.actual_trial, 'env trial not reverted properly. should be in {}, but actually in {}'.format(self.actual_trial, self.env.trial)

                return action_leaf
                
            ## selection step
            else:

                ## get the best child
                action_leaf = self.best_child(node)

                ## update the tree path
                self.tree_actions.append(action_leaf.action)
                self.node_id_path.append(node.node_id)

                ## move in env
                next_state, costs, terminated, node_trial, next_node_id = self.tree_step(action_leaf)

                # see if the next state node already exists as a child of this action leaf
                if next_node_id in action_leaf.children:
                    node = action_leaf.children[next_node_id]
                else:
                    node = self.tree.add_state_node(node_id=next_node_id, cost = costs, terminated=terminated, trial = node_trial, n_afc = self.n_afc, parent=action_leaf)

                ## debugging NB NEED TO FIGURE THIS OUT FOR PA-BAMCP
                # print(next_state, self.env.current, self.env.starts)
                # assert np.array_equal(next_state, self.env.current), 'mismatch between env and tree state\n env: {} \n tree: {}'.format(self.env.current, next_state)


        ## if terminal node, there are no mode action leaves to choose from
        if node.terminated:
            action_leaf = None

        ## revert env
        self.env.set_state(self.actual_state)
        self.env.set_goal(self.actual_goal)
        self.env.set_trial(self.actual_trial)
        assert np.array_equal(self.env.current, self.actual_state), 'env state not reverted properly'
        assert self.env.trial == self.actual_trial, 'env trial not reverted properly'

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

            ## quick sanity check: the depth should match node.trial
            assert depth == node.trial, 'Tree path mismatch:\n node trial: {} \n depth: {}'.format(node.trial, depth)

            ## Discounted cost from the current node to the terminal node
            discounted_cost = np.dot(
                self.tree_costs[depth:],  # Costs from current depth onward
                discount_factors[:tree_len - depth]  # Corresponding discount factors from current depth
            )

            ## update visit counts and performance estimates
            action_leaf.n_action_visits += 1
            node.n_state_visits += 1

            ## Incremental average update for performance
            action_leaf.performance += (
                (discounted_cost - action_leaf.performance) / action_leaf.n_action_visits
            )
            # print('depth:',depth,', n_action_visits:', action_leaf.n_action_visits, 'performance:', action_leaf.performance, 'discounted_cost:', discounted_cost, 'tree costs:', self.tree_costs)

            ## debugging: save updates applied to the first node
            if depth == 0:
                to_append = [np.nan] * self.n_afc
                to_append[action] = discounted_cost
                self.first_node_updates.append(to_append)
                self.first_node_updates_by_depth[tree_len-1].append(to_append)
            
            ## save costs of each step in the tree - i.e. the cost of making each move in the tree
            to_append = [np.nan] * self.n_afc
            to_append[action] = self.tree_costs[depth]
            self.tree_cost_tracker[depth].append(to_append)

            ## updates, conditional on first action
            first_action = self.tree_actions[0]
            to_append = [np.nan] * self.n_afc
            to_append[first_action] = self.tree_costs[depth]
            self.conditional_tree_cost_tracker[action][depth].append(to_append)

            ## debugging: save max and min Q values to normalise Qs
            if action_leaf.performance > self.max_Q[depth]:
                self.max_Q[depth] = action_leaf.performance
            if action_leaf.performance < self.min_Q[depth]:
                self.min_Q[depth] = action_leaf.performance

            ## Move to the next node in the path if not at the end
            if depth < tree_len - 1:
                next_node_id = self.node_id_path[depth+1]
                node = action_leaf.children[next_node_id]


    ## calculate E-E value
    def compute_UCT(self, node, action_leaf): 
        assert action_leaf.n_action_visits > 0 or action_leaf.terminated, 'action leaf has not been visited: {}'.format(action_leaf)
        
        ## standard case
        # exploitation_term = action_leaf.performance
        # exploration_term = self.exploration_constant * sqrt(log(node.n_state_visits) / action_leaf.n_action_visits)

        ## or, depth-dependent exploration constant
        exploitation_term = action_leaf.performance
        exploration_term = self.exploration_constants[action_leaf.trial] * sqrt(log(node.n_state_visits) / action_leaf.n_action_visits)

        
        ### or, min-max normalisation of Qs, 

        ## based on overall min and max Q values (i.e. min and max of all estimates ever recorded at that depth)
        # min_Q = self.min_Q[action_leaf.trial]
        # max_Q = self.max_Q[action_leaf.trial]

        # ## or, based on min and max of current estimates of Qs at that depth
        # # min_Q, max_Q = self.tree.min_max_Q(node=self.tree.root, depth=action_leaf.trial, current_depth=0)

        # norm_term = max_Q - min_Q
        # if norm_term == 0 or norm_term==np.inf:
        #     exploitation_term = action_leaf.performance
        # else:
        #     exploitation_term = (action_leaf.performance - min_Q) / norm_term
        # exploration_term = self.exploration_constants[action_leaf.trial] * sqrt(log(node.n_state_visits) / action_leaf.n_action_visits)
        
        # print('exploration term:', exploration_term, 'exploitation term:', exploitation_term)
        return exploitation_term + exploration_term

    
    ## argmax based on UCT values? 
    def best_child(self, node):
    
        ## get action children
        action_leaves = [node.action_leaves[a] for a in node.action_leaves.keys()]

        ## create deep copy of action leaves
        # action_leaves = [copy.deepcopy(node.action_leaves[a]) for a in node.action_leaves.keys()]
        # print(action_leaves[0])

        ## calculate Q-normalisation term??
        # leaf_perfs = [leaf.performance for leaf in action_leaves]
        # norm_term = np.max(leaf_perfs) - np.min(leaf_perfs)
        # for a, leaf in enumerate(action_leaves):
        #     if norm_term == 0:
        #         action_leaves[a].performance = 0
        #     else:
        #         action_leaves[a].performance = (leaf.performance - np.min(leaf_perfs)) / norm_term
        # norm_term = np.max(leaf_perfs) - np.min(leaf_perfs)
        # Q_diff = np.abs(np.max(leaf_perfs) - np.min(leaf_perfs))
        # norm_term = np.max([Q_diff, 1])
        # norm_term = np.min(leaf_perfs)
        # norm_term = 1

        ## calculate UCT for each action leaf
        UCTs = [self.compute_UCT(node, leaf) for leaf in action_leaves]
        # print('node trial:', node.trial, ', diff in UCTs:', np.max(UCTs) - np.min(UCTs))
        max_UCT = np.max(UCTs)
        max_idx = argm(UCTs, max_UCT)
        best_child = action_leaves[max_idx]

        ## check if the chosen leaf is the one with the highest performance, or is exploratory
        # if best_child.performance == np.max(leaf_perfs):
        #     self.exploitative_steps += 1
        # else:
        #     self.exploratory_steps += 1

        return best_child


    ## tree search --> action loop
    def search(self, n_sims=1000, n_iter=100, lazy=False):

        ## generate new set of root samples
        self.agent.root_samples(obs = self.env.obs, n_samples=n_sims, n_iter=n_iter, lazy=lazy, CE=False, combo=False)

        ## debugging plot
        # plt.figure()
        # plot_r(posterior_mean_p_cost.reshape(self.N,self.N), ax = plt.subplot(), title='posterior sample')
        # plt.show()

        ## debugging Q-vals
        self.Q_tracker = []
        self.first_node_updates = []
        self.first_node_updates_by_depth = []
        self.tree_cost_tracker = []
        self.conditional_tree_cost_tracker = [[] for _ in range(self.n_afc)]
        for t in range(self.env.n_trials):
            self.first_node_updates_by_depth.append([])
            self.tree_cost_tracker.append([])
            for a in range(self.n_afc):
                self.conditional_tree_cost_tracker[a].append([])

        ## init for path sampling in PA-BAMCP (i.e. no need to sample paths for the root, since we are actively considering these)
        actual_trial = self.env.trial
        self.env.sampled_path_actions = {
            actual_trial: self.env.path_actions[actual_trial].copy()
        }
        self.env.sampled_path_states = {
            actual_trial: self.env.path_states[actual_trial].copy()
        }
        self.env.sampled_starts = {
            actual_trial: self.env.starts[actual_trial].copy()
        }
        self.env.sampled_goals = {
            actual_trial: self.env.goals[actual_trial].copy()
        }

        ## if PA-BAMCP, we need to sample new paths at the beginning of each simulation
        for t in range(actual_trial+1, self.env.n_trials):
            _, sampled_path_actions, sampled_path_states, sampled_starts, sampled_goals = self.env.sample_paths_given_future_states(actual_trial)
            self.env.sampled_path_actions[t] = sampled_path_actions
            self.env.sampled_path_states[t] = sampled_path_states
            self.env.sampled_starts[t] = sampled_starts
            self.env.sampled_goals[t] = sampled_goals
        
        
        ## loop through simulations
        for s in range(n_sims):

            ## CHEATING: give agent full knowledge of grid probabilities
            # posterior_p_cost = self.env.p_costs
            
            ## root sampling of new posterior
            posterior_p_cost = self.agent.all_posterior_p_costs[s]
            if self.expt=='free': ## only need to do this in the free-choice expt
                self.agent.dp(posterior_p_cost, expected_cost=True)
            self.env.receive_predictions(posterior_p_cost)

            ## debugging plot
            # plt.figure()
            # # plot_r(self.env.posterior_sample.reshape(self.N,self.N), ax = plt.subplot(), title='posterior sample')
            # plot_action_tree(self.env.Q_inf, self.env.get_obs()['agent'], self.env.get_obs()['goal'], ax = plt.subplot(), title='DP_inf')

            ## selection, expansion, simulation
            action_leaf = self.tree_policy()
            self.rollout_policy(action_leaf)
            
            ## myopic?
            # if action_leaf == False:
            #     self.myopic_rollout(1)
            # else:
            #     self.rollout_policy(action_leaf)
            
            ##backup
            self.backup()

            ## update Q tracker
            try:
                Qs = [self.tree.root.action_leaves[a].performance for a in self.tree.root.action_leaves.keys()]
                self.Q_tracker.append(Qs)
            except:
                pass

        
        ## action selection
        MCTS_estimates = np.full(self.n_afc, np.nan)
        for action, leaf in self.tree.root.action_leaves.items():
            MCTS_estimates[action] = leaf.performance
        assert not np.isnan(np.nansum(MCTS_estimates)), 'no MCTS estimates for {}'.format(self.tree.root)
        max_MCTS = np.nanmax(MCTS_estimates)
        action = argm(MCTS_estimates, max_MCTS)
        
        ## set root for next search
        # next_state = self.tree.root.action_leaves[action].next_state
        # next_root = self.tree.nodes[str(next_state)]

        ## calculate the entropy over actions
        # action_probs = np.exp(MCTS_estimates) / np.sum(np.exp(MCTS_estimates))
        # entropy = -np.nansum(action_probs * np.log(action_probs))
        # print('action probs:', action_probs)
        # print('entropy:', entropy)

        return action, MCTS_estimates
    

### subclasses for different experiments

## free exploration
class MonteCarloTreeSearch_Free(MonteCarloTreeSearch):
    
    def __init__(self, env, agent, tree, exploration_constant=2, discount_factor=0.99):
        super().__init__(env, agent, tree, exploration_constant, discount_factor)

    def init_node_id(self, obs=None, init_info_state = None, trial=None):
        node_id = tuple(np.append(self.actual_state, self.env.costs[self.actual_state[0], self.actual_state[1]]))
        return node_id
    
    def update_trial(self):
        self.actual_state = self.env.current
        self.actual_goal = self.env.goal
        self.actual_trial = self.env.trial

    ## in free choice, there is a meaningful state of the MDP (i.e. agent's current position in grid), which is reflected in belief state
    def check_node(self, node):
        assert np.array_equal(node.belief_state[:2], self.env.current), 'mismatch between node and env state\n node: {} \n env: {}'.format(node, self.env.current)

    ## in free choice, we have a meaningful notion of prev and next state (i.e. current location, and location after taking an action)
    def expand(self, node):
        assert self.env.sim, 'env is not in sim mode'

        ## take action (or path) and get new state
        action = node.untried_action()
        prev_state = self.env.current.copy()
        next_state, _, terminated, _, _ = self.env.step(action)
            
        ## update info for s-a leaf - i.e. the state-action pair
        node.action_leaves[action] = Action_Node(start = prev_state, action=action, goal = next_state, terminated=terminated, trial=node.trial, parent_id=node.node_id)
        node.action_leaves[action].performance = 0
        node.action_leaves[action].norm_performance = 0

        return node.action_leaves[action]

    ## tree step
    def tree_step(self, action_leaf):
        action = action_leaf.action
        next_state, cost, terminated, _, _ = self.env.step(action)
        self.tree_costs.append(cost)
        node_trial = self.actual_trial ##??
        next_node_id = tuple(np.append(next_state, cost))
        return next_state, cost, terminated, node_trial, next_node_id
    
    ## rollout policy
    def rollout_policy(self, action_leaf):

        ## if no action leaf because tree policy has reached a terminal node, return None
        if action_leaf is None:
            return None

        ## init
        total_cost = 0
        depth = 0

        ## rolling out from goal location, can just end here
        if action_leaf.terminated:
            self.tree_costs.append(total_cost)
            return total_cost

        # get the next state and goal
        current = action_leaf.next_state.copy()
        # goal = self.env.goal

        ## begin with cost of current state
        total_cost+=self.env.get_pred_cost(current)

        ## get costs of optimal route
        while True:
            depth+=1
            i, j = current
            action = int(self.agent.A_inf[i, j])  # Ensure action index is int
            
            # Take action and update current state
            direction = self.env.action_to_direction[action]
            current = np.clip(current + direction, 0, self.N - 1)
            
            ## return cost once goal is reached (i.e. don't use the cost of the goal state)
            if np.array_equal(current, self.actual_goal):
                self.tree_costs.append(total_cost)
                return total_cost
            
            ## update costs
            cost = self.env.get_pred_cost(current)
            total_cost += cost*self.discount_factor**depth
    

class MonteCarloTreeSearch_AFC(MonteCarloTreeSearch):

    def __init__(self, env, agent, tree, exploration_constant=2, discount_factor=0.99):
        super().__init__(env, agent, tree, exploration_constant, discount_factor)

    ## node_ids are defined by the informational state, i.e. the counts of low and high cost states in each cell
    def init_node_id(self, obs=None, parent_node_id=None, trial=None):
        
        ## uses sparse representation: frozenset of ((i, j), (low_count, high_count)) for observed cells only

        # Initialize counts from parent node_id (sparse frozenset) if provided
        if parent_node_id is not None:
            counts = dict(parent_node_id)
        else:
            counts = {}
        
        # Add new observations
        for i, j, c in obs:
            i, j = int(i), int(j)
            low, high = counts.get((i, j), (0, 0))
            if c == self.high_cost:
                counts[(i, j)] = (low, high + 1)
            else:
                counts[(i, j)] = (low + 1, high)
        
        return frozenset(counts.items())
    
    ## in AFC, there is no meaningful state of the MDP, so belief state just contains the trial number
    def check_node(self, node):
        # assert np.array_equal(node.belief_state[:2*self.n_afc].reshape(self.n_afc,2), self.env.current), 'mismatch between node and env state\n node: {} \n env: {}'.format(node.belief_state[:2*self.n_afc].reshape(self.n_afc,2), self.env.current)
        # assert node.belief_state[0] == self.env.trial, 'mismatch between node and env trial\n node: {} \n env: {}'.format(node.belief_state[0], self.env.trial)
        assert node.trial == self.env.trial, 'mismatch between node and env trial\n node: {} \n env: {}'.format(node.trial, self.env.trial)

    
    ## in AFC, we define prev and next states in terms of the start and goal states for the chosen path
    def expand(self, node):
        assert self.env.sim, 'env is not in sim mode'

        ### take action (or path) and get new state
        action = node.untried_action()
        terminated = node.trial == self.env.n_trials-1 ## i.e. this action leaf corresponds to the action made in the final trial, so it leads to termination of the day
        # goal = self.env.goals[node.trial][action].copy()
        goal = self.env.sampled_goals[node.trial][action].copy()
            
        ## update info for s-a leaf - i.e. the state-action pair
        # start = self.env.starts[node.trial][action].copy()
        start = self.env.sampled_starts[node.trial][action].copy()
        node.action_leaves[action] = Action_Node(start = start, action=action, goal = goal, terminated=terminated, trial=node.trial, parent_id=node.node_id)
        node.action_leaves[action].performance = 0
        node.action_leaves[action].norm_performance = 0

        return node.action_leaves[action]
    
    def update_trial(self):
        self.actual_trial = self.env.trial ## i.e. the trial that the agent is current faced with in the real env
        self.actual_state = self.env.starts[self.actual_trial].copy() ## i.e. the two possible start states for this trial
        self.actual_goal = self.env.goals[self.actual_trial].copy() ## i.e. the two possible goal states for this trial

    ## tree step
    def tree_step(self, action_leaf):
        
        ## get some initial info about the step
        step_trial = action_leaf.trial
        path_id = action_leaf.action
        assert self.env.trial == action_leaf.trial, 'trial mismatch between env and tree\n env: {} \n tree: {}\n MCTS: {}'.format(self.env.trial, action_leaf.trial, self.actual_trial)

        ## full BAMCP: use actual upcoming paths
        # start_tmp = self.env.starts[step_trial][path_id].copy()
        # goal_tmp = self.env.goals[step_trial][path_id].copy()
        # action_sequence = self.env.path_actions[step_trial][path_id]

        ## PA-BAMCP: sample two paths 
        start_tmp = self.env.sampled_starts[step_trial][path_id].copy()
        goal_tmp = self.env.sampled_goals[step_trial][path_id].copy()
        action_sequence = self.env.sampled_path_actions[step_trial][path_id]

        self.env.set_state(start_tmp)
        self.env.set_goal(goal_tmp)
        # self.env.set_trial(step_trial)

        ## initialise costs and observations for this path
        simulated_obs = []
        # simulated_obs = [np.append(start_tmp, self.env.predicted_costs[start_tmp[0], start_tmp[1]])] ## if the agent hasn't already observed the start state

        ## take path
        states, costs = self.env.take_path(action_sequence)
        # print('start_tmp:', start_tmp, 'goal_tmp:', goal_tmp, 'final state of path:', states[-1], 'actual starts:', self.env.starts[step_trial], 'actual goals:', self.env.goals[step_trial])
        # assert np.array_equal(states[-1], goal_tmp), 'final state of path does not match goal\n final state: {}, goal: {}'.format(states[-1], goal_tmp) ## COMMENTED OUT FOR PA-BAMCP FOR NOW

        ## add back in the start state if it wasn't actually observed in non-sim space
        states = [start_tmp] + states
        costs = [self.env.predicted_costs[start_tmp[0], start_tmp[1]]] + costs

        ## add costs to states to create simulated obs for the tree
        simulated_obs += [np.append(s, c) for s, c in zip(states, costs)]

        ## one-arm bias (NEED TO THINK ABT HOW TO DO THIS FOR PA-BAMCP - DO LATER)
        # aligned_states, orthogonal_states = self.env.path_aligned_states[step_trial][path_id], self.env.path_orthogonal_states[step_trial][path_id]
        # weighted_costs = self.agent.arm_reweighting(self.env.predicted_costs, aligned_states, orthogonal_states) 
        weighted_costs = costs ## for now, just use the actual costs 


        ## save costs
        # self.tree_costs.append(np.sum(costs))
        self.tree_costs.append(np.sum(weighted_costs)) ## NB: we only use the weighted costs for the backup, rather than building the tree
        # if (action_leaf.trial==1) and (action_leaf.action==1):
        #     print(simulated_obs)
        assert len(simulated_obs) == len(weighted_costs), 'sim obs and costs do not match\n sim obs: {}, costs: {}'.format(len(simulated_obs), len(costs))
        assert len(costs) == len(weighted_costs), 'costs and weighted costs do not match\n costs: {}, weighted costs: {}'.format(len(costs), len(weighted_costs))
        assert len(simulated_obs) == len(action_sequence)+1, 'sim obs and action sequence do not match\n sim obs: {}, action seq: {}'.format(len(simulated_obs), len(action_sequence)+1)
        terminated = action_leaf.terminated
        # if terminated:
        #     print(action_leaf.trial, start_tmp, goal_tmp)

        ## get the next node id, i.e. the informational state after taking this path
        next_node_id = self.init_node_id(simulated_obs, action_leaf.parent_id, step_trial)
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
        if not terminated:

            ## full BAMCP
            # next_state = self.env.starts[node_trial].copy()
            
            ## PA-BAMCP
            next_state = self.env.sampled_starts[node_trial].copy()

            self.env.set_trial(node_trial)
            self.env.soft_reset()
        else:
            next_state = np.array([None, None])
            next_goal = np.array([None, None])
            self.env.soft_reset(next_state, next_goal)

        ## some checks
        assert len(self.tree_costs)<=self.env.n_trials, 'tree costs exceed number of trials, tree len: {}, n_trials: {}'.format(len(self.tree_costs), self.env.n_trials)
        return next_state, costs, terminated, node_trial, next_node_id
    
    ## rollout policy
    def rollout_policy(self, action_leaf):

        ## if no action leaf because tree policy has reached a terminal node, return None
        if action_leaf is None:
            return None

        ## first need to get the starting cost r, which is essentially the cost of path choice that corresponds to the action leaf
        first_trial = action_leaf.trial
        path_id = action_leaf.action

        ## full BAMCP: use actual upcoming paths
        # starting_cost = 0
        # for state in self.env.path_states[first_trial][path_id]:
        #     # cost = self.env.get_pred_cost(state)
        #     cost = self.env.predicted_costs[state[0], state[1]]
        #     starting_cost += cost
        # total_cost = starting_cost

        ## or PA-BAMCP: use sampled upcoming paths
        starting_cost = 0
        for state in self.env.sampled_path_states[first_trial][path_id]:
            cost = self.env.predicted_costs[state[0], state[1]]
            starting_cost += cost
        total_cost = starting_cost

        ## or, arm-weighted costs NB NEED TO FIGURE OUT HOW TO DO THIS FOR PA-BAMCP - DO LATER
        # aligned_states, orthogonal_states = self.env.path_aligned_states[first_trial][path_id], self.env.path_orthogonal_states[first_trial][path_id]
        # total_cost = sum(self.agent.arm_reweighting(self.env.predicted_costs, aligned_states, orthogonal_states))

        ## if final trial, just stop here
        if action_leaf.terminated:
            self.tree_costs.append(total_cost)
            return total_cost
        
        ## loop through remaining trials
        depth = 0
        remaining_ro_costs = []
        ro_choices=[]
        for trial in range(first_trial+1, self.env.n_trials):
            depth+=1

            ## greedy: get the total cost of the paths and return the better one
            path_costs = []
            for path_id in range(self.n_afc):
                # path_states = self.env.path_states[trial][path_id] ## full BAMCP
                path_states = self.env.sampled_path_states[trial][path_id] ## PA-BAMCP
                ro_cost = 0
                for state in path_states:
                    cost = self.env.predicted_costs[state[0], state[1]]
                    ro_cost += cost
                path_costs.append(ro_cost)
            best_ro_cost = np.max(path_costs) 
            remaining_ro_costs.append(best_ro_cost)
            ro_choices.append(np.argmax(path_costs))
            total_cost += best_ro_cost * self.discount_factor**depth

            ## or greedy but wrt/ arm-weighted cost
            # path_costs = []
            # for path_id in range(self.n_afc):
            #     aligned_states, orthogonal_states = self.env.path_aligned_states[trial][path_id], self.env.path_orthogonal_states[trial][path_id]
            #     path_cost = sum(self.agent.arm_reweighting(self.env.predicted_costs, aligned_states, orthogonal_states))
            #     path_costs.append(path_cost)
            # best_ro_cost = np.max(path_costs)
            # remaining_ro_costs.append(best_ro_cost)
            # ro_choices.append(np.argmax(path_costs))
            # total_cost += best_ro_cost * self.discount_factor**depth

            ## RANDOM: randomly choose between the paths
            # path_id = np.random.choice(self.n_AFC)
            # path_states = self.env.path_states[action_leaf.trial+1][path_id]
            # ro_cost = 0
            # for state in path_states:
            #     cost = self.env.get_pred_cost(state)
            #     ro_cost += cost

        self.tree_costs.append(total_cost)
        assert len(remaining_ro_costs)+first_trial+1 == self.env.n_trials, 'remaining RO costs do not match number of trials\n n remaining RO costs: {}, n trials: {}'.format(len(remaining_ro_costs), self.env.n_trials)
        return total_cost 
    
    ## myopic rollout - i.e. tree has been cut off at a certain depth, after which point you just do greedy rollouts
    def myopic_rollout(self, myopic_trial):

        ## init
        total_cost = 0
        depth = 0
        for trial in range(myopic_trial, self.env.n_trials):
            depth+=1

            ## GREEDY: get the total cost of the paths and return the better one
            path_costs = []
            for path_id in range(self.n_afc):
                path_states = self.env.path_states[trial][path_id]
                ro_cost = 0
                for state in path_states:
                    # cost = self.env.get_pred_cost(state)
                    cost = self.env.predicted_costs[state[0], state[1]]
                    ro_cost += cost
                path_costs.append(ro_cost)
            total_cost += np.max(path_costs) * self.discount_factor**depth

        # self.tree_costs.append(total_cost)
        self.tree_costs[-1] += total_cost
        return total_cost
    

    ## expected KL divergence
    def expected_KL(self, env_copy):

        ### log determinant of covariance matrix

        ## get prior p and q samples
        prior_p_samples = self.all_posterior_ps
        prior_q_samples = self.all_posterior_qs
        prior_samples = np.vstack([prior_p_samples.T, prior_q_samples.T])

        ## log det of prior covariance matrix 
        # prior_cov = np.cov(prior_samples)
        # prior_LD = np.linalg.slogdet(prior_cov)[1]
        # assert prior_cov.shape[0] == N*2, 'covariance matrix is wrong shape'
        
        ## order the outcomes (counterfactual, then actual. This is to allow reuse of the posterior samples associated with the actual outcome on the next timestep)
        actual_outcome = env_copy.obs.copy()[-1, -1]
        if actual_outcome == env_copy.low_cost:
            ordered_outcomes = [env_copy.high_cost, env_copy.low_cost]
        else:
            ordered_outcomes = [env_copy.low_cost, env_copy.high_cost]
        posterior_LDs = []
        KLs = []

        ## posterior samples under each of the possible outcomes of the action that was just taken
        for o, outcome in enumerate(ordered_outcomes):
            sim_obs = env_copy.obs.copy()
            sim_obs[-1, -1] = outcome
            self.root_samples(sim_obs, n_samples=self.n_sims, n_iter=self.n_iter, lazy=self.lazy, CE=self.CE)
            posterior_samples = np.vstack([np.array(self.all_posterior_ps).T, np.array(self.all_posterior_qs).T])
            
            ## posterior log det
            # posterior_cov = np.cov(posterior_samples)
            # assert posterior_cov.shape == prior_cov.shape, 'prior and posterior covariance matrices do not match: {} vs {}'.format(posterior_cov.shape, prior_cov.shape)
            # LD = np.linalg.slogdet(posterior_cov)[1]
            # posterior_LDs.append(LD)

            ## or, calculate the KL divergence between two multivariate gaussians
            KL = KL_divergence(prior_samples, posterior_samples)
            KLs.append(KL)


        ## expected log det, i.e. the difference between the prior and the expected posterior log dets, weighted by the probability of each outcome
        # p_low = np.mean(prior_p_samples * prior_q_samples)
        # p_high = 1 - p_low
        # if actual_outcome == env_copy.low_cost:
        #     expected_LD = p_low * (posterior_LDs[1] - prior_LD) + p_high * (posterior_LDs[0] - prior_LD)
        # else:
        #     expected_LD = p_low * (posterior_LDs[0] - prior_LD) + p_high * (posterior_LDs[1] - prior_LD)
        # ELDs.append(expected_LD)

        ## expected KL divergence
        p_low = np.mean(prior_p_samples * prior_q_samples)
        p_high = 1 - p_low
        if actual_outcome == env_copy.low_cost:
            expected_KL = p_low * (KLs[1]) + p_high * (KLs[0])
        else:
            expected_KL = p_low * (KLs[0]) + p_high * (KLs[1])
        
        return expected_KL


## parallel function for simulating many trials within the same grid env
def simulate_agent(ppt, env_params=None, MCTS_params=None, sampler_params=None, agents= ['BAMCP', 'CE'], progress=False):
    print(' ') # for some reason need this to get the pbar to appear

    ## or, do this manually
    N = env_params['N']
    n_trials = env_params['n_trials']
    expt_info = env_params['expt_info']
    expt = expt_info['type']
    n_days = env_params['n_days']
    metric = env_params['metric']
    beta_params = env_params['beta_params']

    n_sims = MCTS_params['n_sims']
    exploration_constant = MCTS_params['exploration_constant']
    discount_factor = MCTS_params['discount_factor']

    n_iter = sampler_params['n_iter']
    lazy = sampler_params['lazy']
    
    ## set context prior for each sampling agent
    context_priors = {}
    # for agent in ['BAMCP', 'CE', 'BAMCP w/ CE', 'CE w/ BAMCP']:
    for agent in agents:
        context_priors[agent] = 0.5
    
    ## initiate dictionary to store the results
    sim_out = {}
    for key in data_keys:
        sim_out[key]=[]
    
    ## set seed
    seed=ppt
    seed=os.getpid()
    np.random.seed(seed)

    ## loop through runs of the same grid-trial set
    if progress:
        if n_days > 1:
            pbar = tqdm(total=n_days*n_trials, desc='Grid_'+str(ppt)+', '+str(n_days)+' days, '+str(n_trials)+' trials', position=0, leave=False, ascii=True)
    
    ## loop through days - i.e. different grids drawn from the same prior. we will collect these for saving at the end
    all_day_envs = []
    for day in range(n_days):   

        ## create base grid environment
        env = make_env(N, n_trials, expt_info, beta_params, metric)
        
        ## debugging plot env
        # fig, ax = plt.subplots(1, 1, figsize=(5,5))
        # plot_r(env.p_costs, ax = ax, title=m)
        # plt.show()

        ## copy env so that each agent makes its own observations 
        agent_envs = {}
        for a in agents:
            agent_envs[a] = copy.deepcopy(env)

        ## copy MCTS planners
        MCTSs = {}
        tree_resets = {}
        for a in agents:
            if (a == 'BAMCP') or (a == 'BAMCP w/ CE') or (a=='BAMCP_wrong'):
                MCTSs[a] = None
                tree_resets[a] = True
        # tree_reset = True ## to determine whether tree is reset at the start of each trial

        ## copy of farmers
        farmers = {}
        for a in agents:
            farmers[a] = Farmer(N, context_prior=context_priors[a])

        ## save posterior means (for presentation plot)
        # all_posterior_p_costs_plot = []
        # all_posterior_contexts_plot = []
        # fig, axs = plt.subplots(2,1, figsize =(10,5))
        # plot_r(env.p_costs, ax = axs[0], title='p(low cost)', cbar=True)
        # plot_r(env.costss[0]+1, ax = axs[1], title='Actual costs', cbar=True)
        # plt.show()

        ## initialise farmer agent
        # farmer = Farmer(N, context_prior=context_prior)
        # farmer = Farmer(N, context_prior=context_priors[ag])

        ## loop through trials (i.e. different start and goal states for the same grid)
        if progress:
            if n_days <= 1:
                pbar = tqdm(total=n_trials, desc='Grid_'+str(ppt)+', day '+str(day+1)+'/'+str(n_days), position=ppt+1, leave=False)
        for t in range(n_trials):

            ## loop through agents
            for a, ag in enumerate(agents):
                
                ## initialise the farmer
                farmer = farmers[ag]

                ## tmp fix: fix the prior to the prior that was used at the beginning of the grid (to prevent observations contributing to the posterior on multiple trials)
                farmer.context_prob = context_priors[ag]

                
                ### reset trial 

                ## copy env for our base agents
                if (ag=='BAMCP') or (ag=='CE') or (ag=='BAMCP_wrong'):
                    env_copy = agent_envs[ag]
                    env_copy.reset()
                    env_copy.set_sim(True)

                    ## save the state of these envs for our checker agents, so that we can imbue them with the same knowledge later on 
                    if (ag=='BAMCP') & ('CE w/ BAMCP' in agents):
                        agent_envs['CE w/ BAMCP'] = copy.deepcopy(env_copy)
                        assert np.array_equal(agent_envs['CE w/ BAMCP'].obs, env_copy.obs), 'obs do not match'
                    elif (ag=='CE') & ('BAMCP w/ CE' in agents):
                        agent_envs['BAMCP w/ CE'] = copy.deepcopy(env_copy)
                        assert np.array_equal(agent_envs['BAMCP w/ CE'].obs, env_copy.obs), 'obs do not match'


                ## or, load env for our checker agents
                else:
                    env_copy = agent_envs[ag]
                    env_copy.set_sim(True)

                start = env_copy.current
                current = start
                goal = env_copy.goal
                actions = []
                CE_actions = []
                choice_probs = []
                # context_probs = [] ## i.e. the prob of the context at the start of the trial
                Q_values = []
                CE_Q_values = []
                ELDs = []
                EKLs = []
                

                ## GP-MCTS agent receives info from env
                if ag =='GP-MCTS':
                    agent = GP
                elif (ag == 'BAMCP') or (ag == 'CE') or (ag=='BAMCP w/ CE') or (ag=='CE w/ BAMCP') or (ag=='BAMCP_wrong'):
                    agent = farmer
                agent.get_env_info(env_copy)
                if t==0:
                    assert len(env_copy.obs)==0, 'obs not empty at start of trial'

                ## reset tree
                if ((ag == 'BAMCP') or (ag == 'BAMCP w/ CE') or (ag=='BAMCP_wrong')):
                    # context_probs.append(farmer.context_prob)
                    if tree_resets[ag]:#& tree_reset:
                        tree = Tree(N)
                        if expt == 'free':
                            MCTSs[ag] = MonteCarloTreeSearch_Free(env=env_copy, agent=agent, tree=tree, exploration_constant=exploration_constant, discount_factor=discount_factor)
                        elif expt == 'AFC':
                            MCTSs[ag] = MonteCarloTreeSearch_AFC(env=env_copy, agent=agent, tree=tree, exploration_constant=exploration_constant, discount_factor=discount_factor)
                
                    ## if keeping the tree between trials, need to update tree with new trial info
                    elif (not tree_resets[ag]): #& (not tree_reset):
                        MCTSs[ag].update_trial()
                        tree_resets[ag] = True
                        MCTSs[ag].agent = agent ##???
                    MCTS = MCTSs[ag]
                    assert t == MCTS.actual_trial, 'trial mismatch between env and MCTS\n env: {} \n MCTS: {}'.format(t, MCTS.env.trial)
                assert t == env_copy.trial, 'trial mismatch between simulation and env\n simulation: {} \n env: {}'.format(t, MCTS.env.trial)

            
                ## run trial until goal is reached
                end_trial = False
                terminated=False
                early_terminate = False
                steps = 0
                if expt=='free':
                    max_steps = len(env_copy.o_trajs[t])*1.75
                elif expt=='AFC':
                    max_steps = 100 ## just in case

                while not end_trial:

                    ## plain balanced GP
                    if ag == 'GP':
                        eps = 0.05
                        alpha = 0.4
                        action = env_copy.balanced_policy(current, goal, eps, alpha)
                        actions.append(action)

                        ## action
                        current, _, terminated, truncated, _ = env_copy.step(action)
                        # current = observation['agent']
                        steps += 1

                    ## certainty-equivalent
                    elif (ag == 'CE') or (ag == 'CE w/ BAMCP'):
                        env_copy.set_sim(False)

                        ## get posterior mean grid
                        agent.root_samples(obs=env_copy.obs, n_samples=n_sims, n_iter=n_iter, lazy=lazy, CE=True, combo=False)
                        env_copy.receive_predictions(agent.posterior_mean_p_cost)


                        ## plot for debugging?
                        # _, axs = plt.subplots(1, 3, figsize=(10,5))
                        # plot_r(env_copy.p_costs.reshape(N,N), ax=axs[0], title = 'costs')
                        # plot_traj([env_copy.o_trajs[t], env_copy.a_traj], ax=axs[0])
                        # plot_r(env_copy.predicted_p_costs, ax=axs[1], title = 'posterior mean p cost')
                        # plot_action_tree(agent.Q_inf, current, goal, ax=axs[2], title = 'DP_inf')
                        # plt.show()

                        ## get and take action
                        if expt == 'free':

                            ## dynamic programming under this posterior mean
                            agent.dp(agent.posterior_mean_p_cost, expected_cost=True)

                            ## best action as given by DP solution
                            action = agent.optimal_policy(current, agent.Q_inf)
                            actions.append(action)
                            current, _, terminated, _, _ = env_copy.step(action)
                            
                        elif expt == 'AFC':

                            ## get the cost of each path under the posterior mean
                            path_costs = []
                            for path_id in range(env_copy.n_afc):
                                path_states = env_copy.path_states[t][path_id]
                                path_cost = 0
                                for state in path_states:
                                    # path_cost += env_copy.get_pred_cost(state) ## i.e. sample binary costs from the posterior pqs
                                    path_cost += agent.posterior_mean_p_cost[state[0], state[1]]*env_copy.low_cost + (1-agent.posterior_mean_p_cost[state[0], state[1]])*env_copy.high_cost ## or, use expected costs
                                path_costs.append(path_cost)

                            ## choose the path with the lowest total cost
                            max_cost = np.max(path_costs)
                            action = argm(path_costs, max_cost)
                            actions.append(action)
                            Q_values.append(np.array(path_costs))
                            choice_probs.append(softmax(path_costs))

                            ## take the path
                            env_copy.set_sim(False)
                            env_copy.init_trial(action)
                            start = env_copy.current
                            goal = env_copy.goal
                            assert np.array_equal(start, env_copy.starts[t][action]), 'current state does not match start state\n current: {}, start: {}'.format(env_copy.current, env_copy.starts[t][action])
                            action_sequence = env_copy.path_actions[t][action]
                            _,_ = env_copy.take_path(action_sequence)
                            current = env_copy.current
                            costs = env_copy.trial_obs[:,-1]
                            path_cost = np.sum(costs)
                            terminated=True
                            day_terminated = t == (n_trials-1)
                            # costs = []
                            # for ac in action_sequence:
                            #     current, cost, terminated, _, _ = env_copy.step(ac)
                            #     costs.append(cost)
                            # path_cost = np.sum(costs)
                            day_terminated = t == (n_trials-1)
                        steps += 1
                        leaf_visits = []

                        ## update observations
                        agent.get_env_info(env_copy)


                    ## bamcp
                    elif (ag == 'BAMCP') or (ag == 'BAMCP w/ CE') or (ag=='BAMCP_wrong'):
                        env_copy.set_sim(True)
                        MCTS.actual_state = current
                        
                        ## init MCTS (if resetting the tree for each move, init here. otherwise, this should be outside the trial loop)
                        # tree = Tree(N)
                        # MCTS = MonteCarloTreeSearch(env=env_copy, agent=agent, tree=tree, exploration_constant=exploration_constant, discount_factor=discount_factor)
                        assert MCTS.env.sim == True, 'env not in sim mode'
                    
                        ## search
                        action, MCTS_Q = MCTS.search(n_sims, n_iter=n_iter, lazy=lazy)
                        actions.append(action)
                        Q_values.append(MCTS_Q)
                        choice_probs.append(softmax(MCTS_Q))
                        leaf_visits = [leaf.n_action_visits for leaf in MCTS.tree.root.action_leaves.values()]


                        ### optional: check what the CE agent would have done with the mean of this set of samples

                        ## ensure that all unobserved row and columns are set to the prior mean (i.e. proper CE)
                        all_posterior_ps_tmp = agent.all_posterior_ps.copy()
                        all_posterior_qs_tmp = agent.all_posterior_qs.copy()
                        for i in range(N):
                            for o in env_copy.obs:
                                if o[0] == i:
                                    break
                            else:
                                all_posterior_ps_tmp[:,i] = env_copy.alpha_row / (env_copy.alpha_row + env_copy.beta_row)
                        for j in range(N):
                            for o in env_copy.obs:
                                if o[1] == j:
                                    break
                            else:
                                all_posterior_qs_tmp[:,j] = env_copy.alpha_col / (env_copy.alpha_col + env_copy.beta_col)
                        all_posterior_p_costs_tmp = np.zeros((n_sims, N,N))
                        for s in range(n_sims):
                            all_posterior_p_costs_tmp[s] = np.outer(all_posterior_ps_tmp[s], all_posterior_qs_tmp[s])
                        posterior_mean_p_cost_tmp = np.mean(all_posterior_p_costs_tmp, axis=0)

                        ## get CE's action
                        if expt=='free':
                            agent.dp(posterior_mean_p_cost_tmp, expected_cost=True)
                            action_CE = agent.optimal_policy(current, agent.Q_inf)
                            CE_actions.append(action_CE)
                        elif expt=='AFC':
                            path_costs = []
                            for path_id in range(env_copy.n_afc):
                                path_states = env_copy.path_states[t][path_id]
                                path_cost = 0
                                for state in path_states:
                                    # path_cost += env_copy.get_pred_cost(state) ## i.e. sample binary costs from the posterior pqs
                                    path_cost += posterior_mean_p_cost_tmp[state[0], state[1]]*env_copy.low_cost + (1-posterior_mean_p_cost_tmp[state[0], state[1]])*env_copy.high_cost
                                path_costs.append(path_cost)
                            action_CE = argm(path_costs, np.max(path_costs))
                            CE_actions.append(action_CE)
                            CE_Q_values.append(path_costs)



                        ## plot for debugging?
                        # print('next action: BAMCP action:', env_copy.action_labels[action],', CE action:', env_copy.action_labels[action_CE])
                        # _, axs = plt.subplots(1, 3, figsize=(21,7))
                        # plot_r(env_copy.p_costs.reshape(N,N), ax=axs[0], title = 'p_costs')
                        # a_traj = np.zeros((len(env_copy.a_traj),3))
                        # for i, a in enumerate(env_copy.a_traj):
                        #     a_traj[i,:2] = a
                        #     a_traj[i,2] = env_copy.costs[a[0], a[1]]
                        # # plot_traj([env_copy.o_trajs[t], env_copy.a_traj], ax=axs[0])
                        # plot_traj([env_copy.o_trajs[t], a_traj], ax=axs[0])
                        # plot_r(agent.posterior_mean_p_cost, ax=axs[1], title = 'average posterior p cost')
                        # plot_action_tree(agent.Q_inf, current, goal, ax=axs[2], title = 'CE_DP_inf')
                        # plt.show()
                        # MCTS_Q_labelled = {env_copy.action_labels[k]:v for k,v in enumerate(MCTS_Q)}
                        # if action != action_CE:
                        #     print('MCTS Q:', MCTS_Q_labelled)
                        #     print('n_visits of action leaves:',{env_copy.action_labels[k]:v.n_action_visits for k,v in MCTS.tree.root.action_leaves.items()})

                        ## presentation plot of agent's observations, and the posterior mean (plot all trials)
                        # fig, axs = plt.subplots(2, n_trials, figsize=(n_trials*5, 10))
                        # for trial in range(n_trials):
                        #     observed_costs = np.zeros((N,N)) + np.nan
                        #     for i,j,c in env_copy.obs:
                        #         observed_costs[i,j] = c
                        #     plot_r(observed_costs+1, ax = axs[1, trial], title = f'Grid {day+1}, Trial {ep+1}', cbar=False)
                        #     plot_traj([env.path_states[trial][0], env.path_states[trial][1]], ax = axs[1, trial], expt=expt)
                        # all_posterior_p_costs_plot.append(agent.posterior_mean_p_cost)
                        # all_posterior_contexts_plot.append(farmer.context_prob)
                        # for trial in range(0, t+1):
                        #     context_title = title = r'$p(z_{c}) = $'+str(all_posterior_contexts_plot[trial].round(2))
                        #     title = f'Posterior mean p(low cost)\n{context_title}'
                        #     plot_r(all_posterior_p_costs_plot[trial], ax = axs[0, trial], title = title, cbar=False)
                        #     plot_traj([env.path_states[trial][0], env.path_states[trial][1]], ax = axs[0, trial], expt=expt)
                        # for trial in range(t+1, n_trials):
                        #     fig.delaxes(axs[0,ep])
                        # plt.show()



                        ## take action
                        env_copy.set_sim(False)
                        env_copy.init_trial(action)
                        start = env_copy.current
                        goal = env_copy.goal
                        assert np.array_equal(start, env_copy.starts[t][action]), 'current state does not match start state\n current: {}, start: {}'.format(env_copy.current, env_copy.starts[t][action])
                        if expt=='free':
                            current, cost, terminated, _, _ = env_copy.step(action)
                            next_node_id = np.append(current,cost)
                            steps += 1
                        elif expt=='AFC':
                            action_sequence = env_copy.path_actions[t][action]
                            _, _ = env_copy.take_path(action_sequence)
                            # current = states[-1]
                            current = env_copy.current
                            # path_cost = np.sum(costs)
                            costs = env_copy.trial_obs[:,-1]
                            assert len(costs) == len(action_sequence)+1, 'costs and action sequence do not match\n costs: {}, action sequence: {}'.format(len(costs), len(action_sequence))
                            path_cost = np.sum(costs)
                            terminated = True ## trivially true in AFC
                            day_terminated = t == (n_trials-1)

                            ## update next node id
                            # next_node_id = MCTS.init_node_id(env_copy.obs.copy(), None, t)
                            trial_obs = env_copy.trial_obs.copy()
                            next_node_id = MCTS.init_node_id(trial_obs, MCTS.tree.root.node_id, t)


                        ## check for backtracking
                        if len(actions)>1:
                            # backtracked = np.abs(action-actions[-2]) ==2
                            backtracked = np.array_equal(current, env_copy.a_traj[-3])
                            if backtracked:
                                # print(MCTS.tree.print_tree(MCTS.tree.root))
                                # print('backtracked in state:', current,' back from ', env_copy.a_traj[-2])
                                raise ValueError('backtracked in state:', current,' back from ', env_copy.a_traj[-2], ', en route to ', goal)
                        

                        ## update observations
                        agent.get_env_info(env_copy)

                        ### prune tree
                        if expt=='free': 
                            
                            ## no need to prune at the end of the trial in free-choice expt, since the tree resets at this point
                            if not terminated:
                                MCTS.tree.prune(action, next_node_id)
                                # assert np.array_equal(MCTS.tree.root.belief_state[:2], current), 'error in root update\n env is in: {} but tree is in: {}\n should have taken action {}'.format(current, MCTS.tree.root.belief_state, action)
                                # assert MCTS.tree.root.belief_state[0] == env_copy.trial, 'error in root update\n env trial: {} but tree trial: {}'.format(env_copy.trial, MCTS.tree.root.belief_state[0])
                                assert MCTS.tree.root.trial == env_copy.trial, 'error in root update\n env trial: {} but tree trial: {}'.format(env_copy.trial, MCTS.tree.root.trial)

                        elif expt=='AFC':

                            ## pruning not always successful due to high branching factor, in which case reset the tree
                            if not day_terminated:

                                if next_node_id in MCTS.tree.root.action_leaves[action].children:
                                    MCTS.tree.prune(action, next_node_id)
                                    # assert np.array_equal(MCTS.tree.root.belief_state[2*MCTS.n_afc:], costs), 'error in root update\n root state: {} \n costs: {}'.format(MCTS.tree.root.belief_state[2*MCTS.n_afc:], costs)
                                    # assert np.array_equal(MCTS.tree.root.belief_state[1:], costs), 'error in root update\n root state: {} \n costs: {}'.format(MCTS.tree.root.belief_state[1:], costs)
                                    assert MCTS.tree.root.trial == env_copy.trial, 'error in root update\n env trial: {} but tree trial: {}'.format(env_copy.trial, MCTS.tree.root.trial) ## this might actually be env_copy.trial+1
                                    tree_resets[ag] = False
                                    # print('successful prune after action {}. new root has two leaves with a total of {} children'.format(action, np.sum(len(MCTS.tree.root.action_leaves[a].children) for a in MCTS.tree.root.action_leaves.keys())))
                                    # for a in MCTS.tree.root.action_leaves.keys():
                                    #     print('action:', a, ', n_children:', len(MCTS.tree.root.action_leaves[a].children))
                                    #     for child in MCTS.tree.root.action_leaves[a].children:
                                    #         print(np.sum(np.array(child).reshape(N,N,2), axis=2))
                                    #         print()
                                else:
                                    # print('n_children:', len(MCTS.tree.root.action_leaves[action].children))
                                    # for child in MCTS.tree.root.action_leaves[action].children.values():
                                    #     print(child.node_id)
                                    # print('unsuccessful prune after action {}'.format(action))
                                    # print('failed to find:\n', next_node_id)
                                    # print('next node id:', np.sum(np.array(next_node_id).reshape(N,N,2), axis=2))
                                    # tree_reset = True
                                    tree_resets[ag] = True
                        MCTSs[ag] = MCTS
                    
                    ## get the context prior - i.e. the probability with which samples were drawn
                    context_prior = agent.context_prob
                    # print('prior context:', farmer.context_prob)

                    # get the new context posterior for this agent
                    # print('agent:', ag, ', trial:', e,', day:', day, ', action:', action, ', context prior:', farmer.context_prob)
                    # print('farmers prior context:',farmer.context_prob)
                    context_posterior = farmer.quick_context_posterior(env_copy.obs)
                    # print('posterior context:', context_posterior)
                    # print()

                    ## expected KL
                    # if (ag=='BAMCP') or (ag=='BAMCP w/ CE'):
                    #     expected_KL = MCTS.expected_KL(env_copy)
                    #     EKLs.append(expected_KL)

                    
                    ## prevent endless trial 
                    if steps >= max_steps:
                        early_terminate = True

                    if early_terminate:
                        print('grid ',ppt,': trial ',t,' terminated for agent ',ag,' after ',steps,' steps')
                        # raise ValueError('grid ',m,': trial ',t,' terminated for agent ',ag,' after ',steps,' steps')

                        ## or just skip to the next trial
                        sim_out['agent'].append(agent)
                        sim_out['day'].append(day)
                        sim_out['trial'].append(t)
                        sim_out['grid'].append(ppt)
                        sim_out['start'].append(start)
                        sim_out['goal'].append(goal)
                        sim_out['path_A'].append(env_copy.path_states[t][0])
                        sim_out['path_B'].append(env_copy.path_states[t][1])
                        sim_out['path_A_expected_cost'].append(env_copy.path_expected_costs[t][0])
                        sim_out['path_B_expected_cost'].append(env_copy.path_expected_costs[t][1])
                        sim_out['path_A_actual_cost'].append(env_copy.path_actual_costs[t][0])
                        sim_out['path_B_actual_cost'].append(env_copy.path_actual_costs[t][1])
                        sim_out['path_A_future_overlap'].append(env_copy.path_future_overlaps[t][0])
                        sim_out['path_B_future_overlap'].append(env_copy.path_future_overlaps[t][1])
                        sim_out['abstract_sequence_A'].append(env_copy.sampled_abstract_sequences[t][0])
                        sim_out['abstract_sequence_B'].append(env_copy.sampled_abstract_sequences[t][1])
                        sim_out['context_prior'].append(context_prior)
                        sim_out['context_posterior'].append(context_posterior)
                        sim_out['actions'].append(actions)
                        sim_out['Q_values'].append(Q_values)
                        sim_out['choice_probs'].append(choice_probs)
                        sim_out['leaf_visits'].append(leaf_visits)
                        sim_out['CE_actions'].append(CE_actions)
                        sim_out['CE_Q_values'].append(CE_Q_values)
                        sim_out['optimal_actions'].append(env_copy.o_traj_actions[t])
                        sim_out['costs'].append(np.nan)
                        sim_out['optimal_costs'].append(env_copy.o_traj_costs[t])
                        sim_out['total_cost'].append(np.nan)
                        sim_out['total_optimal_cost'].append(env_copy.o_traj_total_costs[t])
                        sim_out['action_score'].append(np.nan)
                        sim_out['cost_ratio'].append(np.nan)
                        sim_out['n_steps'].append(steps)
                        sim_out['actual_trajectory'].append(env_copy.a_traj)
                        sim_out['optimal_trajectory'].append(env_copy.o_trajs[t])
                        sim_out['observations'].append(env_copy.obs)
                        # sim_out['action_tree'].append(MCTS.tree.action_tree())
                        sim_out['action_tree'].append(np.nan)
                        sim_out['expected_LD'].append(ELDs)
                        sim_out['expected_KL'].append(EKLs)

                        ## discounts
                        sim_out['discounted_costs'].append(np.nan)
                        sim_out['total_discounted_cost'].append(np.nan)
                        discounts = [discount_factor**d for d in range(len(env_copy.o_trajs[t]))] ## NEED TO FIX THIS BUT DON'T HAVE TIME
                        discounted_costs = [c*d for c,d in zip(env_copy.o_traj_costs[t], discounts)]
                        sim_out['discounted_optimal_costs'].append(discounted_costs)
                        sim_out['total_discounted_optimal_cost'].append(np.sum(discounted_costs))
                        
                        ## GP-specific
                        # sim_out['true_k'].append(true_k)
                        # sim_out['RPE'].append(np.nan)
                        # sim_out['posterior_mean'].append(np.nan)
                        # sim_out['theta_MLE'].append(best_theta)

                        end_trial = True

                        ## stop the sim here!
                        return sim_out, env_copy.p_costs

                    ## save data and end the trial
                    elif terminated:
                        sim_out['agent'].append(ag)
                        sim_out['day'].append(day)
                        sim_out['trial'].append(t)
                        sim_out['grid'].append(ppt)
                        sim_out['start'].append(start)
                        sim_out['goal'].append(goal)
                        sim_out['path_A'].append(env_copy.path_states[t][0])
                        sim_out['path_B'].append(env_copy.path_states[t][1])
                        sim_out['path_A_expected_cost'].append(env_copy.path_expected_costs[t][0])
                        sim_out['path_B_expected_cost'].append(env_copy.path_expected_costs[t][1])
                        sim_out['path_A_actual_cost'].append(env_copy.path_actual_costs[t][0])
                        sim_out['path_B_actual_cost'].append(env_copy.path_actual_costs[t][1])
                        sim_out['abstract_sequence_A'].append(env_copy.sampled_abstract_sequences[t][0])
                        sim_out['abstract_sequence_B'].append(env_copy.sampled_abstract_sequences[t][1])
                        sim_out['path_A_future_overlap'].append(env_copy.path_future_overlaps[t][0])
                        sim_out['path_B_future_overlap'].append(env_copy.path_future_overlaps[t][1])
                        sim_out['context_prior'].append(context_prior)
                        sim_out['context_posterior'].append(context_posterior)
                        sim_out['actions'].append(actions)
                        sim_out['Q_values'].append(Q_values)
                        sim_out['choice_probs'].append(choice_probs)
                        sim_out['leaf_visits'].append(leaf_visits)
                        sim_out['CE_actions'].append(CE_actions)
                        sim_out['CE_Q_values'].append(CE_Q_values)
                        sim_out['optimal_actions'].append(env_copy.o_traj_actions[t])
                        # if np.round(env_copy.optimal_cost,4) < np.round(env_copy.accrued_cost,4):
                        #     print(env_copy.optimal_cost, env_copy.accrued_cost)
                        # assert np.round(env_copy.optimal_cost,4) >= np.round(env_copy.accrued_cost,4), 'accrued cost higher than optimal cost'
                        # sim_out['action_score'].append(env_copy.optimal_cost/env_copy.accrued_cost)
                        sim_out['action_score'].append(env_copy.action_score)
                        sim_out['cost_ratio'].append(env_copy.cost_ratio)
                        sim_out['n_steps'].append(steps)
                        sim_out['actual_trajectory'].append(env_copy.a_traj)
                        sim_out['optimal_trajectory'].append(env_copy.o_trajs[t])
                        sim_out['observations'].append(env_copy.obs)
                        # sim_out['action_tree'].append(MCTS.tree.action_tree())
                        sim_out['action_tree'].append(np.nan)
                        sim_out['expected_LD'].append(ELDs)
                        sim_out['expected_KL'].append(EKLs)

                        ### costs

                        ## actual costs
                        sim_out['costs'].append(env_copy.a_traj_costs)
                        sim_out['optimal_costs'].append(env_copy.o_traj_costs[t])
                    
                        # INC START AND END COSTS
                        # sim_out['total_cost'].append(env_copy.a_traj_total_cost) 
                        # sim_out['total_optimal_cost'].append(env_copy.o_traj_total_costs[t])

                        # EXC START AND END COSTS
                        sim_out['total_cost'].append(np.sum(env_copy.a_traj_costs[1:-1]))
                        sim_out['total_optimal_cost'].append(np.sum(env_copy.o_traj_costs[t][1:-1]))


                        ## calculate discounted actual and optimal costs

                        ## INC START AND END COSTS
                        # discounts = [discount_factor**d for d in range(len(env_copy.a_traj_costs))] 
                        # discounted_costs = [c*d for c,d in zip(env_copy.a_traj_costs, discounts)]
                        # sim_out['discounted_costs'].append(discounted_costs)
                        # sim_out['total_discounted_cost'].append(np.sum(discounted_costs))
                        # discounts = [discount_factor**d for d in range(len(env_copy.o_trajs[t]))]
                        # discounted_costs = [c*d for c,d in zip(env_copy.o_traj_costs[t], discounts)]
                        # sim_out['discounted_optimal_costs'].append(discounted_costs)
                        # sim_out['total_discounted_optimal_cost'].append(np.sum(discounted_costs))

                        ## EXC START AND END COSTS
                        discounts = [discount_factor**d for d in range(len(env_copy.a_traj_costs)-2)]
                        discounted_costs = [c*d for c,d in zip(env_copy.a_traj_costs[1:-1], discounts)]
                        sim_out['discounted_costs'].append(discounted_costs)
                        sim_out['total_discounted_cost'].append(np.sum(discounted_costs))
                        discounts = [discount_factor**d for d in range(len(env_copy.o_trajs[t])-2)]
                        discounted_costs = [c*d for c,d in zip(env_copy.o_traj_costs[t][1:-1], discounts)]
                        sim_out['discounted_optimal_costs'].append(discounted_costs)
                        sim_out['total_discounted_optimal_cost'].append(np.sum(discounted_costs))

                        
                        ## update the agent env
                        # agent_envs[a] = copy.deepcopy(env_copy)
                        agent_envs[ag] = env_copy

                        end_trial = True
            
                ## carry over the context prob to the next run, if on the final trial of the day
                if t == (n_trials-1):
                    context_priors[ag] = context_posterior
                # else:
                #     context_priors[ag] = context_prior
                # print('new context prob for agent {}: {}'.format(ag, context_posterior))
            
            if progress:
                pbar.update(1)
        if progress & (n_days <= 1):
            pbar.close()

        ## save the env for this day
        all_day_envs.append(env_copy)

    if progress & (n_days > 1):
        pbar.close()
                    

    return sim_out,all_day_envs
    # return sim_out, _