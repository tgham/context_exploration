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
        self.tree = tree
        self.update_trial()
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
        node_id = self.init_node_id(self.env.obs, None, self.root_trial)

        ## some debugging metrics
        self.exploratory_steps = 0
        self.exploitative_steps = 0
        self.max_Q = np.zeros(self.env.n_trials) - np.inf
        self.min_Q = np.zeros(self.env.n_trials) 

        
        ### node needs to contain paths, actions, starts and goals for that trial so that these can be inherited by the action node
        path_actions = self.env.path_actions[self.root_trial].copy()
        path_states = self.env.path_states[self.root_trial].copy()
        starts = self.env.starts[self.root_trial].copy()
        goals = self.env.goals[self.root_trial].copy()

        ## add state node to the tree
        self.tree.add_state_node(node_id=node_id, cost=None, terminated=False, trial = self.root_trial, n_afc = self.n_afc, parent=None, 
                                path_actions=path_actions, path_states=path_states, starts=starts, goals=goals
                                 )

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
        assert node_trial == node.trial, 'trial mismatch between env and tree\n env: {} \n tree: {}\n MCTS: {}'.format(node_trial, node.trial, self.root_trial)
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
            #     self.env.set_state(self.root_state)
            #     self.env.set_goal(self.root_goal)
            #     self.env.set_trial(self.root_trial)
            #     assert np.array_equal(self.env.current, self.root_state), 'env state not reverted properly'
            #     assert self.env.trial == self.root_trial, 'env trial not reverted properly'
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
                self.env.set_state(self.root_state)
                self.env.set_goal(self.root_goal)
                self.env.set_trial(self.root_trial)
                assert np.array_equal(self.env.current, self.root_state), 'env state not reverted properly'
                assert self.env.trial == self.root_trial, 'env trial not reverted properly. should be in {}, but actually in {}'.format(self.root_trial, self.env.trial)

                return action_leaf
                
            ## selection step
            else:

                ## get the best child
                action_leaf = self.best_child(node)

                ## update the tree path
                self.tree_actions.append(action_leaf.action)
                self.node_id_path.append(node.node_id)

                ## move in env
                _, costs, terminated, node_trial, next_node_id = self.tree_step(action_leaf)

                # see if the next state node already exists as a child of this action leaf
                if next_node_id in action_leaf.children:
                    node = action_leaf.children[next_node_id]
                else:

                    ## full BAMCP: trial info for next node is just inherited from the env (need to sort this out...)
                    # if not terminated:
                    #     next_path_actions = self.env.path_actions[node_trial].copy()
                    #     next_path_states = self.env.path_states[node_trial].copy()
                    #     next_starts = self.env.starts[node_trial].copy()
                    #     next_goals = self.env.goals[node_trial].copy()
                    # else:
                        # next_path_actions= None
                    #     next_path_states= None
                    #     next_starts = None
                    #     next_goals = None

                    ## PA-BAMCP: trial info for the next node (i.e. the node to which the action leaf leads) is sampled
                    if not terminated:
                        _, next_path_actions, next_path_states, next_starts, next_goals = self.env.sample_paths_given_future_states(self.root_trial)
                    else:
                        next_path_actions= None
                        next_path_states= None
                        next_starts = None
                        next_goals = None

                    ## create new node
                    node = self.tree.add_state_node(node_id=next_node_id, cost = costs, terminated=terminated, trial = node_trial, n_afc = self.n_afc, parent=action_leaf,
                                        path_actions=next_path_actions, path_states=next_path_states, starts=next_starts, goals=next_goals
                                                    )

                ## debugging NB NEED TO FIGURE THIS OUT FOR PA-BAMCP
                # print(next_state, self.env.current, self.env.starts)
                # assert np.array_equal(next_state, self.env.current), 'mismatch between env and tree state\n env: {} \n tree: {}'.format(self.env.current, next_state)


        ## if terminal node, there are no mode action leaves to choose from
        if node.terminated:
            action_leaf = None

        ## revert env
        self.env.set_state(self.root_state)
        self.env.set_goal(self.root_goal)
        self.env.set_trial(self.root_trial)
        assert np.array_equal(self.env.current, self.root_state), 'env state not reverted properly'
        assert self.env.trial == self.root_trial, 'env trial not reverted properly'

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
            # assert depth == node.trial, 'Tree path mismatch:\n node trial: {} \n depth: {}'.format(node.trial, depth)

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

        ## check root
        assert self.root_trial == self.env.trial, 'trial mismatch between env and tree at start of search\n env trial: {} \n tree trial: {}'.format(self.env.trial, self.root_trial)
        for a in range(self.n_afc):
            assert np.array_equal(self.tree.root.starts[a], self.env.starts[self.root_trial][a]), 'start state mismatch for action {}\n env start: {} \n tree start: {}'.format(a, self.env.starts[self.root_trial][a], self.tree.root.starts[a])
            assert np.array_equal(self.tree.root.goals[a], self.env.goals[self.root_trial][a]), 'goal state mismatch for action {}\n env goal: {} \n tree goal: {}'.format(a, self.env.goals[self.root_trial][a], self.tree.root.goals[a])
            assert np.array_equal(self.tree.root.path_states[a], self.env.path_states[self.root_trial][a]), 'path state mismatch for action {}\n env path: {} \n tree path: {}'.format(a, self.env.path_states[self.root_trial][a], self.tree.root.path_states[a])

        ## generate new set of root samples
        self.agent.root_samples(obs = self.env.obs, n_samples=n_sims, n_iter=n_iter, lazy=lazy, CE=False, combo=False)

        # debugging plot
        # plt.figure()
        # plot_r(self.agent.posterior_mean_p_cost.reshape(self.N,self.N), ax = plt.subplot(), title='posterior sample')
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
        
        ## loop through simulations
        for s in range(n_sims):
            
            ## root sampling of new posterior
            posterior_p_cost = self.agent.all_posterior_p_costs[s]
            if self.expt=='free': ## only need to do this in the free-choice expt
                self.agent.dp(posterior_p_cost, expected_cost=True)
            self.env.receive_predictions(posterior_p_cost)

            ## selection, expansion, simulation
            action_leaf = self.tree_policy()
            self.rollout_policy(action_leaf)
            
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

        return action, MCTS_estimates
    

### subclasses for different experiments

## free exploration
class MonteCarloTreeSearch_Free(MonteCarloTreeSearch):
    
    def __init__(self, env, agent, tree, exploration_constant=2, discount_factor=0.99):
        super().__init__(env, agent, tree, exploration_constant, discount_factor)

    def init_node_id(self, obs=None, init_info_state = None, trial=None):
        node_id = tuple(np.append(self.root_state, self.env.costs[self.root_state[0], self.root_state[1]]))
        return node_id
    
    def update_trial(self):
        self.root_state = self.env.current
        self.root_goal = self.env.goal
        self.root_trial = self.env.trial

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

        ## one-armed weighting: store the aligned and orthogonal states
        # node.action_leaves[action].aligned_states = self.env.path_aligned_states[node.trial][action] ## full BAMCP
        # node.action_leaves[action].orthogonal_states = self.env.path_orthogonal_states[node.trial][action] ## full BAMCP
        node.action_leaves[action].aligned_states, node.action_leaves[action].orthogonal_states = self.env.get_alignment([[node.path_states[action]]]) ## PA-BAMCP

        return node.action_leaves[action]

    ## tree step
    def tree_step(self, action_leaf):
        action = action_leaf.action
        next_state, cost, terminated, _, _ = self.env.step(action)
        self.tree_costs.append(cost)
        node_trial = self.root_trial ##??
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
            if np.array_equal(current, self.root_goal):
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
            
        ### update info for s-a leaf - i.e. the state-action pair

        ## full BAMCP:
        # start = self.env.starts[node.trial][action].copy()
        # goal = self.env.goals[node.trial][action].copy()
        # assert np.array_equal(start, node.starts[action]), 'mismatch between node and env start states\n node: {} \n env: {}'.format(node.starts[action], start)

        ## PA-BAMCP:
        start = node.starts[action].copy()
        goal = node.goals[action].copy()

        node.action_leaves[action] = Action_Node(start = start, action=action, goal = goal, terminated=terminated, trial=node.trial, parent_id=node.node_id)
        node.action_leaves[action].performance = 0
        node.action_leaves[action].norm_performance = 0

        ### expansion of node attaches a sampled pair of paths to the leaf
        node.action_leaves[action].start = start
        node.action_leaves[action].goal = goal
        node.action_leaves[action].path_actions = node.path_actions[action]
        node.action_leaves[action].path_states = node.path_states[action]
        
        ## one-armed weighting: store the aligned and orthogonal states
        # node.action_leaves[action].aligned_states = self.env.path_aligned_states[node.trial][action] ## full BAMCP
        # node.action_leaves[action].orthogonal_states = self.env.path_orthogonal_states[node.trial][action] ## full BAMCP
        node.action_leaves[action].aligned_states, node.action_leaves[action].orthogonal_states = self.env.get_alignment([[node.path_states[action]]]) ## PA-BAMCP
        
        return node.action_leaves[action]
    
    def update_trial(self):
        self.root_trial = self.env.trial ## i.e. the trial that the agent is current faced with in the real env
        self.root_state = self.env.starts[self.root_trial].copy() ## i.e. the two possible start states for this trial
        self.root_goal = self.env.goals[self.root_trial].copy() ## i.e. the two possible goal states for this trial

        ## PA-BAMCP: root's trial info now needs to accurately reflect the env
        if self.tree.root is not None:
            for a in range(self.n_afc):
                self.tree.root.starts[a] = self.env.starts[self.root_trial][a].copy()
                self.tree.root.goals[a] = self.env.goals[self.root_trial][a].copy()
                self.tree.root.path_states[a] = self.env.path_states[self.root_trial][a]
                self.tree.root.path_actions[a] = self.env.path_actions[self.root_trial][a]
                self.tree.root.action_leaves[a].start = self.env.starts[self.root_trial][a].copy()
                self.tree.root.action_leaves[a].goal = self.env.goals[self.root_trial][a].copy()
                self.tree.root.action_leaves[a].path_states = self.env.path_states[self.root_trial][a]
                self.tree.root.action_leaves[a].path_actions = self.env.path_actions[self.root_trial][a]
                self.tree.root.action_leaves[a].aligned_states = self.env.path_aligned_states[self.root_trial][a]
                self.tree.root.action_leaves[a].orthogonal_states = self.env.path_orthogonal_states[self.root_trial][a]

    ## tree step
    def tree_step(self, action_leaf):
        
        ## get some initial info about the step
        step_trial = action_leaf.trial
        path_id = action_leaf.action
        assert self.env.trial == action_leaf.trial, 'trial mismatch between env and tree\n env: {} \n tree: {}\n MCTS: {}'.format(self.env.trial, action_leaf.trial, self.root_trial)

        ## full BAMCP: use actual upcoming paths
        # start_tmp = self.env.starts[step_trial][path_id].copy()
        # goal_tmp = self.env.goals[step_trial][path_id].copy()
        # action_sequence = self.env.path_actions[step_trial][path_id]
        # assert np.array_equal(start_tmp, action_leaf.start), 'mismatch between node and env start states\n node: {} \n env: {}'.format(action_leaf.start, start_tmp)

        ## PA-BAMCP: sample two paths 
        start_tmp = action_leaf.start.copy()
        goal_tmp = action_leaf.goal.copy()
        action_sequence = action_leaf.path_actions.copy()

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

        ## one-arm bias 
        # aligned_states, orthogonal_states = self.env.path_aligned_states[step_trial][path_id], self.env.path_orthogonal_states[step_trial][path_id] ## full BAMCP
        aligned_states, orthogonal_states = action_leaf.aligned_states, action_leaf.orthogonal_states ## PA-BAMCP
        weighted_costs = self.agent.arm_reweighting(self.env.predicted_costs, aligned_states, orthogonal_states)

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
            # next_state = self.env.sampled_starts[node_trial].copy()
            next_state = None ## sort this out later, but it seems we don't really need this for this task

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

        ### first need to get the starting cost r, which is essentially the cost of path choice that corresponds to the action leaf

        first_trial = action_leaf.trial
        path_id = action_leaf.action

        ## full BAMCP: use actual upcoming paths
        # first_path_states = self.env.path_states[first_trial][path_id]

        ## PA-BAMCP: sample paths
        first_path_states = action_leaf.path_states

        ## unweighted
        # starting_costs = [self.env.predicted_costs[state[0], state[1]] for state in first_path_states]
        # total_cost = sum(starting_costs)

        ## weighted
        aligned_states, orthogonal_states = action_leaf.aligned_states, action_leaf.orthogonal_states
        starting_costs = self.agent.arm_reweighting(self.env.predicted_costs, aligned_states, orthogonal_states)
        total_cost = sum(starting_costs)

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
            _, _, sampled_path_states, _, _ = self.env.sample_paths_given_future_states(self.root_trial) ## PA-BAMCP

            ### greedy: get the total cost of the paths and return the better one
            path_costs = []
            for path_id in range(self.n_afc):

                ## full BAMCP - greedy choice wrt/ predicted costs of actual paths
                # path_states = self.env.path_states[trial][path_id] ## full BAMCP

                ## PA-BAMCP - greedy choice wrt/ predicted costs of sampled paths
                path_states = sampled_path_states[path_id]

                ## unweighted
                costs = [self.env.predicted_costs[state[0], state[1]] for state in path_states]

                ## arm-weighted
                aligned_states, orthogonal_states = self.env.get_alignment([[path_states]])
                costs = self.agent.arm_reweighting(self.env.predicted_costs, aligned_states, orthogonal_states)
                
                ro_cost = sum(costs)
                path_costs.append(ro_cost)
            best_ro_cost = np.max(path_costs) 
            remaining_ro_costs.append(best_ro_cost)
            ro_choices.append(np.argmax(path_costs))
            total_cost += best_ro_cost * self.discount_factor**depth

            
            ### RANDOM: randomly choose between the paths

            # ## full BAMCP - random choice between actual upcoming paths
            # # path_id = np.random.choice(self.n_afc)
            # # path_states = self.env.path_states[trial][path_id] ## full BAMCP

            # ## PA-BAMCP - choose between randomly sampled upcoming paths
            # path_id = np.random.choice(self.n_afc)
            # path_states = sampled_path_states[path_id] ## PA-BAMCP

            # ## unweighted
            # # costs = [self.env.predicted_costs[state[0], state[1]] for state in path_states]

            # ## arm-weighted
            # aligned_states, orthogonal_states = self.env.get_alignment([[path_states]])
            # costs = self.agent.arm_reweighting(self.env.predicted_costs, aligned_states, orthogonal_states)

            # ro_cost = sum(costs)
            # remaining_ro_costs.append(ro_cost)
            # ro_choices.append(path_id)
            # total_cost += ro_cost * self.discount_factor**depth


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
