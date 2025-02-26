import random
from math import sqrt, log
from utils import Node, Action_Node, Tree, make_env, argm, data_keys, KL_divergence, get_next_state
import copy
import numpy as np
from tqdm.auto import tqdm
import os
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from scipy import special

from plotter import *
from agents import GPAgent, Farmer

## base class 
class MonteCarloTreeSearch():

    def __init__(self, env, agent, tree, exploration_constant=2, discount_factor=0.99):
        self.env = env
        self.expt = env.expt
        self.n_afc = self.env.n_afc
        self.agent = agent
        self.update_episode()
        self.tree = tree
        self.N = self.env.N
        self.exploration_constant = exploration_constant
        self.discount_factor = discount_factor
        starting_cost = self.env.costs[self.actual_state[0], self.actual_state[1]]

        ## create id for root node
        node_id = self.init_node_id(self.env.obs, None)

        ## some debugging metrics
        self.exploratory_steps = 0
        self.exploitative_steps = 0

        ## add state node to the tree
        self.tree.add_state_node(state=self.actual_state, cost=starting_cost, node_id=node_id, goal = self.actual_goal, terminated=False, episode = self.actual_episode, n_afc = self.n_afc, parent=None)

    def init_node_id(self, obs=None, init_info_state=None):
        raise NotImplementedError('init_node_id not implemented in subclass')

    ## update MCTS with episode info
    def update_episode(self):
        raise NotImplementedError('episode update not implemented in subclass')

    ## tree step
    def tree_step(self, action_leaf):
        raise NotImplementedError('tree step not implemented in subclass')
    
    ## rollout policy
    def rollout_policy(self, action_leaf):
        raise NotImplementedError('rollout policy not implemented in subclass')

    ## expand the action space of a node
    def expand(self, node):
        assert self.env.sim, 'env is not in sim mode'

        ## take action (or path) and get new state
        action = node.untried_action()
        if self.expt == 'free':
            next_state, _, terminated, _, _ = self.env.step(action)

            ## reset the environment to the actual state
            # self.env.set_state(self.actual_state)

        elif self.expt == '2AFC':
            terminated = node.episode == self.env.n_episodes-1 ## i.e. this action leaf corresponds to the action made in the final episode, so it leads to termination of the block
            # next_state = node.state[:2] #i.e. this stays the same since agent always regens to the same start state. may instead choose to fill this with the states that are actually traversed
            if not terminated:
                # next_state = self.env.starts[node.episode]
                next_state = self.env.goals[node.episode][action].copy()
            else:
                # next_state = np.array([None, None])
                next_state = self.env.goals[node.episode][action].copy()
            
        ## update info for s-a leaf - i.e. the state-action pair
        # prev_state = node.state
        prev_state = self.env.starts[node.episode][action].copy()
        node.action_leaves[action] = Action_Node(prev_state = prev_state, action=action, next_state = next_state, terminated=terminated, episode=node.episode, parent_id=node.node_id)
        node.action_leaves[action].performance = 0
        node.action_leaves[action].norm_performance = 0

        return node.action_leaves[action]
    
    ## debugging method for checking if node and env states match
    def check_state(self, node):
        if self.env.same_SGs:
            assert np.array_equal(node.state[:2], self.env.current), 'mismatch between node and env state\n node: {} \n env: {}'.format(node, self.env.current) ### USE THIS IN THE SUBCLASS DEFINITION FOR THE FREE EXPT TOO
        else:
            # assert np.array_equal(node.state[:2*self.n_afc].reshape(2, self.n_afc), self.env.starts[node.episode]), 'mismatch between node and env state\n node: {} \n env: {}'.format(node.state[:2*self.n_afc].reshape(2, self.n_afc), self.env.starts[node.episode])
            assert np.array_equal(node.state[:2*self.n_afc].reshape(2, self.n_afc), self.env.current), 'mismatch between node and env state\n node: {} \n env: {}'.format(node.state[:2*self.n_afc].reshape(2, self.n_afc), self.env.current)

    ## take an action according to the tree policy, i.e. take the best UCT child and see where it takes you
    def tree_policy(self):

        ## initialise the tree
        node = self.tree.root
        t = 0
        node_episode = self.env.episode
        goal = node.goal
        assert node_episode == node.episode, 'episode mismatch between env and tree\n env: {} \n tree: {}\n MCTS: {}'.format(node_episode, node.episode, self.actual_episode)
        # assert np.array_equal(node.state[:2], self.actual_state), 'mismatch between node and env state\n node: {} \n env: {}'.format(node, self.actual_state)
        self.check_state(node)


        ## create a record of the nodes/leaves visited in the tree
        self.tree_costs = [] ## i.e. the cost associated with each traversal of the tree *under the tree policy*. Hence, this does not include the cost of the current state, which is the starting point of the tree policy, nor does it include the cost of expansion.
        self.tree_path = [] ## i.e. the states and actions visited in the tree. This *does* include the root, because it is from the root that we move to the next leaf (and then next node). 
        self.node_id_path = []

        ## create a copy of the env
        # env_copy = copy.deepcopy(self.env)
        # assert env_copy.sim, 'env copy is not in sim mode'
        # if self.expt == '2AFC':
        #     # env_tmp = copy.deepcopy(self.env)
        #     env_tmp = self.env
        # elif self.expt == 'free':
        #     env_tmp = self.env
        
        ## loop until you reach a leaf node or terminal state
        assert self.env.sim, 'env is not in sim mode'
        while not node.terminated:
            t+=1
            # assert np.array_equal(node.state[:2], self.env.current), 'mismatch between node and env state\n node: {} \n env: {}'.format(node, self.env.current)
            self.check_state(node)

            ## expansion step
            if self.tree.is_expandable(node):
                action_leaf = self.expand(node)
                self.tree_path.append(tuple([node.state, action_leaf.action]))
                self.node_id_path.append(node.node_id)

                ## update counts already?
                # action_leaf.n_action_visits += 1
                # node.n_state_visits += 1

                ## revert env
                self.env.set_state(self.actual_state)
                self.env.set_goal(self.actual_goal)
                self.env.set_episode(self.actual_episode)
                assert np.array_equal(self.env.current, self.actual_state), 'env state not reverted properly'
                assert self.env.episode == self.actual_episode, 'env episode not reverted properly. should be in {}, but actually in {}'.format(self.actual_episode, self.env.episode)

                return action_leaf
                
            ## selection step
            else:

                ## (some debugging vars)
                state_tmp = node.state[:2]

                ## get the best child
                action_leaf = self.best_child(node)
                self.tree_path.append(tuple([node.state, action_leaf.action]))
                self.node_id_path.append(node.node_id)

                ## move in env
                next_state, cost, terminated, node_episode, next_node_id = self.tree_step(action_leaf)

                # see if the next state node already exists as a child of this action leaf
                if next_node_id in action_leaf.children:
                    node = action_leaf.children[next_node_id]
                else:
                    node = self.tree.add_state_node(next_state, cost, next_node_id, goal, terminated, episode = node_episode, n_afc = self.n_afc, parent=action_leaf)

                ## debugging
                assert np.array_equal(next_state, self.env.current), 'mismatch between env and tree state\n env: {} \n tree: {}'.format(self.env.current, next_state)
                # assert np.array_equal(node.state[:2], next_state), 'error in tree policy step {}\n started in {}\n supposed to take action {} to {}\n ended up moving  to {}'.format(t, state_tmp, action_leaf.action, node.state[:2], action_leaf.next_state)

        ## if terminal node, there are no mode action leaves to choose from
        if node.terminated:
            action_leaf = None

        ## revert env
        self.env.set_state(self.actual_state)
        self.env.set_goal(self.actual_goal)
        self.env.set_episode(self.actual_episode)
        assert np.array_equal(self.env.current, self.actual_state), 'env state not reverted properly'
        assert self.env.episode == self.actual_episode, 'env episode not reverted properly'

        ## save tree obs for subsequent rollouts
        # self.tree_obs = self.env.obs_tmp.copy()
        # self.env.flush_obs()
        # assert len(self.tree_obs) == len(self.env.obs)+len(self.tree_path), 'tree obs and path lengths do not match\n tree obs: {}, env.obs: {}, tree path: {}'.format(len(self.tree_obs), len(self.env.obs),len(self.tree_path))

        return action_leaf


    ## backup costs until you reach the root
    def backup(self):
        tree_len = len(self.tree_costs)
        assert tree_len == len(self.tree_path), 'tree costs and path lengths do not match\n n tree costs: {} \n n tree path: {}\ntree path: {}\n tree costs: {}'.format(len(self.tree_costs), len(self.tree_path), self.tree_path, self.tree_costs)

        ## Precompute discount factors
        discount_factors = [self.discount_factor ** d for d in range(tree_len)]

        ## Loop through the tree path
        node = self.tree.root
        for depth, (state, action) in enumerate(self.tree_path):
            
            ## Get the corresponding action leaf
            action_leaf = node.action_leaves[action]

            ## Sanity check: ensure the current node matches the state in the path
            assert np.array_equal(node.state[:2], state[:2]), (
                f'Tree path mismatch:\n node: {node.state[:2]} \n state: {state[:2]}'
            )

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

            ## debugging: save updates applied to the first node
            # if depth == 0:
            #     if action==0:
            #         self.first_node_updates.append([discounted_cost, 0])
            #     elif action==1:
            #         self.first_node_updates.append([0, discounted_cost])


            ## update norm performance
            action_leaves = [leaf for leaf in node.action_leaves.values() if leaf is not None]

            # elif len(action_leaves) == 1:
            #     # If there is only one leaf, set its norm_performance to 1
            #     action_leaves[0].norm_performance = 0
            # else:
            #     pass
            # max_perf = np.max([child.performance for child in node.action_leaves.values() if child is not None])
            # min_perf = np.min([child.performance for child in node.action_leaves.values() if child is not None])
            # norm_term = max_perf - min_perf
            # for leaf in node.action_leaves.values():
            #     if leaf is not None:
            #         if norm_term == 0:
            #             leaf.norm_performance = 0
            #         else:
            #             leaf.norm_performance = (leaf.performance - min_perf) / norm_term

            ## Move to the next node in the path if not at the end
            if depth < tree_len - 1:
                next_node_id = self.node_id_path[depth+1]
                node = action_leaf.children[next_node_id]


    ## calculate E-E value
    def compute_UCT(self, node, action_leaf): 
        exploitation_term = action_leaf.performance
        # exploitation_term = action_leaf.norm_performance
        assert action_leaf.n_action_visits > 0 or action_leaf.terminated, 'action leaf has not been visited: {}'.format(action_leaf)
        exploration_term = self.exploration_constant * sqrt(log(node.n_state_visits) / action_leaf.n_action_visits)
        # print('exploration term:', exploration_term, 'exploitation term:', exploitation_term)
        return exploitation_term + exploration_term

    
    ## argmax based on UCT values? 
    def best_child(self, node):
    
        ## get action children
        action_leaves = [node.action_leaves[a] for a in node.action_leaves.keys()]

        ## create deep copy of action leaves
        # action_leaves = [copy.deepcopy(node.action_leaves[a]) for a in node.action_leaves.keys()]
        # print(action_leaves[0])

        ## some hacky fixes for free expt to prevent backtracking
        if self.expt == 'free':

            ## remove action that keeps you in your current state
            action_leaves = [leaf for leaf in action_leaves if not np.array_equal(leaf.next_state, leaf.prev_state)]
            
            ## remove action that takes you back to previous state in the tree
            if len(self.tree_path) > 0:
                prev_state = self.tree_path[-1][0]
                action_leaves = [leaf for leaf in action_leaves if not np.array_equal(leaf.next_state, prev_state)]

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
        # print('node ep:', node.episode, ', diff in UCTs:', np.max(UCTs) - np.min(UCTs))
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
    def search(self, n_sims=1000, n_futures=0, n_iter=100, lazy=False, reuse_samples=False, correct_prior = True):

        ## root sampling of new posterior
        # self.GP.root_sample(certainty_equivalent=True)

        ## root sampling of new kernel
        # K_inf = self.GP.sample_k()

        ## if samples not provided, generate new set of root samples
        if not reuse_samples:
            self.agent.root_samples(obs = self.env.obs, n_samples=n_sims, n_iter=n_iter, lazy=lazy, CE=False, correct_prior = correct_prior)


        ## debugging plot
        # plt.figure()
        # plot_r(posterior_mean_p_cost.reshape(self.N,self.N), ax = plt.subplot(), title='posterior sample')
        # plt.show()

        ## debugging Q-vals
        self.Q_tracker = []
        self.first_node_updates = []
        
        ## loop through simulations
        for t in range(n_sims):

            ## CHEATING: give agent full knowledge of grid probabilities
            # posterior_p_cost = self.env.p_costs
            
            ## root sampling of new posterior
            posterior_p_cost = self.agent.all_posterior_p_costs[t]
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

    def init_node_id(self, obs=None, init_info_state = None):
        node_id = tuple(np.append(self.actual_state, self.env.costs[self.actual_state[0], self.actual_state[1]]))
        return node_id
    
    def update_episode(self):
        self.actual_state = self.env.current
        self.actual_goal = self.env.goal
        self.actual_episode = self.env.episode

    ## tree step
    def tree_step(self, action_leaf):
        action = action_leaf.action
        next_state, cost, terminated, _, _ = self.env.step(action)
        self.tree_costs.append(cost)
        node_episode = self.actual_episode ##??
        next_node_id = tuple(np.append(next_state, cost))
        return next_state, cost, terminated, node_episode, next_node_id
    
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
    

class MonteCarloTreeSearch_2AFC(MonteCarloTreeSearch):

    def __init__(self, env, agent, tree, exploration_constant=2, discount_factor=0.99):
        super().__init__(env, agent, tree, exploration_constant, discount_factor)

    ## node_ids are defined by the informational state, i.e. the counts of low and high cost states in each cell
    def init_node_id(self, obs=None, init_info_state = None):
        # info_state = np.zeros((self.N, self.N, 2))
        if init_info_state is None:
            init_info_state = np.zeros((self.N, self.N, 2))
        for i,j,c in obs:
            i = int(i)
            j = int(j)
            cost_idx = 1 if c == self.env.high_cost else 0
            init_info_state[i,j,cost_idx] += 1
        node_id = tuple(init_info_state.flatten())
        return node_id
    
    def update_episode(self):
        self.actual_episode = self.env.episode
        # self.actual_state = self.env.current
        self.actual_state = self.env.starts[self.actual_episode].copy()
        self.actual_goal = self.env.goals[self.actual_episode].copy() ## in fact, this will probably be two goals

    ## tree step
    def tree_step(self, action_leaf):
        # print('tree step in ep ', action_leaf.episode, 'action:', action_leaf.action)
        # start_tmp = self.env.current ## will change this if multiple starts are used
        start_tmp = self.env.starts[self.actual_episode][action_leaf.action].copy()
        goal_tmp = self.env.goals[self.actual_episode][action_leaf.action].copy()
        self.env.set_state(start_tmp)
        self.env.set_goal(goal_tmp)
        path_id = action_leaf.action
        assert self.env.episode == action_leaf.episode, 'episode mismatch between env and tree\n env: {} \n tree: {}\n MCTS: {}'.format(self.env.episode, action_leaf.episode, self.actual_episode)
        step_ep = action_leaf.episode
        action_sequence = self.env.path_actions[step_ep][path_id]

        ## initialise costs and observations for this path
        # simulated_obs = []
        simulated_obs = [np.append(start_tmp, self.env.predicted_costs[start_tmp[0], start_tmp[1]])] ## if the agent hasn't already observed the start state

        ## take path
        states, costs = self.env.take_path(action_sequence)
        simulated_obs += [np.append(s, c) for s, c in zip(states, costs)]
        self.tree_costs.append(np.sum(costs))
        # next_state = start_tmp ##  NEED TO CHANGE IF SGS ARE CHANGING 
        terminated = action_leaf.terminated

        ## get the next node id, i.e. the informational state after taking this path
        init_info_state = np.array(action_leaf.parent_id).reshape(self.N, self.N, 2)
        next_node_id = self.init_node_id(simulated_obs, init_info_state)

        ## since the agent has chosen a path to the goal, we need to move the environment to the next episode
        node_episode = step_ep+1
        if not terminated:
            next_state = self.env.starts[node_episode].copy()
            self.env.set_episode(node_episode)
            self.env.soft_reset()
        else:
            next_state = np.array([None, None])
            next_goal = np.array([None, None])
            self.env.soft_reset(next_state, next_goal)

        ## some checks
        assert len(self.tree_costs)<=self.env.n_episodes, 'tree costs exceed number of episodes, tree len: {}, n_eps: {}'.format(len(self.tree_costs), self.env.n_episodes)
        return next_state, costs, terminated, node_episode, next_node_id
    
    ## rollout policy
    def rollout_policy(self, action_leaf):

        ## if no action leaf because tree policy has reached a terminal node, return None
        if action_leaf is None:
            return None

        ## first need to get the starting cost r, which is essentially the cost of path choice that corresponds to the action leaf
        first_episode = action_leaf.episode
        path_id = action_leaf.action
        starting_cost = 0
        for state in self.env.path_states[first_episode][path_id]:
            # cost = self.env.get_pred_cost(state)
            cost = self.env.predicted_costs[state[0], state[1]]
            starting_cost += cost
        total_cost = starting_cost

        ## if final episode, just stop here
        if action_leaf.terminated:
            self.tree_costs.append(total_cost)
            return total_cost
        
        ## loop through remaining episodes
        depth = 0
        remaining_ro_costs = []
        ro_choices=[]
        for ep in range(first_episode+1, self.env.n_episodes):
            depth+=1

            ## GREEDY: get the total cost of the two paths and return the better one
            path_costs = []
            for path_id in range(self.n_afc):
                path_states = self.env.path_states[ep][path_id]
                ro_cost = 0
                for state in path_states:
                    # cost = self.env.get_pred_cost(state)
                    cost = self.env.predicted_costs[state[0], state[1]]
                    ro_cost += cost
                path_costs.append(ro_cost)
            # first_step_action = self.tree_path[0][1]
            # if first_step_action == 0:
            #     first_step_cost = path_costs[0]
            #     print('first step cost:', first_step_cost)
            #     print('RO ep:', ep,', costs:',path_costs)
            remaining_ro_costs.append(np.max(path_costs))
            ro_choices.append(np.argmax(path_costs))
            total_cost += np.max(path_costs) * self.discount_factor**depth

            ## RANDOM: randomly choose between the two paths
            # path_id = np.random.choice(self.n_AFC)
            # path_states = self.env.path_states[action_leaf.episode+1][path_id]
            # ro_cost = 0
            # for state in path_states:
            #     cost = self.env.get_pred_cost(state)
            #     ro_cost += cost

        self.tree_costs.append(total_cost)
        assert len(remaining_ro_costs)+first_episode+1 == self.env.n_episodes, 'remaining RO costs do not match number of episodes\n n remaining RO costs: {}, n episodes: {}'.format(len(remaining_ro_costs), self.env.n_episodes)
        return total_cost 




## parallel function for simulating many episodes within the same mountain env
# def simulate_agent(m, N, env_params=None, metric='cityblock', expt='2AFC', n_episodes=10, agents = ['GP', 'GP-MCTS', 'BAMCP','CE'], n_sims=1000,n_runs=1, correct_prior=True, n_futures=0, n_iter=10, lazy=False, exploration_constant=2, discount_factor=0.95, progress=False, offline=False):
def simulate_agent(m, env_params=None, MCTS_params=None, sampler_params=None, agents= ['BAMCP', 'CE'], progress=False, offline=False):
    print(' ') # for some reason need this to get the pbar to appear

    ## unroll param dictionaries
    # locals().update(env_params)
    # locals().update(MCTS_params)
    # locals().update(sampler_params)

    ## or, do this manually
    N = env_params['N']
    n_episodes = env_params['n_episodes']
    expt = env_params['expt']
    n_runs = env_params['n_runs']
    metric = env_params['metric']
    beta_params = env_params['beta_params']

    n_sims = MCTS_params['n_sims']
    n_futures = MCTS_params['n_futures']
    exploration_constant = MCTS_params['exploration_constant']
    discount_factor = MCTS_params['discount_factor']

    n_iter = sampler_params['n_iter']
    lazy = sampler_params['lazy']
    correct_prior = sampler_params['correct_prior']



    
    ## initiate dictionary to store the results
    sim_out = {}
    for key in data_keys:
        sim_out[key]=[]
    
    ## set seed
    seed=m
    seed=os.getpid()
    np.random.seed(seed)
    
    ## create base mountain environment
    env = make_env(N, n_episodes, expt, beta_params, metric)
    
    ## debugging plot env
    # fig, ax = plt.subplots(1, 1, figsize=(5,5))
    # plot_r(env.p_costs, ax = ax, title=m)
    # plt.show()

    ## loop through runs of the same mountain-episode set
    if progress:
        if n_runs > 1:
            pbar = tqdm(total=n_runs*n_episodes, desc='Mountain_'+str(m)+', '+str(n_runs)+' runs, '+str(n_episodes)+' episodes', position=0, leave=False, ascii=True)
    for run in range(n_runs):   

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
        # tree_reset = True ## to determine whether tree is reset at the start of each episode


        ## initialise farmer
        farmer = Farmer(N)

        ## loop through episodes (i.e. different start and goal states for the same mountain)
        if progress:
            if n_runs <= 1:
                pbar = tqdm(total=n_episodes, desc='Mountain_'+str(m)+', run '+str(run+1)+'/'+str(n_runs), position=m+1, leave=False)
        # for e in range(n_episodes):

        ## TEMP: just interested in first choice
        for e in range(1):

            ## loop through agents
            for a, ag in enumerate(agents):
                
                ### reset episode 

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
                Q_values = []
                CE_Q_values = []
                ELDs = []
                EKLs = []

                ## correct vs incorrect prior for BAMCP agent
                if ag=='BAMCP':
                    correct_prior=True
                elif ag=='BAMCP_wrong':
                    correct_prior=False
                

                ## GP-MCTS agent receives info from env
                if ag =='GP-MCTS':
                    # K_inf = env_copy.K_gen.copy()
                    # K_inf = None
                    agent = GP
                elif (ag == 'BAMCP') or (ag == 'CE') or (ag=='BAMCP w/ CE') or (ag=='CE w/ BAMCP') or (ag=='BAMCP_wrong'):
                    agent = farmer
                agent.get_env_info(env_copy)

                ## reset tree
                if ((ag == 'BAMCP') or (ag == 'BAMCP w/ CE') or (ag=='BAMCP_wrong')) & tree_resets[ag]:#& tree_reset:
                    tree = Tree(N)
                    if expt == 'free':
                        # MCTS = MonteCarloTreeSearch_Free(env=env_copy, agent=agent, tree=tree, exploration_constant=exploration_constant, discount_factor=discount_factor)
                        MCTSs[ag] = MonteCarloTreeSearch_Free(env=env_copy, agent=agent, tree=tree, exploration_constant=exploration_constant, discount_factor=discount_factor)
                    elif expt == '2AFC':
                        # MCTS = MonteCarloTreeSearch_2AFC(env=env_copy, agent=agent, tree=tree, exploration_constant=exploration_constant, discount_factor=discount_factor)
                        MCTSs[ag] = MonteCarloTreeSearch_2AFC(env=env_copy, agent=agent, tree=tree, exploration_constant=exploration_constant, discount_factor=discount_factor)
                
                ## if keeping the tree between episodes, need to update tree with new episode info
                elif ((ag == 'BAMCP') or (ag == 'BAMCP w/ CE') or (ag=='BAMCP_wrong')) & (not tree_resets[ag]): #& (not tree_reset):
                    # MCTS.update_episode()
                    MCTSs[ag].update_episode()
                    # tree_reset = True
                    tree_resets[ag] = True
                MCTS = MCTSs[ag]
                assert e == env_copy.episode, 'episode mismatch between simulation and env\n simulation: {} \n env: {}'.format(e, MCTS.env.episode)
                assert e == MCTS.actual_episode, 'episode mismatch between env and MCTS\n env: {} \n MCTS: {}'.format(e, MCTS.env.episode)

            
                ## run episode until goal is reached
                end_episode = False
                terminated=False
                early_terminate = False
                reuse_samples = False
                steps = 0
                if expt=='free':
                    max_steps = len(env_copy.o_trajs[e])*1.75
                elif expt=='2AFC':
                    max_steps = 100 ## just in case
                max_search_attempts = 3

                while not end_episode:

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

                        search_attempts = 0 # could do nan

                    ## certainty-equivalent
                    elif (ag == 'CE') or (ag == 'CE w/ BAMCP'):
                        env_copy.set_sim(False)

                        ## get posterior mean grid
                        agent.root_samples(obs=env_copy.obs, n_samples=n_sims, n_iter=n_iter, lazy=lazy,CE=True, correct_prior = correct_prior)
                        env_copy.receive_predictions(agent.posterior_mean_p_cost)


                        ## plot for debugging?
                        # _, axs = plt.subplots(1, 3, figsize=(10,5))
                        # plot_r(env_copy.p_costs.reshape(N,N), ax=axs[0], title = 'costs')
                        # plot_traj([env_copy.o_trajs[e], env_copy.a_traj], ax=axs[0])
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
                            
                        elif expt == '2AFC':

                            ## get the cost of each path under the posterior mean
                            path_costs = []
                            for path_id in range(env_copy.n_afc):
                                path_states = env_copy.path_states[e][path_id]
                                path_cost = 0
                                for state in path_states:
                                    # path_cost += env_copy.get_pred_cost(state) ## i.e. sample binary costs from the posterior pqs
                                    path_cost += agent.posterior_mean_p_cost[state[0], state[1]]*env_copy.low_cost + (1-agent.posterior_mean_p_cost[state[0], state[1]])*env_copy.high_cost ## or, use expected costs
                                path_costs.append(path_cost)

                            ## choose the path with the lowest total cost
                            max_cost = np.max(path_costs)
                            action = argm(path_costs, max_cost)
                            actions.append(action)
                            action_sequence = env_copy.path_actions[e][action]
                            costs = []
                            for ac in action_sequence:
                                current, cost, terminated, _, _ = env_copy.step(ac)
                                costs.append(cost)
                            path_cost = np.sum(costs)
                            block_terminated = e == (n_episodes-1)
                        steps += 1
                        search_attempts = 0 # could do nan
                        leaf_visits = []

                        ## update observations
                        agent.get_env_info(env_copy)




                    ## bamcp
                    elif (ag == 'BAMCP') or (ag == 'BAMCP w/ CE') or (ag=='BAMCP_wrong'):
                        env_copy.set_sim(True)
                        MCTS.actual_state = current
                        
                        ## init MCTS (if resetting the tree for each move, init here. otherwise, this should be outside the episode loop)
                        # tree = Tree(N)
                        # MCTS = MonteCarloTreeSearch(env=env_copy, agent=agent, tree=tree, exploration_constant=exploration_constant, discount_factor=discount_factor)
                        assert MCTS.env.sim == True, 'env not in sim mode'
                    
                        ## if online planning (i.e. replan after each step)
                        if not offline:

                            ## search
                            n_sims_tmp = n_sims
                            action, MCTS_Q = MCTS.search(n_sims_tmp, n_futures, n_iter=n_iter, lazy=lazy, reuse_samples=reuse_samples, correct_prior=correct_prior)

                            ## reduce number of sims if near to the goal (A BETTER IDEA WLD BE TO REDUCE THE DISTANCE IF THE TREE HAS REACHED THE GOAL)
                            # dist_to_goal = np.max(cdist([current, goal], [current, goal], metric='cityblock')) 
                            # if dist_to_goal > (N/2):
                            #     n_sims_tmp = n_sims
                            #     action, MCTS_Q = MCTS.search(n_sims_tmp, n_futures, n_iter=n_iter, lazy=lazy, progress=progress, reuse_samples=reuse_samples)
                            # elif (dist_to_goal <= (N/2)) & (dist_to_goal > (N/4)):
                            #     # n_sims_tmp = int(n_sims/2)
                            #     n_sims_tmp = n_sims
                            #     action, MCTS_Q = MCTS.search(n_sims_tmp, n_futures, n_iter=n_iter, lazy=lazy, progress=progress, reuse_samples=reuse_samples)
                            # else:
                            #     # n_sims_tmp = int(n_sims/4)
                            #     n_sims_tmp = n_sims
                            #     action, MCTS_Q = MCTS.search(n_sims_tmp, n_futures,n_iter=n_iter, lazy=lazy,  progress=progress, reuse_samples=reuse_samples)
                            actions.append(action)
                            Q_values.append(MCTS_Q)
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
                            all_posterior_p_costs_tmp = np.zeros((n_sims_tmp, N,N))
                            for s in range(n_sims_tmp):
                                all_posterior_p_costs_tmp[s] = np.outer(all_posterior_ps_tmp[s], all_posterior_qs_tmp[s])
                            posterior_mean_p_cost_tmp = np.mean(all_posterior_p_costs_tmp, axis=0)

                            ## get CE's action
                            if expt=='free':
                                agent.dp(posterior_mean_p_cost_tmp, expected_cost=True)
                                action_CE = agent.optimal_policy(current, agent.Q_inf)
                                CE_actions.append(action_CE)
                            elif expt=='2AFC':
                                path_costs = []
                                for path_id in range(env_copy.n_afc):
                                    path_states = env_copy.path_states[e][path_id]
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
                            # # plot_traj([env_copy.o_trajs[e], env_copy.a_traj], ax=axs[0])
                            # plot_traj([env_copy.o_trajs[e], a_traj], ax=axs[0])
                            # plot_r(agent.posterior_mean_p_cost, ax=axs[1], title = 'average posterior p cost')
                            # plot_action_tree(agent.Q_inf, current, goal, ax=axs[2], title = 'CE_DP_inf')
                            # plt.show()
                            # MCTS_Q_labelled = {env_copy.action_labels[k]:v for k,v in enumerate(MCTS_Q)}
                            # if action != action_CE:
                            #     print('MCTS Q:', MCTS_Q_labelled)
                            #     print('n_visits of action leaves:',{env_copy.action_labels[k]:v.n_action_visits for k,v in MCTS.tree.root.action_leaves.items()})

                            ## take action
                            env_copy.set_sim(False)
                            if expt=='free':
                                current, cost, terminated, _, _ = env_copy.step(action)
                                next_node_id = np.append(current,cost)
                                steps += 1
                            elif expt=='2AFC':
                                action_sequence = env_copy.path_actions[e][action]
                                states, costs = env_copy.take_path(action_sequence)
                                current = states[-1]
                                path_cost = np.sum(costs)
                                terminated = True ## trivially true in 2AFC
                                block_terminated = e == (n_episodes-1)

                                ## update next node id
                                next_node_id = MCTS.init_node_id(env_copy.obs.copy(), None)

                            search_attempts = 0 # could do nan here


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
                                
                                ## no need to prune at the end of the episode in free-choice expt, since the tree resets at this point
                                if not terminated:
                                    MCTS.tree.prune(action, next_node_id)
                                    assert np.array_equal(MCTS.tree.root.state[:2], current), 'error in root update\n env is in: {} but tree is in: {}\n should have taken action {}'.format(current, MCTS.tree.root.state, action)

                            elif expt=='2AFC':

                                ## pruning not always successful due to high branching factor, in which case reset the tree
                                if not block_terminated:

                                    if next_node_id in MCTS.tree.root.action_leaves[action].children:
                                        MCTS.tree.prune(action, next_node_id)
                                        assert np.array_equal(MCTS.tree.root.state[2:], costs), 'error in root update\n env is in: {} but tree is in: {}\n should have taken action {}'.format(current, MCTS.tree.root.state, action) 
                                        # tree_reset = False
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
                    
                    ### log determinant of covariance matrix
                    # if (ag=='BAMCP') or (ag=='BAMCP w/ CE'):

                    #     ## get prior p and q samples
                    #     prior_p_samples = agent.all_posterior_ps
                    #     prior_q_samples = agent.all_posterior_qs
                    #     prior_samples = np.vstack([prior_p_samples.T, prior_q_samples.T])

                    #     ## log det of prior covariance matrix 
                    #     # prior_cov = np.cov(prior_samples)
                    #     # prior_LD = np.linalg.slogdet(prior_cov)[1]
                    #     # assert prior_cov.shape[0] == N*2, 'covariance matrix is wrong shape'
                        
                        
                    #     ## order the outcomes (counterfactual, then actual. This is to allow reuse of the posterior samples associated with the actual outcome on the next timestep)
                    #     actual_outcome = env_copy.obs.copy()[-1, -1]
                    #     if actual_outcome == env_copy.low_cost:
                    #         ordered_outcomes = [env_copy.high_cost, env_copy.low_cost]
                    #     else:
                    #         ordered_outcomes = [env_copy.low_cost, env_copy.high_cost]
                    #     posterior_LDs = []
                    #     KLs = []

                    #     ## posterior samples under each of the possible outcomes of the action that was just taken
                    #     for o, outcome in enumerate(ordered_outcomes):
                    #         sim_obs = env_copy.obs.copy()
                    #         sim_obs[-1, -1] = outcome
                    #         agent.root_samples(sim_obs, n_samples=n_sims, n_iter=n_iter, lazy=lazy, CE=False)
                    #         posterior_samples = np.vstack([np.array(agent.all_posterior_ps).T, np.array(agent.all_posterior_qs).T])
                            
                    #         ## posterior log det
                    #         # posterior_cov = np.cov(posterior_samples)
                    #         # assert posterior_cov.shape == prior_cov.shape, 'prior and posterior covariance matrices do not match: {} vs {}'.format(posterior_cov.shape, prior_cov.shape)
                    #         # LD = np.linalg.slogdet(posterior_cov)[1]
                    #         # posterior_LDs.append(LD)

                    #         ## or, calculate the KL divergence between two multivariate gaussians
                    #         KL = KL_divergence(prior_samples, posterior_samples)
                    #         KLs.append(KL)


                    #     ## expected log det, i.e. the difference between the prior and the expected posterior log dets, weighted by the probability of each outcome
                    #     # p_low = np.mean(prior_p_samples * prior_q_samples)
                    #     # p_high = 1 - p_low
                    #     # if actual_outcome == env_copy.low_cost:
                    #     #     expected_LD = p_low * (posterior_LDs[1] - prior_LD) + p_high * (posterior_LDs[0] - prior_LD)
                    #     # else:
                    #     #     expected_LD = p_low * (posterior_LDs[0] - prior_LD) + p_high * (posterior_LDs[1] - prior_LD)
                    #     # ELDs.append(expected_LD)

                    #     ## expected KL divergence
                    #     p_low = np.mean(prior_p_samples * prior_q_samples)
                    #     p_high = 1 - p_low
                    #     if actual_outcome == env_copy.low_cost:
                    #         expected_KL = p_low * (KLs[1]) + p_high * (KLs[0])
                    #     else:
                    #         expected_KL = p_low * (KLs[0]) + p_high * (KLs[1])
                    #     EKLs.append(expected_KL)

                        ## debugging
                        # print(ag)
                        # if actual_outcome == env_copy.low_cost:
                        #     print('posterior KLs: low = ',KLs[1], ', high = ',KLs[0], ', probs: ',p_low, p_high)
                        # else:
                        #     print('posterior KLs: low = ',KLs[0], ', high = ',KLs[1], ', probs: ',p_low, p_high)
                        # print('expected KL: ',expected_KL)
                        # print()

                        ## reorder the posterior LDs to match low and high cost outcomes (i.e. 0th element is the low cost outcome)
                        # if ordered_outcomes[1] == env_copy.low_cost:
                        #     posterior_LDs = [posterior_LDs[1], posterior_LDs[0]]
                        # expected_LD2 = p_low * (posterior_LDs[0] - prior_LD) + p_high * (posterior_LDs[1] - prior_LD)
                        # assert expected_LD==expected_LD2, 'expected LDs when actual outcome is {} do not match: {} vs {}'.format(actual_outcome, expected_LD, expected_LD2)

                        # CE_deviation = action==action_CE
                        # print('action {} deviate from CE'.format(['did','did not'][CE_deviation]))
                        # print('prior LD: ',prior_LD)
                        # if actual_outcome == env_copy.low_cost:
                        #     print('posterior LDs: low = ',posterior_LDs[1], ', high = ',posterior_LDs[0], ', probs: ',p_low, p_high)
                        # else:
                        #     print('posterior LDs: low = ',posterior_LDs[0], ', high = ',posterior_LDs[1], ', probs: ',p_low, p_high)
                        # print('expected change in LD: ',expected_LD)
                        # print()

                        ## reuse samples associated with the actual outcome on the next timestep
                        # reuse_samples = True



                    ### KL divergence
                    # if (ag=='BAMCP') or (ag=='BAMCP w/ CE'):

                    #     ## get the relevant prior samples, i.e. the p and q samples for the state that has just been reached
                    #     i, j = current
                    #     prior_p_samples = MCTS.all_posterior_p[:,i]
                    #     prior_q_samples = MCTS.all_posterior_q[:,j]

                    #     ##clipping
                    #     prior_joint = np.vstack([prior_p_samples, prior_q_samples])

                    #     ## debugging: plot kde of prior p and q samples
                    #     # fig, axs = plt.subplots(1, 3, figsize=(15,5))
                    #     # sns.kdeplot(prior_p_samples, ax=axs[0], label='prior p')
                    #     # sns.kdeplot(prior_q_samples, ax=axs[0], label='prior q')
                    #     # axs[0].set_title('prior')
                    #     # axs[0].legend()

                    #     ### simulate a set of posterior (root) samples under each of the possible outcomes of the action that was just taken
                    #     kl_divs = []
                    #     for o, outcome in enumerate([env_copy.low_cost, env_copy.high_cost]):
                    #         farmer_copy = copy.deepcopy(agent)
                    #         posterior_p_samples = []
                    #         posterior_q_samples = []
                    #         sim_obs = env_copy.obs.copy()
                    #         sim_obs[-1, -1] = outcome
                    #         state_to_update = np.array([i, j, outcome])
                    #         for t in range(n_sims):
                    #             farmer_copy.root_sample(sim_obs, lazy=True, CE=False, state=state_to_update)
                    #             posterior_p_samples.append(farmer_copy.posterior_p[i])
                    #             posterior_q_samples.append(farmer_copy.posterior_q[j])
                    #         posterior_joint = np.vstack([posterior_p_samples, posterior_q_samples])

                    #         # KL = KL_divergence(prior_joint, posterior_joint)

                    #         ## calculate KL divergence
                    #         kde_prior = gaussian_kde(prior_joint)
                    #         kde_posterior = gaussian_kde(posterior_joint)
                    #         eval_points = posterior_joint
                    #         p_posterior = kde_posterior.evaluate(eval_points)
                    #         p_prior = kde_prior.evaluate(eval_points)
                    #         kl_div = np.mean(np.log(p_posterior / p_prior))
                    #         kl_divs.append(kl_div)

                    #         ## debugging: plot kde of prior p and q samples
                    #         # sns.kdeplot(posterior_p_samples, ax=axs[o+1], label='posterior p')
                    #         # sns.kdeplot(posterior_q_samples, ax=axs[o+1], label='posterior q')
                    #         # axs[o+1].legend()
                    #         # axs[o+1].set_title('posterior for outcome {}'.format(outcome))
                    #     # plt.show()
                        
                    #     # ## compute the expected KL divergence
                    #     p_low = np.mean(prior_p_samples * prior_q_samples) 
                    #     # p_low2 = np.mean(prior_p_samples) * np.mean(prior_q_samples)
                    #     p_high = 1 - p_low
                    #     expected_kl_div = p_low * kl_divs[0] + p_high * kl_divs[1]
                    #     EKLs.append(expected_kl_div)



                    
                    ## prevent endless episode 
                    if steps >= max_steps:
                        early_terminate = True

                    if early_terminate:
                        print('mountain ',m,': episode ',e,' terminated for agent ',ag,' after ',steps,' steps')
                        # raise ValueError('mountain ',m,': episode ',e,' terminated for agent ',ag,' after ',steps,' steps')

                        ## or just skip to the next episode
                        sim_out['agent'].append(agent)
                        sim_out['run'].append(run)
                        sim_out['episode'].append(e)
                        sim_out['mountain'].append(m)
                        sim_out['start'].append(start)
                        sim_out['goal'].append(goal)
                        sim_out['path_A'].append(env_copy.path_states[e][0])
                        sim_out['path_B'].append(env_copy.path_states[e][1])
                        sim_out['actions'].append(actions)
                        sim_out['Q_values'].append(Q_values)
                        sim_out['leaf_visits'].append(leaf_visits)
                        sim_out['CE_actions'].append(CE_actions)
                        sim_out['CE_Q_values'].append(CE_Q_values)
                        sim_out['optimal_actions'].append(env_copy.o_traj_actions[e])
                        sim_out['costs'].append(np.nan)
                        sim_out['optimal_costs'].append(env_copy.o_traj_costs[e])
                        sim_out['total_cost'].append(np.nan)
                        sim_out['total_optimal_cost'].append(env_copy.o_traj_total_costs[e])
                        sim_out['action_score'].append(np.nan)
                        sim_out['cost_ratio'].append(np.nan)
                        sim_out['n_steps'].append(steps)
                        sim_out['actual_trajectory'].append(env_copy.a_traj)
                        sim_out['optimal_trajectory'].append(env_copy.o_trajs[e])
                        sim_out['observations'].append(env_copy.obs)
                        sim_out['search_attempts'].append(search_attempts)
                        # sim_out['action_tree'].append(MCTS.tree.action_tree())
                        sim_out['action_tree'].append(np.nan)
                        sim_out['expected_LD'].append(ELDs)
                        sim_out['expected_KL'].append(EKLs)

                        ## discounts
                        sim_out['discounted_costs'].append(np.nan)
                        sim_out['total_discounted_cost'].append(np.nan)
                        discounts = [discount_factor**d for d in range(len(env_copy.o_trajs[e]))] ## NEED TO FIX THIS BUT DON'T HAVE TIME
                        discounted_costs = [c*d for c,d in zip(env_copy.o_traj_costs[e], discounts)]
                        sim_out['discounted_optimal_costs'].append(discounted_costs)
                        sim_out['total_discounted_optimal_cost'].append(np.sum(discounted_costs))
                        
                        ## GP-specific
                        # sim_out['true_k'].append(true_k)
                        # sim_out['RPE'].append(np.nan)
                        # sim_out['posterior_mean'].append(np.nan)
                        # sim_out['theta_MLE'].append(best_theta)

                        end_episode = True

                        ## stop the sim here!
                        return sim_out, env_copy.p_costs

                    ## save data and end the episode
                    elif terminated:
                        sim_out['agent'].append(ag)
                        sim_out['run'].append(run)
                        sim_out['episode'].append(e)
                        sim_out['mountain'].append(m)
                        sim_out['start'].append(start)
                        sim_out['goal'].append(goal)
                        sim_out['path_A'].append(env_copy.path_states[e][0])
                        sim_out['path_B'].append(env_copy.path_states[e][1])
                        sim_out['actions'].append(actions)
                        sim_out['Q_values'].append(Q_values)
                        sim_out['leaf_visits'].append(leaf_visits)
                        sim_out['CE_actions'].append(CE_actions)
                        sim_out['CE_Q_values'].append(CE_Q_values)
                        sim_out['optimal_actions'].append(env_copy.o_traj_actions[e])
                        # if np.round(env_copy.optimal_cost,4) < np.round(env_copy.accrued_cost,4):
                        #     print(env_copy.optimal_cost, env_copy.accrued_cost)
                        # assert np.round(env_copy.optimal_cost,4) >= np.round(env_copy.accrued_cost,4), 'accrued cost higher than optimal cost'
                        # sim_out['action_score'].append(env_copy.optimal_cost/env_copy.accrued_cost)
                        sim_out['action_score'].append(env_copy.action_score)
                        sim_out['cost_ratio'].append(env_copy.cost_ratio)
                        sim_out['n_steps'].append(steps)
                        sim_out['actual_trajectory'].append(env_copy.a_traj)
                        sim_out['optimal_trajectory'].append(env_copy.o_trajs[e])
                        sim_out['observations'].append(env_copy.obs)
                        sim_out['search_attempts'].append(search_attempts)
                        # sim_out['action_tree'].append(MCTS.tree.action_tree())
                        sim_out['action_tree'].append(np.nan)
                        sim_out['expected_LD'].append(ELDs)
                        sim_out['expected_KL'].append(EKLs)

                        ### costs

                        ## actual costs
                        sim_out['costs'].append(env_copy.a_traj_costs)
                        sim_out['optimal_costs'].append(env_copy.o_traj_costs[e])
                    
                        # INC START AND END COSTS
                        # sim_out['total_cost'].append(env_copy.a_traj_total_cost) 
                        # sim_out['total_optimal_cost'].append(env_copy.o_traj_total_costs[e])

                        # EXC START AND END COSTS
                        sim_out['total_cost'].append(np.sum(env_copy.a_traj_costs[1:-1]))
                        sim_out['total_optimal_cost'].append(np.sum(env_copy.o_traj_costs[e][1:-1]))



                        ## calculate discounted actual and optimal costs

                        ## INC START AND END COSTS
                        # discounts = [discount_factor**d for d in range(len(env_copy.a_traj_costs))] 
                        # discounted_costs = [c*d for c,d in zip(env_copy.a_traj_costs, discounts)]
                        # sim_out['discounted_costs'].append(discounted_costs)
                        # sim_out['total_discounted_cost'].append(np.sum(discounted_costs))
                        # discounts = [discount_factor**d for d in range(len(env_copy.o_trajs[e]))]
                        # discounted_costs = [c*d for c,d in zip(env_copy.o_traj_costs[e], discounts)]
                        # sim_out['discounted_optimal_costs'].append(discounted_costs)
                        # sim_out['total_discounted_optimal_cost'].append(np.sum(discounted_costs))

                        ## EXC START AND END COSTS
                        discounts = [discount_factor**d for d in range(len(env_copy.a_traj_costs)-2)]
                        discounted_costs = [c*d for c,d in zip(env_copy.a_traj_costs[1:-1], discounts)]
                        sim_out['discounted_costs'].append(discounted_costs)
                        sim_out['total_discounted_cost'].append(np.sum(discounted_costs))
                        discounts = [discount_factor**d for d in range(len(env_copy.o_trajs[e])-2)]
                        discounted_costs = [c*d for c,d in zip(env_copy.o_traj_costs[e][1:-1], discounts)]
                        sim_out['discounted_optimal_costs'].append(discounted_costs)
                        sim_out['total_discounted_optimal_cost'].append(np.sum(discounted_costs))

                        
                        ## GP-specific
                        # sim_out['true_k'].append(true_k)
                        # sim_out['RPE'].append(np.mean(np.abs(GP.posterior_mean.reshape(N,N) - env_copy.costs)))
                        # sim_out['posterior_mean'].append(GP.posterior_mean)
                        # sim_out['theta_MLE'].append(best_theta)
                        
                        ## update the agent env
                        # agent_envs[a] = copy.deepcopy(env_copy)
                        agent_envs[ag] = env_copy

                        end_episode = True
            if progress:
                pbar.update(1)
        if progress & (n_runs <= 1):
            pbar.close()
    if progress & (n_runs > 1):
        pbar.close()
                    

    return sim_out,env_copy
    # return sim_out, _