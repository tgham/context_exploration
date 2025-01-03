import random
from math import sqrt, log
from utils import Node, Action_Node, Tree, make_env, argm, data_keys
import copy
import numpy as np
from tqdm.auto import tqdm
import os
from scipy.spatial.distance import cdist

from plotter import *
from agents import GPAgent, Farmer


class MonteCarloTreeSearch():

    def __init__(self, env, agent, tree, exploration_constant=2, discount_factor=0.95):
        self.env = env
        self.agent = agent
        self.tree = tree
        self.action_space = self.env.action_space.n
        self.N = self.env.N
        self.exploration_constant = exploration_constant
        self.discount_factor = discount_factor

        ## get initial state and goal 
        state = self.env.current

        ## (AND THE STARTING COST?)
        starting_cost = self.env.costs[state[0], state[1]]

        ## add state node to the tree
        self.tree.add_state_node(state=state, cost=starting_cost, terminated=False, action_space = self.action_space, parent=None)


    ## expand the action space of a node
    def expand(self, node):

        ## get a copy of the current state, so that the environment can be reset to this state after simulating the action
        actual_state = self.env.current

        ## create copy of env and set state
        # env_copy = copy.deepcopy(self.env)
        # env_copy.set_state(node.state)
        # assert env_copy.sim, 'env is not in sim mode'
        assert self.env.sim, 'env is not in sim mode'

        ## take action and get new state
        action = node.untried_action()
        observation, cost, terminated, truncated, info = self.env.step(action)
        next_state = observation['agent']

        ## update info for s-a leaf - i.e. the state-action pair
        node.action_leaves[action] = Action_Node(prev_state = node.state, action=action, next_state = next_state, terminated=terminated)
        # node.action_leaves[action].performance = cost
        node.action_leaves[action].performance = 0

        ## reset the environment to the actual state
        self.env.set_state(actual_state)

        return node.action_leaves[action]



    ## take an action according to the tree policy, i.e. take the best UCT child and see where it takes you
    def tree_policy(self):
        
        ## create copy of env and set state
        # env_copy = copy.deepcopy(self.env)
        # env_copy.set_sim(True)

        ## get the agent's current location
        actual_state = self.env.current
        assert self.env.sim, 'env is not in sim mode'

        ## initialise the tree
        node = self.tree.root
        t = 0
        self.tree_costs = []
        # self.tree_cost.append(node.cost)
        assert np.array_equal(node.state[:2], actual_state), 'mismatch between node and env state\n node: {} \n env: {}'.format(node, actual_state)

        ## create a record of the nodes/leaves visited in the tree
        self.tree_path = []
        self.node_id_path = []
        
        ## loop until you reach a leaf node or terminal state
        while not node.terminated:
            t+=1

            ## expansion step
            if self.tree.is_expandable(node):
                action_leaf = self.expand(node)
                self.tree_path.append(tuple([node.state, action_leaf.action]))
                self.node_id_path.append(node.node_id)

                ### tree cost here???
                # tree_cost += expanded_node.cost
                # self.tree_cost.append(expanded_node.cost) ## maybe don't need to do this??? because it's included in the rollout too?
                # self.tree_costs.append(action_leaf.next_cost)

                ## update counts already?
                action_leaf.n_action_visits += 1
                node.n_state_visits += 1

                ## revert env
                self.env.set_state(actual_state)

                ## save tree obs for subsequent rollouts
                self.tree_obs = self.env.obs_tmp.copy()
                self.env.flush_obs()
                # if len(self.tree_obs) != len(self.env.obs)+len(self.tree_path):
                #     print('env obs:',self.env.obs)
                #     print('tree obs:',self.tree_obs)
                #     print('tree path:',self.tree_path)
                # assert len(self.tree_obs) == len(self.env.obs)+len(self.tree_path), 'tree obs and path lengths do not match\n tree obs: {}, env.obs: {}, tree path: {}'.format(len(self.tree_obs), len(self.env.obs),len(self.tree_path))

                return action_leaf
                
            ## selection step
            else:

                ## (some debugging vars)
                state_tmp = node.state[:2]
                env_state_tmp = self.env.get_obs()['agent']

                ## get the best child
                action_leaf = self.best_child(node)
                self.tree_path.append(tuple([node.state, action_leaf.action]))
                self.node_id_path.append(node.node_id)

                ## move in env
                observation, cost, terminated, _, _ = self.env.step(action_leaf.action)
                next_state = observation['agent']
                self.tree_costs.append(cost)

                ## create next state node (if it doesn't already exist)
                # history = [tuple(o) for o in self.env.obs_tmp]
                # node = self.tree.add_state_node(next_state, cost, history, terminated, action_space = self.action_space, parent=action_leaf)

                ## or, see if this node is already a child of the action_leaf, otherwise add this child
                if str(np.append(next_state,cost)) in action_leaf.children:
                    node = action_leaf.children[str(np.append(next_state,cost))]
                else:
                    node = self.tree.add_state_node(next_state, cost, terminated, action_space = self.action_space, parent=action_leaf)
                    # action_leaf.children[str(np.append(next_state,cost))] = node
                assert np.array_equal(node.state[:2], next_state), 'error in tree policy step {}\n started in {}\n supposed to take action {} to {}\n ended up moving from {} to {}'.format(t, state_tmp, action_leaf.action, node.state[:2], env_state_tmp, action_leaf.next_state)

                ## update counts already?
                action_leaf.n_action_visits += 1
                node.n_state_visits += 1

            ## if the agent has reached a state that has already been visited, initiate a rollout from there
            # visited_states = np.array([self.tree_path[i][0] for i in range(len(self.tree_path))])
            # if any(np.array_equal(next_state, state) for state in visited_states):
            #     print('tree policy stuck')
            #     break

            # if t>self.N**2:
            #     print('tree policy stuck')
            #     break

        ## revert env
        self.env.set_state(actual_state)

        ## save tree obs for subsequent rollouts
        self.tree_obs = self.env.obs_tmp.copy()
        self.env.flush_obs()
        # assert len(self.tree_obs) == len(self.env.obs)+len(self.tree_path), 'tree obs and path lengths do not match\n tree obs: {}, env.obs: {}, tree path: {}'.format(len(self.tree_obs), len(self.env.obs),len(self.tree_path))

        return action_leaf


    def rollout_policy(self, action_leaf, real_rollout = True, n_futures=None):

        ## init
        total_cost = 0
        max_depth = 100
        # depth = len(self.tree_path)

        ## get the agent's current location and goal
        actual_state = self.env.current
        actual_goal = self.env.goal
        assert self.env.sim, 'env is not in sim mode'
        
        ## set the state from which the rollout is initiated
        # self.env.set_state(action_leaf.next_state)
        # observation = self.env.get_obs()
        
        ## or, make a copy
        env_copy = copy.deepcopy(self.env)
        agent_copy = copy.deepcopy(self.agent) ## this is done so that the GP doesn't change its state. could of course just feed env back into it at end of rollout

        ## standard rollout if this is the first S-G pair
        if real_rollout:
            start = action_leaf.next_state
            env_copy.set_state(action_leaf.next_state)
            
            ## begin with cost of current state
            # starting_cost = action_leaf.
            starting_cost = env_copy.get_pred_cost(start)
            total_cost += starting_cost
            observation = env_copy.get_obs()

            ## reset the depth counter
            self.depth = 0

            ## rolling out from goal location, can just end here
            if action_leaf.terminated:

                ## revert env
                # self.env.set_state(actual_state)

                return total_cost
            

            ## rollout until trial is terminated 
            while True:

                ## prevent infinite rollout
                self.depth += 1
                # if self.depth > max_depth:
                #     # print('exceeded max rolls in {} rollout'.format(['imagined', 'real'][real_rollout]))

                #     # print(env_copy.V_inf)
                #     fig, axs = plt.subplots(1, 3, figsize=(15,5))
                #     # plot_r(env_copy.posterior_mean.reshape(self.N,self.N), ax=axs[0], title = 'posterior mean')
                #     sns.heatmap(agent_copy.posterior_sample.reshape(self.N,self.N), ax=axs[0], cbar=False, annot=True, fmt='.2f')
                #     axs[0].set_title('posterior sample')
                #     plot_action_tree(agent_copy.Q_inf, start, actual_goal, ax=axs[1], title = 'DP_inf')
                #     plot_r(agent_copy.V_inf, ax=axs[2], title = 'V')

                #     ## raise error
                #     raise ValueError('exceeded max rolls in {} rollout, start: {}, goal: {}'.format(['imagined', 'real'][real_rollout], start, goal))
                current = observation['agent']


                ## or, greedy
                # current = observation['agent']
                # action = agent_copy.greedy_policy(current, env_copy.goal, eps = 0.0)

                ## or, optimised rollout 
                action = agent_copy.optimal_policy(current, agent_copy.Q_inf)

                ## take action
                observation, cost, terminated, _, _ = env_copy.step(action)

                ## increment cost
                total_cost += cost * self.discount_factor**self.depth

                ## if terminated return the cost
                if terminated:
                    return total_cost

        ## or, rollout for some new imagined S-G pair
        else:

            ## inherit obs from tree so far and sample new posterior
            # agent_copy.get_env_info(env_copy)
            # GP_copy.root_sample(self.tree_obs, GP_copy.K_inf)
            # agent_copy.root_sample(self.tree_obs)

            ## loop through new start-end pairs
            future_total_costs = []
            for f in range(n_futures):

                # depth = 0
                depth_tmp = self.depth
                total_cost = 0

                ## imagine new start and goal locations
                seed = random.randint(0, 1000)
                start, goal = env_copy.sample_SG()
                _,_ = env_copy.reset(seed=seed, start_goal=[start, goal])
                # start = env_copy.current
                # goal = env_copy.goal
                observation = env_copy.get_obs()
                agent_copy.get_env_info(env_copy)

                ## get DP Q-values for the new start and goal under the current posterior sample
                # agent_copy.dp()

                ## rollout until trial is terminated 
                while True:

                    ## prevent infinite rollout
                    depth_tmp += 1
                    if depth_tmp > max_depth:
                        # print('exceeded max rolls in {} rollout'.format(['imagined', 'real'][real_rollout]))

                        # print(env_copy.V_inf)
                        fig, axs = plt.subplots(1, 3, figsize=(15,5))
                        # plot_r(env_copy.posterior_mean.reshape(self.N,self.N), ax=axs[0], title = 'posterior mean')
                        sns.heatmap(agent_copy.posterior_sample.reshape(self.N,self.N), ax=axs[0], cbar=False, annot=True, fmt='.2f')
                        plot_action_tree(agent_copy.Q_inf, start, goal, ax=axs[1], title = 'DP_inf')
                        plot_r(agent_copy.V_inf, ax=axs[2], title = 'V')

                        ## raise error
                        raise ValueError('exceeded max rolls in {} rollout, start: {}, goal: {}'.format(['imagined', 'real'][real_rollout], start, goal))
                    current = observation['agent']


                    ## optimised rollout 
                    action = agent_copy.optimal_policy(current, agent_copy.Q_inf)

                    # ## or greedy wrt/ distance
                    # action = agent_copy.greedy_policy(current, goal, eps = 0.0)

                    ## take action
                    observation, cost, terminated, _, _ = env_copy.step(action)

                    ## increment cost
                    total_cost += cost * self.discount_factor**depth_tmp

                    ## if terminated, append cost
                    if terminated:
                        future_total_costs.append(total_cost)
                        break

            ## average the future costs
            total_cost = np.mean(future_total_costs)
            return total_cost
            
    ## optimised rollout
    def optimised_rollout_policy(self, action_leaf, real_rollout = True):

        ## get the max Q value of the state that you have just reached from this action leaf
        next_state = action_leaf.next_state
        Qs = self.agent.Q_inf[next_state[0], next_state[1]]
        max_Q = np.nanmax(Qs)

        return max_Q



    ## calculate E-E value
    def compute_UCT(self, node, action_leaf): 
        # exploitation_term = child.total_simulation_cost / child.n_visits
        # exploration_term = exploration_constant * sqrt(2 * log(parent.n_visits) / child.n_visits)
        exploitation_term = action_leaf.performance 
        assert action_leaf.n_action_visits > 0 or action_leaf.terminated, 'action leaf has not been visited: {}'.format(action_leaf)
        exploration_term = self.exploration_constant * sqrt(log(node.n_state_visits) / action_leaf.n_action_visits)
        return exploitation_term + exploration_term

    
    ## argmax based on UCT values? 
    def best_child(self, node):
    
        ## get action children
        action_leaves = [node.action_leaves[a] for a in node.action_leaves.keys()]

        ## remove action that keeps you in your current state
        action_leaves = [leaf for leaf in action_leaves if not np.array_equal(leaf.next_state, leaf.prev_state)]
        
        ## remove action that takes you back to previous state in the tree
        if len(self.tree_path) > 0:
            prev_state = self.tree_path[-1][0]
            action_leaves = [leaf for leaf in action_leaves if not np.array_equal(leaf.next_state, prev_state)]

    
        
        ## or, remove any actions that take you back to a state that has already been visited in the tree
        # if len(self.tree_path) > 0:
        #     # visited_states = {tuple(state) for state, _ in self.tree_path}
        #     # action_leaves = [leaf for leaf in action_leaves if tuple(leaf.next_state) not in visited_states]

        #     visited_states = np.array([self.tree_path[i][0] for i in range(len(self.tree_path))])
        #     action_leaves_tmp = []
        #     for leaf in action_leaves:
        #         if all(not np.array_equal(leaf.next_state, state) for state in visited_states):
        #             action_leaves_tmp.append(leaf)
        #     action_leaves = action_leaves_tmp
            
        #     ## check if the agent has got stuck
        #     if len(action_leaves_tmp) == 0:
        #         print(visited_states, node)
        

        ## calculate UCT for each action leaf
        UCTs = [self.compute_UCT(node, leaf) for leaf in action_leaves]
        max_UCT = np.max(UCTs)
        max_idx = argm(UCTs, max_UCT)
        best_child = action_leaves[max_idx]

        return best_child
    
    ## backup costs until you reach the root
    def backward(self, sim_costs):
        tree_len = len(self.tree_costs)
        path_len = len(self.tree_path)

        ## calculate discount factors
        discount_factors = [self.discount_factor**d for d in range(tree_len)]
        node = self.tree.root

        ## loop through the tree path
        for depth, (state, action) in enumerate(self.tree_path):

            ## get the state node and action leaf
            # state_node_id = self.node_id_path[depth]
            # state_node = self.tree.nodes[state_node_id]
            # action_leaf = state_node.action_leaves[action]
            action_leaf = node.action_leaves[action]
            np.array_equal(node.state[:2], state), 'error in tree path\n node: {} \n state: {}'.format(node, state)

            ## discounted costs from current node to rollout node
            # discounted_tree_cost = 0
            # dist_to_rollout = tree_len - depth
            # for d in range(dist_to_rollout):
            #     discounted_tree_cost += self.tree_costs[d + depth] * self.discount_factor**d
            dist_to_rollout = tree_len - depth
            discounted_tree_cost = np.dot(self.tree_costs[depth:depth + dist_to_rollout], discount_factors[:dist_to_rollout])

            ## calculate cost of the rollout, discounted by the distance from the current node to the rollout node
            total_sim_cost = 0
            first_sim_cost = sim_costs[0] * self.discount_factor**dist_to_rollout
            backup_cost = first_sim_cost + discounted_tree_cost

            ## calculate cost of all future rollouts
            for s in range(1, len(sim_costs)):
                # total_sim_cost += sim_costs[s] * meta_discount**s
                total_sim_cost += sim_costs[s] #* self.discount_factor**(dist_to_rollout + s)

            ## backup + update counts
            # action_leaf.n_action_visits += 1
            # state_node.n_state_visits += 1
            action_leaf.performance = action_leaf.performance + (backup_cost - action_leaf.performance) / action_leaf.n_action_visits

            ## next node
            if depth < path_len-1:
                node = action_leaf.children[str(self.tree_path[depth+1][0])]
            # try:
            #     node = action_leaf.children[str(self.tree_path[depth+1][0])]
            # except:
            #     pass


    ## tree search --> action loop
    def search(self, n_sims=1000, n_futures=1, progress=False):

        if progress:
            pbar = tqdm(total=n_sims, desc='MCTS search', position=0, leave=False, miniters=10, ascii=True, bar_format="{l_bar}{bar}")

        ## root sampling of new posterior
        # self.GP.root_sample(certainty_equivalent=True)

        ## root sampling of new kernel
        # K_inf = self.GP.sample_k()

        ## generate all root samples
        all_posterior_p_costs = []
        # for t in range(n_sims):
        #     self.agent.root_sample(self.env.obs)
        #     all_posterior_p_costs.append(self.agent.posterior_p_cost)
        # posterior_mean_p_cost = np.mean(all_posterior_p_costs, axis=0)

        ## debugging plot
        # plt.figure()
        # plot_r(posterior_mean_p_cost.reshape(self.N,self.N), ax = plt.subplot(), title='posterior sample')
        # plt.show()

        ## precommit to future S-G pairs
        # future_breadth = 5
        # future_depth = 1
        # future_SGs = np.zeros((future_breadth, future_depth, 2,2))
        # for b in range(future_breadth):
        #     for d in range(future_depth):
        #         start, goal = self.env.sample_SG()
        #         future_SGs[b,d,0] = start
        #         future_SGs[b,d,1] = goal

        ## create copy of envs for each future S-G pair
        # future_envs = [copy.deepcopy(self.env) for _ in range(future_breadth)]
        # for f in range(future_breadth):
        #     future_envs[f].set_sim(True)
        #     seed = random.randint(0,1000)
        #     future_envs[f].reset(seed=seed, start_goal=[future_SGs[f,0,0], future_SGs[f,0,1]])
        
        # ## create some future trees
        # future_trees = [Tree(self.N) for _ in range(future_breadth)]
        # future_MCTSs = [MonteCarloTreeSearch(env=env, agent=self.agent, tree=tree, exploration_constant=self.exploration_constant, discount_factor=self.discount_factor) for env, tree in zip(future_envs, future_trees)]
        
        ## loop through simulations
        for t in range(n_sims):

            if progress:
                pbar.update(1)
                
            # ## root sampling of new kernel
            # K_inf = self.GP.sample_k()
            
            ## root sampling of new posterior
            # self.GP.root_sample(self.env.obs, K_inf)
            self.agent.root_sample(self.env.obs)
            self.agent.dp(expected_cost=False)
            self.env.receive_predictions(self.agent.posterior_p_cost)
            all_posterior_p_costs.append(self.agent.posterior_p_cost)

            ## or, use pre-calculated posterior mean every time
            # self.agent.posterior_p_cost = posterior_mean_p_cost
            # self.env.receive_predictions(self.agent.posterior_p_cost)

            ## debugging plot
            # plt.figure()
            # # plot_r(self.env.posterior_sample.reshape(self.N,self.N), ax = plt.subplot(), title='posterior sample')
            # plot_action_tree(self.env.Q_inf, self.env.get_obs()['agent'], self.env.get_obs()['goal'], ax = plt.subplot(), title='DP_inf')

            ## selection, expansion, simulation
            action_leaf = self.tree_policy()
            # initial_sim_cost = self.rollout_policy(action_leaf, real_rollout=True)
            initial_sim_cost = self.optimised_rollout_policy(action_leaf, real_rollout=True)

            ## loop through future imagined episodes
            # future_sim_costs = self.rollout_policy(action_leaf, real_rollout=False, n_futures=n_futures)
            
            ##backup
            sim_costs = [initial_sim_cost
                        #  , future_sim_costs
                         ]
            self.backward(sim_costs)

        if progress:
            pbar.close()

        
        ## action selection
        MCTS_estimates = np.full(4, np.nan)
        for action, leaf in self.tree.root.action_leaves.items():
            MCTS_estimates[action] = leaf.performance
        assert not np.isnan(np.nansum(MCTS_estimates)), 'no MCTS estimates for {}'.format(self.tree.root)
        max_MCTS = np.nanmax(MCTS_estimates)
        action = argm(MCTS_estimates, max_MCTS)
        
        ## set root for next search
        # next_state = self.tree.root.action_leaves[action].next_state
        # next_root = self.tree.nodes[str(next_state)]

        ## mean over posterior samples?
        self.posterior_mean_p_cost = np.mean(all_posterior_p_costs, axis=0)

        return action





## parallel function for simulating many episodes within the same mountain env
def simulate_agent(m, N, params=None, metric='cityblock', true_k=None, n_episodes=10, agents = ['GP', 'GP-MCTS', 'BAMCP','CE'], n_sims=1000, n_futures=0, exploration_constant=2, discount_factor=0.95, progress=False, offline=False):
    
    ## initiate dictionary to store the results
    sim_out = {}
    for key in data_keys:
        sim_out[key]=[]
    
    ## set seed
    seed=m
    seed=os.getpid()
    np.random.seed(seed)
    
    ## create base mountain environment
    env = make_env(N, n_episodes, None, None, metric)
    
    ## debugging plot env
    # fig, ax = plt.subplots(1, 1, figsize=(5,5))
    # plot_r(env.p_costs, ax = ax, title=m)
    # plt.show()

    ## copy env so that each agent makes its own observations 
    agent_envs = [copy.deepcopy(env) for _ in agents]

    ## initialise farmer
    farmer = Farmer(N)

    ## loop through episodes (i.e. different start and goal states for the same mountain)
    print(' ') # for some reason need this to get the pbar to appear
    for e in tqdm(range(n_episodes), desc='Mountain_'+str(m), position=m+1, leave=False):

        ## loop through agents
        for a, ag in enumerate(agents):
            
            ## reset episode 
            # env_copy = copy.deepcopy(agent_envs[a])
            env_copy = agent_envs[a]
            _, _ = env_copy.reset()
            env_copy.set_sim(True)
            start = env_copy.current
            current = start
            goal = env_copy.goal

            ## GP-MCTS agent receives info from env
            if ag =='GP-MCTS':
                # K_inf = env_copy.K_gen.copy()
                # K_inf = None
                agent = GP
            elif (ag == 'BAMCP') or (ag == 'CE'):
                agent = farmer
            agent.get_env_info(env_copy)

            ## initiate tree (if not resetting the tree for each move, init here. otherwise, this should be inside the episode loop)
            tree = Tree(N)
            MCTS = MonteCarloTreeSearch(env=env_copy, agent=agent, tree=tree, exploration_constant=exploration_constant, discount_factor=discount_factor)
        
            ## run episode until goal is reached
            end_episode = False
            terminated=False
            early_terminate = False
            steps = 0
            max_steps = len(env_copy.o_traj)*1.5
            max_search_attempts = 3

            while not end_episode:

                ## plain balanced GP
                if ag == 'GP':
                    eps = 0.05
                    alpha = 0.4
                    action = env_copy.balanced_policy(current, goal, eps, alpha)

                    ## action
                    observation, _, terminated, truncated, info = env_copy.step(action)
                    current = observation['agent']
                    steps += 1

                    search_attempts = 0 # could do nan

                ## certainty-equivalent
                elif ag == 'CE':
                    env_copy.set_sim(False)

                    ## get posterior mean grid
                    all_posterior_p_costs = []
                    for t in range(100):
                        agent.root_sample(env_copy.obs, lazy=True, CE=True)
                        all_posterior_p_costs.append(agent.posterior_p_cost)
                    posterior_mean_p_cost = np.mean(all_posterior_p_costs, axis=0)
                    agent.posterior_p_cost = posterior_mean_p_cost
                    env_copy.receive_predictions(agent.posterior_p_cost)

                    ## dynamic programming under this posterior mean
                    agent.dp(expected_cost=False)

                    ## plot for debugging?
                    # _, axs = plt.subplots(1, 3, figsize=(10,5))
                    # plot_r(env_copy.p_costs.reshape(N,N), ax=axs[0], title = 'costs')
                    # plot_traj([env_copy.o_traj, env_copy.a_traj], ax=axs[0])
                    # plot_obs(env_copy.obs, ax=axs[0])
                    # plot_r(env_copy.predicted_p_costs, ax=axs[1], title = 'posterior mean p cost')
                    # plot_action_tree(agent.Q_inf, current, goal, ax=axs[2], title = 'DP_inf')
                    # plt.show()

                    ## get and take action
                    action = agent.optimal_policy(current, agent.Q_inf)
                    observation, _, terminated, truncated, info = env_copy.step(action)
                    current = observation['agent']
                    steps += 1

                    ## update observations
                    agent.get_env_info(env_copy)

                    search_attempts = 0 # could do nan



                ## bamcp
                elif ag == 'BAMCP':
                    env_copy.set_sim(True)
                    
                    ## init MCTS (if resetting the tree for each move, init here. otherwise, this should be outside the episode loop)
                    # tree = Tree(N)
                    # MCTS = MonteCarloTreeSearch(env=env_copy, agent=agent, tree=tree, exploration_constant=exploration_constant, discount_factor=discount_factor)
                    assert MCTS.env.sim == True, 'env not in sim mode'
                
                    ## if online planning (i.e. replan after each step)
                    if not offline:

                        ## search
                        search_attempts = 0 # could do nan
                        # n_futures = 0
                        action = MCTS.search(n_sims, n_futures, progress=progress)

                        ## plot for debugging?
                        # fig, axs = plt.subplots(1, 3, figsize=(15,5))
                        # plot_r(env_copy.p_costs.reshape(N,N), ax=axs[0], title = 'costs')
                        # plot_traj([env_copy.o_traj, env_copy.a_traj], ax=axs[0])
                        # # plot_obs(env_copy.obs, ax=axs[0])
                        # # MCTS.tree.action_tree()
                        # # plot_action_tree(MCTS.tree.tree_q, current, goal, ax=axs[1], title = 'DP_inf')
                        # plot_r(MCTS.posterior_mean_p_cost, ax=axs[2], title = 'average posterior p cost')
                        # plt.show()
                        
                        ## take action
                        env_copy.set_sim(False)
                        observation, cost, terminated, truncated, info = env_copy.step(action)
                        current = observation['agent']
                        steps += 1

                        ## update observations
                        agent.get_env_info(env_copy)

                        ## prune tree, i.e. remove all nodes that are not children of the new root
                        MCTS.tree.prune(action, np.append(current, cost))

                        assert np.array_equal(MCTS.tree.root.state[:2], current), 'error in root update\n env is in: {} but tree is in: {}\n should have taken action {}'.format(current, MCTS.tree.root.state, action)

                    ## if offline planning (i.e. plan the full trajectory)
                    elif offline:
                        non_stuck_route=False
                        search_attempts = 0
                        while not non_stuck_route:
                            search_attempts += 1

                            ## search
                            action, next_root = MCTS.search(n_sims, n_futures, progress=progress)

                            ## get the trajectory from the tree
                            MCTS.tree.action_tree()
                            traj_states, traj_actions = MCTS.tree.best_traj(start, goal)

                            ## take the trajectory if it leads you to the goal
                            if np.array_equal(traj_states[-1], goal):
                                env_copy.set_sim(False)
                                for state, action in zip(traj_states, traj_actions):
                                    assert np.array_equal(state, current), 'error in trajectory execution\n env is in: {} but tree is in: {}\n should have taken action {}'.format(current, state, action)
                                    observation, _, terminated, truncated, info = env_copy.step(action)
                                    current = observation['agent']
                                    steps += 1
                                    # if terminated:
                                    #     break
                                assert terminated
                                non_stuck_route = True

                            ## if stuck, repeat the search
                            else:
                                print('mountain {}, episode {}: MCTS failed to find a path on search attempt {}'.format(m, e, search_attempts))
                                ## plot the tree???

                                # ## execute the path anyway?
                                # env_copy.set_sim(False)
                                # for state, action in zip(traj_states, traj_actions):
                                #     assert np.array_equal(state, current), 'error in trajectory execution\n env is in: {} but tree is in: {}\n should have taken action {}'.format(current, state, action)
                                #     observation, _, terminated, truncated, info = env_copy.step(action)
                                #     current = observation['agent']
                                #     steps += 1
                                # MCTS.tree.root = MCTS.tree.nodes[str(current)]
                                # env_copy.set_sim(True)

                                ## give up if too many searches
                                print('restarting search from ', MCTS.tree.root.state)
                                if search_attempts>max_search_attempts:
                                    early_terminate = True
                                    print('mountain {}, epsiode {}: search attempts exceeded'.format(m, e))
                                    break

                ## prevent endless episode 
                if steps >= max_steps:
                    early_terminate = True

                if early_terminate:
                    print('mountain ',m,': episode ',e,' terminated for agent ',ag,' after ',steps,' steps')

                    ## or just skip to the next episode
                    sim_out['agent'].append(agent)
                    sim_out['episode'].append(e)
                    sim_out['mountain'].append(m)
                    sim_out['start'].append(start)
                    sim_out['goal'].append(goal)
                    sim_out['actual_cost'].append(np.nan)
                    sim_out['optimal_cost'].append(env_copy.o_traj_total_cost)
                    sim_out['action_score'].append(np.nan)
                    sim_out['cost_ratio'].append(np.nan)
                    sim_out['n_steps'].append(steps)
                    sim_out['actual_trajectory'].append(env_copy.a_traj)
                    sim_out['optimal_trajectory'].append(env_copy.o_traj)
                    sim_out['observations'].append(env_copy.obs)
                    sim_out['search_attempts'].append(search_attempts)
                    # sim_out['action_tree'].append(MCTS.tree.action_tree())
                    sim_out['action_tree'].append(np.nan)
                    
                    ## GP-specific
                    # sim_out['true_k'].append(true_k)
                    # sim_out['RPE'].append(np.nan)
                    # sim_out['posterior_mean'].append(np.nan)
                    # sim_out['theta_MLE'].append(best_theta)

                    end_episode = True

                ## save data and end the episode
                elif terminated:
                    sim_out['agent'].append(ag)
                    sim_out['episode'].append(e)
                    sim_out['mountain'].append(m)
                    sim_out['start'].append(start)
                    sim_out['goal'].append(goal)
                    sim_out['actual_cost'].append(env_copy.a_traj_total_cost)
                    sim_out['optimal_cost'].append(env_copy.o_traj_total_cost)
                    # if np.round(env_copy.optimal_cost,4) < np.round(env_copy.accrued_cost,4):
                    #     print(env_copy.optimal_cost, env_copy.accrued_cost)
                    # assert np.round(env_copy.optimal_cost,4) >= np.round(env_copy.accrued_cost,4), 'accrued cost higher than optimal cost'
                    # sim_out['action_score'].append(env_copy.optimal_cost/env_copy.accrued_cost)
                    sim_out['action_score'].append(env_copy.action_score)
                    sim_out['cost_ratio'].append(env_copy.cost_ratio)
                    sim_out['n_steps'].append(steps)
                    sim_out['actual_trajectory'].append(env_copy.a_traj)
                    sim_out['optimal_trajectory'].append(env_copy.o_traj)
                    sim_out['observations'].append(env_copy.obs)
                    sim_out['search_attempts'].append(search_attempts)
                    # sim_out['action_tree'].append(MCTS.tree.action_tree())
                    sim_out['action_tree'].append(np.nan)
                    
                    ## GP-specific
                    # sim_out['true_k'].append(true_k)
                    # sim_out['RPE'].append(np.mean(np.abs(GP.posterior_mean.reshape(N,N) - env_copy.costs)))
                    # sim_out['posterior_mean'].append(GP.posterior_mean)
                    # sim_out['theta_MLE'].append(best_theta)
                    
                    ## update the agent env
                    # agent_envs[a] = copy.deepcopy(env_copy)
                    agent_envs[a] = env_copy

                    end_episode = True

    return sim_out,env_copy.p_costs
    # return sim_out, _