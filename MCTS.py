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


class MonteCarloTreeSearch():

    def __init__(self, env, agent, tree, exploration_constant=2, discount_factor=0.99):
        self.env = env
        self.actual_state = self.env.current
        self.actual_goal = self.env.goal
        self.agent = agent
        self.tree = tree
        self.action_space = self.env.action_space.n
        self.N = self.env.N
        self.exploration_constant = exploration_constant
        self.discount_factor = discount_factor

        ## cost of current state?
        starting_cost = self.env.costs[self.actual_state[0], self.actual_state[1]]

        ## add state node to the tree
        self.tree.add_state_node(state=self.actual_state, cost=starting_cost, terminated=False, action_space = self.action_space, parent=None)


    ## expand the action space of a node
    def expand(self, node):

        ## create copy of env and set state
        # env_copy = copy.deepcopy(self.env)
        # env_copy.set_state(node.state)
        # assert env_copy.sim, 'env is not in sim mode'
        assert self.env.sim, 'env is not in sim mode'

        ## take action and get new state
        action = node.untried_action()
        next_state, cost, terminated, truncated, _ = self.env.step(action)

        ## or, do this without updating the environment object
        # next_state = np.clip(actual_state + self.env.action_to_direction[action],
        #                         0, self.N - 1)
        # cost = self.env.get_pred_cost(next_state)
        # direction = self.env.action_to_direction[action]
        # next_state = get_next_state(actual_state, direction, self.N)
        # terminated = np.array_equal(next_state, self.env.goal)

        ## update info for s-a leaf - i.e. the state-action pair
        node.action_leaves[action] = Action_Node(prev_state = node.state, action=action, next_state = next_state, terminated=terminated)
        # node.action_leaves[action].performance = cost
        node.action_leaves[action].performance = 0

        ## reset the environment to the actual state
        self.env.set_state(self.actual_state)

        return node.action_leaves[action]



    ## take an action according to the tree policy, i.e. take the best UCT child and see where it takes you
    def tree_policy(self):
        
        ## create copy of env and set state
        # env_copy = copy.deepcopy(self.env)
        # env_copy.set_sim(True)

        assert self.env.sim, 'env is not in sim mode'

        ## initialise the tree
        node = self.tree.root
        t = 0
        self.tree_costs = []
        # self.tree_cost.append(node.cost)
        assert np.array_equal(node.state[:2], self.actual_state), 'mismatch between node and env state\n node: {} \n env: {}'.format(node, self.actual_state)

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
                self.env.set_state(self.actual_state)

                ## save tree obs for subsequent rollouts
                # self.tree_obs = self.env.obs_tmp.copy()
                # self.env.flush_obs()

                return action_leaf
                
            ## selection step
            else:

                ## (some debugging vars)
                # state_tmp = node.state[:2]
                # env_state_tmp = self.env.get_obs()['agent']
                # env_state_tmp = self.env.current

                ## get the best child
                action_leaf = self.best_child(node)
                self.tree_path.append(tuple([node.state, action_leaf.action]))
                self.node_id_path.append(node.node_id)

                ## move in env
                next_state, cost, terminated, _, _ = self.env.step(action_leaf.action)
                self.tree_costs.append(cost)
                cost = self.env.get_pred_cost(next_state)
                self.tree_costs.append(cost)

                ## or, do this without updating the environment object
                # next_state = np.clip(node.state[:2] + self.env.action_to_direction[action_leaf.action],
                #                 0, self.N - 1)
                # direction = self.env.action_to_direction[action_leaf.action]
                # next_state = get_next_state(node.state[:2], direction, self.N)
                # cost = self.env.get_pred_cost(next_state)
                # terminated = action_leaf.terminated

                self.tree_costs.append(cost)

                ## see if the next state node already exists as a child of this action leaf
                state_id = tuple(np.append(next_state, cost))
                if state_id in action_leaf.children:
                    node = action_leaf.children[state_id]
                else:
                    node = self.tree.add_state_node(next_state, cost, terminated, action_space = self.action_space, parent=action_leaf)
                # assert np.array_equal(node.state[:2], next_state), 'error in tree policy step {}\n started in {}\n supposed to take action {} to {}\n ended up moving from {} to {}'.format(t, state_tmp, action_leaf.action, node.state[:2], env_state_tmp, action_leaf.next_state)

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
        self.env.set_state(self.actual_state)

        ## save tree obs for subsequent rollouts
        # self.tree_obs = self.env.obs_tmp.copy()
        # self.env.flush_obs()
        # assert len(self.tree_obs) == len(self.env.obs)+len(self.tree_path), 'tree obs and path lengths do not match\n tree obs: {}, env.obs: {}, tree path: {}'.format(len(self.tree_obs), len(self.env.obs),len(self.tree_path))

        return action_leaf


    def rollout_policy(self, action_leaf, real_rollout = True, n_futures=None):

        ## init
        total_cost = 0
        max_depth = 100
        # depth = len(self.tree_path)
        assert self.env.sim, 'env is not in sim mode'
        
        ## set the state from which the rollout is initiated
        # self.env.set_state(action_leaf.next_state)
        # observation = self.env.get_obs()
        
        ## or, make a copy
        env_copy = copy.deepcopy(self.env)

        ## standard rollout if this is the first S-G pair
        if real_rollout:
            start = action_leaf.next_state
            env_copy.set_state(action_leaf.next_state)

            ## reset the depth counter
            self.depth = 0

            ## rolling out from goal location, can just end here
            if action_leaf.terminated:

                ## revert env
                # self.env.set_state(self.actual_state)

                return total_cost
            
            ## begin with cost of current state
            starting_cost = env_copy.get_pred_cost(start)
            total_cost += starting_cost
            # observation = env_copy.get_obs()
            current = start

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
                # current = observation['agent']


                ## or, greedy
                # current = observation['agent']
                # action = agent_copy.greedy_policy(current, env_copy.goal, eps = 0.0)

                ## or, optimised rollout 
                action = self.agent.optimal_policy(current, self.agent.Q_inf)

                ## take action
                current, cost, terminated, _, _ = env_copy.step(action)
                
                ## if terminated return the cost (i.e. don't use the cost of the goal state)
                if terminated:
                    return total_cost

                ## increment cost
                total_cost += cost * self.discount_factor**self.depth


        ## or, rollout for some new imagined S-G pair
        else:

            ## copy agent
            agent_copy = copy.deepcopy(self.agent) ## this is done so that the agent doesn't change its state in the imagined rollouts. could of course just feed env back into it at end of rollout

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
                # observation = env_copy.get_obs()
                current = start
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
                    # current = observation['agent']


                    ## optimised rollout 
                    action = agent_copy.optimal_policy(current, agent_copy.Q_inf)

                    # ## or greedy wrt/ distance
                    # action = agent_copy.greedy_policy(current, goal, eps = 0.0)

                    ## take action
                    current, cost, terminated, _, _ = env_copy.step(action)

                    ## if terminated return the cost (i.e. don't use the cost of the goal state)
                    if terminated:
                        future_total_costs.append(total_cost)
                        break
                    
                    ## increment cost
                    total_cost += cost * self.discount_factor**depth_tmp


            ## average the future costs
            total_cost = np.mean(future_total_costs)
            return total_cost
            
    ## optimised rollout
    def optimised_rollout_policy(self, action_leaf, real_rollout = True):

        ## init
        total_cost = 0
        depth = 0

        ## rolling out from goal location, can just end here
        if action_leaf.terminated:
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
            # if action == 0:  # Down
            #     current = np.clip((i + 1, j), 0, self.N - 1)
            # elif action == 1:  # Right
            #     current = np.clip((i, j + 1), 0, self.N - 1)
            # elif action == 2:  # Up
            #     current = np.clip((i - 1, j), 0, self.N - 1)
            # elif action == 3:  # Left
            #     current = np.clip((i, j - 1), 0, self.N - 1)
            
            ## return cost once goal is reached (i.e. don't use the cost of the goal state)
            if np.array_equal(current, self.actual_goal):
                return total_cost
            
            ## update costs
            cost = self.env.get_pred_cost(current)
            total_cost += cost*self.discount_factor**depth
        

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
            # assert np.array_equal(node.state[:2], [:2]), 'error in tree path\n node: {} \n state: {}'.format(node.state[:2], state[:2])

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
                node = action_leaf.children[tuple(self.tree_path[depth+1][0])]
            # try:
            #     node = action_leaf.children[str(self.tree_path[depth+1][0])]
            # except:
            #     pass


    ## tree search --> action loop
    def search(self, n_sims=1000, n_futures=0, n_iter=100, lazy=False,  progress=False, reuse_samples=False):

        if progress:
            pbar = tqdm(total=n_sims, desc='MCTS search', position=0, leave=False, miniters=10, ascii=True, bar_format="{l_bar}{bar}")

        ## root sampling of new posterior
        # self.GP.root_sample(certainty_equivalent=True)

        ## root sampling of new kernel
        # K_inf = self.GP.sample_k()

        ## if samples not provided, generate new set of root samples
        if not reuse_samples:
            self.agent.root_samples(obs = self.env.obs, n_samples=n_sims, n_iter=n_iter, lazy=lazy, CE=False)


        ## debugging plot
        # plt.figure()
        # plot_r(posterior_mean_p_cost.reshape(self.N,self.N), ax = plt.subplot(), title='posterior sample')
        # plt.show()
        
        ## loop through simulations
        for t in range(n_sims):

            if progress:
                pbar.update(1)
            
            ## root sampling of new posterior
            posterior_p_cost = self.agent.all_posterior_p_costs[t]
            self.agent.dp(posterior_p_cost, expected_cost=True)
            self.env.receive_predictions(posterior_p_cost)

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

        ## calculate the entropy over actions
        # action_probs = np.exp(MCTS_estimates) / np.sum(np.exp(MCTS_estimates))
        # entropy = -np.nansum(action_probs * np.log(action_probs))
        # print('action probs:', action_probs)
        # print('entropy:', entropy)

        return action





## parallel function for simulating many episodes within the same mountain env
def simulate_agent(m, N, params=None, metric='cityblock', true_k=None, n_episodes=10, agents = ['GP', 'GP-MCTS', 'BAMCP','CE'], n_sims=1000, n_futures=0, n_iter=10, lazy=False, exploration_constant=2, discount_factor=0.95, progress=False, offline=False):
    
    ## initiate dictionary to store the results
    sim_out = {}
    for key in data_keys:
        sim_out[key]=[]
    
    ## set seed
    seed=m
    seed=os.getpid()
    np.random.seed(seed)
    
    ## create base mountain environment
    env = make_env(N, n_episodes, None, params, metric)
    
    ## debugging plot env
    # fig, ax = plt.subplots(1, 1, figsize=(5,5))
    # plot_r(env.p_costs, ax = ax, title=m)
    # plt.show()

    ## copy env so that each agent makes its own observations 
    # agent_envs = [copy.deepcopy(env) for _ in agents]
    agent_envs = {}
    for a in agents:
        agent_envs[a] = copy.deepcopy(env)


    ## initialise farmer
    farmer = Farmer(N)

    ## loop through episodes (i.e. different start and goal states for the same mountain)
    print(' ') # for some reason need this to get the pbar to appear
    for e in tqdm(range(n_episodes), desc='Mountain_'+str(m), position=m+1, leave=False):

        ## loop through agents
        for a, ag in enumerate(agents):
            
            ### reset episode 

            ## copy env for our base agents
            if (ag=='BAMCP') or (ag=='CE'):
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
            EKLs = []

            ## GP-MCTS agent receives info from env
            if ag =='GP-MCTS':
                # K_inf = env_copy.K_gen.copy()
                # K_inf = None
                agent = GP
            elif (ag == 'BAMCP') or (ag == 'CE') or (ag=='BAMCP w/ CE') or (ag=='CE w/ BAMCP'):
                agent = farmer
            agent.get_env_info(env_copy)

            ## initiate tree (if not resetting the tree for each move, init here. otherwise, this should be inside the episode loop)
            if (ag == 'BAMCP') or (ag == 'BAMCP w/ CE'):
                tree = Tree(N)
                MCTS = MonteCarloTreeSearch(env=env_copy, agent=agent, tree=tree, exploration_constant=exploration_constant, discount_factor=discount_factor)
        
            ## run episode until goal is reached
            end_episode = False
            terminated=False
            early_terminate = False
            reuse_samples = False
            steps = 0
            max_steps = len(env_copy.o_trajs[e])*1.75
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
                    # agent.root_sample(env_copy.obs, lazy=True, CE=True)
                    agent.root_samples(obs=env_copy.obs, n_samples=n_sims, lazy=True,CE=True)
                    env_copy.receive_predictions(agent.posterior_mean_p_cost)

                    ## dynamic programming under this posterior mean
                    agent.dp(agent.posterior_mean_p_cost, expected_cost=True)


                    # all_posterior_p_costs = []
                    # all_posterior_p = []
                    # all_posterior_q = []
                    # for t in range(100):
                    #     agent.root_sample(env_copy.obs, lazy=True, CE=True)
                    #     # all_posterior_p_costs.append(agent.posterior_p_cost)
                    #     all_posterior_p.append(agent.posterior_p)
                    #     all_posterior_q.append(agent.posterior_q)
                    # # posterior_mean_p_cost = np.mean(all_posterior_p_costs, axis=0)
                    # posterior_mean_p = np.mean(all_posterior_p, axis=0)
                    # posterior_mean_q = np.mean(all_posterior_q, axis=0)
                    # posterior_mean_p_cost = np.outer(posterior_mean_p, posterior_mean_q)
                    # agent.posterior_p_cost = posterior_mean_p_cost
                    # env_copy.receive_predictions(agent.posterior_p_cost)

                    # ## dynamic programming under this posterior mean
                    # agent.dp(expected_cost=True)

                    ## plot for debugging?
                    # _, axs = plt.subplots(1, 3, figsize=(10,5))
                    # plot_r(env_copy.p_costs.reshape(N,N), ax=axs[0], title = 'costs')
                    # plot_traj([env_copy.o_trajs[e], env_copy.a_traj], ax=axs[0])
                    # plot_r(env_copy.predicted_p_costs, ax=axs[1], title = 'posterior mean p cost')
                    # plot_action_tree(agent.Q_inf, current, goal, ax=axs[2], title = 'DP_inf')
                    # plt.show()

                    ## get and take action
                    action = agent.optimal_policy(current, agent.Q_inf)
                    actions.append(action)
                    current, _, terminated, _, _ = env_copy.step(action)
                    # current = observation['agent']
                    steps += 1

                    ## update observations
                    agent.get_env_info(env_copy)

                    search_attempts = 0 # could do nan



                ## bamcp
                elif (ag == 'BAMCP') or (ag == 'BAMCP w/ CE'):
                    env_copy.set_sim(True)
                    
                    ## init MCTS (if resetting the tree for each move, init here. otherwise, this should be outside the episode loop)
                    # tree = Tree(N)
                    # MCTS = MonteCarloTreeSearch(env=env_copy, agent=agent, tree=tree, exploration_constant=exploration_constant, discount_factor=discount_factor)
                    assert MCTS.env.sim == True, 'env not in sim mode'
                
                    ## if online planning (i.e. replan after each step)
                    if not offline:

                        ## search
                        # action = MCTS.search(n_sims, n_futures, progress=progress)

                        ## reduce number of sims if near to the goal (A BETTER IDEA WLD BE TO REDUCE THE DISTANCE IF THE TREE HAS REACHED THE GOAL)
                        dist_to_goal = np.max(cdist([current, goal], [current, goal], metric='cityblock')) 
                        if dist_to_goal > (N/2):
                            action = MCTS.search(n_sims, n_futures, n_iter=n_iter, lazy=lazy, progress=progress, reuse_samples=reuse_samples)
                        elif (dist_to_goal <= (N/2)) & (dist_to_goal > (N/4)):
                            n_reduced_sims = int(n_sims/2)
                            action = MCTS.search(n_reduced_sims, n_futures, n_iter=n_iter, lazy=lazy, progress=progress, reuse_samples=reuse_samples)
                        else:
                            n_reduced_sims = int(n_sims/4)
                            action = MCTS.search(n_reduced_sims, n_futures,n_iter=n_iter, lazy=lazy,  progress=progress, reuse_samples=reuse_samples)
                        actions.append(action)

                        ##optional: check what the CE agent would have done with the mean of this set of samples (NEED TO THINK ABOUT WHETHER THIS IS RIGHT, SINCE THE UNOBSERVED ROWS/COLS ARE INITIALISED DIFFERENTLY)
                        agent.dp(agent.posterior_mean_p_cost, expected_cost=True)
                        action_CE = agent.optimal_policy(current, agent.Q_inf)
                        CE_actions.append(action_CE)

                        ## plot for debugging?
                        print('BAMCP action:', action,', CE action:', action_CE)
                        fig, axs = plt.subplots(1, 3, figsize=(15,5))
                        plot_r(env_copy.p_costs.reshape(N,N), ax=axs[0], title = 'p_costs')
                        plot_traj([env_copy.o_trajs[e], env_copy.a_traj], ax=axs[0])
                        plot_r(agent.posterior_mean_p_cost, ax=axs[1], title = 'average posterior p cost')
                        plt.show()
                        
                        ## take action
                        env_copy.set_sim(False)
                        current, cost, terminated, _, _ = env_copy.step(action)
                        # current = observation['agent']
                        steps += 1
                        

                        ## update observations
                        agent.get_env_info(env_copy)

                        ## prune tree, i.e. remove all nodes that are not children of the new root
                        if not terminated:
                            MCTS.tree.prune(action, np.append(current, cost))
                            assert np.array_equal(MCTS.tree.root.state[:2], current), 'error in root update\n env is in: {} but tree is in: {}\n should have taken action {}'.format(current, MCTS.tree.root.state, action)
                        search_attempts = 0 # could do nan


                    ## if offline planning (i.e. plan the full trajectory)
                    elif offline:
                        non_stuck_route=False
                        search_attempts = 0
                        while not non_stuck_route:
                            search_attempts += 1

                            ## search
                            action, next_root = MCTS.search(n_sims, n_futures, progress=progress, reuse_samples=reuse_samples)
                            actions.append(action)

                            ## get the trajectory from the tree
                            MCTS.tree.action_tree()
                            traj_states, traj_actions = MCTS.tree.best_traj(start, goal)

                            ## take the trajectory if it leads you to the goal
                            if np.array_equal(traj_states[-1], goal):
                                env_copy.set_sim(False)
                                for state, action in zip(traj_states, traj_actions):
                                    assert np.array_equal(state, current), 'error in trajectory execution\n env is in: {} but tree is in: {}\n should have taken action {}'.format(current, state, action)
                                    current, _, terminated, _, _ = env_copy.step(action)
                                    # current = observation['agent']
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
                                #     observation, _, terminated, truncated = env_copy.step(action)
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

                ### log determinant of covariance matrix
                # if (ag=='BAMCP') or (ag=='BAMCP w/ CE'):

                #     ## get prior p and q samples
                #     prior_p_samples = agent.all_posterior_ps
                #     prior_q_samples = agent.all_posterior_qs
                #     prior_all_samples = np.vstack([prior_p_samples.T, prior_q_samples.T])

                #     ## log det of prior covariance matrix 
                #     prior_cov = np.cov(prior_all_samples)
                #     prior_LD = np.linalg.slogdet(prior_cov)[1]
                #     assert prior_cov.shape[0] == N*2, 'covariance matrix is wrong shape'
                    
                    
                #     ## order the outcomes (counterfactual, then actual. This is to allow reuse of the posterior samples associated with the actual outcome on the next timestep)
                #     actual_outcome = env_copy.obs.copy()[-1, -1]
                #     if actual_outcome == env_copy.low_cost:
                #         ordered_outcomes = [env_copy.high_cost, env_copy.low_cost]
                #     else:
                #         ordered_outcomes = [env_copy.low_cost, env_copy.high_cost]
                #     posterior_LDs = []

                #     ## posterior samples under each of the possible outcomes of the action that was just taken
                #     for o, outcome in enumerate(ordered_outcomes):
                #         sim_obs = env_copy.obs.copy()
                #         sim_obs[-1, -1] = outcome
                #         agent.root_samples(sim_obs, n_samples=n_sims)
                #         posterior_samples = np.vstack([np.array(agent.all_posterior_ps).T, np.array(agent.all_posterior_qs).T])
                #         posterior_cov = np.cov(posterior_samples)
                #         assert posterior_cov.shape == prior_cov.shape, 'prior and posterior covariance matrices do not match: {} vs {}'.format(posterior_cov.shape, prior_cov.shape)
                        
                #         ## posterior log det
                #         posterior_cov
                #         LD = np.linalg.slogdet(posterior_cov)[1]
                #         posterior_LDs.append(LD)


                #     ## expected log det, i.e. the difference between the prior and the expected posterior log dets, weighted by the probability of each outcome
                #     p_low = np.mean(prior_p_samples * prior_q_samples)
                #     p_high = 1 - p_low
                #     if actual_outcome == env_copy.low_cost:
                #         expected_LD = p_low * (posterior_LDs[1] - prior_LD) + p_high * (posterior_LDs[0] - prior_LD)
                #     else:
                #         expected_LD = p_low * (posterior_LDs[0] - prior_LD) + p_high * (posterior_LDs[1] - prior_LD)

                #     ## reorder the posterior LDs to match low and high cost outcomes (i.e. 0th element is the low cost outcome)
                #     # if ordered_outcomes[1] == env_copy.low_cost:
                #     #     posterior_LDs = [posterior_LDs[1], posterior_LDs[0]]
                #     # expected_LD2 = p_low * (posterior_LDs[0] - prior_LD) + p_high * (posterior_LDs[1] - prior_LD)
                #     # assert expected_LD==expected_LD2, 'expected LDs when actual outcome is {} do not match: {} vs {}'.format(actual_outcome, expected_LD, expected_LD2)

                #     CE_deviation = action==action_CE
                #     print('action {} deviate from CE'.format(['did','did not'][CE_deviation]))
                #     print('prior LD: ',prior_LD)
                #     print('posterior LDs: ',posterior_LDs, ', probs: ',p_low, p_high)
                #     print('expected change in LD: ',expected_LD)
                #     print()

                #     ## reuse samples associated with the actual outcome on the next timestep
                #     reuse_samples = True



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
                    sim_out['episode'].append(e)
                    sim_out['mountain'].append(m)
                    sim_out['start'].append(start)
                    sim_out['goal'].append(goal)
                    sim_out['actions'].append(actions)
                    sim_out['CE_actions'].append(CE_actions)
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
                    sim_out['episode'].append(e)
                    sim_out['mountain'].append(m)
                    sim_out['start'].append(start)
                    sim_out['goal'].append(goal)
                    sim_out['actions'].append(actions)
                    sim_out['CE_actions'].append(CE_actions)
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

    return sim_out,env_copy.p_costs
    # return sim_out, _