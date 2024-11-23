import random
from math import sqrt, log
from utils import Node, Action_Node, Tree, make_env, argm
import copy
import numpy as np
from tqdm.auto import tqdm
import os
from scipy.spatial.distance import cdist

from plotter import *
from agents import GPAgent


class MonteCarloTreeSearch():

    def __init__(self, env, GP, tree, exploration_constant=2, discount_factor=0.95):
        self.env = env
        self.GP = GP
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

        ## update info for s-a leaf - i.e. the state-action pair, and the cost of the state that you subsequently reach        
        node.action_leaves[action] = Action_Node(prev_state = node.state, action=action, next_state = next_state, next_cost=cost, terminated=terminated)
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
        assert np.array_equal(node.state, actual_state), 'mismatch between node and env state\n node: {} \n env: {}'.format(node, actual_state)

        ## create a record of the nodes/leaves visited in the tree
        self.tree_path = []
        
        ## loop until you reach a leaf node or terminal state
        while not node.terminated:
            t+=1

            ## expansion step
            if self.tree.is_expandable(node):
                action_leaf = self.expand(node)
                self.tree_path.append(tuple([node.state, action_leaf.action]))

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
                assert len(self.tree_obs) == len(self.env.obs)+len(self.tree_path), 'tree obs and path lengths do not match\n tree obs: {}, env.obs: {}, tree path: {}'.format(len(self.tree_obs), len(self.env.obs),len(self.tree_path))

                return action_leaf
                
            ## selection step
            else:

                ## (some debugging vars)
                # state_tmp = node.state
                # env_state_tmp = self.env.get_obs()['agent']

                ## get the best child
                action_leaf = self.best_child(node)
                self.tree_path.append(tuple([node.state, action_leaf.action]))

                ## move in env
                observation, cost, terminated, _, _ = self.env.step(action_leaf.action)
                next_state = observation['agent'] ## could simplify this so that the step returns the next state rather than the goal too
                self.tree_costs.append(action_leaf.next_cost)

                ## create next state node (if it doesn't already exist)
                node = self.tree.add_state_node(next_state, action_leaf.next_cost, terminated, action_space = self.action_space, parent=action_leaf)
                assert np.array_equal(node.state, next_state), 'error in tree policy step {}\n started in {}\n supposed to take action {} to {}\n ended up moving from {} to {}'.format(t, state_tmp, node.action, node.state, env_state_tmp, action_leaf.next_state)

                ## update counts already?
                action_leaf.n_action_visits += 1
                node.n_state_visits += 1

            ## if the agent has reached a state that has already been visited, initiate a rollout from there
            visited_states = np.array([self.tree_path[i][0] for i in range(len(self.tree_path))])
            if any(np.array_equal(next_state, state) for state in visited_states):
                break

            if t>self.N**2:
                # print('tree policy stuck')
                break

        ## revert env
        self.env.set_state(actual_state)

        ## save tree obs for subsequent rollouts
        self.tree_obs = self.env.obs_tmp.copy()
        self.env.flush_obs()
        assert len(self.tree_obs) == len(self.env.obs)+len(self.tree_path), 'tree obs and path lengths do not match\n tree obs: {}, env.obs: {}, tree path: {}'.format(len(self.tree_obs), len(self.env.obs),len(self.tree_path))

        return action_leaf


    def rollout_policy(self, action_leaf, real_rollout = True, n_futures=None):

        ## init
        total_cost = 0
        max_depth = 100
        depth = 0

        ## get the agent's current location and goal
        actual_state = self.env.current
        actual_goal = self.env.goal
        assert self.env.sim, 'env is not in sim mode'
        
        ## set the state from which the rollout is initiated
        # self.env.set_state(action_leaf.next_state)
        # observation = self.env.get_obs()
        
        ## or, make a copy
        env_copy = copy.deepcopy(self.env)
        GP_copy = copy.deepcopy(self.GP) ## this is done so that the GP doesn't change its state. could of course just feed env back into it at end of rollout

        ## standard rollout if this is the first S-G pair
        if real_rollout:
            env_copy.set_state(action_leaf.next_state)
            
            ## begin with cost of current state
            starting_cost = action_leaf.next_cost
            total_cost += starting_cost
            observation = env_copy.get_obs()

            ## rolling out from goal location, can just end here
            if action_leaf.terminated:

                ## revert env
                # self.env.set_state(actual_state)

                return total_cost
            

            ## rollout until trial is terminated 
            while True:

                ## prevent infinite rollout
                depth += 1
                if depth > max_depth:
                    # print('exceeded max rolls in {} rollout'.format(['imagined', 'real'][real_rollout]))

                    # print(env_copy.V_inf)
                    fig, axs = plt.subplots(1, 3, figsize=(15,5))
                    # plot_r(env_copy.posterior_mean.reshape(self.N,self.N), ax=axs[0], title = 'posterior mean')
                    sns.heatmap(GP_copy.posterior_sample.reshape(self.N,self.N), ax=axs[0], cbar=False, annot=True, fmt='.2f')
                    plot_action_tree(GP_copy.Q_inf, start, goal, ax=axs[1], title = 'DP_inf')
                    plot_r(GP_copy.V_inf, ax=axs[2], title = 'V')

                    ## raise error
                    raise ValueError('exceeded max rolls in {} rollout, start: {}, goal: {}'.format(['imagined', 'real'][real_rollout], start, goal))

                ## or, greedy
                # current = observation['agent']
                # action = self.env.greedy_policy(current, goal, eps = 0.0)
                # action = env_copy.greedy_policy(current, env_copy.goal, eps = 0.0)

                ## or, optimised rollout 
                current = observation['agent']
                action = GP_copy.optimal_policy(current, GP_copy.Q_inf)

                ## take action
                observation, cost, terminated, _, _ = env_copy.step(action)

                ## increment cost
                total_cost += cost * self.discount_factor**depth

                ## if terminated return the cost
                if terminated:
                    return total_cost

        ## or, rollout for some new imagined S-G pair
        else:

            ## inherit obs from tree so far and sample new posterior
            GP_copy.get_env_info(env_copy)
            GP_copy.root_sample(self.tree_obs, GP_copy.K_inf)

            ## loop through new start-end pairs
            future_total_costs = []
            for f in range(n_futures):

                depth = 0
                total_cost = 0

                ## imagine new start and goal locations
                seed = random.randint(0, 1000)
                _,_ = env_copy.reset(seed=seed)
                start = env_copy.current
                goal = env_copy.goal
                observation = env_copy.get_obs()
                GP_copy.get_env_info(env_copy)

                ## get DP Q-values for the new start and goal under the current posterior
                GP_copy.dp(certainty_equivalent=False)

                ## rollout until trial is terminated 
                while True:

                    ## prevent infinite rollout
                    depth += 1
                    if depth > max_depth:
                        # print('exceeded max rolls in {} rollout'.format(['imagined', 'real'][real_rollout]))

                        # print(env_copy.V_inf)
                        fig, axs = plt.subplots(1, 3, figsize=(15,5))
                        # plot_r(env_copy.posterior_mean.reshape(self.N,self.N), ax=axs[0], title = 'posterior mean')
                        sns.heatmap(GP_copy.posterior_sample.reshape(self.N,self.N), ax=axs[0], cbar=False, annot=True, fmt='.2f')
                        plot_action_tree(GP_copy.Q_inf, start, goal, ax=axs[1], title = 'DP_inf')
                        plot_r(GP_copy.V_inf, ax=axs[2], title = 'V')

                        ## raise error
                        raise ValueError('exceeded max rolls in {} rollout, start: {}, goal: {}'.format(['imagined', 'real'][real_rollout], start, goal))


                    ## optimised rollout 
                    current = observation['agent']
                    action = GP_copy.optimal_policy(current, GP_copy.Q_inf)

                    ## take action
                    observation, cost, terminated, _, _ = env_copy.step(action)

                    ## increment cost
                    total_cost += cost * self.discount_factor**depth

                    ## if terminated, append cost
                    if terminated:
                        future_total_costs.append(total_cost)
                        break

            ## average the future costs
            total_cost = np.mean(future_total_costs)
            return total_cost
            
        

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

        ## calculate discount factors
        discount_factors = [self.discount_factor**d for d in range(tree_len)]

        ## loop through the tree path
        for depth, (state, action) in enumerate(self.tree_path):

            ## get the state node and action leaf
            state_node = self.tree.nodes[str(state)]
            action_leaf = state_node.action_leaves[action]

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

            ## calculate cost of all future rollouts, discounted by some meta-discount factor?
            # meta_discount = 0.95
            for s in range(1, len(sim_costs)):
                # total_sim_cost += sim_costs[s] * meta_discount**s
                total_sim_cost += sim_costs[s] * self.discount_factor**(dist_to_rollout + s)

            ## backup + update counts
            # action_leaf.n_action_visits += 1
            # state_node.n_state_visits += 1
            action_leaf.performance = action_leaf.performance + (backup_cost - action_leaf.performance) / action_leaf.n_action_visits


    ## tree search --> action loop
    def search(self, n_trees=1000, n_futures=1, progress=False):

        if progress:
            pbar = tqdm(total=n_trees, desc='MCTS search', position=0, leave=False, miniters=100, ascii=True, bar_format="{l_bar}{bar}")

        ## root sampling of new posterior
        # self.GP.root_sample(certainty_equivalent=True)

        ## loop through trees
        for t in range(n_trees):

            if progress:
                pbar.update(1)
                
            ## root sampling of new kernel
            K_inf = self.GP.sample_k()
            
            ## root sampling of new posterior
            self.GP.root_sample(self.env.obs, K_inf)
            self.GP.dp(certainty_equivalent=False)
            self.env.receive_predictions(self.GP.posterior_sample)
            
            ## debugging plot
            # plt.figure()
            # # plot_r(self.env.posterior_sample.reshape(self.N,self.N), ax = plt.subplot(), title='posterior sample')
            # plot_action_tree(self.env.Q_inf, self.env.get_obs()['agent'], self.env.get_obs()['goal'], ax = plt.subplot(), title='DP_inf')

            ## selection, expansion, simulation
            # sim_costs = []
            action_leaf = self.tree_policy()
            initial_sim_cost = self.rollout_policy(action_leaf, real_rollout=True)

            ## loop through future imagined episodes
            future_sim_costs = self.rollout_policy(action_leaf, real_rollout=False, n_futures=n_futures)
            
            ##backup
            sim_costs = [initial_sim_cost, future_sim_costs]
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
        next_state = self.tree.root.action_leaves[action].next_state
        next_root = self.tree.nodes[str(next_state)]

        return action, next_root





## parallel function for simulating many episodes within the same mountain env
def simulate_agent(m, N, params=None, metric='cityblock', true_k=None, r_noise=0.05, n_episodes=10, agents = ['GP', 'MCTS'], n_trees=1000, n_futures=1, exploration_constant=2, discount_factor=0.95):
    
    ## initiate dictionary to store the results
    sim_out = {
        'agent': [],
        'episode': [],
        'mountain': [],
        'start': [],
        'goal': [],
        'true_k': [],
        'actual_cost': [],
        'optimal_cost': [],
        'action_score': [],
        'cost_ratio': [],
        'n_steps': [],
        'actual_trajectory': [],
        'optimal_trajectory': [],
        'observations': [],
        'RPE':[],
        'search_attempts': [],
        'posterior_mean': [],
        'action_tree': [],
        'theta_MLE': []
    }
    
    ## set seed
    seed=m
    seed=os.getpid()
    np.random.seed(seed)
    
    ## create base mountain environment
    kernel_params = params
    env = make_env(N, true_k, kernel_params, metric, r_noise=r_noise)

    ## copy env so that each agent makes its own observations 
    agent_envs = [copy.deepcopy(env) for _ in agents]

    ## initialise the GP-MCTS agent
    # K_inf = env.K_gen.copy()
    K_inf = None
    GP = GPAgent(N, K_inf, env.metric, r_noise)

    ## loop through episodes (i.e. different start and goal states for the same mountain)
    print(' ') # for some reason need this to get the pbar to appear
    for e in tqdm(range(n_episodes), desc='Mountain_'+str(m), position=m+1, leave=False):

        ## reset episode
        # observation, info = env.reset()
        # start = env.get_obs()['agent']
        # current = start
        # goal = env.get_obs()['goal']

        ## loop through agents
        for a, agent in enumerate(agents):
            
            ## reset episode (IDEALLY THIS WOULD HAPPEN OUTSIDE THE AGENT LOOP, SO THAT THE SAME START AND GOAL ARE USED FOR ALL AGENTS)
            # env_copy = copy.deepcopy(agent_envs[a])
            env_copy = agent_envs[a]
            _, _ = env_copy.reset()
            env_copy.set_sim(True)
            start = env_copy.current
            current = start
            goal = env_copy.goal

            ## GP-MCTS agent receives info from env
            if agent =='MCTS':
                # K_inf = env_copy.K_gen.copy()
                # K_inf = None
                # GP = GPAgent(N, K_inf, env.metric, r_noise)
                GP.get_env_info(env_copy)

                ## also updates its kernel weights, given its observations
                GP.update_k_weights(env_copy.obs)
                best_k_idx = np.argmax(GP.k_weights)
                best_theta = GP.k_params[int(best_k_idx)]
                # print('n_obs = ',len(env_copy.obs))
                # print('k_weights = ', np.round(GP.k_weights,2))
                # print('best guess for theta: {}/np.pi'.format(best_theta/np.pi))

            # ## initiate tree 
            # if agent == 'MCTS':
            #     tree = Tree(N)
            #     MCTS = MonteCarloTreeSearch(env=env_copy, tree=tree)
        
            ## run episode until goal is reached
            end_episode = False
            terminated=False
            early_terminate = False
            steps = 0
            max_steps = len(env_copy.o_traj)*2
            max_search_attempts = 3

            while not end_episode:
                

                ## plain balanced GP
                if agent == 'GP':
                    eps = 0.05
                    alpha = 0.4
                    action = env_copy.balanced_policy(current, goal, eps, alpha)

                    ## action
                    observation, _, terminated, truncated, info = env_copy.step(action)
                    current = observation['agent']
                    steps += 1

                    search_attempts = 0 # could do nan

                ## init MCTS (if resetting the tree for each move, init here. otherwise, this should be outside the episode loop)
                # elif agent == 'MCTS':
                #     env_copy.set_sim(True)
                #     tree = Tree(N)
                #     MCTS = MonteCarloTreeSearch(env=env_copy, tree=tree)
                #     assert MCTS.env.sim == True, 'env not in sim mode'
                #     action, next_root = MCTS.search(n_trees)

                #     ## if online planning, take action and update the new root for the next search
                #     observation, _, terminated, truncated, info = env_copy.step(action)
                #     current = observation['agent']
                #     steps += 1
                #   tree.root = next_root
                #   assert np.array_equal(tree.root.state, current), 'error in root update\n env is in: {} but tree is in: {}\n should have taken action {}'.format(current, tree.root.state, action)
                

                ## or if doing offline planning, search and then execute entire trajectory
                elif agent == 'MCTS':
                    non_stuck_route=False
                    search_attempts = 0
                    
                    
                    ## init MCTS
                    tree = Tree(N)
                    MCTS = MonteCarloTreeSearch(env=env_copy, GP=GP, tree=tree, exploration_constant=exploration_constant, discount_factor=discount_factor)
                    assert MCTS.env.sim == True, 'env not in sim mode'
                    while not non_stuck_route:
                        search_attempts += 1


                        ## search
                        n_futures = 1
                        action, next_root = MCTS.search(n_trees, n_futures, progress=True)

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
                            # print(traj_states)

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
                    print('mountain ',m,': episode ',e,' terminated for agent ',agent,' after ',steps,' steps, cost: ', env_copy.a_traj_total_cost)

                    ## reset
                    # observation, info = env.reset()
                    # start = env.get_obs()['agent']
                    # goal = env.get_obs()['goal']
                    # steps = 0
                    # total_cost = 0

                    ## or just skip to the next episode
                    sim_out['agent'].append(agent)
                    sim_out['episode'].append(e)
                    sim_out['mountain'].append(m)
                    sim_out['start'].append(start)
                    sim_out['goal'].append(goal)
                    sim_out['true_k'].append(true_k)
                    sim_out['actual_cost'].append(np.nan)
                    sim_out['optimal_cost'].append(env_copy.o_traj_total_cost)
                    sim_out['action_score'].append(np.nan)
                    sim_out['cost_ratio'].append(np.nan)
                    sim_out['n_steps'].append(steps)
                    sim_out['RPE'].append(np.nan)
                    sim_out['actual_trajectory'].append(env_copy.a_traj)
                    sim_out['optimal_trajectory'].append(env_copy.o_traj)
                    sim_out['observations'].append(env_copy.obs)
                    sim_out['search_attempts'].append(search_attempts)
                    sim_out['posterior_mean'].append(np.nan)
                    sim_out['action_tree'].append(MCTS.tree.action_tree())
                    sim_out['theta_MLE'].append(best_theta)

                    end_episode = True

                ## save data and end the episode
                elif terminated:
                    sim_out['agent'].append(agent)
                    sim_out['episode'].append(e)
                    sim_out['mountain'].append(m)
                    sim_out['start'].append(start)
                    sim_out['goal'].append(goal)
                    sim_out['true_k'].append(true_k)
                    sim_out['actual_cost'].append(env_copy.a_traj_total_cost)
                    sim_out['optimal_cost'].append(env_copy.o_traj_total_cost)
                    # if np.round(env_copy.optimal_cost,4) < np.round(env_copy.accrued_cost,4):
                    #     print(env_copy.optimal_cost, env_copy.accrued_cost)
                    # assert np.round(env_copy.optimal_cost,4) >= np.round(env_copy.accrued_cost,4), 'accrued cost higher than optimal cost'
                    # sim_out['action_score'].append(env_copy.optimal_cost/env_copy.accrued_cost)
                    sim_out['action_score'].append(env_copy.action_score)
                    sim_out['cost_ratio'].append(env_copy.cost_ratio)
                    sim_out['n_steps'].append(steps)
                    sim_out['RPE'].append(np.mean(np.abs(GP.posterior_mean.reshape(N,N) - env_copy.costs)))
                    sim_out['actual_trajectory'].append(env_copy.a_traj)
                    sim_out['optimal_trajectory'].append(env_copy.o_traj)
                    sim_out['observations'].append(env_copy.obs)
                    sim_out['search_attempts'].append(search_attempts)
                    sim_out['posterior_mean'].append(GP.posterior_mean)
                    sim_out['action_tree'].append(MCTS.tree.action_tree())
                    sim_out['theta_MLE'].append(best_theta)
                    
                    ## update the agent env
                    # agent_envs[a] = copy.deepcopy(env_copy)
                    agent_envs[a] = env_copy

                    end_episode = True


    return sim_out, env_copy.costs