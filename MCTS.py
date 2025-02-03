import random
from math import sqrt, log
from utils import Node, Action_Node, Episode_Node, Episode_Action_Node, Tree, make_env, argm, data_keys, KL_divergence, get_next_state
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
                state_tmp = node.state[:2]
                # env_state_tmp = self.env.get_obs()['agent']
                # env_state_tmp = self.env.current

                ## get the best child
                action_leaf = self.best_child(node)
                self.tree_path.append(tuple([node.state, action_leaf.action]))
                self.node_id_path.append(node.node_id)

                ## move in env
                next_state, cost, terminated, _, _ = self.env.step(action_leaf.action)
                self.tree_costs.append(cost)

                ## or, do this without updating the environment object
                # next_state = np.clip(node.state[:2] + self.env.action_to_direction[action_leaf.action],
                #                 0, self.N - 1)
                # direction = self.env.action_to_direction[action_leaf.action]
                # next_state = get_next_state(node.state[:2], direction, self.N)
                # cost = self.env.get_pred_cost(next_state)
                # terminated = action_leaf.terminated
                # self.tree_costs.append(cost)


                ## see if the next state node already exists as a child of this action leaf
                state_id = tuple(np.append(next_state, cost))
                if state_id in action_leaf.children:
                    node = action_leaf.children[state_id]
                else:
                    node = self.tree.add_state_node(next_state, cost, terminated, action_space = self.action_space, parent=action_leaf)
                # assert np.array_equal(node.state[:2], next_state), 'error in tree policy step {}\n started in {}\n supposed to take action {} to {}\n ended up moving from {} to {}'.format(t, state_tmp, action_leaf.action, node.state[:2], env_state_tmp, action_leaf.next_state)
                assert np.array_equal(node.state[:2], next_state), 'error in tree policy step {}\n started in {}\n supposed to take action {} to {}\n ended up moving  to {}'.format(t, state_tmp, action_leaf.action, node.state[:2], action_leaf.next_state)

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
            assert np.array_equal(node.state[:2], state[:2]), 'error in tree path\n node: {} \n state: {}'.format(node.state[:2], state[:2])

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

        return action, MCTS_estimates






class MonteCarloTreeSearch_2AFC():


    def __init__(self, env, agent, tree, exploration_constant=2, discount_factor=0.99):
        self.env = env
        self.actual_state = self.env.current
        self.actual_goal = self.env.goal
        self.agent = agent
        self.tree = tree
        self.n_AFC = 2
        self.N = self.env.N
        self.episode = self.env.episode
        self.exploration_constant = exploration_constant
        self.discount_factor = discount_factor

        ## cost of current state
        current_cost = self.env.costs[self.actual_state[0], self.actual_state[1]]
        prev_costs = None

        ## add state node to the tree
        self.tree.add_episode_node(start = self.actual_state, goal = self.actual_goal, episode=self.episode, current_cost=current_cost, prev_costs=prev_costs, n_AFC=self.n_AFC, parent=None)


    
    ## expand
    def expand(self, node):

        ## select an untried path
        path_id = node.untried_action()
        actions = self.env.moves[node.episode][path_id]
        # assert self.env.sim
        # next_states = []
        # costs = []
        # for action in actions:
        #     next_state, cost, terminated, truncated, _ = self.env.step(action)
        #     next_states.append(next_state)
        #     costs.append(cost)
        
        ## update info for s-a leaf - i.e. the set of states and costs arising from this path
        # node.action_leaves[move_id] = Episode_Action_Node(prev_state = None, action=move_id, next_state = None, terminated=None)
        episode_terminated = node.episode == self.env.n_episodes
        node.action_leaves[path_id] = Episode_Action_Node(path_id = path_id, episode = node.episode, episode_terminated=episode_terminated)
        node.action_leaves[path_id].performance = 0
        # self.env.set_state(self.actual_state)

        return node.action_leaves[path_id]
    
    ## tree policy
    def tree_policy(self):

        assert self.env.sim, 'env is not in sim mode'

        ## initialise the tree
        node = self.tree.root
        starting_cost = node.current_cost
        t = 0
        episode = self.env.episode
        assert episode == self.tree.root.episode, 'episode mismatch between env and tree\n env: {} \n tree: {}'.format(episode, self.tree.root.episode)
        start = node.start
        goal = node.goal
        self.tree_costs = []
        # assert np.array_equal(node.state[:2], self.actual_state), 'mismatch between node and env state\n node: {} \n env: {}'.format(node, self.actual_state)

        ## create a record of the nodes/leaves visited in the tree
        self.tree_path = []
        self.node_id_path = []
        
        ## loop until you reach a leaf node or terminal state
        while not node.terminated:
            t+=1

            ## expansion step
            if self.tree.is_expandable(node):
                action_leaf = self.expand(node)
                # self.tree_path.append(tuple([node.state, action_leaf.action]))
                # self.tree_path.append(action_leaf.path_id)
                self.tree_path.append(tuple(node.node_id, action_leaf.path_id))
                self.node_id_path.append(node.node_id)

                ## update counts already?
                action_leaf.n_action_visits += 1
                node.n_state_visits += 1

                return action_leaf
                
            ## selection step
            else:

                ## get the best child
                action_leaf = self.best_child(node)
                # self.tree_path.append(tuple([node.state, action_leaf.action]))
                # self.tree_path.append(action_leaf.path_id)
                self.tree_path.append(tuple(node.node_id, action_leaf.path_id))
                self.node_id_path.append(node.node_id)

                ## move in env
                move_id = action_leaf.action
                actions = self.env.moves[node.episode][move_id]
                costs = []
                next_states = []
                for action in actions:
                    next_state, cost, terminated, _, _ = self.env.step(action)
                    next_states.append(next_state)
                    costs.append(cost)
                self.tree_costs.append(np.sum(costs)) ## might choose to discount this

                ## see if the next state node already exists as a child of this action leaf
                episode +=1 ## i.e. having chosen a path, we now progress to the next episode
                episode_id = tuple(costs, episode)
                episode_terminated = episode == self.env.n_episodes
                if episode_id in action_leaf.children:
                    node = action_leaf.children[episode_id]
                else:
                    node = self.tree.add_episode_node(start, goal, episode, costs, self.n_AFC, parent=action_leaf)


                ## update counts already?
                action_leaf.n_action_visits += 1
                node.n_state_visits += 1

                ## next node
                node = action_leaf.children[None]

        ## revert env
        self.env.set_state(self.actual_state)

        return action_leaf
    

    ## rollout policy
    def rollout_policy(self, episode):

        ## if final episode, just stop here
        if episode==self.env.n_episodes:
            return 0
        
        ## OPTIMISED: get the total cost of the two paths and return the better one
        for path_id in range(self.n_AFC):
            path_states = self.env.path_states[episode][path_id]
            ro_costs = []
            total_cost = 0
            for state in path_states:
                cost = self.env.get_pred_cost(state)
                total_cost += cost ## NEED TO DISCOUNT HERE!!!??
            ro_costs.append(total_cost)
        ro_cost = np.max(ro_costs)

        ## GREEDY: randomly choose between the two paths
        # path_id = np.random.choice(self.n_AFC)
        # path_states = self.env.path_states[action_leaf.episode+1][path_id]
        # ro_cost = 0
        # for state in path_states:
        #     cost = self.env.get_pred_cost(state)
        #     ro_cost += cost

        return ro_cost 
    
    ## backup costs to root
    def backward(self, sim_costs):

        ## calculate discount factors
        tree_len = len(self.tree_costs)
        discount_factors = [self.discount_factor**d for d in range(tree_len)]
        node = self.tree.root

        ## loop through the tree path
        for depth, (node_id, path_id) in enumerate(self.tree_path):

            ## get the state node and action leaf
            action_leaf = node.action_leaves[path_id]

            ## discounted costs from current node to rollout node
            dist_to_rollout = tree_len - depth
            discounted_tree_cost = np.dot(self.tree_costs[depth:depth + dist_to_rollout], discount_factors[:dist_to_rollout])

            ## calculate cost of the rollout, discounted by the distance from the current node to the rollout node
            first_sim_cost = sim_costs[0] * self.discount_factor**dist_to_rollout
            backup_cost = first_sim_cost + discounted_tree_cost

            ## backup + update counts
            action_leaf.performance = action_leaf.performance + (backup_cost - action_leaf.performance) / action_leaf.n_action_visits

            ## next node
            if depth < len(self.tree_path)-1:
                node = action_leaf.children[self.tree_path[depth+1][0]]



            
    

    
    ## argmax based on UCT values? SAME AS BEFORE, EXCEPT FOR THE CHECK FOR BACKTRACKING
    def best_child(self, node):
    
        ## get action children
        action_leaves = [node.action_leaves[a] for a in node.action_leaves.keys()]

        ## calculate UCT for each action leaf
        UCTs = [self.compute_UCT(node, leaf) for leaf in action_leaves]
        max_UCT = np.max(UCTs)
        max_idx = argm(UCTs, max_UCT)
        best_child = action_leaves[max_idx]

        return best_child
    
    ## calculate E-E value (SAME AS BEFORE)
    def compute_UCT(self, node, action_leaf): 
        exploitation_term = action_leaf.performance 
        assert action_leaf.n_action_visits > 0 or action_leaf.terminated, 'action leaf has not been visited: {}'.format(action_leaf)
        exploration_term = self.exploration_constant * sqrt(log(node.n_state_visits) / action_leaf.n_action_visits)
        return exploitation_term + exploration_term

