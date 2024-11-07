import random
from math import sqrt, log
from utils import Node, Action_Node, Tree, make_env, argm
import copy
import numpy as np
from tqdm.auto import tqdm


class MonteCarloTreeSearch():

    def __init__(self, env, tree):
        self.env = env
        self.tree = tree
        self.action_space = self.env.action_space.n
        self.N = self.env.N
        self.exploration_constant = 1.0/sqrt(2.0)

        ## visit counts??
        # self.n_state_visits = np.zeros((self.N, self.N))

        ## get initial state and goal 
        observation = self.env.get_obs()
        state = observation['agent']

        ## (AND THE STARTING COST?)
        starting_cost = self.env.costs[state[0], state[1]]

        # self.tree.add_node(Node(state=state, cost=starting_cost, terminated=False, action_space=self.action_space, N=self.N))
        # self.tree.add_node(Node(state=state, action=None, action_space=self.action_space, cost=0, terminated=False, N=self.N))
        
        ## add state node to the tree
        self.tree.add_state_node(state=state, cost=starting_cost, terminated=False, action_space = self.action_space, parent=None)

    
    ## expand the action space of a node
    def expand(self, node):

        ## create copy of env and set state
        env_copy = copy.deepcopy(self.env)
        env_copy.set_state(node.state)
        env_copy.set_sim(True)

        ## take action and get new state
        action = node.untried_action()
        observation, cost, terminated, truncated, info = env_copy.step(action)
        next_state = observation['agent']

        ## update info for s-a leaf - i.e. the state-action pair, and the cost of the state that you subsequently reach
        costs_tmp = np.array([cost, 0])
        cost_tmp = costs_tmp[np.clip(terminated,0,1)] # i.e. cost is 0 if the episode is terminated
        # action_leaf = Node(state=node.state, action=action, next_state = next_state, action_space=self.action_space, cost=cost_tmp, terminated=terminated, N=self.N) 
        # self.tree.add_node(action_leaf, node)

        node.action_leaves[action] = Action_Node(prev_state = node.state, action=action, next_state = next_state, next_cost=cost_tmp, terminated=terminated)

        return node.action_leaves[action]
    
    ## take an action according to the tree policy, i.e. take the best UCT child and see where it takes you
    def tree_policy(self, root = None):

        ## copy env to simulate the tree search
        env_copy = copy.deepcopy(self.env)
        env_copy.set_sim(True)
        if root is None:
            node = self.tree.root ## need some way of setting the current node to the current state
        t = 0

        ## create a record of the nodes/leaves visited
        self.tree_path = []
        terminated=False
        while not terminated:
            t+=1

            ## expansion step
            if self.tree.is_expandable(node):
                action_leaf = self.expand(node)
                self.tree_path.append(tuple([node.state, action_leaf.action]))
                # print('expanding ',node.state,' by taking action ',action_leaf.action,' to ',action_leaf.next_state)
                return action_leaf
                
            ## selection step
            else:
                state_tmp = node.state
                # env_state_tmp = env_copy.get_obs()['agent'] ## DEBUG LINE
                action_leaf = self.best_child(node)
                # print(node.state,'not expandable, so we choose action ',action_leaf.action,' to ',action_leaf.next_state)
                self.tree_path.append(tuple([node.state, action_leaf.action]))
                observation, cost, terminated, _, _ = env_copy.step(action_leaf.action)
                next_state = observation['agent']

                ## DEBUGBING 
                # if not np.array_equal(action_leaf.next_state, next_state):
                #     print('started in ', state_tmp)
                #     print('supposed to take action ', action_leaf.action, ' to ', action_leaf.next_state)
                #     print('ended up moving from ',env_state_tmp,' to', next_state)
                # assert np.array_equal(action_leaf.next_state, next_state)

                ## create next state node (if it doesn't already exist)
                node = self.tree.add_state_node(next_state, action_leaf.next_cost, terminated, action_space = self.action_space, parent=action_leaf)
                assert np.array_equal(node.state, action_leaf.next_state)

            # if t > 10:
            #     print(self.tree_path)
            #     print(node)
                # print(node)
                # print(self.tree.is_expandable(node))
            if t>100:
                # print('selection taking too long')
                terminated=True
                

        return action_leaf

    def rollout_policy(self, action_leaf):

        ## init
        total_cost = 0
        discount_factor = 1
        gamma = 1
        max_rolls = 1000
        rolls = 0
        
        ## set the state from which the rollout is initiated
        env_copy = copy.deepcopy(self.env)
        env_copy.set_state(action_leaf.next_state)
        env_copy.set_sim(True)
        info = env_copy.get_obs()
        target = info['target']
        current = info['agent']

        ## begin with the cost of the state that you have just reached (necessary, since otherwise the cost of this initial state is not included in the total cost)
        starting_cost = action_leaf.next_cost
        total_cost += starting_cost

        ## rolling out from goal location, 0 cost
        if action_leaf.terminated:
            return 0
            # return -action_leaf.next_cost

        ## rollout until trial is terminated 
        while True:

            ## uniform random
            # action = random.randint(0, self.action_space-1)

            ## or, greedy
            current = env_copy.get_obs()['agent']
            action = env_copy.greedy_policy(current, target, eps = 0.0)

            ## or, balanced greedy
            # current = env_copy.get_obs()['agent']
            # target = env_copy.get_obs()['target']
            # action = env_copy.balanced_policy(current, target, eps = 0.05, alpha = 0.1)

            ## take action
            _, cost, terminated, _, _ = env_copy.step(action)

            ## prevent infinite rollout
            rolls += 1
            if rolls > max_rolls:
                # print('exceeded max rolls')
                terminated = True

            ## check if terminated
            if terminated:
                return -total_cost
            
            ## if not terminated, add cost to total cost (i.e. terminal state does not incur cost)
            total_cost += cost * discount_factor
            discount_factor *= gamma
            
            # ## IF YOU WANT THE COST OF THE FINAL STATE, THIS SHOULD COME EARLIER
            # total_cost += cost * discount_factor
            # discount_factor *= gamma
            

    ## calculate E-E value
    def compute_UCT(self, node, action_leaf): 
        # exploitation_term = child.total_simulation_cost / child.n_visits
        # exploration_term = exploration_constant * sqrt(2 * log(parent.n_visits) / child.n_visits)
        exploitation_term = action_leaf.performance
        exploration_term = self.exploration_constant * sqrt(log(node.n_state_visits) / action_leaf.n_action_visits)
        return exploitation_term + exploration_term

    
    ## argmax based on UCT values? 
    def best_child(self, node):
        # children = self.tree.children(node)
        # UCTs = [self.compute_UCT(node, child, exploration_constant) for child in children]
        # max_UCT = np.max(UCTs)
        # max_idx = argm(UCTs, max_UCT)
        # best_child = children[max_idx]
        # assert self.compute_UCT(node, best_child, exploration_constant) == max_UCT
        # return best_child

        ## get action children
        action_leaves = [node.action_leaves[a] for a in node.action_leaves.keys()]

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
        # print('UTCs for ',node.state,':',UCTs,', so we choose action ',max_idx)
        best_child = action_leaves[max_idx]
        assert self.compute_UCT(node, best_child) == max_UCT
        return best_child
        
    
    ## backup costs until you reach the root
    def backward(self, cost):

        ## loop through the tree path
        for state, action in self.tree_path:

            ## get the state node and action leaf
            state_node = self.tree.nodes[str(state)]
            action_leaf = state_node.action_leaves[action]

            ## update counts and performance
            state_node.n_state_visits += 1
            action_leaf.n_action_visits += 1
            action_leaf.total_simulation_cost += cost
            action_leaf.performance = action_leaf.total_simulation_cost / action_leaf.n_action_visits

    # def forward(self):
    #     self._forward(self.tree.root)

    # def _forward(self,node):
    #     best_child = self.best_child(node)
        # print("****** {} ******".format(best_child.state))

        # for child in self.tree.children(best_child):
        #     print("{}: {:0.4f}".format(child.state, child.performance))

        # if len(self.tree.children(best_child)) > 0:
        #     self._forward(best_child)

        ## return the best action

    ## tree search --> action loop
    def search(self, tree, n_trees=1000):

        ## loop through trees
        for t in range(n_trees):
            action_leaf = self.tree_policy()
            simulated_cost = self.rollout_policy(action_leaf)
            self.backward(simulated_cost)
        
        ## get simulated LT costs of adjacent states
        # current_action_leaves = self.tree.children(self.tree.root)
        current_action_leaves = self.tree.root.action_leaves
        MCTS_estimates = np.zeros(4)+np.nan
        for action in current_action_leaves.keys():
            MCTS_estimates[action] = current_action_leaves[action].performance

        ## action selection
        if np.isnan(np.sum(MCTS_estimates)):
            print(self.tree.root)
        max_MCTS = np.nanmax(MCTS_estimates)
        action = argm(MCTS_estimates, max_MCTS)
        assert MCTS_estimates[action] == np.nanmax(MCTS_estimates)
        print(self.tree.root.state, MCTS_estimates, action)
        ## check if this action takes the agent back to the previous state

        ## new root is the state that the agent has just reached
        # next_node = self.best_child(self.tree.root, exploration_constant=0)
        next_state = self.tree.root.action_leaves[action].next_state
        next_node = self.tree.nodes[str(next_state)]
        return action, next_node





## parallel function for simulating many episodes within the same mountain env
def parallel_agent(m, N, params=None, metric='cityblock', true_k=None, inf_k='known', r_noise=0.05, render_mode=None, n_episodes=50, agents = ['GP', 'MCTS'], n_trees=1000):
    
    ## initiate dictionary to store the results
    sim_out = {
        'agent': [],
        'episode': [],
        'mountain': [],
        'start': [],
        'goal': [],
        'true_k': [],
        'inf_k': [],
        'accrued_cost': [],
        'optimal_cost': [],
        'score': [],
        'n_steps': [],
        'actual_trajectory': [],
        'optimal_trajectory': [],
        'observations': [],
        'RPE':[],
    }
    
    ## create base mountain environment
    env = make_env(N, None, metric, true_k, inf_k, render_mode=None, r_noise=r_noise)

    ## copy env so that each agent makes its own observations 
    agent_envs = [copy.deepcopy(env) for _ in agents]

    ## loop through episodes (i.e. different start and goal states for the same mountain)
    print() # for some reason need this to get the pbar to appear
    for e in tqdm(range(n_episodes), desc='Mountain_'+str(m), position=m+1, leave=False):

        ## reset episode
        # observation, info = env.reset()
        # start = env.get_obs()['agent']
        # current = start
        # goal = env.get_obs()['target']

        ## loop through agents
        for a, agent in enumerate(agents):
            
            ## reset episode (IDEALLY THIS WOULD HAPPEN OUTSIDE THE AGENT LOOP, SO THAT THE SAME START AND GOAL ARE USED FOR ALL AGENTS)
            # env_copy = copy.deepcopy(agent_envs[a])
            env_copy = agent_envs[a]
            observation, _ = env_copy.reset()
            start = observation['agent']
            current = start
            goal = observation['target']

            ## initiate tree 
            tree = Tree(N)
            MCTS = MonteCarloTreeSearch(env=env_copy, tree=tree)

        
            ## run episode until goal is reached
            end_episode = False
            terminated=False
            steps = 0

            while not end_episode:
                
                ## init MCTS
                if agent == 'MCTS':
                    env_copy.set_sim(True)
                    # tree = Tree()
                    # MCTS = MonteCarloTreeSearch(env=env_copy, tree=tree)
                    assert MCTS.env.sim == True
                    assert env_copy.sim==True
                    action, next_node = MCTS.search(tree, n_trees)

                ## otherwise, plain balanced GP
                elif agent == 'GP':
                    eps = 0.05
                    alpha = 0.4
                    action = env_copy.balanced_policy(current, goal, eps, alpha)
                    
                
                ## take action
                env_copy.set_sim(False)
                # print(current, goal, action)
                observation, _, terminated, truncated, info = env_copy.step(action)
                assert ~np.array_equal(current,observation['agent'])
                current = observation['agent']
                steps += 1


                ## prevent endless episode 
                if steps >= 50:
                    print('episode ',e,' terminated in mountain ',m,' for agent ',agent,', cost: ', env_copy.accrued_cost)

                    ## reset
                    # observation, info = env.reset()
                    # start = env.get_obs()['agent']
                    # goal = env.get_obs()['target']
                    # steps = 0
                    # total_cost = 0

                    ## or just skip to the next episode
                    sim_out['agent'].append(agent)
                    sim_out['episode'].append(e)
                    sim_out['mountain'].append(m)
                    sim_out['start'].append(start)
                    sim_out['goal'].append(goal)
                    sim_out['true_k'].append(true_k)
                    sim_out['inf_k'].append(inf_k)
                    sim_out['accrued_cost'].append(np.nan)
                    sim_out['optimal_cost'].append(env_copy.optimal_cost)
                    sim_out['score'].append(np.nan)
                    sim_out['n_steps'].append(steps)
                    sim_out['RPE'].append(np.nan)
                    sim_out['actual_trajectory'].append(env_copy.a_traj)
                    sim_out['optimal_trajectory'].append(env_copy.o_traj)
                    sim_out['observations'].append(env_copy.obs)
                    end_episode = True

                ## save data and end the episode
                elif terminated:
                    sim_out['agent'].append(agent)
                    sim_out['episode'].append(e)
                    sim_out['mountain'].append(m)
                    sim_out['start'].append(start)
                    sim_out['goal'].append(goal)
                    sim_out['true_k'].append(true_k)
                    sim_out['inf_k'].append(inf_k)
                    # sim_out['accrued_cost'].append(total_cost)
                    sim_out['accrued_cost'].append(env_copy.accrued_cost)
                    sim_out['optimal_cost'].append(env_copy.optimal_cost)
                    if np.round(env_copy.optimal_cost,4) > np.round(env_copy.accrued_cost,4):
                        print(env_copy.optimal_cost, env_copy.accrued_cost)
                    assert np.round(env_copy.optimal_cost,4) <= np.round(env_copy.accrued_cost,4)
                    sim_out['score'].append(env_copy.optimal_cost/env_copy.accrued_cost)
                    sim_out['n_steps'].append(steps)
                    sim_out['RPE'].append(np.mean(np.abs(env_copy.posterior_mean.reshape(N,N) - env_copy.costs)))
                    sim_out['actual_trajectory'].append(env_copy.a_traj)
                    sim_out['optimal_trajectory'].append(env_copy.o_traj)
                    sim_out['observations'].append(env_copy.obs)
                    
                    ## update the agent env
                    # agent_envs[a] = copy.deepcopy(env_copy)
                    agent_envs[a] = env_copy
                    # print(agent, len(agent_envs[a].obs))
                    # print(agent, agent_envs[a].posterior_mean)
                    # print(agent, env_copy.a_traj, env_copy.obs[:,1], env_copy.obs[:,2])

                    end_episode = True

                ## new root for the next search
                if agent == 'MCTS':
                    tree.root = next_node


    return sim_out, env.costs