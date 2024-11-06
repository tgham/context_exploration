import random
from math import sqrt, log
from utils import Node, Tree, make_env, argm
import copy
import numpy as np
from tqdm.auto import tqdm


class MonteCarloTreeSearch():

    def __init__(self, env, tree):
        self.env = env
        # self.env.sim=True
        self.tree = tree
        self.action_space = self.env.action_space.n
        self.N = self.env.N

        ## visit counts??
        # self.n_state_visits = np.zeros((self.N, self.N))

        ## get initial state and goal 
        observation = self.env.get_obs()
        state = observation['agent']

        ## (AND THE STARTING COST?)
        starting_cost = self.env.costs[state[0], state[1]]

        self.tree.add_node(Node(state=state, action=None, action_space=self.action_space, cost=starting_cost, terminal=False, N=self.N))
        # self.tree.add_node(Node(state=state, action=None, action_space=self.action_space, cost=0, terminal=False, N=self.N))

    def expand(self, node):
        env_copy = copy.deepcopy(self.env)
        env_copy.set_state(node.state)
        env_copy.set_sim(True)
        action = node.untried_action()
        observation, cost, terminated, truncated, info = env_copy.step(action)
        state=observation['agent']
        if not terminated:
            new_node = Node(state=state, action=action, action_space=self.action_space, cost=cost, terminal=terminated, N=self.N) 
        else:
            new_node = Node(state=state, action=action, action_space=self.action_space, cost=0, terminal=terminated, N=self.N)
        self.tree.add_node(new_node, node)
        # print('expanded from ', node, ' to ', new_node)
        return new_node

    def rollout_policy(self, node):

        ## init
        total_cost = 0
        discount_factor = 1
        gamma = 1
        max_rolls = 1000
        rolls = 0
        
        ## set the state from which the rollout is initiated
        env_copy = copy.deepcopy(self.env)
        env_copy.set_state(node.state)
        env_copy.set_sim(True)

        ## (BEGIN WITH COST OF CURRENT STATE?)
        # start = env_copy.get_obs()['agent']
        # total_cost += env_copy.costs[start[0], start[1]]

        if node.terminal:
            return -node.cost

        ## rollout until trial is terminated 
        while True:

            ## uniform random
            # action = random.randint(0, self.action_space-1)

            ## or, greedy
            current = env_copy.get_obs()['agent']
            target = env_copy.get_obs()['target']
            action = env_copy.greedy_policy(current, target, eps = 0.0)

            ## or, balanced greedy
            # current = env_copy.get_obs()['agent']
            # target = env_copy.get_obs()['target']
            # action = env_copy.balanced_policy(current, target, eps = 0.05, alpha = 0.1)

            ## take action and get cost
            _, cost, terminated, _, _ = env_copy.step(action)
            total_cost += cost * discount_factor
            discount_factor *= gamma

            ## prevent infinite rollout
            rolls += 1
            if rolls > max_rolls:
                # print('exceeded max rolls')
                terminated = True

            ## check if terminated
            if terminated:
                return -total_cost
            
            
            # ## IF YOU WANT THE COST OF THE FINAL STATE, THIS SHOULD COME EARLIER
            # total_cost += cost * discount_factor
            # discount_factor *= gamma
            

    ## calculate E-E value
    def compute_UCT(self, parent, child, exploration_constant): # could turn this exploration constant into a param defined at init
        # exploitation_term = child.total_simulation_cost / child.n_visits
        # exploration_term = exploration_constant * sqrt(2 * log(parent.n_visits) / child.n_visits)
        exploitation_term = child.performance
        exploration_term = exploration_constant * sqrt(log(self.tree.n_state_visits[parent.state[0], parent.state[1]]) / child.n_visits)
        return exploitation_term + exploration_term

    
    ## argmax based on UCT values? 
    def best_child(self, node, exploration_constant):
        children = self.tree.children(node)
        UCTs = [self.compute_UCT(node, child, exploration_constant) for child in children]
        max_UCT = np.max(UCTs)
        max_idx = argm(UCTs, max_UCT)
        best_child = children[max_idx]
        assert self.compute_UCT(node, best_child, exploration_constant) == max_UCT
        # best_child = max(children, key=lambda child: self.compute_UCT(node, child, exploration_constant))
        return best_child

    ## take an action according to the tree policy, i.e. take the best UCT child and see where it takes you
    def tree_policy(self, root = None):
        env_copy = copy.deepcopy(self.env)
        env_copy.set_sim(True)
        if root is None:
            node = self.tree.root ## need some way of setting the current node to the current state
        t = 0
        while not node.terminal:
            if self.tree.is_expandable(node):
                # print('expanding ', node) ## if this is a node chosen by UCT, u should expand from the *state* node, not the *state-action* node??
                return self.expand(node) 
                
            else:
                state_tmp = node.state
                env_state_tmp = env_copy.get_obs()['agent']
                # print('fully expanded: ', node)
                node = self.best_child(node, exploration_constant=1.0/sqrt(2.0))
                observation, cost, terminated, _, _ = env_copy.step(node.action)
                state = observation['agent']
                if not np.array_equal(node.state, state):
                    print('started in ', state_tmp)
                    print('supposed to take action ', node.action, ' to ', node.state)
                    print('ended up moving from ',env_state_tmp,' to', state)
                assert np.array_equal(node.state, state)

                # if np.all(state_tmp == state):
                #     print('same state')


        return node

    
    ## backup costs until you reach the root
    def backward(self, node, cost):
        while node:
            # node.n_visits += 1
            # self.n_state_visits[node.state[0], node.state[1]] += 1
            # node.total_simulation_cost += cost
            # node.performance = node.total_simulation_cost / node.n_visits
            # node = self.tree.parent(node)
            node.n_visits += 1
            self.tree.n_state_visits[node.state[0], node.state[1]] += 1
            node.performance = node.performance + (cost - node.performance) / node.n_visits
            node = self.tree.parent(node)



    def forward(self):
        self._forward(self.tree.root)

    def _forward(self,node):
        best_child = self.best_child(node, exploration_constant=0)
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
            node = self.tree_policy()
            simulated_cost = self.rollout_policy(node)
            self.backward(node, simulated_cost)
        
        ## get simulated LT costs of adjacent states
        current_children = self.tree.children(self.tree.root)
        MCTS_estimates = np.zeros(4)+np.nan
        for child in current_children:
            MCTS_estimates[child.action] = child.performance

        ## action selection
        if np.isnan(np.sum(MCTS_estimates)):
            print(self.tree.root)
        max_MCTS = np.nanmax(MCTS_estimates)
        action = argm(MCTS_estimates, max_MCTS)
        assert MCTS_estimates[action] == np.nanmax(MCTS_estimates)
        ## check if this action takes the agent back to the previous state

        ## new root is the state that the agent has just reached
        next_node = self.best_child(self.tree.root, exploration_constant=0)
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
            env_copy = agent_envs[a]
            observation, info = env_copy.reset()
            start = env_copy.get_obs()['agent']
            current = start
            goal = env_copy.get_obs()['target']

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
                    agent_envs[a] = copy.deepcopy(env_copy)

                    end_episode = True

                ## new root for the next search
                if agent == 'MCTS':
                    tree.root = next_node


    return sim_out, env.costs