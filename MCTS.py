import random
from math import sqrt, log
from utils import Node, Tree, make_env
import copy
import numpy as np

class MonteCarloTreeSearch():

    def __init__(self, env, tree):
        self.env = env
        # self.env.sim=True
        self.tree = tree
        self.action_space = self.env.action_space.n
        self.N = self.env.N

        ## get initial state and goal 
        observation = self.env.get_obs()
        state = observation['agent']
        self.tree.add_node(Node(state=state, action=None, action_space=self.action_space, reward=0, terminal=False, N=self.N))

    def expand(self, node):
        env_copy = copy.deepcopy(self.env)
        env_copy.set_state(node.state)
        env_copy.set_sim(True)
        action = node.untried_action()
        observation, reward, terminated, truncated, info = env_copy.step(action)
        state=observation['agent'] 
        new_node = Node(state=state, action=action, action_space=self.action_space, reward=reward, terminal=terminated, N=self.N) 
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
        if node.terminal:
            return -node.reward

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
            

    ## calculate E-E value
    def compute_UCT(self, parent, child, exploration_constant): # could turn this exploration constant into a param defined at init
        exploitation_term = child.total_simulation_reward / child.num_visits
        exploration_term = exploration_constant * sqrt(2 * log(parent.num_visits) / child.num_visits)
        return exploitation_term + exploration_term

    
    ## argmax based on UCT values? 
    def best_child(self, node, exploration_constant):
        children = self.tree.children(node)
        best_child = max(children, key=lambda child: self.compute_UCT(node, child, exploration_constant))
        return best_child
    def second_best_child(self, node, exploration_constant):
        children = self.tree.children(node)
        best_child = max(children, key=lambda child: self.compute_UCT(node, child, exploration_constant))
        children.remove(best_child)
        second_best_child = max(children, key=lambda child: self.compute_UCT(node, child, exploration_constant))
        return second_best_child

    ## take an action according to the tree policy, i.e. take the best UCT child and see where it takes you
    def tree_policy(self):
        env_copy = copy.deepcopy(self.env)
        env_copy.set_sim(True)
        node = self.tree.root
        t = 0
        while not node.terminal:
            if self.tree.is_expandable(node):
                # print('expanding ', node) ## if this is a node chosen by UCT, u should expand from the *state* node, not the *state-action* node??
                return self.expand(node) 
                
            else:
                state_tmp = node.state
                # print('fully expanded: ', node)
                node = self.best_child(node, exploration_constant=1.0/sqrt(2.0))
                # print('best child is ',node)
                # print()
                # observation, reward, terminated, _, _ = self.env.step(node.action) ## I think this needs to actually change the state, bc u need to simulate what happens later.
                observation, reward, terminated, _, _ = env_copy.step(node.action)
                state = observation['agent']
                if not np.array_equal(node.state, state):
                    print(node.state, node.action, state)
                assert np.array_equal(node.state, state)

                # if np.all(state_tmp == state):
                #     print('same state')


        return node

    
    ## backup rewards until you reach the root
    def backward(self, node, value):
        while node:
            node.num_visits += 1
            node.total_simulation_reward += value 
            node.performance = node.total_simulation_reward / node.num_visits
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
        action = np.nanargmax(MCTS_estimates)
        ## check if this action takes the agent back to the previous state
        return action





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
    
    ## create mountain environment
    env = make_env(N, None, metric, true_k, inf_k, render_mode=None, r_noise=r_noise)

    ## loop through episodes (i.e. different start and goal states for the same mountain)
    for e in range(n_episodes):

        ## reset episode
        observation, info = env.reset()
        start = env.get_obs()['agent']
        current = start
        goal = env.get_obs()['target']

        ## loop through agents
        for agent in agents:

            ## create inner loop copy of env
            env_copy = copy.deepcopy(env)
        
            ## run episode until goal is reached
            end_episode = False
            terminated=False
            steps = 0

            while not end_episode:
                
                ## init MCTS
                if agent == 'MCTS':
                    env_copy.set_sim(True)
                    tree = Tree()
                    MCTS = MonteCarloTreeSearch(env=env_copy, tree=tree)
                    assert MCTS.env.sim == True
                    assert env_copy.sim==True
                    action = MCTS.search(tree, n_trees)

                ## otherwise, plain balanced GP
                elif agent == 'GP':
                    eps = 0.05
                    alpha = 0.25
                    action = env_copy.balanced_policy(current, goal, eps, alpha)
                    
                
                ## take action
                env_copy.set_sim(False)
                # print(current, goal, action)
                observation, _, terminated, truncated, info = env_copy.step(action)
                assert ~np.array_equal(current,observation['agent'])
                current = observation['agent']
                steps += 1

                ## prevent endless episode 
                if steps > 50:
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
                    end_episode = True

    return sim_out, env.costs