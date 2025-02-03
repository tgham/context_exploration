from enum import Enum
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register, registry
import pygame
import numpy as np
from plotter import *
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
import scipy
from scipy.spatial.distance import cdist
from scipy.special import softmax
import warnings
import heapq
from collections import defaultdict
from IPython.display import display, clear_output
import uuid
import random
from collections import deque
from minimax_tilting_sampler import TruncatedMVN
import ast
from scipy.spatial import cKDTree as KDTree
import cProfile
import pstats
import subprocess
import time
from numba import jit, njit




## create a mountain environment
def make_env(N, n_episodes, expt, beta_params, metric, seed=None):

    ## register env
    
    # Unregister the environment if it's already registered
    env_id = "mountains/MountainEnv-v0"
    if env_id in registry:
        del registry[env_id]

    # Re-register the updated environment
    register(
        id=env_id,
        entry_point='mountains.envs:MountainEnv',
        max_episode_steps=100,
        kwargs={"size": N},
    )
    
    env = gym.make("mountains/MountainEnv-v0", N=N, n_episodes=n_episodes, expt=expt, beta_params=beta_params, metric=metric, seed=seed)
    return env




## Node class
class Node:

    # __slots__ = ['state', 'n_state_visits', 'cost', 'terminated', 'node_id', 'parent_node_ids', 'N', 'untried_actions', 'action_leaves']

    def __init__(self, state, cost, terminated, action_space, N):
        
        ## state info
        self.state = np.append(state, cost)
        self.n_state_visits = 0
        self.cost = cost
        self.terminated = terminated
        # self.node_id = str(np.append(self.history, self.state))
        self.node_id = tuple(state)
        self.state_id = tuple(state)
        self.parent_node_ids = []
        # self.children_node_ids = []
        self.N = N


        ## define valid actions
        self.untried_actions = list(range(action_space))
        row, col,_ = self.state
        if row == self.N-1:
            self.untried_actions.remove(0)
        if row == 0:
            self.untried_actions.remove(2)
        if col == self.N-1:
            self.untried_actions.remove(1)
        if col == 0:
            self.untried_actions.remove(3)

        ## action leaves
        self.action_leaves = {a: None for a in self.untried_actions}


    def __str__(self):
        action_leaves_msg = {action: leaf.performance if leaf is not None else None for action, leaf in self.action_leaves.items()}
        return "state {}: (visits={}, cost={:0.4f}, terminated={})\n{})".format(
                                                  self.state,
                                                  self.n_state_visits,
                                                  self.cost,
                                                  self.terminated,
                                                  action_leaves_msg
                                                  )

    ## select a random untried action
    def untried_action(self):
        action = random.choice(self.untried_actions)
        self.untried_actions.remove(action)
        return action
    
class Action_Node:

    def __init__(self, prev_state, action, next_state, terminated):
        self.prev_state = prev_state
        self.action = action
        self.total_simulation_cost = 0
        self.performance = None
        self.n_action_visits = 0
        self.next_state = next_state
        self.terminated = terminated
        self.node_id = (self.prev_state, self.action) #+ str(self.next_state)
        self.children={}
        self.children_ids = []

    def __str__(self):
        return "prev_state{}: (action={}, next_state={}, children={}, visits={}, performance={:0.4f})".format(
                                                  self.prev_state,
                                                  self.action,
                                                self.next_state,
                                                  self.children_ids,
                                                  self.n_action_visits,
                                                  self.performance,
                                                  )
    
class Episode_Node:

    # ''' episode nodes are defined by the current state, episode number, and the states+costs observed up until that point'''
    def  __init__(self, start, goal, episode, current_cost, prev_costs, n_AFC, N):
        self.start = start
        self.goal = goal
        self.episode = episode
        self.prev_costs = prev_costs
        self.current_cost = current_cost #i.e. the cost of the starting state of the episode (may not need this)
        self.N = N
        self.node_id = tuple(prev_costs, episode) ## might want to define this differently 
        self.parent_node_ids = []
        self.children_ids = []
        self.children = {}

        ## action leaves
        self.untried_actions = list(range(n_AFC))
        self.action_leaves = {a: None for a in self.untried_actions}

    # def __str__(self):
    #     action_leaves_msg = {action: leaf.performance if leaf is not None else None for action, leaf in self.action_leaves.items()}
    #     return "episode {}: (prev_costs={}, current_cost={:0.4f}, children={}, visits={}, performance={:0.4f})\n{})".format(
    #                                               self.episode,
    #                                               self.prev_costs,
    #                                               self.current_cost,
    #                                               self.children_ids,
    #                                               self.n_state_visits,
    #                                               self.performance,
    #                                               action_leaves_msg
    #                                               )

    ## select a random untried action
    def untried_action(self):
        action = random.choice(self.untried_actions)
        self.untried_actions.remove(action)
        return action
    
class Episode_Action_Node:

    def __init__(self, path_id, episode, episode_terminated):
        self.path_id = path_id
        self.total_simulation_cost = 0
        self.performance = None
        self.n_action_visits = 0
        self.episode_terminated = episode_terminated
        # self.node_id = (self.prev_state, self.action) #+ str(self.next_state)
        self.node_id = (episode, path_id)
        self.children={}
        self.children_ids = []




## Tree class
class Tree:

    def __init__(self,N):
        # self.nodes = {}
        self.root = None
        self.N = N
        self.n_state_visits = np.zeros((N,N))

    ## check if node is expandable
    def is_expandable(self, node):
        return not node.terminated and len(node.untried_actions) > 0

    ## attach action leaf to child state
    def add_state_node(self, state, cost, terminated, action_space, parent=None, episode = None):

        # ## check for existing state node
        # node_id = str(history)
        # if node_id in self.nodes:
        #     # print(state,"already exists")
        #     return self.nodes[node_id]

        
        ## create a new state node
        node = Node(state=state, cost=cost, terminated=terminated, action_space=action_space, N=self.N, episode = episode)
        
        ## store parent-child relationships
        if parent is None:
            self.root = node
            # self.nodes[str(state)].parent = None
        else:
            node.parent_node_ids.append(parent.node_id)
            
            ## add this state node to the children of the previous action leaf
            parent.children_ids.append(node.node_id)
            child_key = tuple(np.append(state, cost))
            parent.children[child_key] = node
            # parent.children[str(np.append(state, cost))] = node
            # parent.children[node.state_id] = node

        return node
    
    ## attach action leaf to child episode
    def add_episode_node(self, start, goal, episode, current_cost, prev_costs, episode_terminated, n_AFC, parent=None):

        ## create a new episode node
        node = Episode_Node(start, goal, episode, current_cost, prev_costs, n_AFC, self.N)
        
        ## store parent-child relationships
        if parent is None:
            self.root = node
        else:
            node.parent_node_ids.append(parent.node_id)
            parent.children_ids.append(node.node_id)
            parent.children[node.node_id] = node

        return node


    def get_children(self, node):
        children = []
        for a, leaf in node.action_leaves.items():
            if leaf is not None:
                # for node_id in leaf.children_ids:
                for child_key in leaf.children.keys():
                    child = leaf.children[child_key]
                    children.append(tuple((a, leaf, child_key, child)))
                    # children.append(tuple((a, self.nodes[node_id].state, self.nodes[node_id])))
        return children

    def parent(self, node):
        parent_node_id = self.nodes[node.node_id].parent_node_id
        if parent_node_id is None:
            return None #i.e. root reached, bc it has no parent
        else:
            return self.nodes[parent_node_id]

    ## calculate value of each S-A node
    def action_tree(self):

        self.tree_q = np.zeros((self.N,self.N,4)) + np.nan
        for sstate in self.nodes.keys():
            state = self.nodes[sstate].state
            for a in self.nodes[sstate].action_leaves.keys():
                try:
                    self.tree_q[state[0], state[1], a] = self.nodes[sstate].action_leaves[a].performance
                except:
                    pass


    def print_tree(self, node, indent="", is_last=True):
        """
        Recursively print the tree structure with markers, visit counts, and values.

        Args:
        - node_id: The ID of the current node.
        - indent: The current indentation string for formatting.
        - is_last: Whether this node is the last child of its parent.
        """
        # Get the current node
        # node = self.nodes[node_id]
        node_label = f"{node.state}"

        # Add branch marker
        branch = "└── " if is_last else "├── "
        print(f"{indent}{branch}Node: {node_label}")

        # Update indentation for children
        child_indent = indent + ("    " if is_last else "│   ")

        # Group children by action
        children_by_action = {}
        for action, leaf, child_id, child_node in self.get_children(node):
            if action not in children_by_action:
                children_by_action[action] = []
            children_by_action[action].append((leaf, child_id, child_node))

        # Find the best action based on performance
        best_action = max(
            children_by_action.items(),
            key=lambda item: item[1][0][0].performance,  # Access the performance of the first leaf
            default=(None, [])
        )[0]


        # Iterate through actions and their corresponding children
        num_actions = len(children_by_action)
        for i, (action, children) in enumerate(children_by_action.items()):
            # Check if this is the last action
            is_action_last = i == num_actions - 1

            # Print the action label (only once per action)
            leaf = children[0][0]  # Assume all children of the same action share the same leaf
            action_label = f"Action {action}, (n_v: {leaf.n_action_visits}, perf: {leaf.performance:.2f})"

            # Highlight the best action in bold (use ANSI escape codes for bold text)
            if action == best_action:
                action_label = f"\033[1m{action_label}\033[0m"

            action_branch = "└── " if is_action_last else "├── "
            print(f"{child_indent}{action_branch}{action_label}")

            # Update child indentation
            sub_child_indent = child_indent + ("    " if is_action_last else "│   ")

            # Print each child for this action
            for j, (leaf, child_id, child_node) in enumerate(children):
                # Check if this is the last child of this action
                is_child_last = j == len(children) - 1

                # Recursively print the child node
                self.print_tree(
                    child_node,
                    indent=sub_child_indent,
                    is_last=is_child_last,
                )

    def max_depth(self, node):
        """
        Recursively calculate the maximum depth of the tree starting from the given node.

        Args:
        - node: The current node (root of the subtree being evaluated).

        Returns:
        - int: The maximum depth of the tree.
        """
        # Base case: If the node has no children, its depth is 1
        if not self.get_children(node):
            return 1

        # Recursive case: Compute the depth for each child
        child_depths = []
        for _, _, _, child_node in self.get_children(node):
            child_depths.append(self.max_depth(child_node))

        # The depth of this node is 1 + max depth of its children
        return 1 + max(child_depths)



    ## prune, i.e. after taking a step, keep only that subtree
    # def prune(self):

    #     ## identify the root's children, i.e. the four adjacent states
    #     # keep_nodes = [str(self.root.state)]
    #     keep_nodes = self.root.node_id
    #     for leaf in self.root.action_leaves.values():
    #         if leaf is not None:
    #             keep_nodes.append(str(leaf.next_state))

    #     for sstate in list(self.nodes.keys()):
    #         if str(self.nodes[sstate].state) not in keep_nodes:
    #             del self.nodes[sstate]

    def prune(self, action, next_state):
        
        ## delete actions not taken
        actions_to_delete = [a for a in self.root.action_leaves.keys() if (a != action) and (self.root.action_leaves[a] is not None)]
        for a in actions_to_delete:
            del self.root.action_leaves[a]

        ## delete subtree for the other state reachable from the root-action pair
        self.root.action_leaves[action].children = {tuple(next_state): self.root.action_leaves[action].children[tuple(next_state)]}

        ## update the root
        self.root = self.root.action_leaves[action].children[tuple(next_state)]
        


    
    ## calculate the best trajectory for any two points, given the tree
    def best_traj(self, start, goal):

        ## get the best action at each state
        best_actions = nanargmax(self.tree_q, axis=2)

        ## get the best trajectory from start to goal
        current = start
        traj_states = [current]
        traj_actions = []
        stuck = False
        while not np.array_equal(current, goal) and not stuck:
            i, j = current
            action = best_actions[i,j]
            action = int(action)
            traj_actions.append(action)
            if action==0:
                current = np.clip((i + 1, j), 0, self.N-1)
            elif action == 1:
                current = np.clip((i, j + 1), 0, self.N-1)
            elif action == 2:
                current = np.clip((i - 1, j), 0, self.N-1)
            elif action == 3:
                current = np.clip((i, j - 1), 0, self.N-1)
            traj_states.append(current)

            ## check if the current state is already in the path
            for s in traj_states[:-1]:
                if np.array_equal(s, current):
                    stuck = True
                    
        
        return traj_states, traj_actions
                    

    
    

### misc utils

## random choice between multiple minima/maxima
def argm(x, extreme_val):
    indices = np.where(x == extreme_val)[0]
    return np.random.choice(indices)

## calculate the angle between two nodes
def node_angle(a,b):
    rad = np.arctan2(b[1]-a[1], b[0]-a[0])
    ang = np.abs(np.degrees(rad))
    ang%=90
    return ang


## sample from the GP
def sample(mean, K, sigma=0.01, high_cost=-0.9, low_cost=-0.1):
    if sigma is None:
        sigma = 0.01 #i.e. just to add to the diagonal of the kernel matrix

    N = int(np.sqrt(len(mean)))

    ## check kernel is valid
    k_check(K)

    # sample
    # if mean is None:
    #     mean = np.zeros(self.N**2)
    # mean = np.zeros(self.N**2)
    # samples = np.random.multivariate_normal(mean, K).reshape(self.N, self.N)

    #normalise
    # high_cost = self.high_cost
    # low_cost = self.low_cost
    # samples = high_cost + (low_cost - high_cost) * (samples - np.min(samples)) / (np.max(samples) - np.min(samples))


    ## or truncated
    lb = np.zeros(N**2) + high_cost
    ub = np.zeros(N**2) + low_cost
    K_tmp = K + sigma**2 * np.eye(N**2)
    tmvn = TruncatedMVN(mean, K_tmp, lb, ub)
    samples = tmvn.sample(1)
    samples = samples.reshape(N, N)

    return samples

## check that kernel is PSD and symmetric
def k_check(K):
    symm = np.allclose(K,K.T)
    if not symm:
        warnings.warn("Kernel matrix is not symmetric.", UserWarning)
    
    eigenvalues = np.linalg.eigvals(K)
    psd = np.all(eigenvalues >= -1e-10)
    if not psd:
        warnings.warn("Kernel matrix is not positive semi-definite.", UserWarning)

    return np.any([not symm, not psd])


## parse strings to lists
def parse_lists(df):
    cols = df.columns[2:]
    for key in cols:
        try:
            df[key] = df[key].apply(lambda x: np.array(ast.literal_eval(x)))
        except:
            pass
    return df


## KL divergence between prior and posterior samples, where samples as assumed to be multivariate Gaussians
def KL_divergence(x, y):
    '''x is the prior, y is the posterior'''

    ## calculate gaussian terms
    cov_x = np.cov(x)
    cov_y = np.cov(y)
    mu_x = np.mean(x, axis=1)
    mu_y = np.mean(y, axis=1)
    d = len(mu_x)
    assert d == cov_x.shape[0], "Mean and covariance dimensions do not match"

    ## trace term
    inv_cov_y = np.linalg.inv(cov_y)
    trace_term = np.trace(inv_cov_y @ cov_x)

    ## log determinant term
    log_det_x = np.linalg.slogdet(cov_x)[1]
    log_det_y = np.linalg.slogdet(cov_y)[1]
    LD_term = log_det_y - log_det_x

    ## mu term
    mean_diff = mu_y - mu_x
    mean_term = mean_diff.T @ inv_cov_y @ mean_diff

    ## combine
    KL = 0.5* (trace_term - d + LD_term + mean_term) 

    return KL


def profile_func(func, *args, **kwargs):

    ## check first of all if a profiler is active, in which case disable it
    if cProfile.Profile().disable() is not None:
        cProfile.Profile().disable()

    ## profile the function
    profiler = cProfile.Profile()
    profiler.enable()
    func(*args, **kwargs)
    profiler.disable()

    ## save profiling report
    func_name = func.__name__
    profile_file = f'{func_name}_profile.pstats'
    profiler.dump_stats(profile_file)
    with open(f'{func_name}_profile.txt', 'w') as f:
        p = pstats.Stats(profiler, stream=f)
        p.sort_stats('cumulative').print_stats(50)

    ## convert to dot file
    dot_file = f'{func_name}_profile.dot'
    subprocess.run(['gprof2dot', '-f', 'pstats', profile_file, '-o', dot_file], check=True)

    ## generate PNG visualization
    png_file = f'{func_name}_profile.png'
    subprocess.run(['dot', '-Tpng', dot_file, '-o', png_file], check=True)

    print(f"Profiling complete. Visualization saved as {png_file}")


## cached function for moving to the next state
def get_next_state(current, direction, N):
    next_state = np.clip(
        current + direction,
        0,
        N - 1
    )
    return next_state

## parallel function for simulating many episodes within the same mountain env
def simulate_agent(m, N, params=None, metric='cityblock', expt='2AFC', n_episodes=10, agents = ['GP', 'GP-MCTS', 'BAMCP','CE'], n_sims=1000, n_futures=0, n_iter=10, lazy=False, exploration_constant=2, discount_factor=0.95, progress=False, offline=False):
    
    ## initiate dictionary to store the results
    sim_out = {}
    for key in data_keys:
        sim_out[key]=[]
    
    ## set seed
    seed=m
    seed=os.getpid()
    np.random.seed(seed)
    
    ## create base mountain environment
    env = make_env(N, n_episodes, expt, params, metric)
    
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
            ELDs = []
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
                    agent.root_samples(obs=env_copy.obs, n_samples=n_sims, n_iter=n_iter, lazy=lazy,CE=True)
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
                    MCTS.actual_state = current
                    
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
                            n_sims_tmp = n_sims
                            action, MCTS_Q = MCTS.search(n_sims_tmp, n_futures, n_iter=n_iter, lazy=lazy, progress=progress, reuse_samples=reuse_samples)
                        elif (dist_to_goal <= (N/2)) & (dist_to_goal > (N/4)):
                            # n_sims_tmp = int(n_sims/2)
                            n_sims_tmp = n_sims
                            action, MCTS_Q = MCTS.search(n_sims_tmp, n_futures, n_iter=n_iter, lazy=lazy, progress=progress, reuse_samples=reuse_samples)
                        else:
                            # n_sims_tmp = int(n_sims/4)
                            n_sims_tmp = n_sims
                            action, MCTS_Q = MCTS.search(n_sims_tmp, n_futures,n_iter=n_iter, lazy=lazy,  progress=progress, reuse_samples=reuse_samples)
                        actions.append(action)

                        ### optional: check what the CE agent would have done with the mean of this set of samples
                        # agent.dp(agent.posterior_mean_p_cost, expected_cost=True)
                        # action_CE = agent.optimal_policy(current, agent.Q_inf)
                        # CE_actions.append(action_CE)

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
                        agent.dp(posterior_mean_p_cost_tmp, expected_cost=True)
                        action_CE = agent.optimal_policy(current, agent.Q_inf)
                        CE_actions.append(action_CE)



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
                        current, cost, terminated, _, _ = env_copy.step(action)
                        steps += 1

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
                            action, MCTS_Q = MCTS.search(n_sims, n_futures, progress=progress, reuse_samples=reuse_samples)
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
                if (ag=='BAMCP') or (ag=='BAMCP w/ CE'):

                    ## get prior p and q samples
                    prior_p_samples = agent.all_posterior_ps
                    prior_q_samples = agent.all_posterior_qs
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
                        agent.root_samples(sim_obs, n_samples=n_sims, n_iter=n_iter, lazy=lazy, CE=False)
                        posterior_samples = np.vstack([np.array(agent.all_posterior_ps).T, np.array(agent.all_posterior_qs).T])
                        
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
                    EKLs.append(expected_KL)

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
                    reuse_samples = True



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

    return sim_out,env_copy.p_costs
    # return sim_out, _


## data-saving/dict stuff
data_keys = [
    'agent',
    'mountain',
    'episode',
    'start',
    'goal',
    'costs',
    'optimal_costs',
    'actions',
    'CE_actions',
    'optimal_actions',
    'total_cost',
    'total_optimal_cost',
    'action_score',
    'cost_ratio',
    'n_steps',
    'actual_trajectory',
    'optimal_trajectory',
    'observations',
    'search_attempts',
    'action_tree',
    'discounted_costs',
    'total_discounted_cost',
    'discounted_optimal_costs',
    'total_discounted_optimal_cost',
    'expected_LD',
    'expected_KL'

    ## GP-specific
    # 'true_k',
    # 'RPE',
    # 'posterior_mean',
    # 'theta_MLE',
]