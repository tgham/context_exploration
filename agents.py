from enum import Enum
import copy
import gymnasium as gym
from gymnasium import spaces
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
from utils import *
from scipy.stats import rankdata, truncnorm
from scipy.linalg import cholesky
from base_kernels import *
from samplers import GridSampler
from MCTS import MonteCarloTreeSearch_2AFC
from tqdm.auto import tqdm
import pandas as pd



    

### base farmer model?
class Farmer:

    def __init__(self, N, context_prior=0.5, metric='cityblock'):

        self.metric = metric
        self.N = N
        self.n_actions = 4
        self.action_to_direction = {0: np.array([1,0]), 1: np.array([0, 1]), 2: np.array([-1, 0]), 3: np.array([0, -1])}
        
        ## initialise context prior prob
        self.context_prob = context_prior
        # print('initialised with context prior:', self.context_prob)


    ### interactions with the environment

    ## function for receiving info from env
    def get_env_info(self, env):
        self.N = env.N 
        self.obs = env.obs
        self.current = env.current
        self.expt = env.expt
        self.goal = env.goal
        self.high_cost = env.high_cost
        self.low_cost = env.low_cost
        self.alpha_row = env.alpha_row
        self.alpha_col = env.alpha_col
        self.beta_row = env.beta_row
        self.beta_col = env.beta_col


    ## generate full set of root samples
    def root_samples(self, obs=None, n_samples=1000, n_iter=100, lazy=True, CE=False, combo=True):
        
        ## hacky: obs should not contain duplicates!
        obs = np.unique(obs, axis=0).tolist() if obs is not None else []

        sampler = GridSampler(self.alpha_row, self.beta_row, self.alpha_col, self.beta_col, self.low_cost, self.high_cost, obs, N=self.N, CE=CE)
        self.all_posterior_ps = np.zeros((n_samples, self.N))
        self.all_posterior_qs = np.zeros((n_samples, self.N))
        self.all_posterior_p_costs = np.zeros((n_samples, self.N, self.N))

        ## if combinatorial task
        if combo:

            if len(obs) > 0:

                ## lazy
                if lazy:

                    ## loop through samples
                    for s in range(n_samples):
                        self.all_posterior_ps[s,:], self.all_posterior_qs[s,:] = sampler.lazy_sample(n_iter = n_iter)
                        self.all_posterior_p_costs[s] = np.outer(self.all_posterior_ps[s], self.all_posterior_qs[s])

                ## full 
                else:

                    ## loop through samples
                    for s in range(n_samples):
                        self.all_posterior_ps[s,:], self.all_posterior_qs[s,:] = sampler.full_sample(n_iter = n_iter)
                        self.all_posterior_p_costs[s] = np.outer(self.all_posterior_ps[s], self.all_posterior_qs[s])


            ## if no obs, just sample from prior
            else:
                for s in range(n_samples):
                    self.all_posterior_ps[s,:] = np.random.beta(sampler.alpha_row, sampler.beta_row, size=self.N)
                    self.all_posterior_qs[s,:] = np.random.beta(sampler.alpha_col, sampler.beta_col, size=self.N)
                    self.all_posterior_p_costs[s] = np.outer(self.all_posterior_ps[s], self.all_posterior_qs[s])

                ## TMP: half the samples are from the correct prior, half are from the incorrect prior
                # for s in range(n_samples//2):
                #     self.all_posterior_ps[s,:] = np.random.beta(sampler.alpha_row, sampler.beta_row, size=self.N)
                #     self.all_posterior_qs[s,:] = np.random.beta(sampler.alpha_col, sampler.beta_col, size=self.N)
                #     self.all_posterior_p_costs[s] = np.outer(self.all_posterior_ps[s], self.all_posterior_qs[s])
                # for s in range(n_samples//2, n_samples):
                #     self.all_posterior_ps[s,:] = np.random.beta(sampler.alpha_col, sampler.beta_col, size=self.N)
                #     self.all_posterior_qs[s,:] = np.random.beta(sampler.alpha_row, sampler.beta_row, size=self.N)
                #     self.all_posterior_p_costs[s] = np.outer(self.all_posterior_ps[s], self.all_posterior_qs[s])
                # np.random.shuffle(self.all_posterior_p_costs)

        ## simpler task
        else:
            
            ### determine context indicators

            ## simple case: certain prior
            # self.col_world_prob = 1 ## tmp, i.e. probability of one context or another

            ## inference case: infer posterior probability, given observations
            context_prior = self.context_prob
            self.context_prob = sampler.context_posterior(context_prior=context_prior)
            # print('sampler prior:', context_prior, ',', 'posterior:', self.context_prob)

            ## use inferred context to sample
            if not CE:
                self.context_indicators = np.random.binomial(1, self.context_prob, size=n_samples) 
                col_context = self.context_indicators.astype(bool)

                ## looping method...
                # for s in range(n_samples):
                #     self.all_posterior_ps[s,:], self.all_posterior_qs[s,:] = sampler.simple_sample(col_context=col_context[s])
                #     self.all_posterior_p_costs[s] = np.outer(self.all_posterior_ps[s], self.all_posterior_qs[s])

                #     ## temp fix: this posterior should also be filled in with 1s and 0s for states where a low and high cost have been observed respectively
                #     for i,j,c in obs:
                #         i = int(i)
                #         j = int(j)
                #         prob = 1 if c == self.low_cost else 0
                #         self.all_posterior_p_costs[s][i,j] = prob

                ## or, all at once?
                n_col_samples = np.sum(self.context_indicators)
                n_row_samples = n_samples - n_col_samples
                posterior_ps_col, posterior_qs_col = sampler.simple_sample(col_context=True, n_samples=n_col_samples)
                posterior_ps_row, posterior_qs_row = sampler.simple_sample(col_context=False, n_samples=n_row_samples)
                self.all_posterior_ps[:n_col_samples,:] = posterior_ps_col
                self.all_posterior_ps[n_col_samples:,:] = posterior_ps_row
                self.all_posterior_qs[:n_col_samples,:] = posterior_qs_col
                self.all_posterior_qs[n_col_samples:,:] = posterior_qs_row

                ## shuffle them all in the same way
                idx = np.random.permutation(n_samples)
                self.all_posterior_ps = self.all_posterior_ps[idx]
                self.all_posterior_qs = self.all_posterior_qs[idx]
                self.context_indicators = np.zeros(n_samples)
                self.context_indicators[:n_col_samples] = 1
                self.context_indicators = self.context_indicators[idx].astype(bool)

                ## posterior costs - i.e. the outer product of each sample's p and q
                self.all_posterior_p_costs = np.einsum('si,sj->sij', self.all_posterior_ps, self.all_posterior_qs)

                ## temp fix: this posterior should also be filled in with 1s and 0s for states where a low and high cost have been observed respectively
                for i,j,c in obs:
                    i = int(i)
                    j = int(j)
                    prob = 1 if c == self.low_cost else 0
                    self.all_posterior_p_costs[:,i,j] = prob
            
                ## posterior means 
                self.posterior_mean_p_cost = np.mean(self.all_posterior_p_costs, axis=0)
                self.posterior_mean_p = np.mean(self.all_posterior_ps, axis=0)
                self.posterior_mean_q = np.mean(self.all_posterior_qs, axis=0)

            ## if CE, no need to loop through samples - just get posterior mean under each context, and then calculate weighted average
            elif CE:
                posterior_ps_col, posterior_qs_col = sampler.simple_sample(col_context=True)
                posterior_ps_row, posterior_qs_row = sampler.simple_sample(col_context=False)
                self.posterior_mean_p_cost = self.context_prob * np.outer(posterior_ps_col, posterior_qs_col) + (1-self.context_prob) * np.outer(posterior_ps_row, posterior_qs_row)
                self.posterior_mean_p = self.context_prob * posterior_ps_col + (1-self.context_prob) * posterior_ps_row
                self.posterior_mean_q = self.context_prob * posterior_qs_col + (1-self.context_prob) * posterior_qs_row


        ## debugging plot - kde of samples
        # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # for p in range(self.N):
        #     sns.kdeplot(self.all_posterior_ps[:,p], ax=axs[0])
        #     sns.kdeplot(self.all_posterior_qs[:,p], ax=axs[1])
        # axs[0].set_title('p')
        # axs[1].set_title('q')
        # plt.show()


    ## quick and cheap context posterior
    def quick_context_posterior(self, obs):
        
        ## hacky fix: obs should not contain duplicates!
        obs = np.unique(obs, axis=0).tolist() if obs is not None else []
        
        sampler = GridSampler(self.alpha_row, self.beta_row, self.alpha_col, self.beta_col, self.low_cost, self.high_cost, obs, N=self.N)
        context_prob = sampler.context_posterior(context_prior=self.context_prob)
        return context_prob


    ## dynamic programming
    def dp(self, posterior_p_cost, expected_cost=True):

        ## use expected cost of each state
        if expected_cost:

            ## p(high cost)
            # dp_costs = self.posterior_p_cost*self.high_cost + (1-self.posterior_p_cost)*self.low_cost
            # dp_costs[self.goal[0], self.goal[1]] = 0
            
            ## p(low cost)
            dp_costs = posterior_p_cost*self.low_cost + (1-posterior_p_cost)*self.high_cost
            dp_costs[self.goal[0], self.goal[1]] = 0

        ## or, sample costs using p and q probabilities 
        else:

            ## p(high cost)
            # dp_costs = np.array([self.low_cost if r > self.posterior_p_cost.flatten()[i] else self.high_cost for i, r in enumerate(np.random.random(self.N**2))]).reshape(self.N, self.N)
            # dp_costs[self.goal[0], self.goal[1]] = 0

            ## p(low cost)
            dp_costs = np.array([self.low_cost if r < posterior_p_cost.flatten()[i] else self.high_cost for i, r in enumerate(np.random.random(self.N**2))]).reshape(self.N, self.N)
            dp_costs[self.goal[0], self.goal[1]] = 0


        self.V_inf, self.Q_inf, self.A_inf = value_iteration(dp_costs, self.goal)

    ## optimal policy, as given by the dynamic programming Q vals
    def optimal_policy(self, current, Q=None):

        if Q is None:
            Q = self.Q_inf

        ## get adjacent states
        next_states = np.clip(np.array([current + self.action_to_direction[i] for i in range(self.n_actions)]), 0, self.N-1)
        next_states_idx = next_states[:, 0]*self.N + next_states[:, 1]
    
        ## choose action with highest Q-value
        current_q = Q[current[0], current[1], :]
        max_current_q = np.nanmax(current_q)
        action = argm(current_q, max_current_q)

        return action
    
    ## greedy wrt/ distance to goal
    def greedy_policy(self, current, goal, eps=0):
        if np.random.rand() < eps:
            return self.random_policy()
        else:
            # distances = cdist([current], [goal], metric=self.metric).flatten()
            ## get adjacent states
            next_states = np.clip(np.array([current + self.action_to_direction[i] for i in range(self.n_actions)]), 0, self.N-1)
            
            ## choose whichever one is closest to the goal
            distances = cdist(next_states, [goal], metric=self.metric).flatten()
            min_distance = np.min(distances)
            action = argm(distances, min_distance)
            # print(next_states, distances, min_distance, action)
            return action
        
    ## choice function
    def softmax(self, Q):
        CPs = (1-self.lapse) * softmax(Q/self.temp) + self.lapse/len(Q)
        return CPs


        
    ## run agent on participant's trial sequence
    def run(self, params, hyperparams, agent = 'CE', df_trials=None, envs=None,fit=True):
        
        ## init expt info
        n_trials = int(df_trials['trial'].max())
        n_days = int(df_trials['day'].max() )
        n_cities = int(df_trials['city'].max())
        N = envs['city_1_grid_1_env_object'][0].N

        ## initialise model's internal variables
        self.n_total_trials = len(df_trials)
        self.df_trials = df_trials
        self.n_afc = 2 ## can sort this out later
        self.p_choice = np.zeros((n_cities, n_days, n_trials, self.n_afc))
        self.p_correct = np.zeros((n_cities, n_days, n_trials))
        self.Q_vals = np.zeros((n_cities, n_days, n_trials, self.n_afc))
        self.actions = np.zeros((n_cities, n_days, n_trials))
        self.context_priors = np.zeros((n_cities, n_days, n_trials))
        self.context_posteriors = np.zeros((n_cities, n_days, n_trials))
        self.leaf_visits = np.zeros((n_cities, n_days, n_trials, self.n_afc))
        self.total_costs = np.zeros((n_cities, n_days, n_trials))
        self.trial_loss = np.zeros(self.n_total_trials)

        ## for extracting some useful trial data...
        self.path_A_past_overlaps = np.zeros((n_cities, n_days, n_trials))
        self.path_B_past_overlaps = np.zeros((n_cities, n_days, n_trials))
        self.path_A_observed_costs = np.zeros((n_cities, n_days, n_trials))
        self.path_B_observed_costs = np.zeros((n_cities, n_days, n_trials))
        self.path_A_observed_no_costs = np.zeros((n_cities, n_days, n_trials))
        self.path_B_observed_no_costs = np.zeros((n_cities, n_days, n_trials))
        
        ## init params and hyperparams
        if agent == 'BAMCP':
            self.temp = params[0]
            self.lapse = params[1]
            n_sims = hyperparams['n_sims']
            exploration_constant = hyperparams['exploration_constant']
            discount_factor = hyperparams['discount_factor']
            n_iter = hyperparams['n_iter']
            lazy = hyperparams['lazy']
        elif agent == 'CE':
            self.temp = params[0]
            self.lapse = params[1]
            n_sims = hyperparams['n_sims']
            n_iter = hyperparams['n_iter']
            lazy = hyperparams['lazy']


        ## loop through cities
        # for city in tqdm(range(n_cities)):
        for city in range(n_cities):

            ## context prior resets
            context_prior = 0.5

            ## loop through days
            for day in range(n_days):
                self.context_prior = context_prior

                ## get the environment for this day
                if envs:
                    env = envs['city_{}_grid_{}_env_object'.format(city+1, day+1)][0]
                env_copy = copy.deepcopy(env)
                env_copy.set_trial(0)
                assert not hasattr(env_copy, 'obs'), 'env_copy.obs should not exist before the first trial: {}'.format(len(env_copy.obs),', city:', city+1, 'day:', day+1)

                ## otherwise, generate a new one
                # else:
                #     env = make_env(N, n_trials, expt_info, beta_params, metric)

                ## FIX FOR OLD ENVS: rename some attributes (episode --> trial, etc.)
                if hasattr(env_copy, 'n_episodes'):
                    env_copy.n_trials = env_copy.n_episodes

                ## initialise planner
                if agent == 'BAMCP':
                    MCTS = None
                    tree_reset = True


                ## loop through trials within day
                for t in range(n_trials):

                    ## reset env/trial
                    self.context_prob = context_prior ## tmp fix: fix the prior to the prior that was used at the beginning of the grid (to prevent observations contributing to the posterior on multiple trials)
                    env_copy.reset()
                    env_copy.set_sim(True)
                    start = env_copy.current
                    current = start
                    goal = env_copy.goal
                    actions = []
                    choice_probs = []

                    ## if extracting useful behavioural measures: check whether any of the agent's observations overlap with the paths of the subsequent trial
                    if agent == 'human':
                        paths = env_copy.path_states[t].copy()
                        obs_list = [tuple(obs[:2]) for obs in env_copy.obs.tolist()]
                        try:
                            A_overlap = set(paths[0]).intersection(set(obs_list))
                            B_overlap = set(paths[1]).intersection(set(obs_list))
                            
                            ## get the number of states that overlap with the paths
                            path_A_past_overlap = len(A_overlap)
                            path_B_past_overlap = len(B_overlap)

                            ## get the number of costs and no-costs that comprise these overlapping states
                            path_A_observed_costs = sum(env_copy.costss[t][obs[0], obs[1]] == self.high_cost for obs in A_overlap)
                            path_A_observed_no_costs = sum(env_copy.costss[t][obs[0], obs[1]] == self.low_cost for obs in A_overlap)
                            path_B_observed_costs = sum(env_copy.costss[t][obs[0], obs[1]] == self.high_cost for obs in B_overlap)
                            path_B_observed_no_costs = sum(env_copy.costss[t][obs[0], obs[1]] == self.low_cost for obs in B_overlap)
                        
                        ## sometimes need to convert each np array to list of tuples...
                        except:
                            paths = [set(map(tuple, path)) for path in paths]
                            A_overlap = set(paths[0]).intersection(set(obs_list))
                            B_overlap = set(paths[1]).intersection(set(obs_list))
                            path_A_past_overlap = len(A_overlap)
                            path_B_past_overlap = len(B_overlap)
                            path_A_observed_costs = np.sum([env_copy.costss[t][obs[0], obs[1]] == self.high_cost for obs in A_overlap])
                            path_A_observed_no_costs = np.sum([env_copy.costss[t][obs[0], obs[1]] == self.low_cost for obs in A_overlap])
                            path_B_observed_costs = np.sum([env_copy.costss[t][obs[0], obs[1]] == self.high_cost for obs in B_overlap])
                            path_B_observed_no_costs = np.sum([env_copy.costss[t][obs[0], obs[1]] == self.low_cost for obs in B_overlap])
                        self.path_A_past_overlaps[city, day, t] = path_A_past_overlap
                        self.path_B_past_overlaps[city, day, t] = path_B_past_overlap
                        self.path_A_observed_costs[city, day, t] = path_A_observed_costs
                        self.path_B_observed_costs[city, day, t] = path_B_observed_costs
                        self.path_A_observed_no_costs[city, day, t] = path_A_observed_no_costs
                        self.path_B_observed_no_costs[city, day, t] = path_B_observed_no_costs
                        assert path_A_past_overlap == path_A_observed_costs + path_A_observed_no_costs, 'path A past overlap does not match observed costs and no-costs\n path A past overlap: {}, path A observed costs: {}, path A observed no-costs: {}'.format(path_A_past_overlap, path_A_observed_costs, path_A_observed_no_costs)
                        assert path_B_past_overlap == path_B_observed_costs + path_B_observed_no_costs, 'path B past overlap does not match observed costs and no-costs\n path B past overlap: {}, path B observed costs: {}, path B observed no-costs: {}'.format(path_B_past_overlap, path_B_observed_costs, path_B_observed_no_costs)

                    ## agent receives info from env
                    self.get_env_info(env_copy)

                    ## agent-specific path selection
                    if agent == 'BAMCP':

                        ## reset tree (or reuse it)
                        if tree_reset:
                            tree = Tree(N)
                            MCTS = MonteCarloTreeSearch_2AFC(env=env_copy, agent=self, tree=tree, exploration_constant=exploration_constant, discount_factor=discount_factor)
                        else:
                            MCTS.update_trial()
                            tree_resets=True
                        assert t == MCTS.actual_trial, 'trial mismatch between env and MCTS\n env: {} \n MCTS: {}'.format(t, MCTS.env.trial)
                        assert MCTS.env.sim == True, 'env not in sim mode'

                        ## search
                        MCTS.actual_state = current
                        action, MCTS_Q = MCTS.search(n_sims, n_iter=n_iter, lazy=lazy)
                        self.actions[city, day, t] = action
                        self.Q_vals[city, day, t] = MCTS_Q
                        self.p_choice[city, day, t] = self.softmax(MCTS_Q)
                        correct_path = np.argmax(env_copy.path_actual_costs[t])
                        self.p_correct[city, day, t] = self.p_choice[city, day, t][correct_path]
                        self.leaf_visits[city, day, t] = MCTS.tree.root.action_leaves[action].n_action_visits

                    elif agent == 'CE':
                        env_copy.set_sim(False)
                        
                        ## get posterior mean grid
                        self.root_samples(obs=env_copy.obs, n_samples=n_sims, n_iter=n_iter, lazy=lazy, CE=True, combo=False)
                        env_copy.receive_predictions(self.posterior_mean_p_cost)

                        ## get the cost of each path under the posterior mean
                        path_costs = []
                        for path_id in range(env_copy.n_afc):
                            path_states = env_copy.path_states[t][path_id]
                            path_cost = 0
                            for state in path_states:
                                # path_cost += env_copy.get_pred_cost(state) ## i.e. sample binary costs from the posterior pqs
                                path_cost += self.posterior_mean_p_cost[state[0], state[1]]*env_copy.low_cost + (1-self.posterior_mean_p_cost[state[0], state[1]])*env_copy.high_cost ## or, use expected costs
                            path_costs.append(path_cost)

                        ## choose the path with the lowest total cost
                        max_cost = np.max(path_costs)
                        action = argm(path_costs, max_cost)
                        self.actions[city, day, t] = action
                        self.Q_vals[city, day, t] = np.array(path_costs)
                        self.p_choice[city, day, t] = self.softmax(np.array(path_costs))
                        correct_path = np.argmax(env_copy.path_actual_costs[t])
                        self.p_correct[city, day, t] = self.p_choice[city, day, t][correct_path]

                    elif agent == 'human':
                        ## need to trivially set predicted costs to 0 to avoid errors when interacting with the environment
                        env_copy.receive_predictions(np.zeros((N, N)))
                        


                    ### take ppt's action if a) we are fitting, or b) we are extracting behavioural measures
                    missed=False
                    if (fit) or (agent == 'human'):

                        ## first check if the participant has made a choice
                        try:
                            missed = pd.isna(df_trials.loc[(df_trials['city'] == city+1) & (df_trials['day'] == day+1) & (df_trials['trial'] == t+1), 'path_chosen'].values[0])
                        except:
                            missed = True

                        ## if the participant has made a choice, then we use their action (rather than the model's)
                        if not missed:
                            action = df_trials.loc[(df_trials['city'] == city+1) & (df_trials['day'] == day+1) & (df_trials['trial'] == t+1), 'path_chosen'].values[0]=='b'                        
                            assert df_trials.loc[(df_trials['city'] == city+1) & (df_trials['day'] == day+1) & (df_trials['trial'] == t+1), 'path_A_expected_cost'].values[0] == env_copy.path_expected_costs[t][0], 'expected cost does not match ppt data\n env: {}, ppt: {}'.format(env_copy.path_expected_costs[t][0], df_trials.loc[(df_trials['city'] == city+1) & (df_trials['day'] == day+1) & (df_trials['trial'] == t+1), 'path_A_expected_cost'].values[0])

                        else:
                            # print('missed in city {}, day {}, trial {}'.format(city+1, day+1, t+1))
                            # print('start: {}, goal: {}'.format(env_copy.starts[t], env_copy.goal))
                            self.p_choice[city, day, t] = np.nan
                            self.p_correct[city, day, t] = np.nan
                            self.Q_vals[city, day, t] = np.nan
                            self.actions[city, day, t] = np.nan
                            self.context_priors[city, day, t] = np.nan
                            self.context_posteriors[city, day, t] = np.nan
                            self.leaf_visits[city, day, t] = np.nan
                            self.total_costs[city, day, t] = np.nan

                    ## only interact with the environment if participant made a choice
                    if not missed:
                        env_copy.set_sim(False)
                        env_copy.init_trial(action)
                        start = env_copy.current
                        goal = env_copy.goal
                        assert np.array_equal(start, env_copy.starts[t][action]), 'current state does not match start state in city {} day {} trial {}\n current: {}, start: {}.\n all starts: {}\n all goals:{}'.format(city, day, t, env_copy.current, env_copy.starts[t][action], env_copy.starts, env_copy.goals)
                        action_sequence = env_copy.path_actions[t][action]
                        _, _ = env_copy.take_path(action_sequence)
                        current = env_copy.current
                        costs = env_copy.trial_obs[:,-1]
                        assert len(costs) == len(action_sequence)+1, 'costs and action sequence do not match\n costs: {}, action sequence: {}'.format(len(costs), len(action_sequence))
                        path_cost = np.sum(costs)
                        self.total_costs[city, day, t] = path_cost
                        day_terminated = t == (n_trials-1)

                    ## else, skip to next trial??
                    else:
                        env_copy.set_trial(t+1)

                    ## update observations
                    self.get_env_info(env_copy)

                    ## update MCTS tree
                    if agent == 'BAMCP':

                        ## prune tree (not always successful due to high branching factor, or if participant made no choice in which case reset the tree)
                        if not missed:
                            init_info_state = np.array(MCTS.tree.root.node_id).reshape(N, N, 2)
                            trial_obs = env_copy.trial_obs.copy()
                            next_node_id = MCTS.init_node_id(trial_obs, init_info_state, t)
                            if not day_terminated:
                                if next_node_id in MCTS.tree.root.action_leaves[action].children:
                                    MCTS.tree.prune(action, next_node_id)
                                    assert np.array_equal(MCTS.tree.root.state[2*MCTS.n_afc:], costs), 'error in root update\n root state: {} \n costs: {}'.format(MCTS.tree.root.state[2*MCTS.n_afc:], costs)
                                    tree_reset = False
                                else:
                                    tree_reset = True
                        else:
                            tree_reset = True

                    ## get the context prior - i.e. the probability with which samples were drawn
                    context_prior = self.context_prob

                    # get the new context posterior for this agent
                    context_posterior = self.quick_context_posterior(env_copy.obs)

                    ## (and save these)
                    self.context_priors[city, day, t] = context_prior
                    self.context_posteriors[city, day, t] = context_posterior

                    ## carry over the context prob to the next run, if on the final trial of the day
                    if t == (n_trials-1):
                        context_prior = context_posterior

        self.loss_func(df_trials)
        return self.loss

            
    ## loss function
    def loss_func(self, df_trials):

        ## flatten + other init
        self.p_choice_flat = self.p_choice[:,:,:,1].flatten() ## i.e. p(choose path B)
        if len(self.p_choice_flat) != len(df_trials):
            # warnings.warn('p_choice_flat length does not match df_trials length. Check your data!')
            print('p_choice_flat length does not match df_trials length for participant {}. Truncating p_choice_flat to match df_trials length.'.format(df_trials['pid'].values[0]))
            self.p_choice_flat = self.p_choice_flat[:len(df_trials)] ## i.e. truncate to match df_trials length
        # self.p_choice_flat = self.p_choice_flat[~np.isnan(self.p_choice_flat)]
        self.ppt_choices = (df_trials['path_chosen']=='b').values


        ## numerical stability
        self.p_choice_flat[(self.p_choice_flat==0) & (self.ppt_choices)] = 0 + np.finfo(float).tiny
        self.p_choice_flat[(self.p_choice_flat==1) & (~self.ppt_choices)] = 1 - np.finfo(float).eps

        ## negative log likelihood
        self.trial_loss[self.ppt_choices] = np.log(self.p_choice_flat[self.ppt_choices])
        self.trial_loss[~self.ppt_choices] = np.log((1-self.p_choice_flat[~self.ppt_choices]))
        self.loss = -np.nansum(self.trial_loss)


    ## pseudo r^2
    def pseudo_r2(self, df_trials):
        
        ## calculate loss under null (random choice)
        n_trials = len(df_trials)
        p_choice_null = np.ones(n_trials) * 0.5
        loss_null = -np.sum(np.log(p_choice_null))

        ## pseudo r^2
        pseudo_r2 = 1 - (self.loss / loss_null)

        ## LLR test?
        llr = 2 * (loss_null - self.loss)
        df = 2
        p_value = scipy.stats.chi2.sf(llr, df)

        return pseudo_r2, p_value
