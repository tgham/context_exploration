from enum import Enum
from itertools import product
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
from MCTS import MonteCarloTreeSearch_AFC
from tqdm.auto import tqdm
import pandas as pd
from scipy.special import beta, logsumexp, digamma, comb, betaln


### base farmer model?
class Farmer:

    def __init__(self, N, context_prior=0.5, metric='cityblock', known_context=False):

        self.metric = metric
        self.N = N
        self.n_actions = 4
        self.action_to_direction = {0: np.array([1,0]), 1: np.array([0, 1]), 2: np.array([-1, 0]), 3: np.array([0, -1])}
        
        ## initialise context prior prob
        self.context_prob = context_prior
        self.known_context = known_context


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
    def root_samples(self, obs=None, n_samples=1000, n_iter=100, lazy=True, CE=False, combo=False):
        
        ## hacky: obs should not contain duplicates!
        obs = np.unique(obs, axis=0).tolist() if obs is not None else []

        sampler = GridSampler(self.alpha_row, self.beta_row, self.alpha_col, self.beta_col, self.low_cost, self.high_cost, obs, N=self.N, CE=CE)
        self.sampler = sampler
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


            ## inference case: infer posterior probability, given observations
            if not self.known_context:
                context_prior = self.context_prob
                self.context_prob = sampler.context_posterior(context_prior=context_prior)

            ## simple case: certain prior
            elif self.known_context:
                self.context_prob = 1.0 if self.known_context == 'column' else 0.0

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

                ## temp fix: this posterior should also be filled in with 1s and 0s for states where a low and high cost have been observed respectively
                for i,j,c in obs:
                    i = int(i)
                    j = int(j)
                    prob = 1 if c == self.low_cost else 0
                    self.posterior_mean_p_cost[i,j] = prob

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
    def run(self, params, hyperparams, agent = 'CE', df_trials=None, envs=None,fit=True, progress=False):
        
        ## init expt info
        try:
            n_trials = int(df_trials['trial'].max())
            n_days = int(df_trials['day'].max() )
            n_cities = int(df_trials['city'].max())
            N = envs['city_1_grid_1_env_object'][0].N
            n_afc = df_trials['path_chosen'].nunique()
        except:
            n_trials = hyperparams['n_trials']
            n_days = hyperparams['n_days']
            n_cities = hyperparams['n_cities']
            N = hyperparams['N']
            n_afc = hyperparams['n_afc']

        ## determine policy - i.e. greedy vs softmax
        if hyperparams is None:
            hyperparams = {}
        if 'greedy' in hyperparams:
            greedy = hyperparams['greedy']
        else:
            greedy = True 

        ## initialise model's internal variables
        self.n_afc = n_afc ## can sort this out later
        self.p_choice = np.zeros((n_cities, n_days, n_trials, self.n_afc))
        self.p_correct = np.zeros((n_cities, n_days, n_trials))
        self.Q_vals = np.zeros((n_cities, n_days, n_trials, self.n_afc))
        self.actions = np.zeros((n_cities, n_days, n_trials))
        self.CE_actions = np.zeros((n_cities, n_days, n_trials)) + np.nan
        self.CE_p_choice = np.zeros((n_cities, n_days, n_trials, self.n_afc)) + np.nan
        self.CE_p_correct = np.zeros((n_cities, n_days, n_trials)) + np.nan
        self.CE_Q_vals = np.zeros((n_cities, n_days, n_trials, self.n_afc)) + np.nan
        self.context_priors = np.zeros((n_cities, n_days, n_trials))
        self.context_posteriors = np.zeros((n_cities, n_days, n_trials))
        self.leaf_visits = np.zeros((n_cities, n_days, n_trials, self.n_afc))
        self.total_costs = np.zeros((n_cities, n_days, n_trials))
        self.path_quality = np.zeros((n_cities, n_days, n_trials)) 
        self.true_context = []
        if fit:
            self.n_total_trials = len(df_trials)
            self.trial_loss = np.zeros(self.n_total_trials)

        ## for extracting some useful trial data...
        self.path_future_overlaps = np.zeros((n_cities, n_days, n_trials, self.n_afc))
        self.path_past_overlaps = np.zeros((n_cities, n_days, n_trials, self.n_afc))
        self.path_past_observed_high_costs = np.zeros((n_cities, n_days, n_trials, self.n_afc))
        self.path_past_observed_low_costs = np.zeros((n_cities, n_days, n_trials, self.n_afc))
        self.path_len = np.zeros((n_cities, n_days, n_trials))
        self.day_costs = np.zeros((n_cities, n_days, n_trials)) ## i.e. the cost of the path chosen by the participant on that trial
        self.distr_diff = np.zeros((n_cities, n_days, n_trials)) 
        
        ## observations on the context-aligned and orthogonal arm of each path
        self.aligned_arm_actual_high_costs = np.zeros((n_cities, n_days, n_trials, self.n_afc))
        self.aligned_arm_actual_low_costs = np.zeros((n_cities, n_days, n_trials, self.n_afc))
        self.orthogonal_arm_actual_high_costs = np.zeros((n_cities, n_days, n_trials, self.n_afc))
        self.orthogonal_arm_actual_low_costs = np.zeros((n_cities, n_days, n_trials, self.n_afc))
        self.aligned_arm_gen_high_costs = np.zeros((n_cities, n_days, n_trials, self.n_afc))
        self.aligned_arm_gen_low_costs = np.zeros((n_cities, n_days, n_trials, self.n_afc))
        self.orthogonal_arm_gen_high_costs = np.zeros((n_cities, n_days, n_trials, self.n_afc))
        self.orthogonal_arm_gen_low_costs = np.zeros((n_cities, n_days, n_trials, self.n_afc))
        self.aligned_arm_cf_gen_high_costs = np.zeros((n_cities, n_days, n_trials, self.n_afc))
        self.aligned_arm_cf_gen_low_costs = np.zeros((n_cities, n_days, n_trials, self.n_afc))
        self.orthogonal_arm_cf_gen_high_costs = np.zeros((n_cities, n_days, n_trials, self.n_afc))
        self.orthogonal_arm_cf_gen_low_costs = np.zeros((n_cities, n_days, n_trials, self.n_afc))

        
        ## init params and hyperparams
        if agent == 'BAMCP':
            self.temp = params[0]
            self.lapse = params[1]
            self.arm_weight = params[2]
            n_sims = hyperparams['n_sims']
            exploration_constant = hyperparams['exploration_constant']
            discount_factor = hyperparams['discount_factor']
            n_iter = hyperparams['n_iter']
        elif (agent == 'CE'):
            self.temp = params[0]
            self.lapse = params[1]
        elif (agent == 'CE_one_arm'):
            self.temp = params[0]
            self.lapse = params[1]
            self.arm_weight = params[2]
        
            # n_sims = hyperparams['n_sims']
            # n_iter = hyperparams['n_iter']
        # elif agent == 'human': ## hacky - need this for CE calcs
        #     n_sims = hyperparams['n_sims']
        #     n_iter = hyperparams['n_iter']

        if progress:
            pbar = tqdm(total=n_cities*n_days*n_trials, desc='Running {} agent'.format(agent), leave=False)

        ## loop through cities
        for city in range(n_cities):
            context_prior = 0.5

            ## loop through days
            for day in range(n_days):

                ## get the environment for this day
                if envs:
                    env = envs['city_{}_grid_{}_env_object'.format(city+1, day+1)][0]

                    ## need to do some fixes for old envs
                    if env.expt == '2AFC':
                        env.expt = 'AFC'
                    
                    ## get context alignment of states
                    env.get_alignment()

                env_copy = copy.deepcopy(env)
                env_copy.set_trial(0)
                assert not hasattr(env_copy, 'obs'), 'env_copy.obs should not exist before the first trial: {}'.format(len(env_copy.obs),', city:', city+1, 'day:', day+1)
                
                ## context prior resets (only if context is unknown)
                if not self.known_context:
                    self.context_prob = context_prior
                elif self.known_context:
                    self.known_context = env_copy.context


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
                    if self.known_context is None: ## only reset if context is unknown
                        self.context_prob = context_prior ## tmp fix: fix the prior to the prior that was used at the beginning of the grid (to prevent observations contributing to the posterior on multiple trials)
                    env_copy.reset()
                    env_copy.set_sim(True)
                    start = env_copy.current
                    current = start
                    goal = env_copy.goal
                    actions = []
                    choice_probs = []

                    ### if extracting useful behavioural measures

                    ## overlaps with previous observations
                    if agent == 'human':
                        paths = env_copy.path_states[t].copy()
                        obs_list = [tuple(obs[:2]) for obs in env_copy.obs.tolist()]
                        obs_list = list(set(obs_list)) # no repeated obs!
                        for i, path in enumerate(paths):
                            try:
                                
                                ## get the number of states that overlap with the paths
                                overlap = set(path).intersection(set(obs_list))
                                path_past_overlap = len(overlap)

                                ## get the number of costs and no-costs that comprise these overlapping states
                                path_past_observed_high_costs = sum(env_copy.costss[t][obs[0], obs[1]] == self.high_cost for obs in overlap)
                                path_past_observed_low_costs = sum(env_copy.costss[t][obs[0], obs[1]] == self.low_cost for obs in overlap)
                                self.path_future_overlaps[city, day, t, i] = env_copy.path_future_overlaps[t][i]
                                self.path_past_overlaps[city, day, t, i] = path_past_overlap
                                self.path_past_observed_high_costs[city, day, t, i] = path_past_observed_high_costs
                                self.path_past_observed_low_costs[city, day, t, i] = path_past_observed_low_costs

                        
                            ## sometimes need to convert each np array to list of tuples...
                            except:
                                # paths = [set(map(tuple, path)) for path in paths]
                                path = set(map(tuple, path))
                                overlap = set(path).intersection(set(obs_list))
                                path_past_overlap = len(overlap)
                                path_past_observed_high_costs = sum(env_copy.costss[t][obs[0], obs[1]] == self.high_cost for obs in overlap)
                                path_past_observed_low_costs = sum(env_copy.costss[t][obs[0], obs[1]] == self.low_cost for obs in overlap)
                                self.path_future_overlaps[city, day, t, i] = env_copy.path_future_overlaps[t][i]
                                self.path_past_overlaps[city, day, t, i] = path_past_overlap
                                self.path_past_observed_high_costs[city, day, t, i] = path_past_observed_high_costs
                                self.path_past_observed_low_costs[city, day, t, i] = path_past_observed_low_costs
                            
                            assert self.path_past_overlaps[city, day, t, i] == self.path_past_observed_high_costs[city, day, t, i] + self.path_past_observed_low_costs[city, day, t, i], 'path {} past overlap does not match observed costs and no-costs\n path past overlap: {}, path observed costs: {}, path observed no-costs: {}'.format(i+1, self.path_past_overlaps[city, day, t, i], self.path_past_observed_high_costs[city, day, t, i], self.path_past_observed_low_costs[city, day, t, i])
                            

                            ## get aligned vs orthogonal states
                            path_states = env_copy.path_states[t][i]
                            aligned_states, orthogonal_states = env_copy.path_aligned_states[t][i], env_copy.path_orthogonal_states[t][i]

                            ## get info on costs on rows and columns
                            observed_high_cost_cols = {obs[1] for obs in obs_list if env_copy.costss[t][obs[0], obs[1]] == self.high_cost}
                            observed_low_cost_cols = {obs[1] for obs in obs_list if env_copy.costss[t][obs[0], obs[1]] == self.low_cost}
                            observed_high_cost_rows = {obs[0] for obs in obs_list if env_copy.costss[t][obs[0], obs[1]] == self.high_cost}
                            observed_low_cost_rows = {obs[0] for obs in obs_list if env_copy.costss[t][obs[0], obs[1]] == self.low_cost}
                            observed_high_cost_states = {tuple(obs) for obs in obs_list if env_copy.costss[t][obs[0], obs[1]] == self.high_cost}
                            observed_low_cost_states = {tuple(obs) for obs in obs_list if env_copy.costss[t][obs[0], obs[1]] == self.low_cost}

                            if env_copy.context == 'column':
                                
                                ### actual costs
                                
                                ## count how many of these aligned states have actual high and low costs
                                self.aligned_arm_actual_high_costs[city, day, t, i] = sum(1 for state in aligned_states if state in observed_high_cost_states)
                                self.aligned_arm_actual_low_costs[city, day, t, i] = sum(1 for state in aligned_states if state in observed_low_cost_states)

                                ## count how many of the orthogonal states have actual high and low costs
                                self.orthogonal_arm_actual_high_costs[city, day, t, i] = sum(1 for state in orthogonal_states if state in observed_high_cost_states)
                                self.orthogonal_arm_actual_low_costs[city, day, t, i] = sum(1 for state in orthogonal_states if state in observed_low_cost_states)

                                
                                ### gen costs
                                
                                ## count how many of these aligned states have observations on the main column
                                self.aligned_arm_gen_high_costs[city, day, t, i] = sum(
                                    1 for state in aligned_states
                                    if (state[1] in observed_high_cost_cols) 
                                    and (tuple(state) not in obs_list) ## exclude states that are themselves observed, since these would be captured in the 'actual' costs above
                                )
                                self.aligned_arm_gen_low_costs[city, day, t, i] = sum(
                                    1 for state in aligned_states 
                                    if (state[1] in observed_low_cost_cols)
                                    and (tuple(state) not in obs_list) ## exclude states that are themselves observed, since these would be captured in the 'actual' costs above
                                )

                                ## count how many of the orthogonal states have observations on their respective columns
                                self.orthogonal_arm_gen_high_costs[city, day, t, i] = sum(
                                    1 for state in orthogonal_states 
                                    if (state[1] in observed_high_cost_cols)
                                    and (tuple(state) not in obs_list) ## exclude states that are themselves observed, since these would be captured in the 'actual' costs above
                                )
                                self.orthogonal_arm_gen_low_costs[city, day, t, i] = sum(
                                    1 for state in orthogonal_states 
                                    if (state[1] in observed_low_cost_cols)
                                    and (tuple(state) not in obs_list) ## exclude states that are themselves observed, since these would be captured in the 'actual' costs above
                                )


                                ### counterfactual generalisation

                                ## how many of the orthogonal states have observations on the main row
                                self.orthogonal_arm_cf_gen_high_costs[city, day, t, i] = sum(
                                    1 for state in orthogonal_states 
                                    if (state[0] in observed_high_cost_rows)
                                    and (tuple(state) not in obs_list) ## exclude states that are themselves observed, since these would be captured in the 'actual' costs above
                                )
                                self.orthogonal_arm_cf_gen_low_costs[city, day, t, i] = sum(
                                    1 for state in orthogonal_states 
                                    if (state[0] in observed_low_cost_rows)
                                    and (tuple(state) not in obs_list) ## exclude states that are themselves observed, since these would be captured in the 'actual' costs above
                                )

                                ## and count how many of the aligned states have observations on their respective rows
                                self.aligned_arm_cf_gen_high_costs[city, day, t, i] = sum(
                                    1 for state in aligned_states 
                                    if (state[0] in observed_high_cost_rows)
                                    and (tuple(state) not in obs_list) ## exclude states that are themselves observed, since these would be captured in the 'actual' costs above
                                )
                                self.aligned_arm_cf_gen_low_costs[city, day, t, i] = sum(
                                    1 for state in aligned_states 
                                    if (state[0] in observed_low_cost_rows)
                                    and (tuple(state) not in obs_list) ## exclude states that are themselves observed, since these would be captured in the 'actual' costs above
                                )
                                
                                ## debugging...
                                # if (t ==3) & (env_copy.objective =='rewards'):
                                #     print('obs high cost rows:', observed_high_cost_rows)
                                #     print('obs low cost rows:', observed_low_cost_rows)
                                #     print('obs high cost cols:', observed_high_cost_cols)
                                #     print('obs low cost cols:', observed_low_cost_cols)
                                #     print('path states', path_states)
                                #     print('aligned_states:', aligned_states)
                                #     print('orthogonal_states:', orthogonal_states)
                                #     print('aligned_arm_gen_high_costs:', self.aligned_arm_gen_high_costs[city, day, t, i])
                                #     print('aligned_arm_gen_low_costs:', self.aligned_arm_gen_low_costs[city, day, t, i])
                                #     print('orthogonal_arm_gen_high_costs:', self.orthogonal_arm_gen_high_costs[city, day, t, i])
                                #     print('orthogonal_arm_gen_low_costs:', self.orthogonal_arm_gen_low_costs[city, day, t, i])
                                #     raise Exception

                            elif env_copy.context == 'row':

                                ### actual costs

                                ## count how many of these aligned states have actual high and low costs
                                self.aligned_arm_actual_high_costs[city, day, t, i] = sum(
                                    1 for state in aligned_states 
                                    if state in observed_high_cost_states)
                                self.aligned_arm_actual_low_costs[city, day, t, i] = sum(
                                    1 for state in aligned_states if state in observed_low_cost_states)

                                ## count how many of the orthogonal states have actual high and low costs
                                self.orthogonal_arm_actual_high_costs[city, day, t, i] = sum(
                                    1 for state in orthogonal_states if state in observed_high_cost_states)
                                self.orthogonal_arm_actual_low_costs[city, day, t, i] = sum(
                                    1 for state in orthogonal_states if state in observed_low_cost_states)

                                
                                ### gen costs
                                
                                ## count how many of these aligned states have observations on the main row
                                self.aligned_arm_gen_high_costs[city, day, t, i] = sum(
                                    1 for state in aligned_states 
                                    if (state[0] in observed_high_cost_rows)
                                    and (tuple(state) not in obs_list) ## exclude states that are themselves observed, since these would be captured in the 'actual' costs above
                                )
                                self.aligned_arm_gen_low_costs[city, day, t, i] = sum(
                                    1 for state in aligned_states 
                                    if (state[0] in observed_low_cost_rows)
                                    and (tuple(state) not in obs_list) ## exclude states that are themselves observed, since these would be captured in the 'actual' costs above
                                )

                                ## count how many of the orthogonal states have observations on their respective rows
                                self.orthogonal_arm_gen_high_costs[city, day, t, i] = sum(
                                    1 for state in orthogonal_states 
                                    if (state[0] in observed_high_cost_rows)
                                    and (tuple(state) not in obs_list) ## exclude states that are themselves observed, since these would be captured in the 'actual' costs above
                                )
                                self.orthogonal_arm_gen_low_costs[city, day, t, i] = sum(
                                    1 for state in orthogonal_states 
                                    if (state[0] in observed_low_cost_rows)
                                    and (tuple(state) not in obs_list) ## exclude states that are themselves observed, since these would be captured in the 'actual' costs above
                                )

                                
                                ### counterfactual generalisation

                                ## how many of the orthogonal states have observations on the main col
                                self.orthogonal_arm_cf_gen_high_costs[city, day, t, i] = sum(
                                    1 for state in orthogonal_states 
                                    if (state[1] in observed_high_cost_cols)
                                    and (tuple(state) not in obs_list) ## exclude states that are themselves observed, since these would be captured in the 'actual' costs above
                                )
                                self.orthogonal_arm_cf_gen_low_costs[city, day, t, i] = sum(
                                    1 for state in orthogonal_states 
                                    if (state[1] in observed_low_cost_cols)
                                    and (tuple(state) not in obs_list) ## exclude states that are themselves observed, since these would be captured in the 'actual' costs above
                                )

                                ## and count how many of the aligned states have observations on their respective cols
                                self.aligned_arm_cf_gen_high_costs[city, day, t, i] = sum(
                                    1 for state in aligned_states 
                                    if (state[1] in observed_high_cost_cols)
                                    and (tuple(state) not in obs_list) ## exclude states that are themselves observed, since these would be captured in the 'actual' costs above
                                )
                                self.aligned_arm_cf_gen_low_costs[city, day, t, i] = sum(
                                    1 for state in aligned_states 
                                    if (state[1] in observed_low_cost_cols)
                                    and (tuple(state) not in obs_list) ## exclude states that are themselves observed, since these would be captured in the 'actual' costs above
                                )

                                ## debugging...
                                # if (t ==3) & (env_copy.objective =='rewards'):
                                #     print('obs high cost rows:', observed_high_cost_rows)
                                #     print('obs low cost rows:', observed_low_cost_rows)
                                #     print('obs high cost cols:', observed_high_cost_cols)
                                #     print('obs low cost cols:', observed_low_cost_cols)
                                #     print('path states', path_states)
                                #     print('aligned_states:', aligned_states)
                                #     print('orthogonal_states:', orthogonal_states)
                                #     print('aligned_arm_gen_high_costs:', self.aligned_arm_gen_high_costs[city, day, t, i])
                                #     print('aligned_arm_gen_low_costs:', self.aligned_arm_gen_low_costs[city, day, t, i])
                                #     print('orthogonal_arm_gen_high_costs:', self.orthogonal_arm_gen_high_costs[city, day, t, i])
                                #     print('orthogonal_arm_gen_low_costs:', self.orthogonal_arm_gen_low_costs[city, day, t, i])
                                #     raise Exception
                    
                        ## misc
                        self.day_costs[city, day, t] = np.nansum(self.total_costs[city, day, :t+1]) ## i.e. costs observed so far today
                        self.path_len[city, day, t] = len(env_copy.path_states[t][0])
                        

                    ## agent receives info from env
                    self.get_env_info(env_copy)

                    ## agent-specific path selection
                    if agent == 'BAMCP':

                        ## reset tree (or reuse it)
                        if tree_reset:
                            tree = Tree(N)
                            MCTS = MonteCarloTreeSearch_AFC(env=env_copy, agent=self, tree=tree, exploration_constant=exploration_constant, discount_factor=discount_factor)
                        else:
                            MCTS.update_trial()
                            tree_resets=True
                        assert t == MCTS.actual_trial, 'trial mismatch between env and MCTS\n env: {} \n MCTS: {}'.format(t, MCTS.env.trial)
                        assert MCTS.env.sim == True, 'env not in sim mode'

                        ## search
                        MCTS.actual_state = current
                        action, MCTS_Q = MCTS.search(n_sims, n_iter=n_iter)
                        self.actions[city, day, t] = action
                        self.Q_vals[city, day, t] = MCTS_Q
                        self.p_choice[city, day, t] = self.softmax(MCTS_Q)
                        correct_path = np.argmax(env_copy.path_actual_costs[t])
                        self.p_correct[city, day, t] = self.p_choice[city, day, t][correct_path]
                        self.leaf_visits[city, day, t] = MCTS.tree.root.action_leaves[action].n_action_visits

                        ## or, do probability matching if not greedy
                        if not greedy:
                            action = np.random.choice(np.arange(len(MCTS_Q)), p=softmax(MCTS_Q))


                        ### let's also calculate CE choice under BAMCP's knowledge 

                        ## get the cost of each path under the posterior mean
                        CE_path_costs = []
                        for path_id in range(env_copy.n_afc):
                            path_states = env_copy.path_states[t][path_id]
                            CE_path_cost = 0
                            for state in path_states:
                                # path_cost += env_copy.get_pred_cost(state) ## i.e. sample binary costs from the posterior pqs
                                CE_path_cost += self.posterior_mean_p_cost[state[0], state[1]]*env_copy.low_cost + (1-self.posterior_mean_p_cost[state[0], state[1]])*env_copy.high_cost ## or, use expected costs
                            CE_path_costs.append(CE_path_cost)

                        ## CE chooses the path with the lowest total cost
                        max_cost = np.max(CE_path_costs)
                        CE_action = argm(CE_path_costs, max_cost)
                        self.CE_actions[city, day, t] = CE_action
                        self.CE_Q_vals[city, day, t] = np.array(CE_path_costs)
                        self.CE_p_choice[city, day, t] = self.softmax(np.array(CE_path_costs))
                        self.CE_p_correct[city, day, t] = self.CE_p_choice[city, day, t][correct_path]


                    elif agent == 'CE':
                        env_copy.set_sim(False)
                        
                        ## get posterior mean grid
                        self.root_samples(obs=env_copy.obs, CE=True, combo=False)
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

                        ## or, do probability matching if not greedy
                        if not greedy:
                            action = np.random.choice(np.arange(len(path_costs)), p=softmax(path_costs))
                    
                    elif agent == 'CE_one_arm':
                        env_copy.set_sim(False)
                        
                        ## get posterior mean grid
                        self.root_samples(obs=env_copy.obs, CE=True, combo=False)
                        env_copy.receive_predictions(self.posterior_mean_p_cost)

                        ## get the cost of each path under the posterior mean (weighted by arm_weight)
                        path_costs = []
                        for path_id in range(env_copy.n_afc):
                            path_states = env_copy.path_states[t][path_id]
                            aligned_states, orthogonal_states = env_copy.path_aligned_states[t][path_id], env_copy.path_orthogonal_states[t][path_id]
                            unweighted_pred_costs = self.posterior_mean_p_cost*env_copy.low_cost + (1-self.posterior_mean_p_cost)*env_copy.high_cost
                            weighted_path_cost = self.arm_reweighting(unweighted_pred_costs, aligned_states, orthogonal_states)
                            path_costs.append(weighted_path_cost)
                            
                            ## debugging
                            # print('path_states:', path_states)
                            # print('aligned_states:', aligned_states)
                            # print('context:', env_copy.context)
                            # print('all path_costs:', [self.posterior_mean_p_cost[state[0], state[1]]*env_copy.low_cost + (1-self.posterior_mean_p_cost[state[0], state[1]])*env_copy.high_cost for state in path_states])
                            # print('all aligned path_costs:', [self.posterior_mean_p_cost[state[0], state[1]]*env_copy.low_cost + (1-self.posterior_mean_p_cost[state[0], state[1]])*env_copy.high_cost for state in aligned_states])
                            # print('all orthogonal path_costs:', [self.posterior_mean_p_cost[state[0], state[1]]*env_copy.low_cost + (1-self.posterior_mean_p_cost[state[0], state[1]])*env_copy.high_cost for state in orthogonal_states])
                            # print('arm weight:', self.arm_weight)
                            # print('path_cost:', weighted_path_cost)
                            # print()
                            # if t==3:
                            #     raise Exception('check this')

                        ## choose the path with the lowest total cost
                        max_cost = np.max(path_costs)
                        action = argm(path_costs, max_cost)
                        self.actions[city, day, t] = action
                        self.Q_vals[city, day, t] = np.array(path_costs)
                        self.p_choice[city, day, t] = self.softmax(np.array(path_costs))
                        correct_path = np.argmax(env_copy.path_actual_costs[t])
                        self.p_correct[city, day, t] = self.p_choice[city, day, t][correct_path]

                        ## or, do probability matching if not greedy
                        if not greedy:
                            action = np.random.choice(np.arange(len(path_costs)), p=softmax(path_costs))

                    elif agent == 'human':
                        
                        ## need to trivially set predicted costs to 0 to avoid errors when interacting with the environment
                        # env_copy.receive_predictions(np.zeros((N, N)))

                        ## or, we might actually calculate the CE-correct answer under the human's observations
                        self.root_samples(obs=env_copy.obs, CE=True, combo=False)
                        env_copy.receive_predictions(self.posterior_mean_p_cost)
                        path_costs = []
                        for path_id in range(env_copy.n_afc):
                            path_states = env_copy.path_states[t][path_id]
                            path_cost = 0
                            for state in path_states:
                                path_cost += self.posterior_mean_p_cost[state[0], state[1]]*env_copy.low_cost + (1-self.posterior_mean_p_cost[state[0], state[1]])*env_copy.high_cost ## or, use expected costs
                            path_costs.append(path_cost)
                        max_cost = np.max(path_costs)
                        CE_action = argm(path_costs, max_cost)
                        self.CE_actions[city, day, t] = CE_action
                        self.CE_Q_vals[city, day, t] = np.array(path_costs)

                        
                        ### get the difference in distributions over total costs of the two paths

                        # ## sample PMFs over total costs for each path
                        # n_samples = 50000
                        # self.root_samples(obs = env_copy.obs, n_samples=n_samples, CE=False, combo=False)
                        # sample_total_costs = np.zeros((n_samples, env_copy.n_afc))
                        # for s in range(n_samples):
                            
                        #     ## sample binary grid
                        #     sample_costs = np.array([self.high_cost if r>self.all_posterior_p_costs[s].flatten()[ri] else self.low_cost for ri, r in enumerate(np.random.random(self.N**2))]).reshape(self.N, self.N)

                        #     ## sum costs along each path
                        #     for path_id in range(env_copy.n_afc):
                        #         path_states = env_copy.path_states[t][path_id]
                        #         path_cost = 0
                        #         for state in path_states:
                        #             path_cost += sample_costs[state[0], state[1]]
                        #         sample_total_costs[s, path_id] = path_cost


                        ### OR, vectorized sampling approach:

                        # --- 1. PRE-CALCULATION (Do this once outside the sampling loop) ---
                        # Create the Path Weight Matrix W (N_path x N^2)
                        # W = np.zeros((env_copy.n_afc, self.N**2))
                        # for path_id in range(env_copy.n_afc):
                        #     path_states = env_copy.path_states[t][path_id]
                        #     flat_indices = [state[0] * self.N + state[1] for state in path_states]
                        #     W[path_id, flat_indices] = 1


                        # # --- 2. VECTORIZED SAMPLING (Replaces your N_samples loop) ---
                        # n_samples = 10000
                        # self.root_samples(obs = env_copy.obs, n_samples=n_samples, CE=False, combo=False)
                        # p_costs_flat = self.all_posterior_p_costs.reshape(n_samples, self.N**2)
                        # random_draws = np.random.random((n_samples, self.N**2))
                        # sample_costs_binary = (random_draws < p_costs_flat).astype(int) 
                        # sample_costs_vectorized = sample_costs_binary * self.high_cost + (1 - sample_costs_binary) * self.low_cost

                        # # Step 3: Vectorized Path Summation
                        # sample_total_costs = sample_costs_vectorized @ W.T 


                        ## or, only calculate the difference if the CE's belief does indeed favour the better path - i.e. if CE_aciton == correct_path
                        # if CE_action == correct_path:
                        #     self.distr_diff[city, day, t] = path_distr_diff(sample_total_costs[:,0], sample_total_costs[:,1], metric, correct_path)
                        # else:
                        #     self.distr_diff[city, day, t] = np.nan

                        # print('t{}, distr diff: {}'.format(t, self.distr_diff[city, day, t]))
                        
                        


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
                            assert np.isclose(df_trials.loc[(df_trials['city'] == city+1) & (df_trials['day'] == day+1) & (df_trials['trial'] == t+1), 'path_A_expected_cost'].values[0], env_copy.path_expected_costs[t][0], rtol=1e-5), 'expected cost does not match ppt data\n env: {}, ppt: {}'.format(env_copy.path_expected_costs[t][0], df_trials.loc[(df_trials['city'] == city+1) & (df_trials['day'] == day+1) & (df_trials['trial'] == t+1), 'path_A_expected_cost'].values[0])

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
                        self.path_quality[city, day, t] = self.total_costs[city, day, t]/self.path_len[city, day, t] ## i.e. cost as a proportion of path length
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

                    ## update progress bar
                    if progress:
                        pbar.update(1)
            self.true_context.append(env_copy.context)
        if progress:
            pbar.close()

        ## if we are fitting, calculate the loss
        if fit:
            self.loss_func(df_trials)
            return self.loss
        
        ## or, if we are running our own simulations, give the simulation output
        elif (not fit) & (df_trials is None): 
            sim_out ={
                'participant':[],
                'agent':[],
                'city':[],
                'day':[],
                'trial':[],
                'context':[],
                'actions':[],
                'CE_actions':[],
                'distr_diff':[],
                'p_choice_A':[],
                'p_choice_B':[],
                'p_choice_C':[],
                'p_correct':[],
                'Q_a':[],
                'Q_b':[],
                'Q_c':[],
                'CE_p_choice_A':[],
                'CE_p_choice_B':[],
                'CE_p_choice_C':[],
                'CE_p_correct':[],
                'CE_Q_a':[],
                'CE_Q_b':[],
                'CE_Q_c':[],
                'leaf_visits_a':[],
                'leaf_visits_b':[],
                'leaf_visits_c':[],
            }
            for c in range(n_cities):
                for d in range(n_days):
                    for t in range(n_trials):
                        sim_out['participant'].append(envs['participant'])
                        sim_out['agent'].append(agent)
                        sim_out['city'].append(c+1)
                        sim_out['day'].append(d+1)
                        sim_out['trial'].append(t+1)
                        sim_out['actions'].append(self.actions[c][d][t])
                        sim_out['CE_actions'].append(self.CE_actions[c][d][t])
                        sim_out['distr_diff'].append(self.distr_diff[c][d][t])
                        sim_out['context'].append(self.true_context[c])
                        sim_out['p_correct'].append(self.p_correct[c][d][t])
                        sim_out['p_choice_A'].append(self.p_choice[c][d][t][0])
                        sim_out['p_choice_B'].append(self.p_choice[c][d][t][1])
                        sim_out['Q_a'].append(self.Q_vals[c][d][t][0])
                        sim_out['Q_b'].append(self.Q_vals[c][d][t][1])
                        sim_out['leaf_visits_a'].append(self.leaf_visits[c][d][t][0])
                        sim_out['leaf_visits_b'].append(self.leaf_visits[c][d][t][1])
                        sim_out['CE_p_correct'].append(self.CE_p_correct[c][d][t])
                        sim_out['CE_p_choice_A'].append(self.CE_p_choice[c][d][t][0])
                        sim_out['CE_p_choice_B'].append(self.CE_p_choice[c][d][t][1])
                        sim_out['CE_Q_a'].append(self.CE_Q_vals[c][d][t][0])
                        sim_out['CE_Q_b'].append(self.CE_Q_vals[c][d][t][1])
                        if self.n_afc==3:
                            sim_out['p_choice_C'].append(self.p_choice[c][d][t][2])
                            sim_out['leaf_visits_c'].append(self.leaf_visits[c][d][t][2])
                            sim_out['Q_c'].append(self.Q_vals[c][d][t][2])
                            sim_out['CE_p_choice_C'].append(self.CE_p_choice[c][d][t][2])
                            sim_out['CE_Q_c'].append(self.CE_Q_vals[c][d][t][2])
                        else:
                            sim_out['p_choice_C'].append(np.nan)
                            sim_out['leaf_visits_c'].append(np.nan)
                            sim_out['Q_c'].append(np.nan)
                            sim_out['CE_p_choice_C'].append(np.nan)
                            sim_out['CE_Q_c'].append(np.nan)
            return sim_out

            
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


    ## calculate weighted cost based on aligned vs orthogonal states
    def arm_reweighting(self, costs, aligned_states, orthogonal_states):
        """
        Calculate weighted cost for a path based on aligned and orthogonal state costs.
        
        Args:
            costs: NxN array of costs for each state in the grid
            aligned_states: set of (row, col) tuples for states on the context-aligned arm
            orthogonal_states: set of (row, col) tuples for states on the orthogonal arm
            
        Returns:
            weighted_cost: the total weighted cost for the path
        """
        
        ## calculate weights based on arm_weight parameter
        # arm_weight > 0: favour aligned arm (reduce orthogonal weight)
        # arm_weight < 0: favour orthogonal arm (reduce aligned weight)
        aligned_weight = 1 - max(0.0, -self.arm_weight)
        orthogonal_weight = 1 - max(0.0, self.arm_weight)
        
        ## calculate weighted cost
        weighted_cost = 0
        for state in aligned_states:
            weighted_cost += aligned_weight * costs[state[0], state[1]]
        for state in orthogonal_states:
            weighted_cost += orthogonal_weight * costs[state[0], state[1]]
            
        return weighted_cost
