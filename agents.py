from abc import ABC, abstractmethod
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
from MCTS import MonteCarloTreeSearch_AFC
from tqdm.auto import tqdm
import pandas as pd
from scipy.special import beta, logsumexp, digamma, comb, betaln


### base farmer model?
class Farmer(ABC):

    def __init__(self, context_prior=0.5, known_context=False,
                 temp=1, lapse=0, arm_weight=0, horizon=3, real_future_paths=True,
                 exploration_constant=None, discount_factor=None, n_samples=None):
        
        ## initialise context prior prob
        self.context_prob = context_prior
        self.known_context = known_context

        ## behavioural parameters
        self.temp = temp
        self.lapse = lapse
        self.arm_weight = arm_weight
        self._cache_arm_weights()

        ## MCTS parameters
        self.horizon = horizon
        self.real_future_paths = real_future_paths
        self.exploration_constant = exploration_constant
        self.discount_factor = discount_factor
        self.n_samples = n_samples

        ## MCTS object (will be initialised when running BAMCP agent)
        self.mcts = None

        ## environment reference (set via get_env_info)
        self.env = None
        

    ## create a sampler object from the environment's current state
    def make_sampler(self):
        """Delegate to the environment's task-specific sampler factory."""
        if self.env is None:
            raise ValueError("Environment not set. Call get_env_info() first.")
        return self.env.make_sampler()

    ### interactions with the environment

    ## function for receiving info from env
    def get_env_info(self, env):
        self.env = env

    ## agent-specific calculation of Q values based on posterior
    @abstractmethod
    def compute_Q(self, env_copy, tree_reset=True):
        pass


    ## quick and cheap context posterior
    def quick_context_posterior(self, obs=None):
        sampler = self.make_sampler()
        context_prob = sampler.context_posterior(context_prior=self.context_prob)
        return context_prob
    

    ## choice function
    def softmax(self, Q):
        CPs = (1-self.lapse) * softmax(Q/self.temp) + self.lapse/len(Q)
        return CPs

        
    ## run agent on participant's trial sequence
    def run(self, hyperparams, agent = 'CE', df_trials=None, envs=None,fit=True, progress=False):
        
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
            self.greedy = hyperparams['greedy']
        else:
            self.greedy = True

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
        self.path_future_rel_overlaps = np.zeros((n_cities, n_days, n_trials, self.n_afc))
        self.path_future_irrel_overlaps = np.zeros((n_cities, n_days, n_trials, self.n_afc))
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
        self.aligned_arm_len = np.zeros((n_cities, n_days, n_trials, self.n_afc))
        self.orthogonal_arm_len = np.zeros((n_cities, n_days, n_trials, self.n_afc))

        if progress:
            pbar = tqdm(total=n_cities*n_days*n_trials, desc='Running {} agent'.format(agent), leave=False)

        ## loop through cities
        for city in range(n_cities):
            if not self.known_context:
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
                    env.path_aligned_states, env.path_orthogonal_states = env.get_alignment(env.path_states)

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
                self.mcts = None
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
                                path_past_observed_high_costs = sum(env_copy.costss[t][obs[0], obs[1]] == env_copy.high_cost for obs in overlap)
                                path_past_observed_low_costs = sum(env_copy.costss[t][obs[0], obs[1]] == env_copy.low_cost for obs in overlap)
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
                                path_past_observed_high_costs = sum(env_copy.costss[t][obs[0], obs[1]] == env_copy.high_cost for obs in overlap)
                                path_past_observed_low_costs = sum(env_copy.costss[t][obs[0], obs[1]] == env_copy.low_cost for obs in overlap)
                                self.path_future_overlaps[city, day, t, i] = env_copy.path_future_overlaps[t][i]
                                self.path_past_overlaps[city, day, t, i] = path_past_overlap
                                self.path_past_observed_high_costs[city, day, t, i] = path_past_observed_high_costs
                                self.path_past_observed_low_costs[city, day, t, i] = path_past_observed_low_costs
                            
                            assert self.path_past_overlaps[city, day, t, i] == self.path_past_observed_high_costs[city, day, t, i] + self.path_past_observed_low_costs[city, day, t, i], 'path {} past overlap does not match observed costs and no-costs\n path past overlap: {}, path observed costs: {}, path observed no-costs: {}'.format(i+1, self.path_past_overlaps[city, day, t, i], self.path_past_observed_high_costs[city, day, t, i], self.path_past_observed_low_costs[city, day, t, i])
                            

                            ## get aligned vs orthogonal states
                            path_states = env_copy.path_states[t][i]
                            aligned_states, orthogonal_states = env_copy.path_aligned_states[t][i], env_copy.path_orthogonal_states[t][i]
                            self.aligned_arm_len[city, day, t, i] = len(aligned_states)
                            self.orthogonal_arm_len[city, day, t, i] = len(orthogonal_states)

                            ## get info on costs on rows and columns
                            observed_high_cost_cols = {obs[1] for obs in obs_list if env_copy.costss[t][obs[0], obs[1]] == env_copy.high_cost}
                            observed_low_cost_cols = {obs[1] for obs in obs_list if env_copy.costss[t][obs[0], obs[1]] == env_copy.low_cost}
                            observed_high_cost_rows = {obs[0] for obs in obs_list if env_copy.costss[t][obs[0], obs[1]] == env_copy.high_cost}
                            observed_low_cost_rows = {obs[0] for obs in obs_list if env_copy.costss[t][obs[0], obs[1]] == env_copy.low_cost}
                            observed_high_cost_states = {tuple(obs) for obs in obs_list if env_copy.costss[t][obs[0], obs[1]] == env_copy.high_cost}
                            observed_low_cost_states = {tuple(obs) for obs in obs_list if env_copy.costss[t][obs[0], obs[1]] == env_copy.low_cost}

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
                                    # and (tuple(state) not in obs_list) ## exclude states that are themselves observed, since these would be captured in the 'actual' costs above
                                )
                                self.aligned_arm_gen_low_costs[city, day, t, i] = sum(
                                    1 for state in aligned_states 
                                    if (state[1] in observed_low_cost_cols)
                                    # and (tuple(state) not in obs_list) ## exclude states that are themselves observed, since these would be captured in the 'actual' costs above
                                )

                                ## count how many of the orthogonal states have observations on their respective columns
                                self.orthogonal_arm_gen_high_costs[city, day, t, i] = sum(
                                    1 for state in orthogonal_states 
                                    if (state[1] in observed_high_cost_cols)
                                    # and (tuple(state) not in obs_list) ## exclude states that are themselves observed, since these would be captured in the 'actual' costs above
                                )
                                self.orthogonal_arm_gen_low_costs[city, day, t, i] = sum(
                                    1 for state in orthogonal_states 
                                    if (state[1] in observed_low_cost_cols)
                                    # and (tuple(state) not in obs_list) ## exclude states that are themselves observed, since these would be captured in the 'actual' costs above
                                )
                                

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
                            
                            ## axis overlaps with future states
                            self.path_future_rel_overlaps[city, day, t, i] = env_copy.path_future_rel_overlaps[t][i]
                            self.path_future_irrel_overlaps[city, day, t, i] = env_copy.path_future_irrel_overlaps[t][i]
                    
                        ## misc
                        self.day_costs[city, day, t] = np.nansum(self.total_costs[city, day, :t+1]) ## i.e. costs observed so far today
                        self.path_len[city, day, t] = len(env_copy.path_states[t][0])
                        

                    ## agent receives info from env
                    self.get_env_info(env_copy)

                    ## agent-specific path selection
                    Q_vals = self.compute_Q(env_copy, tree_reset)
                    action_probs = self.softmax(Q_vals)

                    ## action selection
                    assert not np.isnan(np.nansum(Q_vals)), 'no Q estimates": {}'.format(Q_vals)
                    if self.greedy:
                        max_Q = np.nanmax(Q_vals)
                        action = argm(Q_vals, max_Q)
                    else:
                        action = np.random.choice(len(Q_vals), p=action_probs)

                    self.actions[city, day, t] = action
                    self.Q_vals[city, day, t] = Q_vals
                    self.p_choice[city, day, t] = action_probs
                    correct_path = np.argmax(env_copy.path_actual_costs[t])
                    self.p_correct[city, day, t] = self.p_choice[city, day, t][correct_path]



                    # elif agent == 'human': CAN JUST DO CE AGENT HERE
                        
                    #     ## need to trivially set predicted costs to 0 to avoid errors when interacting with the environment
                    #     # env_copy.receive_predictions(np.zeros((N, N)))

                    #     ## or, we might actually calculate the CE-correct answer under the human's observations
                    #     self.root_samples(obs=env_copy.obs, CE=True)
                    #     env_copy.receive_predictions(self.posterior_mean_p_cost)
                    #     path_costs = []
                    #     for path_id in range(env_copy.n_afc):
                    #         path_states = env_copy.path_states[t][path_id]
                    #         path_cost = 0
                    #         for state in path_states:
                    #             path_cost += self.posterior_mean_p_cost[state[0], state[1]]*env_copy.low_cost + (1-self.posterior_mean_p_cost[state[0], state[1]])*env_copy.high_cost ## or, use expected costs
                    #         path_costs.append(path_cost)
                    #     max_cost = np.max(path_costs)
                    #     CE_action = argm(path_costs, max_cost)
                    #     self.CE_actions[city, day, t] = CE_action
                    #     self.CE_Q_vals[city, day, t] = np.array(path_costs)

                        
                        ### get the difference in distributions over total costs of the two paths

                        # ## sample PMFs over total costs for each path
                        # n_samples = 50000
                        # self.root_samples(obs = env_copy.obs, n_samples=n_samples, CE=False)
                        # sample_total_costs = np.zeros((n_samples, env_copy.n_afc))
                        # for s in range(n_samples):
                            
                        #     ## sample binary grid
                        #     sample_costs = np.array([env_copy.high_cost if r>self.all_posterior_p_costs[s].flatten()[ri] else env_copy.low_cost for ri, r in enumerate(np.random.random(env_copy.N**2))]).reshape(env_copy.N, env_copy.N)

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
                        # W = np.zeros((env_copy.n_afc, env_copy.N**2))
                        # for path_id in range(env_copy.n_afc):
                        #     path_states = env_copy.path_states[t][path_id]
                        #     flat_indices = [state[0] * env_copy.N + state[1] for state in path_states]
                        #     W[path_id, flat_indices] = 1


                        # # --- 2. VECTORIZED SAMPLING (Replaces your N_samples loop) ---
                        # n_samples = 10000
                        # self.root_samples(obs = env_copy.obs, n_samples=n_samples, CE=False)
                        # p_costs_flat = self.all_posterior_p_costs.reshape(n_samples, env_copy.N**2)
                        # random_draws = np.random.random((n_samples, env_copy.N**2))
                        # sample_costs_binary = (random_draws < p_costs_flat).astype(int) 
                        # sample_costs_vectorized = sample_costs_binary * env_copy.high_cost + (1 - sample_costs_binary) * env_copy.low_cost

                        # # Step 3: Vectorized Path Summation
                        # sample_total_costs = sample_costs_vectorized @ W.T 


                        ## or, only calculate the difference if the CE's belief does indeed favour the better path - i.e. if CE_aciton == correct_path
                        # if CE_action == correct_path:
                        #     self.distr_diff[city, day, t] = path_distr_diff(sample_total_costs[:,0], sample_total_costs[:,1], correct_path)
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
                        # _, _ = env_copy.take_path(action_sequence)
                        states, costs = env_copy.path_step(action)
                        current = env_copy.current
                        assert np.array_equal(costs, env_copy.trial_obs[:,-1]), 'costs do not match trial observations\n costs: {}, trial_obs: {}'.format(costs, env_copy.trial_obs[:,-1])
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
                    if (not missed) and (not day_terminated):
                        tree_reset = self.update_tree(env_copy, action)
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
                'temp': [],
                'lapse': [],
                'arm_weight': [],
                'horizon': [],
                'real_future_paths': []
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
                        sim_out['temp'].append(self.temp)
                        sim_out['lapse'].append(self.lapse)
                        sim_out['arm_weight'].append(self.arm_weight)
                        sim_out['horizon'].append(self.horizon)
                        sim_out['real_future_paths'].append(self.real_future_paths)

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
    def _cache_arm_weights(self):
        """Cache the aligned/orthogonal weights based on arm_weight parameter."""
        if self.arm_weight is None:
            self._aligned_weight = 1.0
            self._orthogonal_weight = 1.0
        else:
            # arm_weight > 0: favour aligned arm (reduce orthogonal weight)
            # arm_weight < 0: favour orthogonal arm (reduce aligned weight)
            self._aligned_weight = 1 - max(0.0, -self.arm_weight)
            self._orthogonal_weight = 1 - max(0.0, self.arm_weight)
    

## define subclasses
class BAMCP(Farmer):
    def __init__(self, context_prior=0.5, known_context=False,
                 temp=1, lapse=0, arm_weight=0, horizon=3, real_future_paths=True,
                 exploration_constant=None, discount_factor=None, n_samples=None):
        super().__init__(context_prior, known_context,temp, lapse, arm_weight, horizon, real_future_paths, exploration_constant, discount_factor,n_samples)


    ## initialise MCTS object for tree search
    def init_mcts(self, env, reset=True):
        """
        Initialise or update the MCTS object for tree search.
        
        Args:
            env: The environment to use for MCTS.
            reset: If True, create a new MCTS object. If False, update the existing one.
        """
        if reset:
            tree = Tree()
            self.mcts = MonteCarloTreeSearch_AFC(
                env=env, 
                tree=tree, 
                exploration_constant=self.exploration_constant, 
                discount_factor=self.discount_factor, 
                horizon=self.horizon, 
                real_future_paths=self.real_future_paths, 
                aligned_weight=self._aligned_weight, 
                orthogonal_weight=self._orthogonal_weight
            )
        else:
            self.mcts.update_trial()


    ## generate full set of root samples from posterior
    def root_samples(self, n_samples=1000):
        """
        Generic BAMCP root sampling: infer the context posterior, then
        delegate to self.sampler.sample_mdps() to draw n_samples MDPs
        from the task-specific posterior.
        """

        ## create task-specific sampler
        self.sampler = self.make_sampler()

        ## update context posterior
        if not self.known_context:
            context_prior = self.context_prob
            self.context_prob = self.sampler.context_posterior(context_prior=context_prior)
        elif self.known_context:
            self.context_prob = 1.0 if self.known_context == 'column' else 0.0

        ## draw context indicators (True = column context)
        self.context_indicators = np.random.binomial(1, self.context_prob, size=n_samples).astype(bool)

        ## delegate MDP sampling to the sampler
        self.all_posterior_p_costs = self.sampler.sample_mdps(n_samples, self.context_indicators)
        
        ## optional: compute CE mean too
        self.posterior_mean_p_cost = self.sampler.mean_mdp(context_prob = self.context_prob)

    ## tree search using this agent's internal MCTS object
    def search(self):
        """
        Perform MCTS search using this agent's internal MCTS object.
        
        Args:
            n_samples: Number of MCTS samples to use for the search.
            
        Returns:
            action: The selected action.
            MCTS_estimates: The Q-value estimates for each action.
        """
        if self.mcts is None:
            raise ValueError("MCTS object has not been initialized. Call run() with agent='BAMCP' first.")

        ## check root
        assert self.mcts.root_trial == self.mcts.env.trial, 'trial mismatch between env and tree at start of search\n env trial: {} \n tree trial: {}'.format(self.mcts.env.trial, self.mcts.root_trial)
        for a in range(self.mcts.n_afc):
            assert np.array_equal(self.mcts.tree.root.starts[a], self.mcts.env.starts[self.mcts.root_trial][a]), 'start state mismatch for action {}\n env start: {} \n tree start: {}'.format(a, self.mcts.env.starts[self.mcts.root_trial][a], self.mcts.tree.root.starts[a])
            assert np.array_equal(self.mcts.tree.root.goals[a], self.mcts.env.goals[self.mcts.root_trial][a]), 'goal state mismatch for action {}\n env goal: {} \n tree goal: {}'.format(a, self.mcts.env.goals[self.mcts.root_trial][a], self.mcts.tree.root.goals[a])
            assert np.array_equal(self.mcts.tree.root.path_states[a], self.mcts.env.path_states[self.mcts.root_trial][a]), 'path state mismatch for action {}\n env path: {} \n tree path: {}'.format(a, self.mcts.env.path_states[self.mcts.root_trial][a], self.mcts.tree.root.path_states[a])

        ## generate new set of root samples
        self.root_samples(n_samples=self.n_samples)

        ## debugging Q-vals
        self.mcts.Q_tracker = []
        self.mcts.return_tracker = []
        self.mcts.first_node_updates = []
        self.mcts.first_node_updates_by_depth = []
        self.mcts.tree_cost_tracker = []
        self.mcts.conditional_tree_cost_tracker = [[] for _ in range(self.mcts.n_afc)]
        for t in range(self.mcts.env.n_trials):
            self.mcts.first_node_updates_by_depth.append([])
            self.mcts.tree_cost_tracker.append([])
            for a in range(self.mcts.n_afc):
                self.mcts.conditional_tree_cost_tracker[a].append([])
        
        ## loop through simulations
        for s in range(self.n_samples):
            
            ## root sampling of new posterior
            posterior_p_cost = self.all_posterior_p_costs[s]
            self.mcts.env.receive_predictions(posterior_p_cost)

            ## selection, expansion, simulation
            action_leaf = self.mcts.tree_policy()
            self.mcts.rollout_policy(action_leaf)
            
            ##backup
            self.mcts.backup()

            ## update Q tracker
            try:
                Qs = [self.mcts.tree.root.action_leaves[a].performance for a in self.mcts.tree.root.action_leaves.keys()]
                self.mcts.Q_tracker.append(Qs)
            except:
                pass

        ## return final Q estimates
        MCTS_estimates = np.full(self.mcts.n_afc, np.nan)
        for action, leaf in self.mcts.tree.root.action_leaves.items():
            MCTS_estimates[action] = leaf.performance

        return MCTS_estimates


    ## get MCTS Q estimates
    def compute_Q(self, env_copy, tree_reset=True):

        ## reset tree (or reuse it)
        self.init_mcts(env=env_copy, reset=tree_reset)
        assert self.mcts.env.trial == self.mcts.root_trial, 'trial mismatch between env and MCTS\n env: {} \n MCTS: {}'.format(env_copy.trial, self.mcts.root_trial)
        assert self.mcts.env.sim == True, 'env not in sim mode'
        self.mcts.root_state = env_copy.current

        ## search
        MCTS_Q = self.search()

        # for a in range(self.n_afc):
        #     self.leaf_visits[city, day, t, a] = self.mcts.tree.root.action_leaves[a].n_action_visits

        ## debug plot
        # print(t,': BAMCP Q vals:', MCTS_Q, 'CE path costs:', CE_path_costs, 'actual path costs:', env_copy.path_actual_costs[t])
        # for a in range(self.n_afc):
        #     print(MCTS.tree.root.action_leaves[a])
        #     print(env_copy.starts[t][a])
        #     print(env_copy.goals[t][a])
        # fig, axs = plt.subplots(1,1, figsize=(5,5))
        # plot_r(self.posterior_mean_p_cost, axs, title = 'Posterior reward distribution\nmean root sample\npost obs')
        # plot_traj([env_copy.path_states[self.mcts.root_trial][c] for c in range(self.n_afc)], ax = axs)
        # plot_obs(env_copy.obs, ax = axs, text=True)
        # plt.show()


        # ### let's also calculate CE choice under BAMCP's knowledge 

        # ## get the cost of each path under the posterior mean
        # CE_path_costs = []
        # for path_id in range(env_copy.n_afc):
        #     path_states = env_copy.path_states[t][path_id]
        #     CE_path_cost = 0
        #     for state in path_states:
        #         # path_cost += env_copy.get_pred_cost(state) ## i.e. sample binary costs from the posterior pqs
        #         CE_path_cost += self.posterior_mean_p_cost[state[0], state[1]]*env_copy.low_cost + (1-self.posterior_mean_p_cost[state[0], state[1]])*env_copy.high_cost ## or, use expected costs
        #     CE_path_costs.append(CE_path_cost)

        # ## CE chooses the path with the lowest total cost
        # max_cost = np.max(CE_path_costs)
        # CE_action = argm(CE_path_costs, max_cost)
        # self.CE_actions[city, day, t] = CE_action
        # self.CE_Q_vals[city, day, t] = np.array(CE_path_costs)
        # self.CE_p_choice[city, day, t] = self.softmax(np.array(CE_path_costs))
        # self.CE_p_correct[city, day, t] = self.CE_p_choice[city, day, t][correct_path]

        return MCTS_Q
    

    ## update the MCTS tree
    def update_tree(self, env_copy, action):

        ## prune tree (not always successful due to high branching factor, or if participant made no choice in which case reset the tree)
        init_info_state = self.mcts.tree.root.node_id
        trial_obs = env_copy.trial_obs.copy()
        t = self.mcts.root_trial
        next_node_id = self.mcts.init_node_id(trial_obs, init_info_state, t)

        if next_node_id in self.mcts.tree.root.action_leaves[action].children:
            self.mcts.tree.prune(action, next_node_id)
            # assert np.array_equal(self.mcts.tree.root.belief_state[2*self.mcts.n_afc:], costs), 'error in root update\n root state: {} \n costs: {}'.format(self.mcts.tree.root.belief_state[2*self.mcts.n_afc:], costs)
            # assert np.array_equal(self.mcts.tree.root.belief_state[1:], costs), 'error in root update\n root state: {} \n costs: {}'.format(self.mcts.tree.root.belief_state[2*self.mcts.n_afc:], costs)
            assert self.mcts.tree.root.trial == t+1, 'trial mismatch after pruning\n env trial: {}, MCTS trial: {}'.format(t, self.mcts.tree.root.trial)
            tree_reset = False
        else:
            tree_reset = True

        ## hacky: unless full BAMCP with real future paths and full horizon, reset the tree
        if (not self.real_future_paths) or (self.horizon < (env_copy.n_trials-t)):
            tree_reset = True
    
        return tree_reset
        



class CE(Farmer):
    def __init__(self, context_prior=0.5, known_context=False,
                 temp=1, lapse=0, arm_weight=0, horizon=None, real_future_paths=None,
                 exploration_constant=None, discount_factor=None, n_samples=None):
        super().__init__(context_prior, known_context,temp, lapse, arm_weight, horizon, real_future_paths, exploration_constant, discount_factor,n_samples)

    ## method for computing posterior mean MDP
    def mean_grid(self, n_samples=1000):

        ## create task-specific sampler
        self.sampler = self.make_sampler()
            
        ## update context posterior
        if not self.known_context:
            context_prior = self.context_prob
            self.context_prob = self.sampler.context_posterior(context_prior=context_prior)
        elif self.known_context:
            self.context_prob = 1.0 if self.known_context == 'column' else 0.0

        ## delegate mean MDP computation to the sampler
        self.posterior_mean_p_cost = self.sampler.mean_mdp(context_prob=self.context_prob)

    
    ## act based on posterior mean grid
    def compute_Q(self, env_copy, tree_reset=True):
        
        ## get posterior mean grid
        env_copy.set_sim(False)
        self.mean_grid()
        env_copy.receive_predictions(self.posterior_mean_p_cost)

        ## get the cost of each path under the posterior mean (weighted by arm_weight)
        t = env_copy.trial
        path_costs = []
        for path_id in range(env_copy.n_afc):
            path_states = env_copy.path_states[t][path_id]
            aligned_states, orthogonal_states = env_copy.path_aligned_states[t][path_id], env_copy.path_orthogonal_states[t][path_id]
            unweighted_pred_costs = self.posterior_mean_p_cost*env_copy.low_cost + (1-self.posterior_mean_p_cost)*env_copy.high_cost
            total_weighted_path_costs = env_copy.arm_reweighting(unweighted_pred_costs, aligned_states, orthogonal_states, self._aligned_weight, self._orthogonal_weight)
            path_costs.append(total_weighted_path_costs)
        
        ## debugging
        # print(t,': CE_one_arm path costs:', path_costs)
        # fig, axs = plt.subplots(1,1, figsize=(5,5))
        # plot_r(self.posterior_mean_p_cost, axs, title = 'Posterior reward distribution\nmean root sample\npost obs')
        # plot_traj([env_copy.path_states[t][c] for c in range(self.n_afc)], ax = axs)
        # plot_obs(env_copy.obs, ax = axs, text=True)
        # plt.show()
        # if t==3:
        #     raise Exception('check this')

        return np.array(path_costs)

    
    ## trivially need to do this
    def update_tree(self, env_copy, action):
        return True  # default: always reset