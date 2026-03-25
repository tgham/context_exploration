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
from base_kernels import *
from MCTS import MonteCarloTreeSearch_AFC, MonteCarloTreeSearch_Bandit
from tqdm.auto import tqdm
import pandas as pd
from scipy.special import beta, logsumexp, digamma, comb, betaln


### base farmer model?
class Farmer(ABC):

    def __init__(self,
                 temp=1, lapse=0, horizon=3,
                 exploration_constant=None, discount_factor=None, n_samples=None,
                 **task_params):

        ## behavioural parameters
        self.temp = temp
        self.lapse = lapse

        ## task-specific parameters (passed through to env via receive_task_params)
        self.task_params = task_params

        ## MCTS parameters
        self.horizon = horizon
        self.exploration_constant = exploration_constant
        self.discount_factor = discount_factor
        self.n_samples = n_samples
    
    
    ### general methods for sampling, choice, fitting etc.

    ## create a sampler object from the environment's current state
    def init_sampler(self, env):
        """Delegate to the environment's task-specific sampler factory."""

        ## pass task-specific parameters to env (not all envs support this)
        if hasattr(env, 'receive_task_params'):
            env.receive_task_params(self.task_params)

        ## make sampler if there is not one, otherwise we just need to update the sampler's observations
        # if not hasattr(self, 'sampler') or self.sampler is None:
        #     self.sampler = env.make_sampler()
        # else:
        #     self.sampler.set_obs(env.obs)
        self.sampler = env.make_sampler()

        ## if dealing with context priors, give this to the sampler
        if hasattr(self, 'context_prior'):
            self.sampler.context_prior = self.context_prior

            
    ## agent-specific calculation of Q values based on posterior
    @abstractmethod
    def compute_Q(self, env_copy, tree_reset=True):
        pass

    ## agent-specific CE (certainty-equivalent) Q values
    @abstractmethod
    def compute_CE_Q(self, env_copy):
        pass

    ## choice function
    def softmax(self, Q):
        CPs = (1-self.lapse) * softmax(Q/self.temp) + self.lapse/len(Q)
        return CPs


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
    

    ### bespoke methods 
        
    ## run agent on participant's trial sequence
    def run(self, hyperparams, agent = 'CE', df_trials=None, envs=None,fit=True, yoked=False, progress=False):
        
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
            
            ## expts 1-2 - unknown context
            start_of_day_context_prior = 0.5 

            ## loop through days
            for day in range(n_days):

                ## get the environment for this day
                if envs:
                    env = envs['city_{}_grid_{}_env_object'.format(city+1, day+1)][0]

                    ## need to do some fixes for old envs
                    if env.expt == '2AFC':
                        env.expt = 'AFC'
                    
                    ## get context alignment of states
                    env.path_aligned_states, env.path_orthogonal_states, env.path_weights = env.get_alignment(env.path_states)

                env_copy = copy.deepcopy(env)
                assert not hasattr(env_copy, 'obs'), 'env_copy.obs should not exist before the first trial: {}'.format(len(env_copy.obs),', city:', city+1, 'day:', day+1)
                
                ## expt 3 - context is known
                if env_copy.context == 'column':
                    start_of_day_context_prior = 1.0
                elif env_copy.context == 'row':
                    start_of_day_context_prior = 0.0
                
                ## context prior resets
                self.context_prior = start_of_day_context_prior


                ## FIX FOR OLD ENVS: rename some attributes (episode --> trial, etc.)
                if hasattr(env_copy, 'n_episodes'):
                    env_copy.n_trials = env_copy.n_episodes

                ## initialise planner
                self.mcts = None
                tree_reset = True


                ## loop through trials within day
                for t in range(n_trials):

                    ## reset env/trial
                    env_copy.reset()
                    env_copy.set_sim(True)

                    ### if extracting useful behavioural measures, i.e. yoking to human choices

                    ## overlaps with previous observations
                    if yoked:
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
                                # self.aligned_arm_actual_high_costs[city, day, t, i] = sum(1 for state in aligned_states if state in observed_high_cost_states)
                                # self.aligned_arm_actual_low_costs[city, day, t, i] = sum(1 for state in aligned_states if state in observed_low_cost_states)

                                # ## count how many of the orthogonal states have actual high and low costs
                                # self.orthogonal_arm_actual_high_costs[city, day, t, i] = sum(1 for state in orthogonal_states if state in observed_high_cost_states)
                                # self.orthogonal_arm_actual_low_costs[city, day, t, i] = sum(1 for state in orthogonal_states if state in observed_low_cost_states)

                                
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
                                # self.aligned_arm_actual_high_costs[city, day, t, i] = sum(
                                #     1 for state in aligned_states 
                                #     if state in observed_high_cost_states)
                                # self.aligned_arm_actual_low_costs[city, day, t, i] = sum(
                                #     1 for state in aligned_states if state in observed_low_cost_states)

                                # ## count how many of the orthogonal states have actual high and low costs
                                # self.orthogonal_arm_actual_high_costs[city, day, t, i] = sum(
                                #     1 for state in orthogonal_states if state in observed_high_cost_states)
                                # self.orthogonal_arm_actual_low_costs[city, day, t, i] = sum(
                                #     1 for state in orthogonal_states if state in observed_low_cost_states)

                                
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
                        
                    
                    ## agent-specific path selection
                    Q_vals = self.compute_Q(env_copy, tree_reset)
                    action_probs = self.softmax(Q_vals)

                    ## action selection
                    assert not np.isnan(np.nansum(Q_vals)), 'no Q estimates": {}'.format(Q_vals)
                    if self.greedy:
                        max_Q = np.nanmax(Q_vals)
                        action = np.argmax(Q_vals)
                    else:
                        action = np.random.choice(len(Q_vals), p=action_probs)

                    self.actions[city, day, t] = action
                    self.Q_vals[city, day, t] = Q_vals
                    self.p_choice[city, day, t] = action_probs
                    correct_path = np.argmax(env_copy.path_actual_costs[t])
                    self.p_correct[city, day, t] = self.p_choice[city, day, t][correct_path]

                    ## let's also calculate the CE choice under the current agent's knowledge
                    CE_Q_vals = self.compute_CE_Q(env_copy)
                    CE_action = np.argmax(CE_Q_vals)
                    CE_action_probs = self.softmax(CE_Q_vals)
                    self.CE_Q_vals[city, day, t] = CE_Q_vals
                    self.CE_actions[city, day, t] = CE_action
                    self.CE_p_choice[city, day, t] = CE_action_probs
                    self.CE_p_correct[city, day, t] = self.CE_p_choice[city, day, t][correct_path]


                    ### take ppt's action if a) we are fitting, or b) we are extracting behavioural measures by yoking to ppt's choices
                    missed=False
                    if (fit) or (yoked):

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
                        action_sequence = env_copy.path_actions[t][action]
                        _, cost, _, _, _ = env_copy.step(action)
                        self.total_costs[city, day, t] = cost
                        self.path_quality[city, day, t] = self.total_costs[city, day, t]/self.path_len[city, day, t] ## i.e. cost as a proportion of path length
                        day_terminated = t == (n_trials-1)


                    ## else, skip to next trial??
                    else:
                        env_copy._trial += 1 
                        print('skipping city {}, day {}, trial {} because participant missed their choice'.format(city+1, day+1, t+1))

                    
                    ## update MCTS tree
                    if (not missed) and (not day_terminated):
                        tree_reset = self.update_tree(env_copy, action)
                    else:
                        tree_reset = True
                    
                    ## update the sampler with the new observations
                    self.sampler.set_obs(env_copy.obs)
                    
                    ## get the context prior - i.e. the probability with which samples were drawn
                    context_prior = self.context_prior ## this is the prior that was used to generate the samples for this trial, which we will need to calculate the context posterior

                    ## update the context prior for the next trial
                    context_posterior = self.sampler.update_context_posterior(start_of_day_context_prior)
                    self.context_prior = context_posterior
                    self.context_priors[city, day, t] = context_prior
                    self.context_posteriors[city, day, t] = context_posterior

                    ## carry over the context prob to the next run, if on the final trial of the day
                    if t == (n_trials-1):
                        start_of_day_context_prior = context_posterior

                        ## also need to clear sampler
                        self.sampler = None

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



## define subclasses
class BAMCP(Farmer):

    _mcts_class = MonteCarloTreeSearch_AFC

    def __init__(self,
                 temp=1, lapse=0, horizon=3,
                 exploration_constant=None, discount_factor=None, n_samples=None,
                 **task_params):
        super().__init__(temp, lapse, horizon, exploration_constant, discount_factor, n_samples,
                         **task_params)
        self.arm_weight = task_params.get('arm_weight', 0)


    ## CE Q values: act based on posterior mean grid (AFC-specific)
    def compute_CE_Q(self, env_copy):

        ## get task-specific sampler
        self.init_sampler(env_copy)

        ## get posterior mean grid
        self.posterior_mean_MDP = self.sampler.mean_mdp()

        ## get the cost of each path under the posterior mean (weighted by arm_weight)
        t = env_copy.trial
        path_costs = []
        for path_id in range(env_copy.n_afc):
            path = env_copy.path_states[t][path_id]
            path_weight_idx = env_copy.path_weights[t][path_id]
            try:
                weighted_costs = [float(env_copy.costs[x, y]) * env_copy.sim_weight_map[path_weight_idx[k]] for k, (x, y) in enumerate(path)]
            except:
                weighted_costs = [float(env_copy.costss[t][x, y]) * env_copy.sim_weight_map[path_weight_idx[k]] for k, (x, y) in enumerate(path)]
            path_costs.append(sum(weighted_costs))

        CE_Q = np.array(path_costs)
        return CE_Q


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
            self.mcts = self._mcts_class(
                env=env,
                tree=tree,
                exploration_constant=self.exploration_constant,
                discount_factor=self.discount_factor,
                horizon=self.horizon,
            )
        else:
            self.mcts.refresh_env(env)


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

        ## generate new set of root samples
        self.all_posterior_MDPs = self.sampler.sample_mdps(self.n_samples)

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
            # posterior_MDP = self.all_posterior_MDPs[s]
            # self.mcts.env.receive_predictions(posterior_MDP)
            self.mcts.env = self.all_posterior_MDPs[s]

            ## selection, expansion, simulation
            action_leaf = self.mcts.traverse_tree()
            self.mcts.rollout(action_leaf)
            
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

        ## get task-specific sampler
        self.init_sampler(env_copy)

        ## reset tree (or reuse it)
        self.init_mcts(env=env_copy, reset=tree_reset)
        assert self.mcts.env.trial == self.mcts.root_trial, 'trial mismatch between env and MCTS\n env: {} \n MCTS: {}'.format(env_copy.trial, self.mcts.root_trial)
        assert self.mcts.env.sim == True, 'env not in sim mode'


        ## search
        MCTS_Q = self.search()

        # ## debugging plot
        # # toplot_mean = np.mean([MDP.costs == self.sampler.low_cost for MDP in self.all_posterior_MDPs], axis=0) ## mean of binary grids
        # toplot_mean = self.sampler.mean_mdp() ## or actual posterior mean
        # fig, axs = plt.subplots(1,1, figsize=(5,5))
        # plot_r(toplot_mean, axs, title = 'Posterior reward distribution\nmean of all root samples\nMCTS Q: {}'.format(np.round(MCTS_Q,2)))
        # plot_traj([env_copy.path_states[self.mcts.root_trial][c] for c in range(self.mcts.n_afc)], ax = axs)
        # plot_obs(env_copy.obs, ax = axs, text=True)
        # plt.show()

        return MCTS_Q
    

    ## update the MCTS tree
    def update_tree(self, env_copy, action):

        ## prune tree (not always successful due to high branching factor, or if participant made no choice in which case reset the tree)
        init_info_state = self.mcts.tree.root.node_id
        trial_obs = env_copy.trial_obs.copy()
        t = self.mcts.root_trial
        next_node_id = self.mcts.init_node_id(trial_obs, init_info_state)

        if next_node_id in self.mcts.tree.root.action_leaves[action].children:
            self.mcts.tree.prune(action, next_node_id)
            # assert np.array_equal(self.mcts.tree.root.belief_state[2*self.mcts.n_afc:], costs), 'error in root update\n root state: {} \n costs: {}'.format(self.mcts.tree.root.belief_state[2*self.mcts.n_afc:], costs)
            # assert np.array_equal(self.mcts.tree.root.belief_state[1:], costs), 'error in root update\n root state: {} \n costs: {}'.format(self.mcts.tree.root.belief_state[2*self.mcts.n_afc:], costs)
            assert self.mcts.tree.root.trial == t+1, 'trial mismatch after pruning\n env trial: {}, MCTS trial: {}'.format(t, self.mcts.tree.root.trial)
            tree_reset = False
        else:
            tree_reset = True

        ## hacky: unless full BAMCP with real future paths and full horizon, reset the tree
        # if (not self.real_future_paths) or (self.horizon < (env_copy.n_trials-t-1)):
        if self.horizon < (env_copy.n_trials-t-1):
            tree_reset = True
    
        return tree_reset
        



class CE(Farmer):
    def __init__(self,
                 temp=1, lapse=0, horizon=None,
                 exploration_constant=None, discount_factor=None, n_samples=None,
                 **task_params):
        super().__init__(temp, lapse, horizon, exploration_constant, discount_factor, n_samples,
                         **task_params)
        self.arm_weight = task_params.get('arm_weight', 0)

    
    ## CE Q values: act based on posterior mean grid (AFC-specific)
    def compute_CE_Q(self, env_copy):

        ## get task-specific sampler
        self.init_sampler(env_copy)

        ## get posterior mean grid
        self.posterior_mean_MDP = self.sampler.mean_mdp()

        ## get the cost of each path under the posterior mean (weighted by arm_weight)
        t = env_copy.trial
        path_costs = []
        for path_id in range(env_copy.n_afc):
            path = env_copy.path_states[t][path_id]
            path_weight_idx = env_copy.path_weights[t][path_id]
            try:
                weighted_costs = [float(env_copy.costs[x, y]) * env_copy.sim_weight_map[path_weight_idx[k]] for k, (x, y) in enumerate(path)]
            except:
                weighted_costs = [float(env_copy.costss[t][x, y]) * env_copy.sim_weight_map[path_weight_idx[k]] for k, (x, y) in enumerate(path)]
            path_costs.append(sum(weighted_costs))

        CE_Q = np.array(path_costs)
        return CE_Q

    ## act based on posterior mean grid
    def compute_Q(self, env_copy, tree_reset=None):
        CE_Q = self.compute_CE_Q(env_copy)
        return CE_Q


    ## trivially need to do this
    def update_tree(self, env_copy, action):
        return True  # default: always reset
    




# ---------------------------------------------------------------------------
# 3. BAMCP agent for bandits
# ---------------------------------------------------------------------------
class BanditBAMCP(BAMCP):
    """
    BAMCP agent for multi-armed bandits.

    Inherits all planning logic (init_sampler, init_mcts, search, compute_Q)
    from BAMCP. Only the run loop differs to suit the bandit task structure.
    """

    _mcts_class = MonteCarloTreeSearch_Bandit

    def __init__(self, n_samples=200, exploration_constant=2,
                 discount_factor=0.99, horizon=5, temp=1.0, lapse=0.0):
        super().__init__(
            temp=temp, lapse=lapse, horizon=horizon,
            exploration_constant=exploration_constant,
            discount_factor=discount_factor,
            n_samples=n_samples,
        )

    ## CE Q values: posterior mean reward for each arm
    def compute_CE_Q(self, env_copy):
        self.init_sampler(env_copy)
        mean_ps = self.sampler.mean_probs()

        ## expected reward = p(success) * r_dist for each arm
        r_dist = env_copy.r_dist
        CE_Q = np.array([mean_ps[a] * r_dist[a] for a in range(env_copy.n_afc)])
        return CE_Q

    def run(self, env, greedy=False, verbose=True):
        """Run the agent on the bandit for n_trials, returning trial-by-trial data."""
        n_trials = env.n_trials
        n_arms = env.n_afc

        Q_history = np.zeros((n_trials, n_arms))
        p_choice_history = np.zeros((n_trials, n_arms))
        actions = np.zeros(n_trials, dtype=int)
        rewards = np.zeros(n_trials)
        chose_optimal = np.zeros(n_trials, dtype=bool)
        bayes_regret = np.zeros(n_trials)
        post_probs = np.zeros((n_trials, n_arms))

        ## CE tracking — what would the CE agent have chosen with the same info?
        CE_Q_history = np.zeros((n_trials, n_arms))
        CE_actions = np.zeros(n_trials, dtype=int)
        CE_chose_optimal = np.zeros(n_trials, dtype=bool)
        CE_bayes_regret = np.zeros(n_trials)

        optimal_arm = np.argmax(env.p_dist)

        env.reset()

        for t in range(n_trials):
            env_copy = copy.deepcopy(env)
            env_copy.set_sim(True)

            Q = self.compute_Q(env_copy)
            probs = self.softmax(Q)
            mean_probs = self.sampler.mean_probs()

            ## CE Q values under the same posterior
            CE_Q = self.compute_CE_Q(env_copy)
            CE_action = int(np.argmax(CE_Q))

            Q_history[t] = Q
            p_choice_history[t] = probs
            post_probs[t] = mean_probs
            CE_Q_history[t] = CE_Q
            CE_actions[t] = CE_action
            CE_chose_optimal[t] = (CE_action == optimal_arm)
            CE_bayes_regret[t] = env.p_dist[optimal_arm] - env.p_dist[CE_action]

            if greedy:
                max_Q = np.nanmax(Q)
                best_arms = np.where(Q == max_Q)[0]
                action = int(np.random.choice(best_arms))

                # debugging
                if action != CE_action:
                    if verbose:
                        print(f"Trial {t+1}: Greedy action {action} differs from CE action {CE_action} with Q values {Q} and CE Q values {CE_Q}")
            else:
                max_Q = np.nanmax(Q)
                best_arms = np.where(Q == max_Q)[0]
                if len(best_arms) > 1:
                    action = int(np.random.choice(best_arms))
                else:
                    action = int(best_arms[0])

            env.set_sim(False)
            _, reward, terminated, truncated, _ = env.step(action)

            actions[t] = action
            rewards[t] = reward
            chose_optimal[t] = (action == optimal_arm)
            bayes_regret[t] = env.p_dist[optimal_arm] - env.p_dist[action]

            self.init_sampler(env)

            if verbose:
                print(f"  trial {t+1:>3}/{n_trials}  \n Q={np.round(Q, 3)}  CE_Q={np.round(CE_Q, 3)}"
                      f"p={np.round(probs, 3)} \n pulled arm {action} with reward {reward:.0f}")

            if terminated:
                break

        return {
            'Q': Q_history,
            'p_choice': p_choice_history,
            'actions': actions,
            'rewards': rewards,
            'cumulative_reward': np.cumsum(rewards),
            'chose_optimal': chose_optimal,
            'true_probs': env.p_dist.copy(),
            'optimal_arm': optimal_arm,
            'bayes_regret': bayes_regret,
            'post_probs': post_probs,
            'CE_Q': CE_Q_history,
            'CE_actions': CE_actions,
            'CE_chose_optimal': CE_chose_optimal,
            'CE_bayes_regret': CE_bayes_regret,
        }


class BanditCE(BanditBAMCP):
    """CE (certainty-equivalent) agent for multi-armed bandits.
    Picks the arm with the highest posterior mean expected reward — no tree search."""

    def __init__(self, temp=1.0, lapse=0.0):
        super().__init__(n_samples=0, exploration_constant=0,
                         discount_factor=1.0, horizon=0,
                         temp=temp, lapse=lapse)

    def compute_Q(self, env_copy, tree_reset=None):
        return self.compute_CE_Q(env_copy)