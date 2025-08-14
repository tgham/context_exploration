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
    
    ## calculate the KL divergence summed over all grid
    def grid_KL(self, prior_obs, posterior_obs, context = 'column'):

        prior_sampler = GridSampler(self.alpha_row, self.beta_row, self.alpha_col, self.beta_col, self.low_cost, self.high_cost, prior_obs, N=self.N)
        posterior_sampler = GridSampler(self.alpha_row, self.beta_row, self.alpha_col, self.beta_col, self.low_cost, self.high_cost, posterior_obs, N=self.N)

        if prior_obs is None:
            n_prior_obs = 0
        else:
            n_prior_obs = len(prior_obs)
        n_posterior_obs = len(posterior_obs)

        KLs = []
        if context == 'column':
            for i in range(self.N):

                ## define prior params - that is, the params before the new observations are made
                low_counts_prior = prior_sampler.low_counts_cols[i]
                high_counts_prior = prior_sampler.high_counts_cols[i]
                alpha_prior = prior_sampler.alpha_col + low_counts_prior
                beta_prior = prior_sampler.beta_col + high_counts_prior

                ## repeat for posterior params
                low_counts_post = posterior_sampler.low_counts_cols[i]
                high_counts_post = posterior_sampler.high_counts_cols[i]
                alpha_post = posterior_sampler.alpha_col + low_counts_post
                beta_post = posterior_sampler.beta_col + high_counts_post

                ## calculate the difference in low and high counts
                low_counts_diff = low_counts_post - low_counts_prior
                high_counts_diff = high_counts_post - high_counts_prior

                ## calculate the KL divergence
                KL = self.KL_divergence(alpha_prior, beta_prior, alpha_post, beta_post, low_counts_diff, high_counts_diff) 
                KLs.append(KL)

        elif context == 'row':
            for j in range(self.N):

                ## define prior params - that is, the params before the new observations are made
                low_counts_prior = prior_sampler.low_counts_rows[j]
                high_counts_prior = prior_sampler.high_counts_rows[j]
                alpha_prior = prior_sampler.alpha_row + low_counts_prior
                beta_prior = prior_sampler.beta_row + high_counts_prior

                ## repeat for posterior params
                low_counts_post = posterior_sampler.low_counts_rows[j]
                high_counts_post = posterior_sampler.high_counts_rows[j]
                alpha_post = posterior_sampler.alpha_row + low_counts_post
                beta_post = posterior_sampler.beta_row + high_counts_post

                ## calculate the difference in low and high counts
                low_counts_diff = low_counts_post - low_counts_prior
                high_counts_diff = high_counts_post - high_counts_prior

                ## calculate the KL divergence
                KL = self.KL_divergence(alpha_prior, beta_prior, alpha_post, beta_post, low_counts_diff, high_counts_diff) 
                KLs.append(KL)
        
        return np.sum(KLs)
    

    ## KL divergence between two beta distributions
    def KL_divergence(self, alpha_q, beta_q, alpha_p, beta_p, low_counts_diff, high_counts_diff):

        """
        Calculate the KL divergence between a prior Q and posterior P beta distribution.
        The prior is a beta distribution with parameters alpha and beta,
        and the posterior is a beta distribution with parameters alpha + s and beta + f,
        where s is the number of successes and f is the number of failures."""

        assert alpha_q + low_counts_diff == alpha_p, 'alpha_q + s should equal alpha_p, but got {} + {} != {}'.format(alpha_q, low_counts_diff, alpha_p)
        KL = np.log((beta(alpha_p, beta_p) / beta(alpha_q, beta_q))) - low_counts_diff*(digamma(alpha_q) - digamma(alpha_q + beta_q)) - high_counts_diff*(digamma(beta_q) - digamma(alpha_q + beta_q))

        return KL
    

    ## calculate the expected KL of an obs sequence of length n 
    # def expected_KL(self, prior_obs, new_obs, context):
        
    #     ## loop through possible sequences of n binary outcomes (0=low, 1=high)
    #     EKL_total = 0.0
    #     n_obs = len(new_obs)
    #     for seq in product([self.low_cost, self.high_cost], repeat=n_obs):

    #         ## initialise a farmer based on the prior observations (+ the new hypothetical obs)
    #         prior_sampler = GridSampler(self.alpha_row, self.beta_row, self.alpha_col, self.beta_col, self.low_cost, self.high_cost, prior_obs, N=self.N, CE=True) ## can use CE since we are only interested in the prior mean p and q
    #         prior_sampler.simple_sample(col_context=context=='column')
            
    #         ## loop through hypothetical observations
    #         obs_tmp = prior_obs.copy() if prior_obs is not None else np.array([])
    #         p_seq = 1.0
            
    #         ### update the sampled after each hypothetical observation??
    #         if context == 'column':
    #             for oi, outcome in enumerate(seq):

    #                 ## initialise a farmer based on the prior observations (+ the new hypothetical obs)
    #                 # prior_sampler = GridSampler(self.alpha_row, self.beta_row, self.alpha_col, self.beta_col, self.low_cost, self.high_cost, obs_tmp, N=self.N, CE=True) ## can use CE since we are only interested in the prior mean p and q
                    
    #                 ## append prior obs with hypothetical obs
    #                 i, j, _ = new_obs[oi]
    #                 obs_tmp = np.vstack([obs_tmp, [i,j, outcome]]) if len(obs_tmp)>0 else np.array([[i,j, outcome]])

    #                 ## get prior predictive probability of a low cost in this state
    #                 # prior_sampler.simple_sample(col_context=True)
    #                 p_obs = prior_sampler.col_probs[0][int(j)]

    #                 ## update prob of observing this sequence
    #                 p_seq *= p_obs if outcome == self.low_cost else (1 - p_obs)



    #         elif context == 'row':
    #             for oi, outcome in enumerate(seq):

    #                 ## initialise a farmer based on the prior observations (+ the new hypothetical obs)
    #                 # prior_sampler = GridSampler(self.alpha_row, self.beta_row, self.alpha_col, self.beta_col, self.low_cost, self.high_cost, obs_tmp, N=self.N, CE=True) ## can use CE since we are only interested in the prior mean p and q
                    
    #                 ## append prior obs with hypothetical obs
    #                 i, j, _ = new_obs[oi]
    #                 obs_tmp = np.vstack([obs_tmp, [i,j, outcome]]) if len(obs_tmp)>0 else np.array([[i,j, outcome]])

    #                 ## get prior predictive probability of a low cost in this state
    #                 # prior_sampler.simple_sample(col_context=False)
    #                 p_obs = prior_sampler.row_probs[0][int(i)]

    #                 ## update prob of observing this sequence
    #                 p_seq *= p_obs if outcome == self.low_cost else (1 - p_obs)
            
    #         # Compute KL for this hypothetical final posterior
    #         KL_seq = self.grid_KL(prior_obs, obs_tmp, context)
    #         # print('seq:', seq, 'p_seq:', p_seq, 'KL_seq',KL_seq)
    #         EKL_total += p_seq * KL_seq

    #     return EKL_total

    def beta_binomial_pmf(self, k, n, a, b):
        """Beta-binomial predictive probability p(k | a, b, n)."""
        return comb(n, k) * np.exp(betaln(a + k, b + n - k) - betaln(a, b))

    def expected_KL(self, prior_obs, new_obs, context):
        """
        Compute the expected KL divergence for a planned set of observations
        without enumerating all binary sequences, using the beta-binomial formula.
        """
        EKL_total = 0.0
        
        # Initialise prior sampler to get current alpha/beta params after prior_obs
        prior_sampler = GridSampler(self.alpha_row, self.beta_row,
                                    self.alpha_col, self.beta_col,
                                    self.low_cost, self.high_cost,
                                    prior_obs, N=self.N, CE=True)
        prior_sampler.simple_sample(col_context=(context == 'column'))

        # Group planned observations by row (row context) or col (column context)
        if context == 'row':
            group_key = lambda obs: obs[0]  # group by row index
            alphas = prior_sampler.alpha_row
            betas = prior_sampler.beta_row
        else:
            group_key = lambda obs: obs[1]  # group by column index
            alphas = prior_sampler.alpha_col
            betas = prior_sampler.beta_col

        grouped = {}
        for i, j, _ in new_obs:
            key = group_key((i, j, None))
            grouped.setdefault(key, []).append((i, j))
        
        # Iterate over each row/col group independently
        for idx, obs_list in grouped.items():
            n_i = len(obs_list)       # planned obs in this row/col
            # a_i = alphas[idx]
            # b_i = betas[idx]
            a_i = alphas ## hacky - these need to be alpha + low_counts etc.
            b_i = betas
            
            # Sum over k successes
            E_KL_i = 0.0
            for k in range(n_i + 1):
                p_k = self.beta_binomial_pmf(k, n_i, a_i, b_i)
                KL_k = self.KL_divergence(a_i, b_i, a_i + k, b_i + n_i - k, k, n_i-k)
                E_KL_i += p_k * KL_k
            print('n_i:', n_i, 'idx:', idx, 'obs_list:', obs_list, 'E_KL_i:', E_KL_i)
            
            EKL_total += E_KL_i
        
        return EKL_total





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
        greedy = hyperparams['greedy'] if 'greedy' in hyperparams else True 

        ## initialise model's internal variables
        self.n_afc = n_afc ## can sort this out later
        self.p_choice = np.zeros((n_cities, n_days, n_trials, self.n_afc))
        self.p_correct = np.zeros((n_cities, n_days, n_trials))
        self.Q_vals = np.zeros((n_cities, n_days, n_trials, self.n_afc))
        self.actions = np.zeros((n_cities, n_days, n_trials))
        self.CE_actions = np.zeros((n_cities, n_days, n_trials))
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
        self.path_past_overlaps = np.zeros((n_cities, n_days, n_trials, self.n_afc))
        self.path_past_observed_costs = np.zeros((n_cities, n_days, n_trials, self.n_afc))
        self.path_past_observed_no_costs = np.zeros((n_cities, n_days, n_trials, self.n_afc))
        self.path_len = np.zeros((n_cities, n_days, n_trials))
        self.day_costs = np.zeros((n_cities, n_days, n_trials)) ## i.e. the cost of the path chosen by the participant on that trial

        
        ## init params and hyperparams
        if agent == 'BAMCP':
            self.temp = params[0]
            self.lapse = params[1]
            n_sims = hyperparams['n_sims']
            exploration_constant = hyperparams['exploration_constant']
            discount_factor = hyperparams['discount_factor']
            n_iter = hyperparams['n_iter']
        elif agent == 'CE':
            self.temp = params[0]
            self.lapse = params[1]
            # n_sims = hyperparams['n_sims']
            # n_iter = hyperparams['n_iter']
        # elif agent == 'human': ## hacky - need this for CE calcs
        #     n_sims = hyperparams['n_sims']
        #     n_iter = hyperparams['n_iter']

        if progress:
            pbar = tqdm(total=n_cities*n_days*n_trials, desc='Running {} agent'.format(agent), leave=False)

        ## loop through cities
        for city in range(n_cities):

            ## context prior resets
            context_prior = 0.5

            ## loop through days
            for day in range(n_days):
                self.context_prior = context_prior

                ## get the environment for this day
                if envs:
                    env = envs['city_{}_grid_{}_env_object'.format(city+1, day+1)][0]

                    ## need to do some fixes for old envs
                    if env.expt == '2AFC':
                        env.expt = 'AFC'
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

                    ### if extracting useful behavioural measures

                    ## overlaps with previous observations
                    if agent == 'human':
                        paths = env_copy.path_states[t].copy()
                        obs_list = [tuple(obs[:2]) for obs in env_copy.obs.tolist()]
                        for i, path in enumerate(paths):
                            try:
                                
                                ## get the number of states that overlap with the paths
                                overlap = set(path).intersection(set(obs_list))
                                path_past_overlap = len(overlap)

                                ## get the number of costs and no-costs that comprise these overlapping states
                                path_past_observed_costs = sum(env_copy.costss[t][obs[0], obs[1]] == self.high_cost for obs in overlap)
                                path_past_observed_no_costs = sum(env_copy.costss[t][obs[0], obs[1]] == self.low_cost for obs in overlap)
                                self.path_past_overlaps[city, day, t, i] = path_past_overlap
                                self.path_past_observed_costs[city, day, t, i] = path_past_observed_costs
                                self.path_past_observed_no_costs[city, day, t, i] = path_past_observed_no_costs
                        
                            ## sometimes need to convert each np array to list of tuples...
                            except:
                                # paths = [set(map(tuple, path)) for path in paths]
                                path = set(map(tuple, path))
                                overlap = set(path).intersection(set(obs_list))
                                path_past_overlap = len(overlap)
                                path_past_observed_costs = sum(env_copy.costss[t][obs[0], obs[1]] == self.high_cost for obs in overlap)
                                path_past_observed_no_costs = sum(env_copy.costss[t][obs[0], obs[1]] == self.low_cost for obs in overlap)
                                self.path_past_overlaps[city, day, t, i] = path_past_overlap
                                self.path_past_observed_costs[city, day, t, i] = path_past_observed_costs
                                self.path_past_observed_no_costs[city, day, t, i] = path_past_observed_no_costs
                            
                            assert self.path_past_overlaps[city, day, t, i] == self.path_past_observed_costs[city, day, t, i] + self.path_past_observed_no_costs[city, day, t, i], 'path {} past overlap does not match observed costs and no-costs\n path past overlap: {}, path observed costs: {}, path observed no-costs: {}'.format(i+1, self.path_past_overlaps[city, day, t, i], self.path_past_observed_costs[city, day, t, i], self.path_past_observed_no_costs[city, day, t, i])
                    
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
                            action = np.random.choice(np.arange(len(MCTS_Q)), p=softmax(MCTS_Q))

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
                'p_choice_A':[],
                'p_choice_B':[],
                'p_choice_C':[],
                'p_correct':[],
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
                        sim_out['context'].append(self.true_context[c])
                        sim_out['p_choice_A'].append(self.p_choice[c][d][t][0])
                        sim_out['p_choice_B'].append(self.p_choice[c][d][t][1])
                        if self.n_afc==3:
                            sim_out['p_choice_C'].append(self.p_choice[c][d][t][2])
                        else:
                            sim_out['p_choice_C'].append(np.nan)
                        sim_out['p_correct'].append(self.p_correct[c][d][t])
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
