from enum import Enum
import gymnasium as gym
from gymnasium import spaces
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
from utils import *
from scipy.stats import rankdata, truncnorm
from scipy.linalg import cholesky
from minimax_tilting_sampler import TruncatedMVN
from base_kernels import *
from samplers import GridSampler




class GPAgent:

    def __init__(self, N, K_inf, metric='cityblock',inf_noise=0.01):

        self.metric = metric
        # self.K_inf = K_inf
        # self.N = int(np.sqrt(len(K_inf)))
        self.N = N
        self.n_actions = 4
        self.inf_noise = inf_noise
        self.action_to_direction = {0: np.array([0, 1]), 1: np.array([0, -1]), 2: np.array([1, 0]), 3: np.array([-1, 0])}
        
        ## define grid of kernel parameters and their weights
        n_param_vals = 10
        lb = 0
        ub = np.pi 
        ub -= (ub-lb)/(n_param_vals)
        self.k_params = np.linspace(lb, ub, n_param_vals)
        self.k_weights = np.ones(n_param_vals) / n_param_vals
        # self.k_params = {
        #     'theta': np.linspace(0, np.pi, n_param_vals),
        # }
        

        ## if using a known kernel
        if K_inf is not None:
            self.K_inf = K_inf

        ## else, sample a kernel
        else:
            self.init_kernels()
            self.K_inf = self.sample_k()

    ## initialise kernel set
    def init_kernels(self):

        ## init kernel set
        x = np.arange(self.N)
        y = np.arange(self.N)
        X,Y = np.meshgrid(x,y)
        locations = np.column_stack([X.ravel(), Y.ravel()])
        kernel_set = BaseKernels(locations)

        ## create kernel for each parameter in grid
        self.all_Ks = []
        for theta in self.k_params:
            K_inf = kernel_set.rbf_1D(dim=0, theta=theta, sigma_f=2, length_scale=0.5)
            self.all_Ks.append(K_inf)
        self.all_Ks = np.array(self.all_Ks)
    
    ## sample kernel
    def sample_k(self):
        K_idx = np.random.choice(np.arange(len(self.all_Ks)), p=self.k_weights)
        self.current_theta = self.k_params[K_idx]
        K_inf = self.all_Ks[K_idx]
        return K_inf
    
    ## kernel weight update
    def update_k_weights(self, obs):
        lls = []
        for k_inf in self.all_Ks:
            ll = self.likelihood(k_inf, obs)
            lls.append(ll)
        self.k_weights = softmax(lls)

        ## get MLE param value
        # best_idx = self.k_weights
        # self.best_theta = self.k_params[best_idx]
    
    
    
    ### interactions with the environment

    ## function for receiving info from env
    def get_env_info(self, env):
        self.N = env.N
        self.obs = env.obs.copy()
        self.current = env.current
        self.goal = env.goal
                     

    ## root sampling of surface
    def root_sample(self, obs, K_inf):

        ## calculate posterior mean 
        self.posterior_mean, self.posterior_cov, self.posterior_var = self.post_pred(K_inf = K_inf, obs=self.obs, pred = 'all')

        ## sample from posterior
        self.posterior_sample = sample(self.posterior_mean, self.posterior_cov).flatten()

    ## dynamic programming
    def dp(self, certainty_equivalent = False):

        ## dynamic programming to get optimal Q-values, given the agent's knowledge of the environment
        if certainty_equivalent:
            dp_costs = self.posterior_mean.reshape(self.N, self.N).copy()
            # dp_costs += self.expl_beta * np.sqrt(self.posterior_var.reshape(self.N, self.N)) #UCB
        else:
            dp_costs = self.posterior_sample.reshape(self.N, self.N).copy()
        dp_costs[self.goal[0], self.goal[1]] = 0
        self.V_inf, self.Q_inf, self.A_inf = value_iteration(dp_costs, self.goal)


    ## posterior prediction
    def post_pred(self, K_inf, obs, pred='all', sigma=0.01):
        # sigma = self.inf_noise
        if isinstance(pred, str):
            pred_idx = np.arange(self.N**2)
        else:
            pred_idx = pred

        if obs is not None:

            ## centre around 0 
            obs_idx = obs[:,0]* self.N + obs[:,1]
            obs_costs = obs[:, 2].copy()
            centring = 0.5
            # centring = np.mean(obs_costs)
            obs_costs += centring
            # obs_costs += 0.5
            assert np.all(obs_costs < 0.5), f"Adjusted observations out of bounds: {obs_costs}"


            
            # Covariance matrix of the already observed points
            K_obs = K_inf[obs_idx][:, obs_idx]

            
            # Covariance matrix between input points (i.e. points to be predicted) and observed points
            K_pred = K_inf[pred_idx][:, obs_idx]
            
            # inversion covariance matrix
            # inv_K = np.linalg.inv(K_obs + sigma**2 * np.eye(len(obs_idx)))
            inv_K = np.linalg.solve(K_obs + sigma**2 * np.eye(len(obs_idx)), np.eye(len(obs_idx)))

            
            # Posterior mean calculation
            post_mean = K_pred @ inv_K @ obs_costs
            # post_var = K_inf[pred_idx][:, pred_idx] - K_pred @ inv_K @ K_pred.T
            post_cov = K_inf[pred_idx][:, pred_idx] - K_pred @ inv_K @ K_pred.T
            post_var = np.diag(post_cov)

        
        ## or if starting from nothing, just return the prior
        elif obs is None:
            post_mean = np.zeros(len(pred_idx))  # Zero prior mean
            post_cov = K_inf[pred_idx][:, pred_idx]  # Full prior covariance
            post_var = np.diag(post_cov)

        ## revert back to prior mean
        post_mean -= 0.5
        # post_mean -= centring

        ## check for any non-negativity
        # assert np.all((obs_costs-0.5)<0), 'obs is not all negative: {}'.format(self.obs)
        # if np.any(post_mean >= 0):
        #     # print(f"Positive posterior mean detected: obs_costs: {obs_costs-0.5}, post_mean: {post_mean}")
        #     plot_r(post_mean.reshape(self.N,self.N), ax=plt.subplot(), title='post mean')

        #     ## plot obs 
        #     plot_obs(obs, ax=plt.subplot(), text=True)

        #     ## plot kernel
        #     plt.figure()
        #     plot_kernel(K_inf, ax=plt.subplot(), title='K_inf')
        #     plt.figure()
        #     sample_tmp = sample(np.zeros(self.N**2), K_inf).reshape(self.N, self.N)
        #     plot_r(sample_tmp, ax=plt.subplot(), title='sample')

        ## clipping (cheap fix)
        post_mean = np.clip(post_mean, -0.9, -0.1)

        assert np.all(post_mean < 0), 'post mean is not all negative: \n{}'.format(post_mean.reshape(self.N, self.N))

        return post_mean, post_cov, post_var
    
    ## posterior prediction weighted by the likelihood of the observations under each kernel
    def weighted_post_pred(self, obs, pred='all'):
        if isinstance(pred, str):
            pred_idx = np.arange(self.N**2)
        else:
            pred_idx = pred    
        post_means = []
        post_vars = []
        lls = []
        
        ## imperfect memory of observations
        # obs = self.obs[-10:]

        if obs is not None:

            ## loop through possible kernels
            for k_inf in self.all_Ks:
                post_mean, post_var = self.post_pred(k_inf, obs = obs, pred=pred_idx)
                post_means.append(post_mean)
                post_vars.append(post_var)

                ## calculate marginal likelihood of obs given this kernel
                ll = self.likelihood(k_inf, obs)
                lls.append(ll)
            
            ## weight each posterior prediction by the corresponding marginal likelihood
            self.k_weights = softmax(lls)
            post_mean = np.sum([self.k_weights[i] * post_means[i] for i in range(len(self.k_weights))], axis=0)

            ## weighted posterior covariance
            # post_var = np.sum([k_weights[i] * (post_vars[i] + (post_means[i] - post_mean) @ (post_means[i] - post_mean).T) for i in range(len(k_weights))], axis=0)
            post_var = np.sum([self.k_weights[i] * (post_vars[i] + np.outer(post_means[i] - post_mean, post_means[i] - post_mean)) for i in range(len(self.k_weights))], axis=0) ## need to check if this is correct

        ## or if starting from nothing, just return the prior
        elif obs is None:
            post_mean = np.zeros(len(pred_idx)) - 0.5
            # post_var = self.K_inf[pred_idx][:, pred_idx]
            post_var = np.zeros(len(pred_idx))

        return post_mean, post_var
    
    ## compute log marginal likelihood of set of observations, given the inference kernel
    def likelihood(self, K_inf, obs, sigma=0.01):
        # sigma = self.inf_noise
        n_obs = len(obs)
        obs_idx = obs[:,0]* self.N + obs[:,1]
        obs_costs = obs[:, 2] #i.e. y
        K_tmp = K_inf[obs_idx][:,obs_idx] 
        K_tmp = K_tmp + ((sigma**2) * np.eye(n_obs))
        k_check(K_tmp)
        
        ## cholesky decomp
        L = scipy.linalg.cholesky(K_tmp, lower=True, check_finite=False)
        alpha = scipy.linalg.cho_solve((L, True), obs_costs, check_finite=False)

        ## calculate log likelihood terms
        log_det = np.sum(np.log(np.diag(L))) 
        quad_form = 0.5 * (obs_costs@alpha)
        norm_term = 0.5 * n_obs * np.log(2*np.pi)
        ll = -quad_form - log_det - norm_term

        return ll
    
    ## generate random observations from current true GP kernel
    def gen_obs(self, samples, n_obs):
        obs_idx = np.random.randint(0, self.N**2, size=n_obs)

        ## map these observations to the grid and get the reward values
        obs_coords = np.column_stack(np.unravel_index(obs_idx, (self.N, self.N)))
        obs_costs = samples[obs_coords[:, 0], obs_coords[:, 1]]
        obs = np.column_stack([obs_idx, obs_coords, obs_costs])

        return obs


    
    
    ### define some policies

    ## random
    def random_policy(self):
        action = np.random.choice(self.n_actions)
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
            return action
        

    ## greedy wrt/ both distance to goal and cost, i.e. some combination of the two
    def balanced_policy(self, current, goal, eps=0, alpha=0.5):
        if np.random.rand() < eps:
            return self.random_policy()
        else:

            
            ## get adjacent states
            next_states = np.clip(np.array([current + self.action_to_direction[i] for i in range(self.n_actions)]), 0, self.N-1)
            next_states_idx = next_states[:, 0]*self.N + next_states[:, 1]
            
            ## myopic UCB
            next_q = self.posterior_sample.reshape(self.N, self.N)[next_states[:, 0], next_states[:, 1]]
            # next_q = self.posterior_mean.reshape(self.N, self.N)[next_states[:, 0], next_states[:, 1]]
            # next_var = self.posterior_var.reshape(self.N, self.N)[next_states[:, 0], next_states[:, 1]]
            # next_q = next_q + self.expl_beta * np.sqrt(next_var)

            ## ensure post_mean is negative
            if next_q.max() > 0:
                next_q -= next_q.max()

            
            ## weight the distance to the goal by the cost of the state
            distances = cdist(next_states, [goal], metric=self.metric).flatten()
            combined_q = alpha * softmax(-distances) + (1-alpha) * softmax(next_q)
            max_combined_q = np.max(combined_q)
            action = argm(combined_q, max_combined_q)
            return action
        
    
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
    

### farmer model?
class Farmer:

    def __init__(self, N, metric='cityblock'):

        self.metric = metric
        self.N = N
        self.n_actions = 4
        # self.action_to_direction = {0: np.array([0, 1]), 1: np.array([0, -1]), 2: np.array([1, 0]), 3: np.array([-1, 0])}
        self.action_to_direction = {0: np.array([1,0]), 1: np.array([0, 1]), 2: np.array([-1, 0]), 3: np.array([0, -1])}

    ### interactions with the environment

    ## function for receiving info from env
    def get_env_info(self, env):
        self.N = env.N
        self.obs = env.obs
        self.current = env.current
        self.goal = env.goal
        self.high_cost = env.high_cost
        self.low_cost = env.low_cost
        self.alpha_row = env.alpha_row
        self.alpha_col = env.alpha_col
        self.beta_row = env.beta_row
        self.beta_col = env.beta_col

    ## root sampling of surface
    def root_sample(self, obs=None, n_iter=100, lazy=True):
        sampler = GridSampler(self.alpha_row, self.beta_row, self.alpha_col, self.beta_col, obs, N=self.N)

        ## lazy
        if lazy:
            self.posterior_p, self.posterior_q = sampler.lazy_sample(n_iter = n_iter)

        ## full
        else:
            self.posterior_p, self.posterior_q = sampler.sample(n_iter = n_iter)

        self.posterior_p_cost = np.outer(self.posterior_p, self.posterior_q)


    ## dynamic programming
    def dp(self, expected_cost=True):

        ## use expected cost of each state
        if expected_cost:
            dp_costs = self.posterior_p_cost*self.high_cost + (1-self.posterior_p_cost)*self.low_cost
            dp_costs[self.goal[0], self.goal[1]] = 0
            # dp_costs = self.posterior_p_cost*self.low_cost + (1-self.posterior_p_cost)*self.high_cost
            # dp_costs[self.goal[0], self.goal[1]] = 1

        ## or, sample costs using p and q probabilities 
        else:
            dp_costs = np.array([self.high_cost if np.random.random() < self.posterior_p[i] else self.low_cost for i in range(self.N)]).reshape(self.N, self.N)
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