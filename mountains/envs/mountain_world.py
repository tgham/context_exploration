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


# from PIL import image


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3

    north_east = 4
    north_west = 5
    south_east = 6
    south_west = 7


class MountainEnv(gym.Env):
    # metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, N, params=None, metric = 'cityblock', true_k=None, inf_k='known', known_costs = True, render_mode=None, r_noise=0.05,size=5):
        
        ### GP inits

        ## initialise the GP grid
        self.N = N
        x = np.arange(N)
        y = np.arange(N)
        X,Y = np.meshgrid(x,y)
        self.locations = np.column_stack([X.ravel(), Y.ravel()])

        ## set the kernel parameters
        if params is None:
            self.c = 1
            self.scale = 1
            self.theta = 0
            # self.theta = np.pi/3
            self.sigma_f = 2
            # self.length_scale = self.N/5
            self.length_scale = 1
            self.period = self.N/5
            self.periodic_length_scale = self.N/2
            self.periodic_theta = np.pi/3
            self.expl_beta = 0.
        else:
            self.c = params[0]
            self.scale = params[1]
            self.theta = params[2]
            self.sigma_f = params[3]
            self.length_scale = params[4]
            self.periodic_length_scale = params[5]
            self.period = params[6]
            self.periodic_theta = params[7]

        self.min_cost, self.max_cost = -0.9, -0.1

        ## initialise the kernels
        self.K_lin = self.linear()
        self.K_lin_x = self.linear_1D(0)
        self.K_lin_y = self.linear_1D(1)
        self.K_rbf = self.rbf()
        self.K_rbf_x = self.rbf_1D(0)
        self.K_rbf_y = self.rbf_1D(1)
        self.K_periodic_x = self.periodic(0)
        self.K_periodic_y = self.periodic(1)
        self.all_Ks = [
            # self.K_lin, self.K_lin_x, self.K_lin_y, 
            self.K_rbf, self.K_rbf_x, self.K_rbf_y, 
            self.K_periodic_x, 
            # self.K_periodic_y
            ]
        self.k_weights = np.zeros(len(self.all_Ks))
        self.r_noise = r_noise

        

        ## define mountain costs as samples from the GP
        ## (for now, let's just use the RBF kernel)
        self.true_k = true_k
        if true_k == 'lin':
            self.K_gen = self.K_lin
        elif true_k == 'lin_x':
            self.K_gen = self.K_lin_x
        elif true_k == 'lin_y':
            self.K_gen = self.K_lin_y
        elif true_k == 'rbf':
            self.K_gen = self.K_rbf
        elif true_k == 'rbf_x':
            self.K_gen = self.K_rbf_x
        elif true_k == 'rbf_y':
            self.K_gen = self.K_rbf_y
        elif true_k == 'periodic_x':
            self.K_gen = self.K_periodic_x
        elif true_k == 'periodic_y':
            self.K_gen = self.K_periodic_y
        else: #default
            self.K_gen = self.K_rbf
        mean = np.zeros(self.N**2) - 0.5
        self.costs = self.sample(mean, self.K_gen) 
        self.cost_threshold = 1 
        self.known_costs = known_costs

        ## determine how inferences are made (i.e. with full knowledge of the kernel, or with a weighted combination of kernels)
        self.inf_k = inf_k
        if inf_k == 'known':
            self.K_inf = self.K_gen
            self.inference_func = lambda obs, pred: self.post_pred(K_inf = self.K_inf, obs=obs, pred = pred)
        elif inf_k == 'weighted':
            self.inference_func = lambda obs, pred: self.weighted_post_pred(obs = obs, pred = pred)

        
        ### gym inits

        ## sizes
        self.window_size = 512

        # Observations are dictionaries with the agent's and the goal's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        size = 5
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "goal": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        ## init trial info
        # self.starts = []
        # self.goals = []
        self.starts = [[0, 0],              [0, self.N-1],    [self.N-1, 0],    [self.N-1, self.N-1]]
        self.goals = [[self.N-1, self.N-1], [self.N-1, 0],    [0, self.N-1],    [0, 0]]
        self.n_eps = 0

        # define actions, depending on metric
        self.metric = metric

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        if self.metric == 'cityblock':
            self.action_space = spaces.Discrete(4)
            self._action_to_direction = {
                Actions.right.value: np.array([1, 0]),
                Actions.up.value: np.array([0, 1]),
                Actions.left.value: np.array([-1, 0]),
                Actions.down.value: np.array([0, -1]),
            }
            self.n_actions = 4
        elif self.metric == 'chebyshev':
            self.action_space = spaces.Discrete(8)
            self._action_to_direction = {
                Actions.right.value: np.array([1, 0]),
                Actions.up.value: np.array([0, 1]),
                Actions.left.value: np.array([-1, 0]),
                Actions.down.value: np.array([0, -1]),

                Actions.north_east.value: np.array([1, 1]),
                Actions.north_west.value: np.array([-1, 1]),
                Actions.south_east.value: np.array([1, -1]),
                Actions.south_west.value: np.array([-1, -1]),
            }
            self.n_actions = 8

        # assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        # if self.render_mode == "human":
        #     self.sim = False
        # elif self.render_mode == "MCTS":
        #     self.sim = True
        self.sim = False

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None


    ### RL env inits

    ## get info from current state
    def get_obs(self):
        return {"agent": self._agent_location, "goal": self._goal_location}
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._goal_location, ord=1
            )
        }
    # def get_current(self):
    #     return self._agent_location.copy()
    # def get_goal(self):
    #     return self._goal_location.copy()

    @property
    def current(self):
        return self._agent_location
    
    @property
    def goal(self):
        return self._goal_location
    


    ## reset the environment
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)


        ## sample start and goal locations
        dist = 0
        min_dist = self.N*0.75
        angle = 0
        angle_tolerance = 0.5
        angle_bounds = [45*(1+angle_tolerance), 45*(1-angle_tolerance)]
        row_or_col = 1
        goal_val = 0
        start_val = 0
        min_val = 0.6
        t = 0
        while (dist<min_dist) or (row_or_col>0) or (angle>angle_bounds[0]) or (angle<angle_bounds[1]): #or (goal_val<min_val) or (start_val<min_val):
            self._agent_location = self.np_random.integers(0, self.N, size=2, dtype=int)
            self._goal_location = self.np_random.integers(
                0, self.N, size=2, dtype=int
            )

            ## distance criterion
            dist = np.max(cdist([self._agent_location, self._goal_location], [self._agent_location, self._goal_location], metric='cityblock'))

            ## same row/col criterion
            row_or_col = np.sum(self._agent_location == self._goal_location)

            ## angle criterion
            angle = node_angle(self._agent_location, self._goal_location)

            ## value criterion
            goal_val = self.costs[self._goal_location[0], self._goal_location[1]]
            goal_val = 1

            start_val = self.costs[self._agent_location[0], self._agent_location[1]]
            start_val = 1
            t+=1

            # if t>10:
            #     print('cant find start and end', dist, angle, t)

            # last goal distance criterion
            # self.starts

        ## for sanity check, just place agent and goal in opposite corners
        # self._agent_location = np.array(self.starts[self.n_eps%4])
        # self._goal_location = np.array(self.goals[self.n_eps%4])
        # self.n_eps += 1



        ## initialise trial info
        self.terminated = False
        observation = self.get_obs()
        info = self._get_info()
        current_cost = self.costs[self._agent_location[0], self._agent_location[1]] #+ np.random.normal(0, self.r_noise)

        ## reset obs on each trial
        # loc_idx = self._agent_location[0]*self.N + self._agent_location[1]
        # self.obs = np.array([loc_idx, self._agent_location[0], self._agent_location[1], current_cost], ndmin=2)

        ## or, observations accumulate over trials, and agent observes starting position
        if not self.sim:
            loc_idx = self._agent_location[0]*self.N + self._agent_location[1]
            if not hasattr(self, 'obs') or self.obs is None:
                self.obs = np.array([[loc_idx, self._agent_location[0], self._agent_location[1], current_cost]]) 
            else:
                # print(len(self.obs))
                self.obs = np.vstack([self.obs, [loc_idx, self._agent_location[0], self._agent_location[1], current_cost]])
            self.obs_tmp = self.obs.copy()

        ## or, observations accumulate over trials, but agent doesn't observe starting position
        # loc_idx = self._agent_location[0]*self.N + self._agent_location[1]
        # if not hasattr(self, 'obs') or self.obs is None:
        #     self.obs = None
        #     self.obs_tmp = np.array([[loc_idx, self._agent_location[0], self._agent_location[1], current_cost]])
        # else:
        #####     self.obs = np.vstack([self.obs, [loc_idx, self._agent_location[0], self._agent_location[1], current_cost]])
        #     self.obs_tmp = self.obs.copy()

        ## or, if simulating some unknown future environment, the observations are given by the previous tree, so we trivially have obs already
        elif self.sim:
            loc_idx = self._agent_location[0]*self.N + self._agent_location[1]
            self.obs_tmp = np.vstack([self.obs, [loc_idx, self._agent_location[0], self._agent_location[1], current_cost]])



        ## calculate posterior mean 
        self.posterior_mean, self.posterior_cov, self.posterior_var = self.inference_func(obs = self.obs, pred='all')

        ## sample from posterior
        self.root_sample(certainty_equivalent=False)

        ## dynamic programming to get the true optimal trajectory
        dp_costs = self.costs.copy()
        dp_costs[self._goal_location[0], self._goal_location[1]] = 0
        self.V_true, self.Q_true, self.A_true = self.value_iteration(dp_costs=dp_costs)

        ## get the costs of this optimal trajectory
        self.optimal_trajectory()

        ## initialise actual trajectory as list of tuples
        self.a_traj = [tuple(self._agent_location)]
        self.action_scores = []
        
        return observation, info
    
    ## custom functions for manually editing the env
    def set_state(self, state):
        self._agent_location = state
    def set_sim(self, sim):
        self.sim = sim
    def set_obs(self, obs):
        self.obs = obs
    def flush_obs(self): ## necessary for MCTS
        self.obs_tmp = self.obs.copy()


    ## take a step in the environment
    def step(self, action):
        
        ## first, get the ranking of the best actions to take under the *true* optimal policy, given the agent's current position
        current_Q_vals = self.Q_true[self._agent_location[0], self._agent_location[1], :]
        action_ranking = rankdata(current_Q_vals, method='max') - 1

        ## get the score of the action that will  actually be taken, given the ranking of the optimal actions
        action_score = action_ranking[action] + 1
        action_score /= self.n_actions

        ## or, score the action based on the normalised Q-values of the available actions
        norm_Q_vals = (current_Q_vals - np.min(current_Q_vals)) / (np.max(current_Q_vals) - np.min(current_Q_vals))
        action_score = norm_Q_vals[action]
        
        ## take the actual action 
        direction = self._action_to_direction[action] 

        ## move to the new state
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.N - 1
        )


        ## get the predicted cost of the new state (for MCTS)
        # predicted_cost = self.posterior_mean[self._agent_location[0]*self.N + self._agent_location[1]]
        predicted_cost = self.posterior_sample[self._agent_location[0]*self.N + self._agent_location[1]]
        var_cost = self.posterior_var[self._agent_location[0]*self.N + self._agent_location[1]]
        
        ## get the observed cost of the current state
        current_cost = self.costs[self._agent_location[0], self._agent_location[1]] #+ np.random.normal(0, self.r_noise)

        ## return the real cost if not simulating
        if not self.sim:
            cost = current_cost
            
            ## update observation and trajectory arrays - i.e. agent observes along the way
            # self.a_traj.append(tuple(self._agent_location))
            # loc_idx = self._agent_location[0]*self.N + self._agent_location[1]
            # self.obs = np.vstack([self.obs, [loc_idx, self._agent_location[0], self._agent_location[1], current_cost]])
            # self.obs_tmp = np.vstack([self.obs_tmp, [loc_idx, self._agent_location[0], self._agent_location[1], current_cost]])
            
            ## calculate new posterior for next trial
            # self.posterior_mean, self.posterior_var = self.inference_func(obs = self.obs, pred='all')

            ## or, temporarily store observations until the end of the trial (no need to calculate posterior at each step, since there are no new observations)
            self.a_traj.append(tuple(self._agent_location))
            loc_idx = self._agent_location[0]*self.N + self._agent_location[1]
            self.obs_tmp = np.vstack([self.obs_tmp, [loc_idx, self._agent_location[0], self._agent_location[1], cost]])

            ## store info on optimality of the choice, given the agent's current position
            self.action_scores.append(action_score)


        ## return the predicted cost if simulating
        elif self.sim:
            if self.known_costs:
                cost = current_cost
            else:
                cost = predicted_cost 
                cost += self.expl_beta * np.sqrt(var_cost) #UCB

            ## still need to store obs_tmp along the way for subsequent posterior inference
            self.a_traj.append(tuple(self._agent_location))
            loc_idx = self._agent_location[0]*self.N + self._agent_location[1]
            self.obs_tmp = np.vstack([self.obs_tmp, [loc_idx, self._agent_location[0], self._agent_location[1], cost]])
                

        # An episode is done iff the agent has reached the goal
        if np.array_equal(self._agent_location, self._goal_location):
            self.terminated = True
            cost=0 ## cost of final state is 0
            cost = self.expl_beta * np.sqrt(var_cost)
        
            ## update observation array only once the episode is complete
            if not self.sim:
                self.obs = self.obs_tmp.copy()

                ## (REMOVE COST OF FIRST AND FINAL STATE??)
                # self.a_traj.pop(0)
                # self.a_traj.pop(-1)

                ## sum of costs of route
                self.a_traj_costs = [self.costs[x, y] for x, y in self.a_traj]
                self.a_traj_total_cost = np.sum(self.a_traj_costs)

                ## scores for the trial
                self.action_score = np.mean(self.action_scores)
                self.cost_ratio = self.o_traj_total_cost / self.a_traj_total_cost

        else:
            self.terminated = False
        observation = self.get_obs()
        info = self._get_info()
        terminated = self.terminated
        truncated=False
        return observation, cost, terminated, truncated, info



    ### define some policies

    ## random
    def random_policy(self):
        return self.action_space.sample()
    
    ## greedy wrt/ distance to goal
    def greedy_policy(self, current, goal, eps=0):
        if np.random.rand() < eps:
            return self.random_policy()
        else:
            # distances = cdist([current], [goal], metric=self.metric).flatten()
            ## get adjacent states
            next_states = np.clip(np.array([current + self._action_to_direction[i] for i in range(self.n_actions)]), 0, self.N-1)
            
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
            next_states = np.clip(np.array([current + self._action_to_direction[i] for i in range(self.n_actions)]), 0, self.N-1)
            next_states_idx = next_states[:, 0]*self.N + next_states[:, 1]
            
            ## myopic UCB
            next_q = self.posterior_mean.reshape(self.N, self.N)[next_states[:, 0], next_states[:, 1]]
            next_var = self.posterior_var.reshape(self.N, self.N)[next_states[:, 0], next_states[:, 1]]
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
        next_states = np.clip(np.array([current + self._action_to_direction[i] for i in range(self.n_actions)]), 0, self.N-1)
        next_states_idx = next_states[:, 0]*self.N + next_states[:, 1]
    
        ## choose action with highest Q-value
        current_q = Q[current[0], current[1], :]
        max_current_q = np.nanmax(current_q)
        action = argm(current_q, max_current_q)

        return action
        
        
    
    
    #### GP inits

    ### define the kernels

    ## linear kernels
    
    # linear kernel over x,y, i.e. similarity as a function of the distance from the origin (0,0)
    def linear(self):
        dists = np.sqrt(self.locations[:, 0]**2 + self.locations[:, 1]**2)
        K = np.outer(dists, dists) + self.c
        return K
    
    # linear kernel over x-distance (0) or y-distance (1), i.e. similarity as a function of the distance from (0,:) or (:,0), where the basis vectors are determined by the angle theta (in radians)    
    def linear_1D(self, dim = 0):

        # define basis vectors
        rotation = np.array([[np.cos(self.theta), -np.sin(self.theta)], [np.sin(self.theta), np.cos(self.theta)]])
        rotated_locations = np.dot(self.locations, rotation)
        # dists = np.subtract.outer(rotated_locations[:, dim], rotated_locations[:, dim])

        # calculate distances
        dists = rotated_locations[:,dim] * self.scale**2
        K = np.outer(dists, dists) + self.c
        return K

    

    ## RBFs 

    # RBF kernel over x,y (i.e. Euclidean distance)
    def rbf(self):
        dists = cdist(self.locations, self.locations, metric='euclidean')
        K = self.sigma_f**2 * np.exp(-0.5 * (dists / self.length_scale)**2)
        return K

    # RBF kernel over just x-distance or y-distance
    def rbf_1D(self, dim=0):

        # define basis vectors
        rotation = np.array([[np.cos(self.theta), -np.sin(self.theta)], [np.sin(self.theta), np.cos(self.theta)]])
        rotated_locations = np.dot(self.locations, rotation)

        # calculate distances
        dists = np.subtract.outer(rotated_locations[:, dim], rotated_locations[:, dim])
        K = self.sigma_f**2 * np.exp(-0.5 * (dists / self.length_scale)**2)
        return K
    

    ## periodic kernel
    def periodic(self, dim=0):
        rotation = np.array([[np.cos(self.periodic_theta), -np.sin(self.periodic_theta)], [np.sin(self.periodic_theta), np.cos(self.periodic_theta)]])
        rotated_locations = np.dot(self.locations, rotation)
        dists = np.subtract.outer(rotated_locations[:, dim], rotated_locations[:, dim])
        K = self.sigma_f**2 * np.exp(-2 * np.sin(np.pi * dists / self.period)**2 / self.periodic_length_scale**2)
        return K


    ## sample from the GP
    def sample(self, mean, K):

        ## check kernel is valid
        self.k_check(K)

        # sample
        # if mean is None:
        #     mean = np.zeros(self.N**2)
        # mean = np.zeros(self.N**2)
        # samples = np.random.multivariate_normal(mean, K).reshape(self.N, self.N)

        #normalise
        # min_cost = self.min_cost
        # max_cost = self.max_cost
        # samples = min_cost + (max_cost - min_cost) * (samples - np.min(samples)) / (np.max(samples) - np.min(samples))


        ## or truncated
        lb = np.zeros(self.N**2) + self.min_cost
        ub = np.zeros(self.N**2) + self.max_cost
        # K_tmp = K + 1e-5 * np.eye(self.N**2)
        K_tmp = K + self.r_noise**2 * np.eye(self.N**2)
        tmvn = TruncatedMVN(mean, K_tmp, lb, ub)
        samples = tmvn.sample(1)
        samples = samples.reshape(self.N, self.N)

        return samples
    

    ## sample from posterior and re-compute Q-vals etc..
    def root_sample(self, certainty_equivalent=False):

        ## sample from posterior
        self.posterior_sample = self.sample(self.posterior_mean, self.posterior_cov).flatten()
        # assert np.all(self.posterior_sample < 0), 'post sample is not all negative: {}'.format(self.posterior_sample)

        ## dynamic programming to get the optimal Q-vals given the agent's knowledge of the environment
        if certainty_equivalent:
            dp_costs = self.posterior_mean.reshape(self.N, self.N).copy()
            # dp_costs += self.expl_beta * np.sqrt(self.posterior_var.reshape(self.N, self.N)) #UCB
        else:
            dp_costs = self.posterior_sample.reshape(self.N, self.N).copy()
        dp_costs[self._goal_location[0], self._goal_location[1]] = 0

        if self.known_costs:
            self.V_inf = self.V_true
            self.Q_inf = self.Q_true
            self.A_inf = self.A_true
        else:
            self.V_inf, self.Q_inf, self.A_inf = self.value_iteration(dp_costs=dp_costs)

    

    ## use GP regression to predict posterior distribution of rewards, given these observations,based on the currently inferred kernel
    def post_pred(self, K_inf, obs, pred='all', sigma=0.01):
        sigma = self.r_noise
        if isinstance(pred, str):
            pred_idx = np.arange(self.N**2)
        else:
            pred_idx = pred

        if obs is not None:

            ## centre around 0 
            obs_idx = obs[:, 0].astype(int)
            obs_rewards = obs[:, 3].copy()
            obs_rewards += 0.5 
            # obs_rewards /=10
            
            # Covariance matrix of the already observed points
            K_obs = K_inf[obs_idx][:, obs_idx]
            
            # Covariance matrix between input points (i.e. points to be predicted) and observed points
            K_pred = K_inf[pred_idx][:, obs_idx]
            
            # inversion covariance matrix
            # inv_K = np.linalg.inv(K_obs + sigma**2 * np.eye(len(obs_idx)))
            inv_K = np.linalg.solve(K_obs + sigma**2 * np.eye(len(obs_idx)), np.eye(len(obs_idx)))

            
            # Posterior mean calculation
            post_mean = K_pred @ inv_K @ obs_rewards
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

        ## check for any non-negativity
        assert np.all((obs_rewards-0.5)<0), 'obs is not all negative: {}'.format(self.obs)
        # assert np.all(post_mean < 0), 'post mean is not all negative: {}'.format(post_mean)

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
    

    ## check that kernel is PSD and symmetric
    def k_check(self, K):
        symm = np.allclose(K,K.T)
        if not symm:
            warnings.warn("Kernel matrix is not symmetric.", UserWarning)
        
        eigenvalues = np.linalg.eigvals(K)
        psd = np.all(eigenvalues >= -1e-10)
        if not psd:
            warnings.warn("Kernel matrix is not positive semi-definite.", UserWarning)

        return np.any([not symm, not psd])
    
    ## compute log marginal likelihood of set of observations, given the inference kernel
    def likelihood(self, K_inf, obs, sigma=0.01):
        n_obs = len(obs)
        obs_idx = obs[:, 0].astype(int) #i.e. x
        obs_rewards = obs[:, 3] #i.e. y
        K_tmp = K_inf[obs_idx][:,obs_idx] 
        K_tmp = K_tmp + ((sigma**2) * np.eye(n_obs))
        self.k_check(K_tmp)
        
        ## cholesky decomp
        L = scipy.linalg.cholesky(K_tmp, lower=True, check_finite=False)
        alpha = scipy.linalg.cho_solve((L, True), obs_rewards, check_finite=False)

        ## calculate log likelihood terms
        log_det = np.sum(np.log(np.diag(L))) 
        quad_form = 0.5 * (obs_rewards@alpha)
        norm_term = 0.5 * n_obs * np.log(2*np.pi)
        ll = -quad_form - log_det - norm_term

        return ll
    


    ## calculate the optimal trajectory between the two points, as given by the true DP solution
    def optimal_trajectory(self):
        start = self._agent_location
        current = start.copy()
        goal = self._goal_location
        self.o_traj = [tuple(current)]
        self.o_traj_costs = [self.costs[current[0], current[1]]]
        
        visited = set()
        while not np.array_equal(current, goal):
            i, j = current
            action = int(self.A_true[i, j])  # Ensure action index is int
            visited.add(tuple(current))
            
            # Take action and update current state
            if action == 0:  # Down
                current = np.clip((i + 1, j), 0, self.N - 1)
            elif action == 1:  # Right
                current = np.clip((i, j + 1), 0, self.N - 1)
            elif action == 2:  # Up
                current = np.clip((i - 1, j), 0, self.N - 1)
            elif action == 3:  # Left
                current = np.clip((i, j - 1), 0, self.N - 1)
            
            if tuple(current) in visited:
                print(f"Cycle detected from {start} to {goal} at state {current}. Exiting to prevent infinite loop.")
                print(self.A_true)
                break
            
            # Update trajectory
            self.o_traj.append(tuple(current))
            self.o_traj_costs.append(self.costs[current[0], current[1]])
        
        # Compute total cost
        self.o_traj_total_cost = np.sum(self.o_traj_costs)



    ## generate random observations from current true GP kernel
    def gen_obs(self, samples, n_obs):
        obs_idx = np.random.randint(0, self.N**2, size=n_obs)

        ## map these observations to the grid and get the reward values
        obs_coords = np.column_stack(np.unravel_index(obs_idx, (self.N, self.N)))
        obs_rewards = samples[obs_coords[:, 0], obs_coords[:, 1]]
        obs = np.column_stack([obs_idx, obs_coords, obs_rewards])

        return obs
    
    ## dynamic programming
    def value_iteration(self, dp_costs, max_iters = 1000, theta = 0.0001, discount = 0.99):
        
        ## init tables
        V = np.zeros((self.N, self.N))
        A = np.zeros((self.N, self.N))
        Q = np.zeros((self.N, self.N, self.n_actions))

        ## determine whether to use true costs or inferred costs
        # if self.known_costs:
        #     dp_costs = self.costs.copy()
        # else:
        #     dp_costs = self.posterior_mean.reshape(self.N, self.N).copy()

        ## set cost of goal to 0
        start = self._agent_location.copy()
        goal = self._goal_location.copy()
        dp_costs[goal[0], goal[1]] = 0

        assert np.all(dp_costs <= 0), 'costs are not all negative: {}'.format(dp_costs)

        ## loop through states
        for i in range(max_iters):
            delta = 0
            for x in range(self.N):
                for y in range(self.N):
                    
                    ## (make sure the goal state has value 0)
                    if (x, y) == tuple(goal):
                        # V[x, y] = 0
                        continue

                    v = V[x, y]
                    q = np.zeros(self.n_actions)

                    ## loop through actions and get the discounted value of each of the next states
                    for a in range(self.n_actions):

                        ## allow wall moves
                        # next_state = np.clip([x, y] + self._action_to_direction[a], 0, self.N-1)
                        # q[a] = dp_costs[next_state[0], next_state[1]] + discount*V[next_state[0], next_state[1]]

                        ## or, don't allow wall moves
                        next_state = [x, y] + self._action_to_direction[a]
                        if (next_state[0] >= 0) and (next_state[0] < self.N) and (next_state[1] >= 0) and (next_state[1] < self.N):
                            q[a] = dp_costs[next_state[0], next_state[1]] + discount*V[next_state[0], next_state[1]]
                        else:
                            q[a] = np.nan

                        ## update the Q-table
                        Q[x, y, a] = q[a]

                    ## use the best action to update the value of the current state
                    V[x, y] = np.nanmax(q)

                    # A[x, y] = np.argmax(q)
                    A[x, y] = argm(q, np.nanmax(q))

                    ## check if converged
                    delta = max(delta, np.abs(v - V[x, y]))
            
            if delta < theta:
                # print('DP converged after {} iterations'.format(i))
                break

            if i == max_iters-1:
                print('DP did not converge after {} iterations'.format(i))

        ## need to check if this has lead to a valid policy

        return V, Q, A
