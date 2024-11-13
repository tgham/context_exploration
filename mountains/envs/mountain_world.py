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
from scipy.stats import rankdata
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
            self.scale = 1.0
            self.theta = 0
            self.sigma_f = 1.0
            self.length_scale = self.N/5
            self.period = self.N/5
            self.periodic_length_scale = self.N/2
            self.periodic_theta = np.pi/3
        else:
            self.c = params[0]
            self.scale = params[1]
            self.theta = params[2]
            self.sigma_f = params[3]
            self.length_scale = params[4]
            self.periodic_length_scale = params[5]
            self.period = params[6]
            self.periodic_theta = params[7]



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
        self.costs = self.sample(self.K_gen) *-1
        self.cost_threshold = 1 
        self.r_noise = r_noise
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

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        size = 5
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
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
        # self.sim = False

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
        return {"agent": self._agent_location, "target": self._target_location}
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    ## reset the environment
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)


        ## sample start and goal locations
        dist = 0
        min_dist = self.N*0.85
        angle = 0
        angle_tolerance = 0.4
        angle_bounds = [45*(1+angle_tolerance), 45*(1-angle_tolerance)]
        row_or_col = 1
        goal_val = 0
        start_val = 0
        min_val = 0.6
        # while (dist<min_dist) or (row_or_col>0) or (angle>angle_bounds[0]) or (angle<angle_bounds[1]) or (goal_val<min_val) or (start_val<min_val):
        #     self._agent_location = self.np_random.integers(0, self.N, size=2, dtype=int)
        #     self._target_location = self.np_random.integers(
        #         0, self.N, size=2, dtype=int
        #     )

        #     ## distance criterion
        #     dist = np.max(cdist([self._agent_location, self._target_location], [self._agent_location, self._target_location], metric='cityblock'))

        #     ## angle criterion
        #     row_or_col = np.sum(self._agent_location == self._target_location)
        #     angle = node_angle(self._agent_location, self._target_location)

        #     ## value criterion
        #     goal_val = self.costs[self._target_location[0], self._target_location[1]]
        #     goal_val = 1

        #     start_val = self.costs[self._agent_location[0], self._agent_location[1]]
            # start_val = 1

            ## last goal distance criterion
            # self.starts

        ## for sanity check, just place agent and target in opposite corners
        # self._agent_location = np.array([0, 0])
        # self._target_location = np.array([self.N-1, self.N-1])
        self._agent_location = np.array(self.starts[self.n_eps%4])
        self._target_location = np.array(self.goals[self.n_eps%4])
        self.n_eps += 1



        ## initialise trial info
        self.t = 0
        self.terminated = False
        observation = self.get_obs()
        current_cost = self.costs[self._agent_location[0], self._agent_location[1]] + np.random.normal(0, self.r_noise)
        info = self._get_info()
        self.accrued_cost = current_cost # is the cost incurred in the first state?? if not, this is set to 0

        ## reset obs on each trial
        # loc_idx = self._agent_location[0]*self.N + self._agent_location[1]
        # self.obs = np.array([loc_idx, self._agent_location[0], self._agent_location[1], current_cost], ndmin=2)

        ## or, observations accumulate over trials, and agent observes starting position
        # loc_idx = self._agent_location[0]*self.N + self._agent_location[1]
        # if not hasattr(self, 'obs') or self.obs is None:
        #     self.obs = np.array([[loc_idx, self._agent_location[0], self._agent_location[1], current_cost]]) 
        # else:
        #     # print(len(self.obs))
        #     self.obs = np.vstack([self.obs, [loc_idx, self._agent_location[0], self._agent_location[1], current_cost]])
        # self.obs_tmp = self.obs.copy()

        ## or, observations accumulate over trials, but agent doesn't observe starting position
        loc_idx = self._agent_location[0]*self.N + self._agent_location[1]
        if not hasattr(self, 'obs') or self.obs is None:
            self.obs = None
            self.obs_tmp = np.array([[loc_idx, self._agent_location[0], self._agent_location[1], current_cost]])
        else:
            self.obs = np.vstack([self.obs, [loc_idx, self._agent_location[0], self._agent_location[1], current_cost]])
            self.obs_tmp = self.obs.copy()




        ## initialise actual trajectory as list of tuples
        self.a_traj = [tuple(self._agent_location)]
        self.action_scores = []

        ## get the cost of the optimal trajectory, and use this to set the cost threshold
        # self.costs *= -1
        self.o_traj, self.o_route_cost = self.optimal_trajectory()
        self.o_traj.pop(-1)
        self.o_traj.pop(0)
        self.optimal_cost = np.sum(self.o_route_cost) #np.sum(self.o_route_cost[1:]) # is a cost incurred in the first state??
        # self.costs *= -1
        
        ## posterior inference over the whole environment
        self.posterior_mean, self.posterior_cov = self.inference_func(obs = self.obs, pred='all')
        # print(self.posterior_mean, self.obs)
        # print(len(self.obs_tmp))

        # if (self.render_mode == "human") or (self.render_mode == "MCTS"):
        if (self.render_mode == "human") or (self.render_mode == "MCTS"):

            ## posterior prediction using known kernel
            # self.posterior_mean, self.posterior_cov = self.post_pred(self.K_inf, self.obs)

            ## posterior prediction using weighted kernel
            # self.posterior_mean = self.weighted_post_pred()

            self.render()

        ## dynamic programming to get the true optimal trajectory, and the optimal trajectory given the agent's knowledge of the environment
        self.V_true, self.Q_true = self.value_iteration(dp_costs=self.costs.copy())
        if self.known_costs:
            self.V_inf = self.V_true
            self.Q_inf = self.Q_true
        else:
            self.V_inf, self.Q_inf = self.value_iteration(dp_costs=self.posterior_mean.reshape(self.N, self.N).copy())

        return observation, info
    
    ## custom function for manually setting the state (for MCTS?)
    def set_state(self, state):
        self._agent_location = state

    def set_sim(self, sim):
        self.sim = sim


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
        self.t += 1
        direction = self._action_to_direction[action] 

        ## move to the new state
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.N - 1
        )


        ## get the predicted cost of the new state (for MCTS)
        predicted_cost = self.posterior_mean.reshape(self.N, self.N)[self._agent_location[0], self._agent_location[1]]
        
        ## get the observed cost of the current state
        current_cost = self.costs[self._agent_location[0], self._agent_location[1]] + np.random.normal(0, self.r_noise)
        self.accrued_cost += current_cost

        ## return the real cost if not simulating
        if not self.sim:
            cost = current_cost
            
            ## update observation and trajectory arrays - i.e. agent observes along the way
            # self.a_traj.append(tuple(self._agent_location))
            # loc_idx = self._agent_location[0]*self.N + self._agent_location[1]
            # self.obs = np.vstack([self.obs, [loc_idx, self._agent_location[0], self._agent_location[1], current_cost]])

            ## temporarily store observations until the end of the trial
            self.a_traj.append(tuple(self._agent_location))
            loc_idx = self._agent_location[0]*self.N + self._agent_location[1]
            self.obs_tmp = np.vstack([self.obs_tmp, [loc_idx, self._agent_location[0], self._agent_location[1], current_cost]])

            ## store info on optimality of the choice, given the agent's current position
            self.action_scores.append(action_score)

        ## return the predicted cost if simulating
        elif self.sim:
            if self.known_costs:
                cost = current_cost
            else:
                cost = predicted_cost

        # An episode is done iff the agent has reached the target
        if np.array_equal(self._agent_location, self._target_location):
            self.terminated = True
            cost=0 ## cost of final state is 0
        
            ## update observation array only once the episode is complete
            if not self.sim:
                self.obs = self.obs_tmp.copy()

                ## (REMOVE COST OF FIRST AND FINAL STATE??)
                self.a_traj.pop(0)
                self.a_traj.pop(-1)

            ## sum of costs of route
            self.a_route_cost = [self.costs[x, y] for x, y in self.a_traj]
            self.actual_cost = np.sum(self.a_route_cost)

            ## average action score
            self.episode_score = np.mean(self.action_scores)

        else:
            self.terminated = False
            reward = 0
        observation = self.get_obs()
        info = self._get_info()
        terminated = self.terminated

        if (self.render_mode == "human"):
            ## posterior prediction using known kernel
            # self.posterior_mean, self.posterior_cov = self.post_pred(self.K_inf, self.obs)

            ## posterior prediction using weighted kernel
            # self.posterior_mean = self.weighted_post_pred()

            self.posterior_mean, self.posterior_cov = self.inference_func(obs = self.obs, pred='all')
            self.render()      

        truncated=False
        return observation, cost, terminated, truncated, info

    ## rendering funcs
    def render(self):

        self.posterior_mean, self.posterior_cov = self.inference_func(obs = self.obs, pred='all')
        
        # Clear the current output
        clear_output(wait=True)
        
        # Check if we already have a figure, if not create one
        if not hasattr(self, 'fig') or not plt.fignum_exists(self.fig.number):
            # self.fig, self.axs = plt.subplots(1, 2, figsize=(7.5, 15))
            if self.inf_k == 'known':
                self.fig, self.axs = plt.subplots(1, 2, figsize=(7.5, 15))
            elif self.inf_k == 'weighted':
                self.fig, self.axs = plt.subplots(1, 3, figsize=(15, 15))

        else:
            # Clear the existing axes
            for ax in self.axs:
                ax.clear()
        
        # Plot the full reward distribution, and the posterior distribution
        # title = 't={}\naccrued cost: {}\nthreshold: {}'.format(self.t, np.round(self.accrued_cost,2), np.round(self.cost_threshold,2))
        title = 't={}, (sub-)optimality: {}%'.format(self.t, np.round(100*self.accrued_cost/self.optimal_cost))
        title = ''
        plot_r(self.costs, self.axs[0])
        plot_r(self.posterior_mean.reshape(self.N, self.N), self.axs[1], title=title)

        ## plot the kernel weights
        if self.inf_k == 'weighted':
            plot_k_weights(self.k_weights, self.axs[2], title='Kernel weights')



        ## plot the optimal trajectory and trajectory so far
        plot_traj([self.o_traj, self.a_traj], self.axs[0])
        plot_traj([self.o_traj, self.a_traj], self.axs[1])
        
        # Plot the agent and target positions
        plot_state(self._agent_location, self._target_location, self.axs[0], title = "True mountain surface: \n"+title)
        plot_state(self._agent_location, self._target_location, self.axs[1], title = "Posterior mountain surface: \n"+title)
        
        
        # Adjust the layout to prevent overlapping
        plt.tight_layout()
        
        # Display the plot
        display(self.fig)
        
        # Instead of closing the figure, we'll just clear the current display
        clear_output(wait=True)
        
        # # Add a small delay to allow the plot to update
        if self.terminated:
            # print(self.msg)
            plt.pause(2)
        else:
            plt.pause(0.4)


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    ### define some policies

    ## random
    def random_policy(self):
        return self.action_space.sample()
    
    ## greedy wrt/ distance to target
    def greedy_policy(self, current, target, eps=0):
        if np.random.rand() < eps:
            return self.random_policy()
        else:
            # distances = cdist([current], [target], metric=self.metric).flatten()
            ## get adjacent states
            next_states = np.clip(np.array([current + self._action_to_direction[i] for i in range(self.n_actions)]), 0, self.N-1)
            
            ## choose whichever one is closest to the target
            distances = cdist(next_states, [target], metric=self.metric).flatten()
            min_distance = np.min(distances)
            action = argm(distances, min_distance)
            return action
        

    ## greedy wrt/ both distance to target and cost, i.e. some combination of the two
    def balanced_policy(self, current, target, eps=0, alpha=0.5, MCTS = False):
        if np.random.rand() < eps:
            return self.random_policy()
        else:

            
            ## get adjacent states
            next_states = np.clip(np.array([current + self._action_to_direction[i] for i in range(self.n_actions)]), 0, self.N-1)
            next_states_idx = next_states[:, 0]*self.N + next_states[:, 1]
            
            ## myopic
            # if not MCTS:
            # next_q, next_cov = self.inference_func(obs = self.obs, pred=next_states_idx)
            next_q = self.posterior_mean.reshape(self.N, self.N)[next_states[:, 0], next_states[:, 1]]
            # next_q = [self.posterior_mean.reshape(self.N, self.N)[next_states[i, 0], next_states[i, 1]] for i in range(len(next_states))]
            # self.posterior_mean.reshape(self.N, self.N)[self._agent_location[0], self._agent_location[1]]

            ## estimates provided by MCTS
            # elif MCTS:
            #     next_q = MCTS_estimates
            
            ## ensure post_mean is non-negative
            # if next_q.min() < 0:
            #     next_q -= next_q.min()

            ## ensure post_mean is negative
            if next_q.max() > 0:
                next_q -= next_q.max()

            
            ## weight the distance to the target by the cost of the state
            distances = cdist(next_states, [target], metric=self.metric).flatten()
            combined_q = alpha * softmax(-distances) + (1-alpha) * softmax(next_q)
            max_combined_q = np.max(combined_q)
            action = argm(combined_q, max_combined_q)
            return action
        
    ## optimal policy, as given by the dynamic programming solution
    def optimal_policy(self, current):

        ## get adjacent states
        next_states = np.clip(np.array([current + self._action_to_direction[i] for i in range(self.n_actions)]), 0, self.N-1)
        next_states_idx = next_states[:, 0]*self.N + next_states[:, 1]
    
        ## choose action with highest Q-value
        current_q = self.Q_inf[current[0], current[1], :]
        max_current_q = np.max(current_q)
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
    def sample(self, K):

        ## check kernel is valid
        self.k_check(K)

        # sample
        mean = np.zeros(self.N**2)
        samples = np.random.multivariate_normal(mean, K).reshape(self.N, self.N)

        #normalise
        lower, upper = 0.1, 0.9
        # lower+=0.01
        samples = lower + (upper - lower) * (samples - np.min(samples)) / (np.max(samples) - np.min(samples))

        # make all samples non-negative
        samples += np.abs(np.min(samples))
        return samples
    

    ## use GP regression to predict posterior distribution of rewards, given these observations,based on the currently inferred kernel
    def post_pred(self, K_inf, obs, pred='all', sigma=0.01):
        if isinstance(pred, str):
            pred_idx = np.arange(self.N**2)
        else:
            pred_idx = pred

        if obs is not None:
            obs_idx = obs[:, 0].astype(int)
            obs_rewards = obs[:, 3]
            # obs_idx = self.obs[:, 0].astype(int)
            # obs_rewards = self.obs[:, 3]
            
            # Covariance matrix of the already observed points
            K_obs = K_inf[obs_idx][:, obs_idx]
            
            # Covariance matrix between input points (i.e. points to be predicted) and observed points
            K_pred = K_inf[pred_idx][:, obs_idx]
            
            # inversion covariance matrix
            inv_K = np.linalg.inv(K_obs + sigma**2 * np.eye(len(obs_idx)))
            
            # Posterior mean calculation
            post_mean = K_pred @ inv_K @ obs_rewards
            post_cov = K_inf[pred_idx][:, pred_idx] - K_pred @ inv_K @ K_pred.T
        
        ## or if starting from nothing, just return the prior
        elif obs is None:
            post_mean = np.zeros(len(pred_idx)) - 0.5
            post_cov = K_inf[pred_idx][:, pred_idx]

        
        return post_mean, post_cov
    

    ## posterior prediction weighted by the likelihood of the observations under each kernel
    def weighted_post_pred(self, obs, pred='all'):
        if isinstance(pred, str):
            pred_idx = np.arange(self.N**2)
        else:
            pred_idx = pred    
        post_means = []
        post_covs = []
        lls = []
        
        ## imperfect memory of observations
        # obs = self.obs[-10:]

        ## loop through possible kernels
        for k_inf in self.all_Ks:
            post_mean, post_cov = self.post_pred(k_inf, obs = obs, pred=pred_idx)
            post_means.append(post_mean)
            post_covs.append(post_cov)

            ## calculate marginal likelihood of obs given this kernel
            ll = self.likelihood(k_inf, self.obs)
            lls.append(ll)
        
        ## weight each posterior prediction by the corresponding marginal likelihood
        self.k_weights = softmax(lls)
        post_mean = np.sum([self.k_weights[i] * post_means[i] for i in range(len(self.k_weights))], axis=0)

        ## weighted posterior covariance
        # post_cov = np.sum([k_weights[i] * (post_covs[i] + (post_means[i] - post_mean) @ (post_means[i] - post_mean).T) for i in range(len(k_weights))], axis=0)
        post_cov = np.sum([self.k_weights[i] * (post_covs[i] + np.outer(post_means[i] - post_mean, post_means[i] - post_mean)) for i in range(len(self.k_weights))], axis=0) ## need to check if this is correct

        return post_mean, post_cov
    

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
    


    ## calculate the optimal trajectory between the two points (i.e. the trajectory with the lowest cumulative cost)
    def optimal_trajectory(self, h_w = 0):

        # Initialize the open list (priority queue) and closed list (visited nodes)
        start, goal = [self._agent_location, self._target_location]
        start = tuple(map(int, start))
        goal = tuple(map(int, goal))
        open_list = []

        ## weighted combination of g(n) (actual cost) and h(n) (heuristic for step count)
        heapq.heappush(open_list, (-(h_w * self.heuristic(start, goal)), 0, start, []))
        closed_list = set()

        # Pop the node with the lowest total cost from the priority queue
        while open_list:
            estimated_total_cost, current_cost, current_point, path = heapq.heappop(open_list)
            if current_point in closed_list:
                continue
            
            # Add the current point to the path
            path = path + [current_point]
            
            # If we reached the goal, return the path and the accumulated reward
            if current_point == goal:
                route_cost = [self.costs[x, y] for x, y in path]
                return path, route_cost
            
            # Mark this point as visited
            closed_list.add(current_point)
            
            # Explore neighbors
            for neighbor in self.get_neighbors(current_point):
                if neighbor not in closed_list:
                    # Calculate new cost to reach this neighbor
                    new_cost = current_cost + self.costs[neighbor]
                    
                    # Add the neighbor to the open list with its weighted total cost
                    # weighted_total_cost = (1 - h_w) * new_cost + h_w * self.heuristic(neighbor, goal)
                    weighted_total_cost = -(new_cost + h_w * self.heuristic(neighbor, goal))
                    heapq.heappush(open_list, (weighted_total_cost, new_cost, neighbor, path))

        
        # If there's no path found, return empty
        return [], []
    
    ## h(n), i.e. the estimated cost to reach the goal from the current point (although I think this is just taking into account distance rather than reward)
    def heuristic(self, current, goal):
        x1, y1 = current
        x2, y2 = goal
        if self.metric == 'chebyshev':
            return max(abs(x2 - x1), abs(y2 - y1))
        elif self.metric == 'cityblock':
            return abs(x2 - x1) + abs(y2 - y1)
        
    ## get all possible neighbours for a given point in the grid
    def get_neighbors(self, point):
        x, y = point
        neighbors = []
        if self.metric == 'chebyshev':
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue  # Skip the current point itself
                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < self.N and 0 <= new_y < self.N:
                        neighbors.append((new_x, new_y))
        elif self.metric == 'cityblock':
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < self.N and 0 <= new_y < self.N:
                    neighbors.append((new_x, new_y))
        return neighbors


    ## generate random observations from current true GP kernel
    def gen_obs(self, samples, n_obs):
        obs_idx = np.random.randint(0, self.N**2, size=n_obs)

        ## map these observations to the grid and get the reward values
        obs_coords = np.column_stack(np.unravel_index(obs_idx, (self.N, self.N)))
        obs_rewards = samples[obs_coords[:, 0], obs_coords[:, 1]]
        obs = np.column_stack([obs_idx, obs_coords, obs_rewards])

        return obs
    
    ## dynamic programming
    def value_iteration(self, dp_costs, max_iters = 1000, theta = 0.001, discount = 0.99):
        
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
        info = self.get_obs()
        start = info['agent']
        goal = info['target']
        dp_costs[goal[0], goal[1]] = 0

        ## loop through states
        for i in range(max_iters):
            delta = 0
            for x in range(self.N):
                for y in range(self.N):
                    v = V[x, y]
                    q = np.zeros(self.n_actions)

                    ## loop through actions and get the discounted value of each of the next states
                    for a in range(self.n_actions):
                        next_state = np.clip([x, y] + self._action_to_direction[a], 0, self.N-1)
                        q[a] = dp_costs[next_state[0], next_state[1]] + discount*V[next_state[0], next_state[1]]

                        ## update the Q-table
                        Q[x, y, a] = q[a]

                    ## use the best action to update the value of the current state
                    V[x, y] = np.max(q)
                    A[x, y] = np.argmax(q)


                    ## check if converged
                    delta = max(delta, np.abs(v - V[x, y]))
            
            if delta < theta:
                # print('DP converged after {} iterations'.format(i))
                break

        return V, Q
