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
from value_iteration import value_iteration
from scipy.stats import rankdata, truncnorm
from scipy.linalg import cholesky
from minimax_tilting_sampler import TruncatedMVN
from base_kernels import *


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

    def __init__(self, N, n_episodes=1, true_k=None, beta_params=None, metric = 'cityblock', size=5):
        
        ## seed?
        # seed = np.random.randint(0, 1000)
        # np.random.seed(seed)

        ## initialise the GP grid
        self.N = N
        x = np.arange(N)
        y = np.arange(N)
        X,Y = np.meshgrid(x,y)
        self.locations = np.column_stack([X.ravel(), Y.ravel()])

        ### initialise farm
        init_done = False
        while not init_done:
            self.high_cost, self.low_cost = -0.9, -0.1
            default_param = 0.5
            self.alpha_row = beta_params['alpha_row']
            self.beta_row = beta_params['beta_row']
            self.alpha_col = beta_params['alpha_col']
            self.beta_col = beta_params['beta_col']
            self.row_p = np.random.beta(self.alpha_row,self.beta_row, self.N)
            self.col_q = np.random.beta(self.alpha_col, self.beta_col, self.N)
            # self.col_q = np.ones(self.N)
            self.p_costs = np.outer(self.row_p, self.col_q)
            # self.p_costs = 1 - self.p_costs

            ## prob = p(high cost)
            # self.costs = np.array([self.high_cost if r<self.p_costs.flatten()[ri] else self.low_cost for ri, r in enumerate(np.random.random(self.N**2))]).reshape(self.N, self.N)

            ## prob = p(low cost)
            # self.costs = np.array([self.high_cost if r>self.p_costs.flatten()[ri] else self.low_cost for ri, r in enumerate(np.random.random(self.N**2))]).reshape(self.N, self.N)
            
            ### gym inits

            ## sizes
            self.window_size = 512

            # Observations are dictionaries with the agent's and the goal's location.
            size = 5
            self.observation_space = spaces.Dict(
                {
                    "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                    "goal": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                }
            )

            ## init trial info
            # self.starts = [[0, 0],              [0, self.N-1],    [self.N-1, 0],    [self.N-1, self.N-1]]
            # self.goals = [[self.N-1, self.N-1], [self.N-1, 0],    [0, self.N-1],    [0, 0]]
            self.starts = []
            self.goals = []
            self.costss = []
            self.o_trajs = []
            self.o_traj_costs = []
            self.o_traj_total_costs = []
            self.o_traj_actions = []
            self.n_episodes = n_episodes
            for e in range(n_episodes):
                try:
                    start, goal = self.sample_SG()
                    self.starts.append(start)
                    self.goals.append(goal)

                    ## prob = p(high cost)
                    # self.costss.append(np.array([self.high_cost if r<self.p_costs.flatten()[ri] else self.low_cost for ri, r in enumerate(np.random.random(self.N**2))]).reshape(self.N, self.N))
                    
                    ## prob = p(low cost)
                    self.costss.append(np.array([self.high_cost if r>self.p_costs.flatten()[ri] else self.low_cost for ri, r in enumerate(np.random.random(self.N**2))]).reshape(self.N, self.N))

                except:
                    # print("couldn't find start and goal. retrying...")
                    break
            if len(self.starts)==self.n_episodes:
                init_done = True
                self.e = 0

 

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

        self.sim = False

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
    def reset(self, seed=None, start_goal=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        ## set costs, given p_costs
        # self.costs = np.array([self.high_cost if r>self.p_costs.flatten()[ri] else self.low_cost for ri, r in enumerate(np.random.random(self.N**2))]).reshape(self.N, self.N)
        self.costs = self.costss[self.e]

        ## set start and end
        if start_goal is not None: 
            self._agent_location = np.array(start_goal[0], dtype=int)
            self._goal_location = np.array(start_goal[1], dtype=int)
        else:
            # self._agent_location, self._goal_location = self.sample_SG()
            self._agent_location = np.array(self.starts[self.e])
            self._goal_location = np.array(self.goals[self.e])

            ## get true Q vals (u only do this if it's a real reset, rather than a reset for an imagined rollout)

            ## pq = p(high cost)
            # dp_costs = self.p_costs*self.high_cost + (1-self.p_costs)*self.low_cost ## standard case (i.e. pq = p(high cost))
            # dp_costs[self._goal_location[0], self._goal_location[1]] = 0

            ## pq = p(low cost)
            dp_costs = self.p_costs*self.low_cost + (1-self.p_costs)*self.high_cost
            dp_costs[self._goal_location[0], self._goal_location[1]] = 0

            self.V_true, self.Q_true, self.A_true = value_iteration(dp_costs=dp_costs, goal=self._goal_location)
            self.optimal_trajectory(self._agent_location, self._goal_location)


        ## initialise trial info
        self.terminated = False
        observation = self.get_obs()
        info = self._get_info()
        # current_cost = self.get_cost(self._agent_location)
        current_cost = self.costs[self._agent_location[0], self._agent_location[1]]

        ## reset obs on each trial
        # self.obs = np.array([self._agent_location[0], self._agent_location[1], current_cost], ndmin=2)

        ## or, observations accumulate over trials, and agent observes starting position
        if not self.sim:
            if not hasattr(self, 'obs') or self.obs is None:
                self.obs = np.array([[self._agent_location[0], self._agent_location[1], current_cost]]) 
            else:
                # print(len(self.obs))
                self.obs = np.vstack([self.obs, [self._agent_location[0], self._agent_location[1], current_cost]])
            self.obs_tmp = self.obs.copy()

        ## or, observations accumulate over trials, but agent doesn't observe starting position
        # if not self.sim:
        #     if not hasattr(self, 'obs') or self.obs is None:
        #         self.obs = np.array([])
        #         self.obs_tmp = np.array([[self._agent_location[0], self._agent_location[1], current_cost]])
        #         self.obs_start_tmp = self.obs_tmp.copy()
        #     else:
        #     ####     self.obs = np.vstack([self.obs, [self._agent_location[0], self._agent_location[1], current_cost]])
        #         self.obs_tmp = self.obs.copy()
        #         self.obs_tmp = np.vstack([self.obs_tmp, [self._agent_location[0], self._agent_location[1], current_cost]])
        #         self.obs_start_tmp = self.obs_tmp.copy()

        ## or, if simulating some unknown future environment, the observations are given by the previous tree, so we trivially have obs already
        elif self.sim:
            self.obs_tmp = np.vstack([self.obs, [self._agent_location[0], self._agent_location[1], current_cost]])
            self.obs_start_tmp = self.obs_tmp.copy()


        ## dynamic programming to get the true optimal trajectory
        # if not self.sim:
        #     dp_costs = self.costs.copy()
        #     self.V_true, self.Q_true, self.A_true = value_iteration(dp_costs=dp_costs, goal=self._goal_location)

        #     ## get the costs of this optimal trajectory
        #     self.optimal_trajectory()

        ## initialise actual trajectory as list of tuples
        if not self.sim:

            ## start from start
            self.a_traj = [tuple(self._agent_location)]

            ## or start from nothing
            # self.a_traj = []
            
            self.action_scores = []
        
        return observation, info
    
    ## get some S-G pairs
    def sample_SG(self):

        ## sample start and goal locations
        dist = 0
        min_dist = self.N*0.6
        angle = 0
        angle_tolerance = 0.5
        angle_bounds = [45*(1+angle_tolerance), 45*(1-angle_tolerance)]
        row_or_col = 1
        t = 0
        worth_it = False
        new = False
        new_rc = False
        route_optimality_tolerance = 0.75
        while (dist<min_dist) or (row_or_col>0) or (angle>angle_bounds[0]) or (angle<angle_bounds[1]) or (not worth_it) or (not new) or (not new_rc):
            agent_location = self.np_random.integers(0, self.N, size=2, dtype=int)
            goal_location = self.np_random.integers(
                0, self.N, size=2, dtype=int
            )

            ## distance criterion
            dist = np.max(cdist([agent_location, goal_location], [agent_location, goal_location], metric='cityblock'))

            ## same row/col criterion
            row_or_col = np.sum(agent_location == goal_location)

            ## angle criterion
            angle = node_angle(agent_location, goal_location)

            ## check if start or goal have appeared already
            if len(self.starts)==0:
                new = True
            else:
                for s, g in zip(self.starts, self.goals):
                    if np.array_equal(agent_location, s) or np.array_equal(goal_location, g):
                        new = False
                    else:
                        new = True

            ## check if start or goal is in the same row or column as another start or goal
            if len(self.starts)==0:
                new_rc = True
            else:
                for s, g in zip(self.starts, self.goals):
                    if np.sum(agent_location == s)>0 or np.sum(agent_location == g)>0 or np.sum(goal_location == s)>0 or np.sum(goal_location == g)>0:
                        new_rc = False
                    else:
                        new_rc = True
            

            ## checkpoint before doing DP
            if (dist<min_dist) or (row_or_col>0) or (angle>angle_bounds[0]) or (angle<angle_bounds[1]) or (not new) or (not new_rc):
                continue

            ### comparison of optimal vs manhattan routes

            # ## standard case (i.e. pq = p(high cost))
            # dp_costs = self.p_costs*self.high_cost + (1-self.p_costs)*self.low_cost 
            # dp_costs[goal_location[0], goal_location[1]] = 0
            
            ## alternative case (i.e. pq = p(low cost))
            dp_costs = self.p_costs*self.low_cost + (1-self.p_costs)*self.high_cost
            dp_costs[goal_location[0], goal_location[1]] = 0

            self.V_true, self.Q_true, self.A_true = value_iteration(dp_costs=dp_costs, goal=goal_location)
            o_traj, o_traj_costs, o_traj_total_cost, o_traj_actions = self.optimal_trajectory(agent_location, goal_location)


            ## by length
            # n_steps_opt = len(self.o_traj)-1
            # worth_it = n_steps_opt > dist

            ## or, by cost (i.e. vs manhattan vertical-first or horizontal-first)
            manhattan_costs = self.manhattan_trajectory(agent_location, goal_location)
            # worth_it = (manhattan_costs[0]/o_traj_total_cost) >= route_optimality_tolerance and (manhattan_costs[1]/o_traj_total_cost) >= route_optimality_tolerance ## p(high cost)
            # worth_it = (manhattan_costs[0]/o_traj_total_cost) <= route_optimality_tolerance and (manhattan_costs[1]/o_traj_total_cost) >= route_optimality_tolerance ## p(low cost)
            worth_it = (o_traj_total_cost/manhattan_costs[0]) <= route_optimality_tolerance and (o_traj_total_cost/manhattan_costs[1]) <= route_optimality_tolerance ## p(low cost)

            t+=1
            if t>100:
                raise ValueError('cant find start and end')

        ## for sanity check, just place agent and goal in opposite corners
        # self._agent_location = np.array(self.starts[self.n_eps%4])
        # self._goal_location = np.array(self.goals[self.n_eps%4])
        # self.n_eps += 1

        self.o_trajs.append(o_traj)
        self.o_traj_costs.append(o_traj_costs)
        self.o_traj_total_costs.append(o_traj_total_cost)
        self.o_traj_actions.append(o_traj_actions)

        return agent_location, goal_location

    
    ## custom functions for manually editing the env
    def set_state(self, state):
        self._agent_location = state
    def set_sim(self, sim):
        self.sim = sim
    def set_obs(self, obs):
        self.obs = obs
    def flush_obs(self): ## necessary for MCTS
        self.obs_tmp = self.obs.copy()
        if len(self.obs_tmp)==0:
            self.obs_tmp = self.obs_start_tmp.copy()

    ## get cost, given p(cost)
    def get_cost(self, state):

        ## pq = p(high cost)
        # return self.high_cost if np.random.random() < self.p_costs[state[0], state[1]] else self.low_cost
        
        ## pq = p(low cost)
        return self.high_cost if np.random.random() > self.p_costs[state[0], state[1]] else self.low_cost
    def get_pred_cost(self, state):
        
        ## pq = p(high cost)
        # return self.high_cost if np.random.random() < self.predicted_p_costs[state[0], state[1]] else self.low_cost

        ## pq = p(low cost)
        return self.high_cost if np.random.random() > self.predicted_p_costs[state[0], state[1]] else self.low_cost
    
    ## functions for receiving predictions from the agent
    def receive_predictions(self, predicted_p_costs):
        # self.posterior_mean = posterior_mean
        # self.posterior_cov = posterior_cov
        # self.posterior_var = posterior_var
        # self.predicted_costs = predicted_costs.reshape(self.N, self.N)
        self.predicted_p_costs = predicted_p_costs.reshape(self.N, self.N)

    ## take a step in the environment
    def step(self, action):
        
        ## first, get the ranking of the best actions to take under the *true* optimal policy, given the agent's current position
        current_Q_vals = self.Q_true[self._agent_location[0], self._agent_location[1], :]
        action_ranking = rankdata(current_Q_vals, method='max') - 1

        ## get the score of the action that will  actually be taken, given the ranking of the optimal actions
        action_score = action_ranking[action] + 1
        action_score /= self.n_actions ## may be more suitable to divide by len(actions) in case of wall states

        ## or, score the action based on the normalised Q-values of the available actions
        norm_Q_vals = (current_Q_vals - np.nanmin(current_Q_vals)) / (np.nanmax(current_Q_vals) - np.nanmin(current_Q_vals))
        action_score = norm_Q_vals[action]
        
        ## take the actual action 
        direction = self._action_to_direction[action] 

        ## move to the new state
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.N - 1
        )


        ## get the predicted cost of the new state
        # predicted_cost = self.predicted_costs[self._agent_location[0]*self.N + self._agent_location[1]]
        # predicted_cost = self.predicted_costs[self._agent_location[0], self._agent_location[1]]
        predicted_cost = self.get_pred_cost(self._agent_location)
        
        ## get the actual cost of the current state
        # current_cost = self.get_cost(self._agent_location)
        current_cost = self.costs[self._agent_location[0], self._agent_location[1]]


        ## return the real cost if not simulating
        if not self.sim:
            cost = current_cost
            
            ## update observation and trajectory arrays - i.e. agent observes along the way
            self.a_traj.append(tuple(self._agent_location))
            self.obs = np.vstack([self.obs, [self._agent_location[0], self._agent_location[1], current_cost]])
            self.obs_tmp = np.vstack([self.obs_tmp, [self._agent_location[0], self._agent_location[1], current_cost]])
            
            ## calculate new posterior for next trial
            # self.posterior_mean, self.posterior_var = self.inference_func(obs = self.obs, pred='all')

            ## or, temporarily store observations until the end of the trial (no need to calculate posterior at each step, since there are no new observations)
            # self.a_traj.append(tuple(self._agent_location))
            # self.obs_tmp = np.vstack([self.obs_tmp, [self._agent_location[0], self._agent_location[1], cost]])

            ## store info on optimality of the choice, given the agent's current position
            self.action_scores.append(action_score)


        ## return the predicted cost if simulating
        elif self.sim:
            # cost = current_cost
            cost = predicted_cost 
            # cost += self.expl_beta * np.sqrt(var_cost) #UCB

            ## still need to store obs_tmp along the way for subsequent posterior inference
            # self.a_traj.append(tuple(self._agent_location))
            self.obs_tmp = np.vstack([self.obs_tmp, [self._agent_location[0], self._agent_location[1], cost]])
                

        # An episode is done iff the agent has reached the goal
        if np.array_equal(self._agent_location, self._goal_location):
            self.terminated = True
            cost=0 ## cost of final state is 0
            # cost = self.expl_beta * np.sqrt(var_cost)
        
            ## update observation array only once the episode is complete
            if not self.sim:
                self.obs = self.obs_tmp.copy()

                ## sum of costs of route INC START AND END
                # self.a_traj_costs = [self.costs[x, y] for x, y in self.a_traj]
                # self.a_traj_total_cost = np.sum(self.a_traj_costs)

                ## sum of costs of route EXC START AND END
                self.a_traj_costs = [self.costs[x, y] for x, y in self.a_traj[1:-1]]
                self.a_traj_total_cost = np.sum(self.a_traj_costs)

                ## scores for the trial
                self.action_score = np.nanmean(self.action_scores)
                self.cost_ratio = self.o_traj_total_costs[self.e] / self.a_traj_total_cost

                ## update ep counter
                self.e += 1

        else:
            self.terminated = False
        observation = self.get_obs()
        info = self._get_info()
        terminated = self.terminated
        truncated=False
        return observation, cost, terminated, truncated, info    


    ## calculate the optimal trajectory between the two points, as given by the true DP solution
    def optimal_trajectory(self, start, goal):
        current = start.copy()

        ## start with the current state
        o_traj = [tuple(current)]
        expected_cost = self.p_costs[current[0], current[1]]*self.low_cost + (1-self.p_costs[current[0], current[1]])*self.high_cost
        o_traj_costs = [expected_cost]

        ## or start from nothing
        # o_traj = []
        # o_traj_costs = []

        ## save the optimal actions
        o_traj_actions = []

        ## Loop until the goal is reached
        visited = set()
        while True:
            i, j = current
            action = int(self.A_true[i, j])  # Ensure action index is int
            o_traj_actions.append(action)
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
            
            ## check if goal has been reached (THIS SHOULD COME BEFORE THE APPEND IF WE DON'T WANT TO INCLUDE THE GOAL STATE
            # if np.array_equal(current, goal):
            #     break

            # Update trajectory and expected cost
            o_traj.append(tuple(current))
            expected_cost = self.p_costs[current[0], current[1]]*self.low_cost + (1-self.p_costs[current[0], current[1]])*self.high_cost
            o_traj_costs.append(expected_cost)

            ## check if goal has been reached (THIS SHOULD COME BEFORE THE APPEND IF WE DON'T WANT TO INCLUDE THE GOAL STATE
            if np.array_equal(current, goal):
                break

        ## calculate the total cost of the trajectory INC START AND END
        # o_traj_total_cost = np.sum(o_traj_costs)

        ## calculate the total cost of the trajectory EXC START AND END
        o_traj_costs = o_traj_costs[1:-1]
        o_traj_total_cost = np.sum(o_traj_costs)

        return o_traj, o_traj_costs, o_traj_total_cost, o_traj_actions


    ## calculate the cost of the simplest manhattan paths
    def manhattan_trajectory(self, start, goal):
        x1, y1 = start
        x2, y2 = goal
        
        ## horizontal-first trajectory (i.e. move in x direction first)
        horizontal_trajectory = [start]
        while x1 != x2:
            if x2 > x1:
                x1 += 1  # Move right
            else:
                x1 -= 1  # Move left
            horizontal_trajectory.append((x1, y1))
        while y1 != y2:
            if y2 > y1:
                y1 += 1  # Move up
            else:
                y1 -= 1  # Move down
            horizontal_trajectory.append((x1, y1))
        
        ## vertical-first trajectory (i.e. move in y direction first)
        x1, y1 = start
        vertical_trajectory = [start]
        while y1 != y2:
            if y2 > y1:
                y1 += 1
            else:
                y1 -= 1
            vertical_trajectory.append((x1, y1))
        while x1 != x2:
            if x2 > x1:
                x1 += 1
            else:
                x1 -= 1
            vertical_trajectory.append((x1, y1))

        ## calculate the costs of these trajectories
        # horizontal_trajectory_costs = [self.costs[x, y] for x, y in horizontal_trajectory]
        # vertical_trajectory_costs = [self.costs[x, y] for x, y in vertical_trajectory]
        # horizontal_trajectory_costs = [self.p_costs[x, y] for x, y in horizontal_trajectory]
        # vertical_trajectory_costs = [self.p_costs[x, y] for x, y in vertical_trajectory]
        horizontal_trajectory_costs = [self.p_costs[x, y]*self.low_cost + (1-self.p_costs[x, y])*self.high_cost for x, y in horizontal_trajectory]
        vertical_trajectory_costs = [self.p_costs[x, y]*self.low_cost + (1-self.p_costs[x, y])*self.high_cost for x, y in vertical_trajectory]
        manhattan_costs = [np.sum(horizontal_trajectory_costs), np.sum(vertical_trajectory_costs)]

        return manhattan_costs