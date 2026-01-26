from enum import Enum
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
from itertools import permutations, combinations


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


class GridEnv(gym.Env):

    def __init__(self, N, n_trials=1, expt_info={'type':'free'}, beta_params=None, metric = 'cityblock', size=5, seed=None):
        
        ## seed
        if seed is not None:
            seed = np.random.randint(0, 1000)
        self.seed = seed
        np.random.seed(self.seed)

        ## initialise the grid
        self.N = N
        x = np.arange(N)
        y = np.arange(N)
        X,Y = np.meshgrid(x,y)
        self.locations = np.column_stack([X.ravel(), Y.ravel()])
        self.expt = expt_info['type']
        self.context = expt_info['context']
        self.n_afc = expt_info['n_afc'] if 'n_afc' in expt_info else 2
        if self.expt_info['objective'] is not None:
            self.objective = expt_info['objective']
        else:
            self.objective = 'costs'


        ### misc gym inits

        ## sizes
        self.window_size = 512

        # Observations are dictionaries with the agent's and the goal's location.
        size = 5
        # Observation space is a 2D location on the grid
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0]),  # Minimum x and y
            high=np.array([self.N - 1, self.N - 1]),  # Maximum x and y
            dtype=np.int32
        )

        # define actions, depending on metric
        self.metric = metric

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        if self.metric == 'cityblock':
            self.action_space = spaces.Discrete(4)
            # self.action_to_direction = {
            #     Actions.right.value: np.array([1, 0]),
            #     Actions.up.value: np.array([0, 1]),
            #     Actions.left.value: np.array([-1, 0]),
            #     Actions.down.value: np.array([0, -1]),
            # }
            self.action_to_direction = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
            self.direction_to_action = {tuple(v): k for k, v in enumerate(self.action_to_direction)}
            self.n_actions = 4

            ## action labels (NB these deviate from env action space, bc axes are flipped for plotting
            self.action_labels = ['down', 'right', 'up', 'left']

        elif self.metric == 'chebyshev':
            self.action_space = spaces.Discrete(8)
            self.action_to_direction = {
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

        ## define abstract sequences
        if self.N%2==1:
            self.path_len = self.N-1
        else:
            self.path_len = self.N-2
        max_turns =1
        self.abstract_sequences = self.generate_abstract_sequences(self.path_len, max_turns)

        ### initialise grid
        init_done = False
        t=0
        while not init_done:
            SG_found = False
            paths_found = False
            if self.objective == 'rewards':
                self.high_cost, self.low_cost = 0, 1
            elif self.objective == 'costs':
                self.high_cost, self.low_cost = -1, -0
            else:
                raise ValueError('objective must be either rewards or costs')
            self.alpha_row = beta_params['alpha_row']
            self.beta_row = beta_params['beta_row']
            self.alpha_col = beta_params['alpha_col']
            self.beta_col = beta_params['beta_col']

            ## combinatorial
            # self.row_p = np.random.beta(self.alpha_row,self.beta_row, self.N)
            # self.col_q = np.random.beta(self.alpha_col, self.beta_col, self.N)
            # self.p_costs = np.outer(self.row_p, self.col_q)

            
            ### determine context
            if self.context == 'row':
                self.row_p = np.random.beta(self.alpha_row,self.beta_row, self.N)
                self.col_q = np.ones(self.N)
                self.p_costs = np.outer(self.row_p, self.col_q)
            elif self.context == 'column':
                self.row_p = np.ones(self.N)
                self.col_q = np.random.beta(self.alpha_col, self.beta_col, self.N)
                self.p_costs = np.outer(self.row_p, self.col_q)


            ####  define costs

            ### define actual binary costs for each trial, assuming they regenerate each time
            # self.costss = []
            # for t in range(self.n_trials):

                ## prob = p(high cost)
                # self.costss.append(np.array([self.high_cost if r<self.p_costs.flatten()[ri] else self.low_cost for ri, r in enumerate(np.random.random(self.N**2))]).reshape(self.N, self.N))
                
                ## prob = p(low cost)
                # self.costss.append(np.array([self.high_cost if r>self.p_costs.flatten()[ri] else self.low_cost for ri, r in enumerate(np.random.random(self.N**2))]).reshape(self.N, self.N))

            ### or, define actual binary costs for each trial, assuming they are the same across trials
            self.costss = []
            costs = np.array([self.high_cost if r>self.p_costs.flatten()[ri] else self.low_cost for ri, r in enumerate(np.random.random(self.N**2))]).reshape(self.N, self.N)
            self.costss = [costs for t in range(n_trials)]

            ## check if decent spread of high and low costs - i.e. mean cost should be around the mean of the two costs, +/- some tolerance
            mean_cost = np.mean(costs)
            ideal_mean_cost = (self.high_cost + self.low_cost)/2
            tol = 0.05
            if (mean_cost<ideal_mean_cost-tol) or (mean_cost>ideal_mean_cost+tol):
                continue

            ## init trial info, depending on the expt type
            self.starts = []
            self.goals = []
            self.dominant_axis_A = []
            self.dominant_axis_B = []
            self.dominant_axis_C = []
            self.path_states = []
            self.path_actions = []
            self.sampled_abstract_sequences = []
            self.context_alignment = []
            self.path_actual_costs = []
            self.path_expected_costs = []
            self.o_trajs = []
            self.o_traj_costs = []
            self.o_traj_total_costs = []
            self.o_traj_actions = []
            self.n_trials = n_trials


            ## if AFC, we use the same SG pair all the way through
            if self.expt=='AFC':
                try:
                    start, goal = self.sample_SG()
                    SG_found=True
                except:
                    continue

            ## generate relevant trial info for each trial
            for t in range(n_trials):

                try:
                
                    ## free movement
                    if self.expt == 'free':
                        start, goal = self.sample_SG()
                        self.starts.append(start)
                        self.goals.append(goal)
                        SG_found=True
                        paths_found=True

                        ## get info about optimal path
                        o_traj, o_traj_costs, o_traj_total_cost, o_traj_actions = self.optimal_trajectory(start, goal)
                        self.o_trajs.append(o_traj)
                        self.o_traj_costs.append(o_traj_costs)
                        self.o_traj_total_costs.append(o_traj_total_cost)
                        self.o_traj_actions.append(o_traj_actions)

                    ## AFC
                    elif self.expt=='AFC':
                        max_turns=1
                        sampled_abstract_sequences, path_actions, path_states, starts, goals = self.sample_paths()
                        self.starts.append(starts)
                        self.goals.append(goals)
                        self.path_states.append(path_states)
                        self.path_actions.append(path_actions)
                        self.sampled_abstract_sequences.append(sampled_abstract_sequences)

                        ## more info on A and B wrt/ orientation and context
                        
                        dominant_axis_list = [self.dominant_axis_A, self.dominant_axis_B, self.dominant_axis_C]
                        for si, s_a_s in enumerate(sampled_abstract_sequences):
                            alignment_tmp = []
                            
                            # determine the dominant axis - i.e. is the path more vertical or horizontal?
                            if s_a_s[0]> s_a_s[1]:
                                dominant_axis_list[si].append('vertical')
                                if self.context == 'row':
                                    alignment_tmp.append('orthogonal')
                                elif self.context == 'column':
                                    alignment_tmp.append('aligned')
                            elif s_a_s[0]< s_a_s[1]:
                                dominant_axis_list[si].append('horizontal')
                                if self.context == 'row':
                                    alignment_tmp.append('aligned')
                                elif self.context == 'column':
                                    alignment_tmp.append('orthogonal')
                            elif s_a_s[0]==s_a_s[1]:
                                dominant_axis_list[si].append('L-shaped')
                                alignment_tmp.append('L-shaped')
                            self.context_alignment.append(alignment_tmp)
                        SG_found = True


                        ## get info about optimal path (WILL CHANGE THIS LATER SINCE THE NOTION OF OPTIMAL IS DIFFERENT FOR AFC)
                        # o_traj, o_traj_costs, o_traj_total_cost, o_traj_actions = self.optimal_trajectory(start, goal)
                        self.o_trajs.append([])
                        self.o_traj_costs.append([])
                        self.o_traj_total_costs.append(np.nan)
                        self.o_traj_actions.append([])

                        ## save expected costs of the paths
                        path_actual_costs = []
                        path_expected_costs = []
                        for path in self.path_states[t]:
                            
                            ## pq = p(high cost)
                            # path_cost = np.sum([self.p_costs[x, y]*self.high_cost + (1-self.p_costs[x, y])*self.low_cost for x, y in path]) 

                            ## pq = p(low cost)
                            # path_expected_cost = np.sum([self.p_costs[x, y]*self.low_cost + (1-self.p_costs[x, y])*self.high_cost for x, y in path]) 
                            path_expected_cost = np.sum([self.p_costs[x, y]*self.compound_cost(self.low_cost,t) + (1-self.p_costs[x, y])*self.compound_cost(self.high_cost,t) for x, y in path])  ## if using compound costs
                            path_actual_cost = np.sum([self.compound_cost(self.costss[t][x, y],t) for x, y in path])
                            path_expected_costs.append(path_expected_cost)
                            path_actual_costs.append(path_actual_cost)
                        self.path_expected_costs.append(path_expected_costs)
                        self.path_actual_costs.append(path_actual_costs)
                        paths_found = True

                except:
                    break

            ## hacky: randomise the order of the SGs and paths, so that A and B are not always in the same order
            # if self.expt=='AFC' and SG_found and paths_found:
            #     for t in range(self.n_trials):
            #         order = np.random.permutation(self.n_afc)
            #         self.starts[t] = [self.starts[t][i] for i in order]
            #         self.goals[t] = [self.goals[t][i] for i in order]
            #         self.path_states[t] = [self.path_states[t][i] for i in order]
            #         self.path_actions[t] = [self.path_actions[t][i] for i in order]
            #         if order[0]==0:
            #             ## many things stay as is
            #             pass
            #         elif order[0]==1:
            #             # swap dominant_axis_A and dominant_axis_B
            #             tmp = self.dominant_axis_A[t]
            #             self.dominant_axis_A[t] = self.dominant_axis_B[t]
            #             self.dominant_axis_B[t] = tmp

            #         self.sampled_abstract_sequences[t] = [self.sampled_abstract_sequences[t][i] for i in order]

                    


            ## if all the SGs and paths have been found, then we then need to check overlap across trials
            if len(self.starts)==self.n_trials:
        
                ## hacky fix: make sure all coordinates in the path_states list are tuples of int, rather than tuples of int64
                for t in range(self.n_trials):
                    for p, path in enumerate(self.path_states[t]):
                        self.path_states[t][p] = [tuple([int(x) for x in state]) for state in path]
                
                ## get info on path overlaps in AFC expt
                if self.expt == 'AFC':
                    self.most_overlap = []
                    self.path_future_overlaps = []
                    self.path_future_row_overlaps = np.zeros((self.n_trials, self.n_afc))
                    self.path_future_col_overlaps = np.zeros((self.n_trials, self.n_afc))
                    self.path_future_row_and_col_overlaps = np.zeros((self.n_trials, self.n_afc))
                    self.path_future_rel_overlaps = np.zeros((self.n_trials, self.n_afc))
                    self.path_future_irrel_overlaps = np.zeros((self.n_trials, self.n_afc))

                    ## check that no starts or goals are shared across trials
                    all_starts = [tuple(s) for trial_starts in self.starts for s in trial_starts]
                    all_goals = [tuple(g) for trial_goals in self.goals for g in trial_goals]
                    n_distinct_starts = len(set(all_starts))
                    n_distinct_goals = len(set(all_goals))
                    n_distinct_starts_and_goals = len(set(all_starts + all_goals))
                    if (n_distinct_starts != self.n_afc * self.n_trials) or (n_distinct_goals != self.n_afc * self.n_trials) or (n_distinct_starts_and_goals != 2 * self.n_afc * self.n_trials):
                        continue
                        
                    for t in range(self.n_trials-1):

                        ### calculate the number of states in the current path that appear in the future set

                        ## no repeats (e.g. if [x,y] appears in trials 2 and 3, only count it once)
                        # future_states = []
                        # for next_t in range(t+1, self.n_trials):
                        #     for next_path in self.path_states[next_t]:
                        #         future_states.extend(next_path)
                        # n_intersections = []
                        # for path in self.path_states[t]:
                        #     intersections = set(path).intersection(set(future_states))
                        #     n_intersections.append(len(intersections) -2) ## -2 if start and end are shared
                        # self.path_n_intersections.append(n_intersections)

                        ## overlaps
                        n_overlaps = []
                        for path in self.path_states[t]:
                            path = path.copy()
                            intersections = []
                            for next_t in range(t+1, self.n_trials):
                                for next_path in self.path_states[next_t]:
                                    next_path = next_path.copy()
                                    intersection = set(path).intersection(set(next_path))
                                    
                                    ## if path and next_path share start and end, remove them from the intersection
                                    if np.array_equal(path[0], next_path[0]):
                                        intersection = intersection - set([path[0]])
                                    if np.array_equal(path[-1], next_path[-1]):
                                        intersection = intersection - set([path[-1]])


                                    ## allow double counting (e.g. if [x,y] appears in trials 2 and 3, count it twice)
                                    # intersections.extend(intersection)

                                    ## or, prevent double counting of states by seeing if they have already been counted (e.g. if [x,y] appears in trials 2 and 3, count it once))
                                    for state in intersection:
                                        if state not in intersections:
                                            intersections.append(state)
                            n_overlaps.append(len(intersections))
                        self.path_future_overlaps.append(n_overlaps)
                        self.most_overlap.append(np.argmax(n_overlaps))
                


                        ## for each path, check how many subsequently visited states are on rows or columns covered by these paths
                        for p, path in enumerate(self.path_states[t]):
                            row_overlap = []
                            col_overlap = []

                            ## get the rows and columns that are covered by the first path
                            p_rows = [state[0] for state in path]
                            p_cols = [state[1] for state in path]

                            ## loop through the future paths and check how many of them cover these rows and columns
                            for next_t in range(t+1, self.n_trials):
                                rel_overlap_tmp = []
                                irrel_overlap_tmp = []
                                for next_path in self.path_states[next_t]:
                                    for state in next_path:
                                        if state[0] in p_rows:
                                            row_overlap.append(state)
                                            if self.context=='row':
                                                rel_overlap_tmp.append(state)
                                            elif self.context=='column':
                                                irrel_overlap_tmp.append(state)
                                        if state[1] in p_cols:
                                            col_overlap.append(state)
                                            if self.context=='column':
                                                rel_overlap_tmp.append(state)
                                            elif self.context=='row':
                                                irrel_overlap_tmp.append(state)

                                ## count total number of overlapping states (inc duplicates)
                            #     self.path_future_rel_irrel[t, next_t, p, 0] = len(rel_overlap_tmp)
                            #     self.path_future_rel_irrel[t, next_t, p, 1] = len(irrel_overlap_tmp)
                            # total_row_overlap = len(row_overlap)
                            # total_col_overlap = len(col_overlap)

                            ## count total number of overlapping states (exc duplicates)
                            total_row_overlap = len(set(row_overlap))
                            total_col_overlap = len(set(col_overlap))
                            self.path_future_row_overlaps[t, p] = total_row_overlap
                            self.path_future_col_overlaps[t, p] = total_col_overlap
                            if self.context == 'row':
                                self.path_future_rel_overlaps[t, p] = total_row_overlap
                                self.path_future_irrel_overlaps[t, p] = total_col_overlap
                            elif self.context == 'column':
                                self.path_future_rel_overlaps[t, p] = total_col_overlap
                                self.path_future_irrel_overlaps[t, p] = total_row_overlap

                    ## trivially, the final trial has no future overlaps
                    self.path_future_overlaps.append([0 for a in range(self.n_afc)]) 
                    self.path_future_row_overlaps[-1, :] = np.zeros(self.n_afc)
                    self.path_future_col_overlaps[-1, :] = np.zeros(self.n_afc)
                    self.path_future_row_and_col_overlaps[-1, :] = np.zeros(self.n_afc)
                    self.path_future_rel_overlaps[-1, :] = np.zeros(self.n_afc)
                    self.path_future_irrel_overlaps[-1, :] = np.zeros(self.n_afc)

                    
                    ## for the relevant context, get the ratio between paths of relevant future overlaps, and then irrelevant future overlaps
                    relevant_first_overlaps = self.path_future_rel_overlaps[0]
                    irrelevant_first_overlaps = self.path_future_irrel_overlaps[0]
                    relevant_overlap_ratio = np.max(relevant_first_overlaps) / np.min(relevant_first_overlaps)
                    irrelevant_overlap_ratio = np.max(irrelevant_first_overlaps) / np.min(irrelevant_first_overlaps)
                    overlap_ratio_tol = 2

                    ## must be at least overlap_ratio_tol times as many overlaps in one path than the other
                    # if relevant_overlap_ratio >= overlap_ratio_tol:
                    #     self.same_overlaps = False
                    #     self._trial = 0
                    #     init_done = True

                    ## or, even more restrictive: the context-aligned path must have more relevant overlaps
                    # if (relevant_overlap_ratio >= overlap_ratio_tol) & (relevant_first_overlaps[1]>relevant_first_overlaps[0]):
                    #     self.same_overlaps = False
                    #     self._trial = 0
                    #     init_done = True

                    ## or, very restrictive: the context-aligned path must have more relevant overlaps, BUT the other path must have more irrelevant overlaps?
                    # if (relevant_overlap_ratio >= overlap_ratio_tol) & (relevant_first_overlaps[1]>relevant_first_overlaps[0]) & (irrelevant_overlap_ratio>=overlap_ratio_tol) & (irrelevant_first_overlaps[1]<irrelevant_first_overlaps[0]):
                    #     self.same_overlaps = False
                    #     self._trial = 0
                    #     init_done = True

                    ## as above, but no constraint on ordering - i.e. one of them must have more relevant overlaps, and the other must have more irrelevant overlaps
                    if (relevant_overlap_ratio >= overlap_ratio_tol) & (irrelevant_overlap_ratio>=overlap_ratio_tol) & (((relevant_first_overlaps[0]<relevant_first_overlaps[1]) & (irrelevant_first_overlaps[0]>irrelevant_first_overlaps[1])) or ((relevant_first_overlaps[0]>relevant_first_overlaps[1]) & (irrelevant_first_overlaps[0]<irrelevant_first_overlaps[1]))):
                        self.same_overlaps = False
                        self._trial = 0
                        init_done = True

                    ## or, very very restrictive: as above, but also require first path to have more overlaps in total...
                    # total_first_overlaps = self.path_future_row_and_col_overlaps[0]
                    # if (relevant_overlap_ratio >= overlap_ratio_tol) & (relevant_first_overlaps[1]>relevant_first_overlaps[0]) & (irrelevant_overlap_ratio>=overlap_ratio_tol) & (irrelevant_first_overlaps[1]<irrelevant_first_overlaps[0]) & (total_first_overlaps[0]>total_first_overlaps[1]):
                    #     self.same_overlaps = False
                    #     self._trial = 0
                    #     init_done = True


                    ## or just skip this if debugging...
                    # self.same_overlaps = True
                    # init_done = True
                    # self._trial = 0

            t+=1
            if t>500:
                raise ValueError('couldnt initialise env. SG: ', SG_found, 'paths: ', paths_found)

        self.sim = False

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
    
    @property
    def trial(self):
        return self._trial


    ## reset the environment
    def reset(self, seed=None, start_goal=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        ## set costs, given p_costs
        # self.costs = np.array([self.high_cost if r>self.p_costs.flatten()[ri] else self.low_cost for ri, r in enumerate(np.random.random(self.N**2))]).reshape(self.N, self.N)
        self.costs = self.costss[self._trial]

        ## set start and end
        if start_goal is not None: 
            self._agent_location = np.array(start_goal[0], dtype=int)
            self._goal_location = np.array(start_goal[1], dtype=int)
        else:
            # self._agent_location, self._goal_location = self.sample_SG()
            self._agent_location = np.array(self.starts[self._trial])
            self._goal_location = np.array(self.goals[self._trial])

            
            ## get true Q vals (PROBABLY only need to do this if we're interested in optimal paths, as in the free-choice expt)
            if self.expt=='free':
                
                ## pq = p(high cost)
                # dp_costs = self.p_costs*self.high_cost + (1-self.p_costs)*self.low_cost ## standard case (i.e. pq = p(high cost))
                # dp_costs[self._goal_location[0], self._goal_location[1]] = 0

                ## pq = p(low cost)
                dp_costs = self.p_costs*self.low_cost + (1-self.p_costs)*self.high_cost
                dp_costs[self._goal_location[0], self._goal_location[1]] = 0

                self.V_true, self.Q_true, self.A_true = value_iteration(dp_costs=dp_costs, goal=self._goal_location)
                # self.optimal_trajectory(self._agent_location, self._goal_location)

        ## initialise trial info
        self.terminated = False
        # observation = self.get_obs()
        # info = self._get_info()
        # current_cost = self.get_cost(self._agent_location)
        # current_cost = self.costs[self._agent_location[0], self._agent_location[1]]

        ## reset obs on each trial
        # self.obs = np.array([self._agent_location[0], self._agent_location[1], current_cost], ndmin=2)

        ## or, observations accumulate over trials, and agent observes starting position
        # if not self.sim:
        #     if not hasattr(self, 'obs') or self.obs is None:
        #         self.obs = np.array([[self._agent_location[0], self._agent_location[1], current_cost]]) 
        #     else:
        #         # print(len(self.obs))
        #         self.obs = np.vstack([self.obs, [self._agent_location[0], self._agent_location[1], current_cost]])
        #     self.obs_tmp = self.obs.copy()

        ## or, observations accumulate over trials, but agent doesn't observe starting position
        if not self.sim:
            if not hasattr(self, 'obs') or self.obs is None:
                self.obs = np.array([])


        ## dynamic programming to get the true optimal trajectory
        # if not self.sim:
        #     dp_costs = self.costs.copy()
        #     self.V_true, self.Q_true, self.A_true = value_iteration(dp_costs=dp_costs, goal=self._goal_location)

        #     ## get the costs of this optimal trajectory
        #     self.optimal_trajectory()

        ## initialise actual trajectory as list of tuples (MIGHT NEED THIS FOR THE FREE CHOICE EXPT?)
        # if not self.sim:

        #     ## start from start
        #     self.a_traj = [tuple(self._agent_location)]

        #     ## or start from nothing
        #     # self.a_traj = []
            
        self.action_scores = []
        
        # return observation, info

    ## init trial - i.e. in the AFC task, once a start has been chosen, initialise the start, goal and obs for that trial
    def init_trial(self, action):
        self._agent_location = np.array(self.starts[self._trial][action])
        # if isinstance(self._agent_location[0], tuple):
        #     print('self._agent_location is a tuple":', self._agent_location, self._trial)
        self._goal_location = np.array(self.goals[self._trial][action])
        self.terminated = False
        if self.sim:
            current_cost = self.predicted_costs[self._agent_location[0], self._agent_location[1]]
        else:
            try:
                current_cost = self.costs[self._agent_location[0], self._agent_location[1]]
            except:
                print('error: ', self.costs[self._agent_location[0]][self._agent_location[1]])
                current_cost = self.costs[self._agent_location[0], self._agent_location[1]]

        ## if using compound costs
        current_cost = self.compound_cost(current_cost, self._trial)
        self.trial_obs = np.array([[self._agent_location[0], self._agent_location[1], current_cost]])
        self.a_traj = [tuple(self._agent_location)]

    ## soft reset (allows simulation of future trials without full copying of env) - i.e. update only the start and goal locations
    def soft_reset(self, start=None, goal=None):
        if start is not None:
            self._agent_location = start
        else:
            self._agent_location = np.array(self.starts[self._trial])
        if goal is not None:
            self._goal_location = goal
        else:
            self._goal_location = np.array(self.goals[self._trial])
        self.terminated=False
    
    ## get some S-G pairs
    def sample_SG(self):

        ## sample start and goal locations
        dist = 0
        min_dist = self.N*0.75
        angle = 0
        angle_tolerance = 0.75
        angle_bounds = [45*(1+angle_tolerance), 45*(1-angle_tolerance)]
        row_or_col = 1
        t = 0
        worth_it = False
        new = False
        new_rc = False
        route_optimality_tolerance = 1
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

            ### comparison of optimal vs manhattan routes (only necessary if we're interested in the optimal path, as in the free-choice expt)
            if self.expt == 'free':

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

            ## or, if we're interested in AFC, then we don't need to do this
            elif self.expt == 'AFC':
                worth_it = True

            t+=1
            if t>200:
                raise ValueError('cant find start and end')

        ## for sanity check, just place agent and goal in opposite corners
        # self._agent_location = np.array(self.starts[self.n_trials%4])
        # self._goal_location = np.array(self.goals[self.n_trials%4])
        # self.n_trials += 1

        ## what kind of quadrilateral is this? e.g. is the long edge vertical or horizontal?
        dx = goal_location[0] - agent_location[0]
        dy = goal_location[1] - agent_location[1]
        if abs(dx) > abs(dy):
            self.quad_type = 'horizontal'
        else:
            self.quad_type = 'vertical'

        return agent_location, goal_location
    

    ## sample paths and SGs for AFC expt
    def sample_paths(self):
        diff_axes = False
        while not diff_axes:

            ## sample a pair of abstract sequences
            # seq_idxs = np.random.choice(len(abstract_sequences), size=self.n_afc, replace=False) 

            ## or, ensure that we don't sample the first or last (i.e. straight line) sequences
            seq_idxs = np.random.randint(1, len(self.abstract_sequences)-1, size=self.n_afc) 
            sampled_abstract_sequences = [self.abstract_sequences[i] for i in seq_idxs]

            if self.n_afc == 2:
                if len(self.starts)==0:
                    
                    ## choose the longest vertical and horizontal paths
                    # if self.context == 'column':
                    #     sampled_abstract_sequences = [abstract_sequences[0], abstract_sequences[-1]]
                    # elif self.context == 'row':
                    #     sampled_abstract_sequences = [abstract_sequences[-1], abstract_sequences[0]]
                    # diff_axes = True

                    ## or, one is a long path, the other is an L path, as long as they have different axes
                    # first_or_last = np.random.choice([0, -1])
                    # sampled_abstract_sequences[0] = abstract_sequences[first_or_last]
                    # L_path_idx = np.random.choice([i for i in range(2,len(abstract_sequences)-2)])
                    # sampled_abstract_sequences[1] = abstract_sequences[L_path_idx]
                    # if ((sampled_abstract_sequences[0][0]>sampled_abstract_sequences[0][1]) and (sampled_abstract_sequences[1][0]<sampled_abstract_sequences[1][1])) or ((sampled_abstract_sequences[0][0]<sampled_abstract_sequences[0][1]) and (sampled_abstract_sequences[1][0]>sampled_abstract_sequences[1][1])):
                    #     diff_axes = True

                    ## or, one of each L, but for consistency let's keep the first one dominant in  the direction of the context
                    # if self.context == 'column':
                    #     if ((sampled_abstract_sequences[0][0]<sampled_abstract_sequences[0][1]) and (sampled_abstract_sequences[1][0]>sampled_abstract_sequences[1][1])):
                    #         diff_axes = True
                    # elif self.context == 'row':
                    #     if ((sampled_abstract_sequences[0][0]>sampled_abstract_sequences[0][1]) and (sampled_abstract_sequences[1][0]<sampled_abstract_sequences[1][1])):
                    #         diff_axes = True

                    ## or, as above, but no need to keep consistency - as long as one is dominant in each direction
                    # one_vertical = False
                    # one_horizontal = False
                    # for s_a_s in sampled_abstract_sequences:
                    #     if s_a_s[0] > s_a_s[1]:
                    #         one_vertical = True
                    #     elif s_a_s[0] < s_a_s[1]:
                    #         one_horizontal = True
                    # if one_vertical and one_horizontal:
                    #     diff_axes = True
                    

                    ## or, two Ls of the same kind, where each arm is the same length 
                    sampled_abstract_sequences = [self.abstract_sequences[len(self.abstract_sequences)//2], self.abstract_sequences[len(self.abstract_sequences)//2]]
                    diff_axes = True
                else:

                    ## dominant in the same way
                    # if ((sampled_abstract_sequences[0][0]>sampled_abstract_sequences[0][1]) and (sampled_abstract_sequences[1][0]>sampled_abstract_sequences[1][1])) or ((sampled_abstract_sequences[0][0]<sampled_abstract_sequences[0][1]) and (sampled_abstract_sequences[1][0]<sampled_abstract_sequences[1][1])):
                    
                    ## one of each
                    if ((sampled_abstract_sequences[0][0]>sampled_abstract_sequences[0][1]) and (sampled_abstract_sequences[1][0]<sampled_abstract_sequences[1][1])) or ((sampled_abstract_sequences[0][0]<sampled_abstract_sequences[0][1]) and (sampled_abstract_sequences[1][0]>sampled_abstract_sequences[1][1])):
                        diff_axes = True

                    ## choose the two longest vertical paths
                    # sampled_abstract_sequences = [abstract_sequences[-1], abstract_sequences[-1]]
                    # diff_axes=True
    
            elif self.n_afc > 2:
                
                ## one long vertical, one long horizontal, one right-angled L (i.e. the middle abstract sequence)
                if len(self.starts)==0:
                    if self.context == 'column':
                        sampled_abstract_sequences = [self.abstract_sequences[0], self.abstract_sequences[-1], self.abstract_sequences[len(self.abstract_sequences)//2]]
                    elif self.context == 'row':
                        sampled_abstract_sequences = [self.abstract_sequences[-1], self.abstract_sequences[0], self.abstract_sequences[len(self.abstract_sequences)//2]]
                    diff_axes = True

                
                ## sample any sequences, as long as at least one is vertically dominant and one is horizontally dominant
                else:
                    one_vertical = False
                    one_horizontal = False
                    for s_a_s in sampled_abstract_sequences:
                        if s_a_s[0] > s_a_s[1]:
                            one_vertical = True
                        elif s_a_s[0] < s_a_s[1]:
                            one_horizontal = True
                    if one_vertical and one_horizontal:
                        diff_axes = True
                    
        
        ## set path criteria
        diff_starts = False
        n_common_within_trial = np.inf
        if len(self.path_states)>0:
            n_common_across_trials = np.inf
        else:
            n_common_across_trials = 0
        max_common_within_trial = 1
        max_common_across_trials = (self.path_len-1)/2
        vals_diff = False
        
        ## debugging
        # max_common_within_trial = np.inf
        # max_common_across_trials = np.inf


        ### get the concrete sequences
        max_attempts = 1000
        attempt=0
        while (not diff_starts) or (n_common_within_trial > max_common_within_trial) or (n_common_across_trials > max_common_across_trials) or (not vals_diff):
            attempt+=1
            if attempt>max_attempts:
                raise RuntimeError(f"Exceeded maximum attempts ({max_attempts}) while generating paths and start-goal pairs for trial {len(self.starts)}. Failed using sequences {sampled_abstract_sequences}; paths {path_states};\n criteria: diff starts: {diff_starts}, n common within trial: {n_common_within_trial}, n common across trials: {n_common_across_trials}, max common within trial: {max_common_within_trial}, max common across trials: {max_common_across_trials}")
            path_states = []
            path_actions = []
            starts = []
            goals = []
            
            ## if placing in opposite corners, determine which diagonal pairing to use (i.e. top left and bottom right, top right and bottom left, bottom left and top right, bottom right and top left)
            if len(self.starts)==0:
                corner_pairing = np.random.choice([0,1,2,3])

            for s_a_s_i, s_a_s in enumerate(sampled_abstract_sequences):
                in_grid = False
                
                while not in_grid:
                    transformation = np.random.choice(['none', 'x', 'y', 'xy']) ## i.e. reflect along an axis
                    reverse = np.random.choice([True, False]) ## i.e. start and goal are reversed

                    ## optional: force Ls to slot right into the corner?
                    slot = True

                    ## or, if first trial, put them in either top left and bottom right, or top right and bottom left corners
                    if len(self.starts)==0:
                        # transformation = 'xy'
                        # reverse = False
                        if corner_pairing == 0:
                            if s_a_s_i==0:
                                start = np.array([0, 0])
                                if slot:
                                    transformation = 'x' 
                            elif s_a_s_i==1:
                                start = np.array([self.N-1, self.N-1])
                                if slot:
                                    transformation = 'y' 
                        elif corner_pairing == 1:
                            if s_a_s_i==0:
                                start = np.array([0, self.N-1])
                                if slot:
                                    transformation = 'xy' 
                            elif s_a_s_i==1:
                                start = np.array([self.N-1, 0])
                                if slot:
                                    transformation = 'none' 
                        elif corner_pairing == 2:
                            if s_a_s_i==0:
                                start = np.array([self.N-1, 0])
                                if slot:
                                    transformation = 'none' 
                            elif s_a_s_i==1:
                                start = np.array([0, self.N-1])
                                if slot:
                                    transformation = 'xy' 
                        elif corner_pairing == 3:
                            if s_a_s_i==0:
                                start = np.array([self.N-1, self.N-1])
                                if slot:
                                    transformation = 'y' 
                            elif s_a_s_i==1:
                                start = np.array([0, 0])
                                if slot:
                                    transformation = 'x' 
                        path, actions = self.generate_concrete_sequence(s_a_s[0], s_a_s[1], start=start, transformation=transformation, reverse=reverse)

                    ## otherwise, random location
                    else:
                        start = np.random.randint(0, self.N-1, size=2)
                        path, actions = self.generate_concrete_sequence(s_a_s[0], s_a_s[1], start=start, transformation=transformation, reverse=reverse)


                    ## if these paths fall outside the grid, push them back in
                    if np.any(path[:,0] > self.N-1):
                        excess = np.max(path[:,0]) - (self.N-1)
                        path[:,0] = path[:,0] - excess
                    if np.any(path[:,0] < 0):
                        excess = np.min(path[:,0])
                        path[:,0] = path[:,0] - excess
                    if np.any(path[:,1] > self.N-1):
                        excess = np.max(path[:,1]) - (self.N-1)
                        path[:,1] = path[:,1] - excess
                    if np.any(path[:,1] < 0):
                        excess = np.min(path[:,1])
                        path[:,1] = path[:,1] - excess
                    start = path[0]

                    ## check to see if all states are in the grid
                    if np.all(path >= 0) and np.all(path < self.N):
                        in_grid = True
            
                path_states.append(path)
                path_actions.append(actions)
                starts.append(path[0])
                goals.append(path[-1])
            
            ## check that the two start and goal locations are different
            n_distinct_starts = len(set([tuple(s) for s in starts]))
            n_distinct_goals = len(set([tuple(g) for g in goals]))
            if (n_distinct_starts == self.n_afc) and (n_distinct_goals == self.n_afc):
                diff_starts = True

            ## check that the start or goal of path A is not a state that appears on the other path
            for i in range(self.n_afc):
                path_A = set(map(tuple, path_states[i]))
                for j in range(self.n_afc):
                    if i != j:
                        path_B = set(map(tuple, path_states[j]))
                        if tuple(starts[i]) in path_B or tuple(goals[i]) in path_B or tuple(starts[j]) in path_A or tuple(goals[j]) in path_A:
                            diff_starts = False
                            break                

            ## check overlap between paths across/within trials
            path_states = [tuple(map(tuple, path)) for path in path_states]
            n_common_within_trial, n_common_across_trials = self.check_overlap(path_states,0)

            
            ### optional: prevent overlaps with t1

            ## no t2-t1 overlaps
            # if len(self.starts)==1:
            #     if n_common_across_trials>0:
            #         n_common_across_trials = np.inf ## i.e. force failure unless there are 0 overlaps

            ## or, no overlaps with t1 on any trial
            if len(self.starts)>0:
                common_with_t1 = 0
                for path_t1 in self.path_states[0]:
                    for path_tn in path_states:
                        common_with_t1 += len(set(path_t1).intersection(path_tn))
                if common_with_t1>0:
                    n_common_across_trials = np.inf ## i.e. force failure unless there are 0 overlaps

            
            ### check that the costs of the paths are sufficiently different

            ## absolute difference in value
            t = len(self.starts)
            path_costs = [np.sum([self.costss[t][x, y] for x, y in path]) for path in path_states]
            cost_tol = self.N/3
            vals_diff = np.abs(max(path_costs) - min(path_costs)) >= cost_tol

        ## final randomisation of order of the two options
        order = np.random.permutation(self.n_afc)
        sampled_abstract_sequences = [sampled_abstract_sequences[i] for i in order]
        path_states = [path_states[i] for i in order]
        path_actions = [path_actions[i] for i in order]
        starts = [starts[i] for i in order]
        goals = [goals[i] for i in order]
        
        return sampled_abstract_sequences, path_actions, path_states, starts, goals



    ## count number of overlapping states
    def check_overlap(self, paths, minus=2):

        ## within trial
        n_common_within_trial = 0
        for path1, path2 in combinations(paths, 2):  # Generate all unique pairs of paths
            n_common_within_trial += len(set(path1).intersection(set(path2))) - minus

        ## across trials
        if len(self.path_states)>0:
            common_across_trials = []
            for trial_paths in self.path_states:
                for tp in trial_paths:
                    for cp in paths:
                        common_across_trials.append(len(set(tp).intersection(set(cp)))-minus)
            n_common_across_trials = np.max(common_across_trials)
        else:
            n_common_across_trials = 0
        return n_common_within_trial, n_common_across_trials
        





    ## generate all possible Manhattan paths from start to goal
    # def generate_paths(self, start, goal):

    #     ## determine direction of cardinal movement
    #     dx = goal[0] - start[0]
    #     dy = goal[1] - start[1]

    #     ## convert these dx and dy values into actions, as given by action_to_direction
    #     x_actions = np.repeat(0, abs(dx)) if dx>0 else np.repeat(2, abs(dx))
    #     y_actions = np.repeat(1, abs(dy)) if dy>0 else np.repeat(3, abs(dy))
    #     moves = np.concatenate([x_actions, y_actions])

    #     ## generate possible permutations of these moves
    #     unique_paths = set(permutations(moves))

    #     return unique_paths


    ### some more functions for path building

    ## count number of turns in a sequence of moves
    def count_turns(self, moves):
        """Count the number of turns (non-consecutive action changes) in a movement sequence."""
        if isinstance(moves[0], tuple):
            turns = 0
            for i in range(1, len(moves)):
                if moves[i] != moves[i - 1]:  # A turn occurs when the direction changes
                    turns += 1
        elif isinstance(moves[0], int) or isinstance(moves[0], np.int64):

            turns = sum(m1 != m2 and (m1 + m2) % 2 == 1 for m1, m2 in zip(moves[:-1], moves[1:]))
        else:
            raise ValueError("Invalid input type: %s" % type(moves[0]))
        return turns
    
    ## generate abstract and concrete sequences
    def generate_abstract_sequences(self, path_len, max_turns):
        """Generate unique movement sequences based on abstract structure (number of horizontal/vertical moves)."""
        
        abstract_structures = []  # Store unique (num_right, num_up) pairs

        for num_right in range(path_len + 1):
            num_up = path_len - num_right  # Remaining moves must be 'up'
            if num_right > 0 or num_up > 0:
                # Check if at least one valid sequence exists with max_turns constraint
                example_sequence = [(1, 0)] * num_right + [(0, 1)] * num_up
                for perm in permutations(example_sequence):
                    if self.count_turns(perm) <= max_turns:
                        abstract_structures.append((num_right, num_up))
                        break  # Only need one example to confirm it's valid

        return abstract_structures

    def generate_concrete_sequence(self, num_right, num_up, start = np.array([0,0]),transformation='none', reverse=False):
        """Convert an abstract sequence into a concrete state sequence."""
        if transformation == 'none':
            moves = [(1, 0)] * num_right + [(0, 1)] * num_up
        elif transformation == 'x':
            moves = [(-1, 0)] * num_right + [(0, 1)] * num_up
        elif transformation == 'y':
            moves = [(1, 0)] * num_right + [(0, -1)] * num_up
        elif transformation =='xy':
            moves = [(-1, 0)] * num_right + [(0, -1)] * num_up
        else:
            raise ValueError("Invalid transformation")
        # random.shuffle(moves)  # Randomize order while preserving counts
        
        state = start.copy()
        path = [state.copy()]
        for move in moves:
            state += np.array(move)
            path.append(state.copy())

        ## if reverse, flip the path round
        if reverse:
            path = path[::-1]
            moves = moves[::-1]
            moves = [(-m[0], -m[1]) for m in moves]


        ## convert moves from tuples to integers
        moves = [self.direction_to_action[tuple(m)] for m in moves]
        return np.array(path), moves

    
    ## custom functions for manually editing the env
    def set_state(self, state):
        self._agent_location = state
    def set_goal(self, goal):
        self._goal_location = goal
    def set_trial(self, trial):
        self._trial = trial
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

        ## ensure coordinates of state are integers
        state = np.array(state, dtype=int)

        ## pq = p(high cost)
        # return self.high_cost if np.random.random() < self.predicted_p_costs[state[0], state[1]] else self.low_cost

        ## pq = p(low cost)
        return self.high_cost if np.random.random() > self.predicted_p_costs[state[0], state[1]] else self.low_cost
    
    ## define way in which costs become compounded over trials
    def compound_cost(self, cost, trial):
        cc = cost ## do nothing
        # cc = cost * (trial + 1)  ## i.e. cost increases linearly with trial number
        # cc = cost * 2**trial ## i.e. cost increases exponentially with trial number
        return cc
    
    
    ## functions for receiving predictions from the agent
    def receive_predictions(self, predicted_p_costs):
        # self.posterior_mean = posterior_mean
        # self.posterior_cov = posterior_cov
        # self.posterior_var = posterior_var
        # self.predicted_costs = predicted_costs.reshape(self.N, self.N)
        self.predicted_p_costs = predicted_p_costs.reshape(self.N, self.N)

        ## fill in the grid with highs and lows based on predicted_p_costs
        self.predicted_costs = np.array([self.high_cost if r>self.predicted_p_costs.flatten()[ri] else self.low_cost for ri, r in enumerate(np.random.random(self.N**2))]).reshape(self.N, self.N)


    ## take a step in the environment
    def step(self, action):

        self.terminated = False

        
        ## get the score of the current action (only necessary if not simulating)
        if not self.sim:

            ## action score given by true Q-values, as defined by DP solution
            if self.expt == 'free': 
                current_Q_vals = self.Q_true[self._agent_location[0], self._agent_location[1], :]
                
                ## get the ranking of the best actions to take under the *true* optimal policy, given the agent's current position
                # action_ranking = rankdata(current_Q_vals, method='max') - 1

                # ## get the score of the action that will  actually be taken, given the ranking of the optimal actions
                # action_score = action_ranking[action] + 1
                # action_score /= self.n_actions ## may be more suitable to divide by len(actions) in case of wall states

                ## or, score the action based on the normalised Q-values of the available actions
                norm_Q_vals = (current_Q_vals - np.nanmin(current_Q_vals)) / (np.nanmax(current_Q_vals) - np.nanmin(current_Q_vals))
                action_score = norm_Q_vals[action]

            ## action score is for the whole path (need to do this later...)
            elif self.expt == 'AFC':
                action_score = 1

        
        ## take the actual action 
        direction = self.action_to_direction[action] 

        ## move to the new state
        # self._agent_location = np.clip(
        #     self._agent_location + direction, 0, self.N - 1
        # )
        self._agent_location = get_next_state(self._agent_location, direction, self.N)

        ## get the predicted and actual costs of the new state, sampling using the p(cost) values
        # current_cost = self.get_cost(self._agent_location)
        # predicted_cost = self.get_pred_cost(self._agent_location)

        ## or, use pre-sampled costs
        current_cost = self.costs[self._agent_location[0], self._agent_location[1]]
        predicted_cost = self.predicted_costs[self._agent_location[0], self._agent_location[1]]

        ## costs are compounded with each trial??
        current_cost = self.compound_cost(current_cost, self._trial)
        predicted_cost = self.compound_cost(predicted_cost, self._trial)

        ## return the real cost if not simulating
        if not self.sim:
            cost = current_cost.copy()
            
            ## update observation and trajectory arrays - i.e. agent observes along the way
            self.a_traj.append(tuple(self._agent_location))
            self.trial_obs = np.vstack([self.trial_obs, [self._agent_location[0], self._agent_location[1], current_cost]])

            ## store info on optimality of the choice, given the agent's current position
            # self.action_scores.append(action_score)


        ## return the predicted cost if simulating
        elif self.sim:
            # cost = current_cost
            cost = predicted_cost.copy()


        # An trial is done iff the agent has reached the goal
        if np.array_equal(self._agent_location, self._goal_location):
            self.terminated = True
            if self.expt=='free':
                cost=0 ## cost of final state is 0 (MIGHT WANT TO CHANGE THIS...)
        
            ## update observation array only once the trial is complete
            if not self.sim:
                if self._trial==0:
                    assert len(self.obs)==0, 'obs should be empty at the start of the trial'

                if len(self.obs)==0: ## i.e. first trial, so just copy the trial_obs
                    self.obs = self.trial_obs.copy()
                else: ## otherwise, append the trial_obs to the obs from the previous trials
                    self.obs = np.vstack([self.obs, self.trial_obs])

                ## sum of costs of route INC START AND END
                self.a_traj_costs = [self.costs[x, y] for x, y in self.a_traj]
                self.a_traj_total_cost = np.sum(self.a_traj_costs)

                ## sum of costs of route EXC START AND END
                # self.a_traj_costs = [self.costs[x, y] for x, y in self.a_traj[1:-1]]
                # self.a_traj_total_cost = np.sum(self.a_traj_costs)

                ## scores for the trial
                self.action_score = np.nanmean(self.action_scores)
                if self.expt == 'free':
                    self.cost_ratio = self.o_traj_total_costs[self._trial] / self.a_traj_total_cost
                elif self.expt == 'AFC':
                    self.cost_ratio = 1 ## sort this out later

                ## update trial counter
                self._trial += 1


        # observation = self.get_obs()
        # info = self._get_info()
        truncated=False
        agent_loc = self._agent_location
        info = {} ## make this empty since gym requires it
        return agent_loc, cost, self.terminated, truncated, info 
    
    ## take path
    def take_path(self, action_sequence):
        states = []
        costs = []
        for action in action_sequence:
            current, cost, terminated, _, _ = self.step(action)
            states.append(current)
            costs.append(cost)
        return states, costs
        


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
        o_traj_total_cost = np.sum(o_traj_costs)

        ## calculate the total cost of the trajectory EXC START AND END
        # o_traj_costs = o_traj_costs[1:-1]
        # o_traj_total_cost = np.sum(o_traj_costs)

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