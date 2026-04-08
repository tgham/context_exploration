from enum import Enum
import gymnasium as gym
from gymnasium import spaces
gym.logger.set_level(40)  # Sets logger to 'Error' only, silencing warnings
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
import copy
from collections import defaultdict
from IPython.display import display, clear_output
from utils import *
from scipy.stats import rankdata, truncnorm
from scipy.linalg import cholesky
from base_kernels import *
from itertools import permutations, combinations
from samplers import GridSampler
from numba import njit



class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3

class GridEnv(gym.Env):

    def __init__(self, N, n_trials, expt_info, beta_params=None, seed=None):
        
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
        if expt_info['objective'] is not None:
            self.objective = expt_info['objective']
        else:
            self.objective = 'costs'

        ### misc gym inits
        
        # Observation space is a 2D location on the grid
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0]),  # Minimum x and y
            high=np.array([self.N - 1, self.N - 1]),  # Maximum x and y
            dtype=np.int32
        )
        self.info = {} ## dummy info for compatibility with gym API

        ## define actions

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self.action_space = spaces.Discrete(4)
        self.action_to_direction = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        self.direction_to_action = {tuple(v): k for k, v in enumerate(self.action_to_direction)}
        self.n_actions = 4

        ## action labels (NB these deviate from env action space, bc axes are flipped for plotting
        self.action_labels = ['down', 'right', 'up', 'left']

        ## define abstract sequences
        if self.N%2==1:
            self.path_len = self.N-1
        else:
            self.path_len = self.N-2
        max_turns =1
        self.abstract_sequences = self.generate_abstract_sequences(self.path_len, max_turns)

        ## initialise grid
        init_done = False
        t=0
        while not init_done:
            paths_found = False
            if self.objective == 'rewards':
                self.high_cost, self.low_cost = 0, 1
            elif self.objective == 'costs':
                self.high_cost, self.low_cost = -1, -0
            elif self.objective == 'both':
                self.high_cost, self.low_cost = -1, 1
            self.alpha_row = beta_params['alpha_row']
            self.beta_row = beta_params['beta_row']
            self.alpha_col = beta_params['alpha_col']
            self.beta_col = beta_params['beta_col']
            
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

            ## actual binary costs for each trial, assuming they are the same across trials
            self.costs = np.array([self.high_cost if r>self.p_costs.flatten()[ri] else self.low_cost for ri, r in enumerate(np.random.random(self.N**2))]).reshape(self.N, self.N)

            ## check if decent spread of high and low costs - i.e. mean cost should be around the mean of the two costs, +/- some tolerance
            mean_cost = np.mean(self.costs)
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
            self.n_trials = n_trials
            self.path_aligned_states = []  ## states on each path that are aligned with context
            self.path_orthogonal_states = []  ## states on each path that are orthogonal to context


            ## generate relevant trial info for each trial
            for t in range(n_trials):

                try:
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


                    ## save expected costs of the paths
                    path_actual_costs = []
                    path_expected_costs = []
                    for path in self.path_states[t]:
                        
                        ## pq = p(high cost)
                        # path_cost = np.sum([self.p_costs[x, y]*self.high_cost + (1-self.p_costs[x, y])*self.low_cost for x, y in path]) 

                        ## pq = p(low cost)
                        path_expected_cost = np.sum([self.p_costs[x, y]*self.low_cost + (1-self.p_costs[x, y])*self.high_cost for x, y in path])
                        path_actual_cost = np.sum([self.costs[x, y] for x, y in path])
                        path_expected_costs.append(path_expected_cost)
                        path_actual_costs.append(path_actual_cost)
                    self.path_expected_costs.append(path_expected_costs)
                    self.path_actual_costs.append(path_actual_costs)
                    
                    paths_found = True

                except:
                    break


            ## if all the SGs and paths have been found, then we then need to check overlap across trials
            if len(self.starts)==self.n_trials:
        
                ## hacky fix: make sure all coordinates in the path_states list are tuples of int, rather than tuples of int64
                for t in range(self.n_trials):
                    for p, path in enumerate(self.path_states[t]):
                        self.path_states[t][p] = [tuple([int(x) for x in state]) for state in path]
                
                ## get info on path overlaps 
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
                                if (path[0][0], path[0][1]) == (next_path[0][0], next_path[0][1]):
                                    intersection = intersection - set([path[0]])
                                if (path[-1][0], path[-1][1]) == (next_path[-1][0], next_path[-1][1]):
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
                    self.path_aligned_states, self.path_orthogonal_states, self.path_weights = self.get_alignment(self.path_states)
                    self.get_future_states()
                    self.enumerate_valid_paths()

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
                # self.path_aligned_states, self.path_orthogonal_states, self.path_weights  = self.get_alignment(self.path_states)
                # self.get_future_states()
                # self.enumerate_valid_paths()

            t+=1
            if t>500:
                raise ValueError('couldnt initialise env. paths: ', paths_found)

        self.sim = False

    ## get info from current state
    @property
    def trial(self):
        return self._trial
    @property
    def current(self):
        return self.starts[self._trial]
    
    ## set the horizon trial in the env
    def set_trunc_trial(self, trunc_trial):
        self.trunc_trial = trunc_trial

    def sim_clone(self, costs):
        """
        Returns a lightweight shallow copy of the environment, injecting
        the newly sampled MDP components (e.g. costs).
        """
        sim_env = copy.copy(self)

        # Inject the new underlying sampled MDP dynamics
        sim_env.costs = costs
        sim_env.sim = True

        # # Explicitly shallow-copy attributes that might be mutated in-place
        # if hasattr(self, 'trial_obs') and self.trial_obs is not None:
        #     if isinstance(self.trial_obs, np.ndarray):
        #         sim_env.trial_obs = self.trial_obs.copy()
        #     else:
        #         sim_env.trial_obs = copy.copy(self.trial_obs)

        # if hasattr(self, 'obs') and self.obs is not None:
        #     if isinstance(self.obs, np.ndarray):
        #         sim_env.obs = self.obs.copy()
        #     else:
        #         sim_env.obs = copy.copy(self.obs)

        return sim_env

    ## reset the environment
    def reset(self, seed=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        ## initialise trial info
        self.terminated = False
        self.set_trunc_trial(self.n_trials-1)

        ## initialise obs if first trial
        if self._trial == 0:
            self.obs = np.empty((0, 3), dtype=int)
        

        ## hack... remove this once we've created new envs
        if not hasattr(self, 'costs'):
            self.costs = self.costss[self._trial]
        if not hasattr(self, 'info'):
            self.info = {}
        if not hasattr(self, 'path_weights'):
            self.path_aligned_states, self.path_orthogonal_states, self.path_weights = self.get_alignment(self.path_states)


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

                    ## choose the two of the longest vertical paths
                    # sampled_abstract_sequences = [self.abstract_sequences[-2], self.abstract_sequences[-2]]
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
            path_costs = [np.sum([self.costs[x, y] for x, y in path]) for path in path_states]
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
    

    def sample_paths_given_future_states(self, t):
        """
        Sample paths that only contain states from future_states for trial t.
        Uses precomputed valid paths from enumerate_valid_paths().
        
        Args:
            t: Trial index to sample paths for
        """
        
        if not hasattr(self, 'valid_future_path_info') or self.valid_future_path_info is None:
            raise RuntimeError("valid_future_path_info is not precomputed. Call enumerate_valid_paths() first.")
        
        if len(self.valid_future_path_info[t]) < self.n_afc:
            raise RuntimeError(f"Not enough valid paths found in future_states for trial {t}. Found {len(self.valid_future_path_info[t])}, need {self.n_afc}.")
        
        # Separate paths by dominant axis
        vertical_dominant = []  # more vertical moves than horizontal
        horizontal_dominant = []  # more horizontal moves than vertical
        
        for path_info in self.valid_future_path_info[t]:
            path, actions, abstract_seq = path_info
            num_vertical, num_horizontal = abstract_seq
            if num_vertical > num_horizontal:
                vertical_dominant.append(path_info)
            elif num_horizontal > num_vertical:
                horizontal_dominant.append(path_info)
            # Skip L-shaped paths with equal arms for the diff_axes requirement
        
        if self.n_afc == 2 and (len(vertical_dominant) == 0 or len(horizontal_dominant) == 0):
            raise RuntimeError("Cannot find paths with different dominant axes in future_states.")
        
        # Try to find a valid pair of paths
        max_attempts = 1000
        max_common_within_trial = 1
        
        for attempt in range(max_attempts):
            if self.n_afc == 2:
                # Pick one from each category
                v_idx = np.random.randint(len(vertical_dominant))
                h_idx = np.random.randint(len(horizontal_dominant))
                selected = [vertical_dominant[v_idx], horizontal_dominant[h_idx]]
            else:
                # For n_afc > 2, randomly sample from all valid paths
                indices = np.random.choice(len(self.valid_future_path_info[t]), size=self.n_afc, replace=False)
                selected = [self.valid_future_path_info[t][i] for i in indices]
            
            path_states = [sel[0] for sel in selected]
            path_actions = [sel[1] for sel in selected]
            sampled_abstract_sequences = [sel[2] for sel in selected]
            starts = [path[0] for path in path_states]
            goals = [path[-1] for path in path_states]
            
            # Check criteria
            # 1. Different start and goal locations
            n_distinct_starts = len(set([tuple(s) for s in starts]))
            n_distinct_goals = len(set([tuple(g) for g in goals]))
            if n_distinct_starts != self.n_afc or n_distinct_goals != self.n_afc:
                continue
            
            # 2. Start/goal of one path not on another path
            diff_starts = True
            for i in range(self.n_afc):
                path_A = set(map(tuple, path_states[i]))
                for j in range(self.n_afc):
                    if i != j:
                        path_B = set(map(tuple, path_states[j]))
                        if tuple(starts[i]) in path_B or tuple(goals[i]) in path_B or tuple(starts[j]) in path_A or tuple(goals[j]) in path_A:
                            diff_starts = False
                            break
                if not diff_starts:
                    break
            if not diff_starts:
                continue
            
            # 3. Check overlap within trial
            path_states_tuples = [tuple(map(tuple, path)) for path in path_states]
            n_common_within_trial, _ = self.check_overlap(path_states_tuples, 0)
            if n_common_within_trial > max_common_within_trial:
                continue
            
            # All criteria met!
            break
        else:
            raise RuntimeError(f"Exceeded maximum attempts ({max_attempts}) while selecting valid path pairs from enumerated paths.")
        
        # Final randomisation of order
        # order = np.random.permutation(self.n_afc)
        # sampled_abstract_sequences = [sampled_abstract_sequences[i] for i in order]
        # path_states_tuples = [path_states_tuples[i] for i in order]
        # path_actions = [path_actions[i] for i in order]
        # starts = [starts[i] for i in order]
        # goals = [goals[i] for i in order]

        ## for consistency, let's keep the first one vertically dominant and the second one horizontally dominant (if n_afc=2)
        if self.n_afc == 2:
            if sampled_abstract_sequences[0][0] < sampled_abstract_sequences[0][1]:
                sampled_abstract_sequences[0], sampled_abstract_sequences[1] = sampled_abstract_sequences[1], sampled_abstract_sequences[0]
                path_states_tuples[0], path_states_tuples[1] = path_states_tuples[1], path_states_tuples[0]
                path_actions[0], path_actions[1] = path_actions[1], path_actions[0]
                starts[0], starts[1] = starts[1], starts[0]
                goals[0], goals[1] = goals[1], goals[0]
        
        return sampled_abstract_sequences, path_actions, path_states_tuples, starts, goals

    def enumerate_valid_paths(self):
        """
        For each trial, enumerate all valid L-shaped paths (at most 1 turn) of the required length 
        that lie entirely within future_states for that trial.
        
        Requires self.future_states to be precomputed (via get_future_states).
        
        Stores:
            self.valid_future_path_info: List of n_trials lists, where each inner list contains
                (path, actions, abstract_seq) tuples:
                    - path: numpy array of shape (path_len+1, 2) with coordinates
                    - actions: list of action indices
                    - abstract_seq: (num_vertical, num_horizontal) tuple
        """
        if not hasattr(self, 'future_states') or self.future_states is None:
            raise RuntimeError("future_states not computed. Call get_future_states() first.")
        
        self.valid_future_path_info = []
        
        for t in range(self.n_trials):
            future_states = self.future_states[t]
            valid_paths_tmp = []
            
            # Get all valid starting points for this trial
            valid_coords = np.argwhere(future_states == 1)
            
            # For each abstract sequence (num_right moves, num_up moves) with at most 1 turn
            for abstract_seq in self.abstract_sequences:
                num_right, num_up = abstract_seq
                
                # Try all 4 transformations and 2 reverse options
                for transformation in ['none', 'x', 'y', 'xy']:
                    for reverse in [False, True]:
                        
                        # For each potential starting point
                        for start_coord in valid_coords:
                            start = start_coord.copy()
                            path, actions = self.generate_concrete_sequence(
                                num_right, num_up, 
                                start=start, 
                                transformation=transformation, 
                                reverse=reverse
                            )
                            
                            # Check if entire path is within bounds and in future_states
                            if np.any(path < 0) or np.any(path >= self.N):
                                continue
                            
                            # Check all states are in future_states
                            all_valid = True
                            for state in path:
                                if future_states[state[0], state[1]] != 1:
                                    all_valid = False
                                    break
                            
                            if all_valid:
                                # Determine the actual abstract sequence after transformation/reverse
                                # by counting vertical vs horizontal moves
                                if len(path) > 1:
                                    diffs = np.diff(path, axis=0)
                                    num_vertical = np.sum(np.abs(diffs[:, 0]))  # changes in row
                                    num_horizontal = np.sum(np.abs(diffs[:, 1]))  # changes in column
                                    actual_abstract = (num_vertical, num_horizontal)
                                else:
                                    actual_abstract = (0, 0)
                                
                                valid_paths_tmp.append((path, actions, actual_abstract))
            
            # Remove duplicates (same path coordinates) for this trial
            trial_valid_paths = []
            seen = set()
            for path, actions, abstract_seq in valid_paths_tmp:
                path_tuple = tuple(map(tuple, path))
                if path_tuple not in seen:
                    seen.add(path_tuple)
                    trial_valid_paths.append((path, actions, abstract_seq))
            
            self.valid_future_path_info.append(trial_valid_paths)


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
    def set_sim(self, sim):
        self.sim = sim
    def set_sim_weights(self, aligned_weight, orthogonal_weight):
        self.sim_weight_map = np.array([aligned_weight, orthogonal_weight])

    def receive_task_params(self, task_params):
        arm_weight = task_params.get('arm_weight', None)
        if arm_weight is not None:
            aligned = 1 - max(0.0, -arm_weight)
            orthogonal = 1 - max(0.0, arm_weight)
            self.set_sim_weights(aligned, orthogonal)

    ## receiving predictions from the agent
    def receive_predictions(self, costs):
        self.costs = costs


    ## construct a task-specific sampler for this environment
    def make_sampler(self):
        """Create a GridSampler from this environment's parameters and the given observations."""
        return GridSampler(self)

    
    ## take path using pre-computed path_states (avoids looping through individual step calls)
    def step(self, action):
        """
        
        Args:
            action: index of the path to take from path_states[self._trial]
        
        Returns standard env.step outputs:
            states: list of states visited
            costs: list of costs at each state
        """
        path = self.path_states[self._trial][action]

        ## apply reward-func weighting in sim mode
        if self.sim:
            path_weight_idx = self.path_weights[self._trial][action]
            costs = [float(self.costs[x, y]) * self.sim_weight_map[path_weight_idx[k]] for k, (x, y) in enumerate(path)]
            self.trial_obs = [(x, y, costs[k]) for k, (x, y) in enumerate(path)]
        else:
            path_arr = np.array(path, dtype=np.int64)
            costs = self.costs[path_arr[:, 0], path_arr[:, 1]].astype(np.float64)
            self.trial_obs = np.column_stack([path_arr, costs])
            self.obs = np.vstack([self.obs, self.trial_obs])
            
            
        # Mark as terminated if done final trial
        self.terminated = self._trial >= self.n_trials - 1

        ## mark as truncated if we have reached horizon
        truncated = self._trial >= self.trunc_trial

        # Update trial counter
        self._trial += 1

        cost = sum(costs)
        return self.trial_obs, cost, self.terminated, truncated, self.info
        
    

    ## get information on which states are orthogonal vs aligned to the context
    def get_alignment(self, path_states):
        """
        Classify states in paths as aligned or orthogonal to the context for all trials, 
        and store them directly as attributes.
        
        For column context: aligned states are those where movement is vertical (row changes, column stays same)
        For row context: aligned states are those where movement is horizontal (column changes, row stays same)
            
        Stores:
            Populates self.path_aligned_states and self.path_orthogonal_states for all trials
        """

        ## init list
        # if not hasattr(self, 'path_aligned_states'):
        #     self.path_aligned_states = []
        # if not hasattr(self, 'path_orthogonal_states'):
        #     self.path_orthogonal_states = []
        path_aligned_states = []
        path_orthogonal_states = []
        path_weights = []

        # Process all trials
        n_trials_tmp = len(path_states)
        for trial_idx in range(n_trials_tmp):
            trial_aligned_states = []
            trial_orthogonal_states = []
            trial_weights = []
            n_afc_tmp = len(path_states[trial_idx])
            for choice_idx in range(n_afc_tmp):
                ps = path_states[trial_idx][choice_idx]
                aligned_states = set()
                orthogonal_states = set()

                # Iterate through consecutive state pairs to determine movement direction
                for idx in range(len(ps) - 1):
                    current_state = ps[idx]
                    next_state = ps[idx + 1]

                    if self.context == 'column':
                        # Aligned: vertical movement (column stays constant)
                        if current_state[1] == next_state[1] and current_state[0] != next_state[0]:
                            aligned_states.add(tuple(current_state))
                            aligned_states.add(tuple(next_state))
                        # Orthogonal: horizontal movement (row stays constant)
                        elif current_state[0] == next_state[0] and current_state[1] != next_state[1]:
                            orthogonal_states.add(tuple(current_state))
                            orthogonal_states.add(tuple(next_state))

                    elif self.context == 'row':
                        # Aligned: horizontal movement (row stays constant)
                        if current_state[0] == next_state[0] and current_state[1] != next_state[1]:
                            aligned_states.add(tuple(current_state))
                            aligned_states.add(tuple(next_state))
                        # Orthogonal: vertical movement (column stays constant)
                        elif current_state[1] == next_state[1] and current_state[0] != next_state[0]:
                            orthogonal_states.add(tuple(current_state))
                            orthogonal_states.add(tuple(next_state))

                # corner states should only be counted as aligned
                corner_states = aligned_states.intersection(orthogonal_states)
                orthogonal_states -= corner_states

                # Pre-convert sets to numpy arrays for faster arm_reweighting
                aligned_arr = np.array(list(aligned_states)) if aligned_states else np.empty((0, 2), dtype=int)
                orthogonal_arr = np.array(list(orthogonal_states)) if orthogonal_states else np.empty((0, 2), dtype=int)

                # Build weight category mask: 0=aligned, 1=orthogonal
                n_states = len(ps)
                categories = np.zeros(n_states, dtype=np.int8)
                for idx in range(n_states):
                    state = tuple(ps[idx])
                    if state in aligned_states:
                        categories[idx] = 0
                    elif state in orthogonal_states:
                        categories[idx] = 1
                    else:
                        raise ValueError(f"State {state} in path not classified as aligned or orthogonal. This should not happen.")

                trial_aligned_states.append(aligned_arr)
                trial_orthogonal_states.append(orthogonal_arr)
                trial_weights.append(categories)

            # Store results as attributes
            path_aligned_states.append(trial_aligned_states)
            path_orthogonal_states.append(trial_orthogonal_states)
            path_weights.append(trial_weights)

        ## sometimes, we're only returning aligned and orthogonal states for a single trial, and a single choice, so we can just return the arrays for that trial and choice rather than the whole list of lists
        if n_trials_tmp == 1 and n_afc_tmp == 1:
            return path_aligned_states[0][0], path_orthogonal_states[0][0], path_weights[0][0]

        return path_aligned_states, path_orthogonal_states, path_weights


    ## get grid of upcoming states
    def get_future_states(self):
        """
        For each trial, compute which intersections may potentially be visited on any of the upcoming trials.
        
        Stores:
            self.future_intersections: List of n_trials N*N binary arrays. 
                For trial t, future_states[t][x, y] = 1 if state (x, y) 
                appears on any path in any trial from t+1 onwards, 0 otherwise.
                The final trial has all zeros (no upcoming trials).
        """
        self.future_states = []
        
        for t in range(self.n_trials):
            # Create an N*N binary array for this trial
            future_grid = np.zeros((self.N, self.N), dtype=int)
            
            # Look at all future trials (t+1 onwards)
            for future_t in range(t + 1, self.n_trials):
                # Look at all paths in the future trial
                for path in self.path_states[future_t]:
                    # Mark each state on this path
                    for state in path:
                        future_grid[state[0], state[1]] = 1
            
            self.future_states.append(future_grid)


    def get_future_axis_overlaps(self, trial, path):
        """
        For a given trial and path, count the number of future states that fall on the path's 
        columns or rows (context-dependent relevant vs irrelevant).
        
        - In column context: relevant = count of future states on path's columns,
                             irrelevant = count of future states on path's rows
        - In row context: relevant = count of future states on path's rows,
                          irrelevant = count of future states on path's columns
        
        Args:
            trial: The trial index
            path: List of (row, col) tuples representing the path states
            
        Returns:
            n_relevant_future_overlaps: Number of future states on context-relevant axis
            n_irrelevant_future_overlaps: Number of future states on context-irrelevant axis
        """
        if not hasattr(self, 'future_states') or self.future_states is None:
            raise RuntimeError("future_states not computed. Call get_future_states() first.")
        
        # Get future states grid for this trial
        future_grid = self.future_states[trial]
        
        # Get coordinates of future states
        future_coords = np.argwhere(future_grid == 1)
        future_states_set = {(coord[0], coord[1]) for coord in future_coords}
        
        # Get the set of columns and rows in the path
        path_cols = {state[1] for state in path}
        path_rows = {state[0] for state in path}
        
        # Count future states that fall on the path's columns
        n_future_on_path_cols = sum(1 for state in future_states_set if state[1] in path_cols)
        
        # Count future states that fall on the path's rows
        n_future_on_path_rows = sum(1 for state in future_states_set if state[0] in path_rows)
        
        # Return relevant and irrelevant based on context
        if self.context == 'column':
            n_relevant_future_overlaps = n_future_on_path_cols
            n_irrelevant_future_overlaps = n_future_on_path_rows
        elif self.context == 'row':
            n_relevant_future_overlaps = n_future_on_path_rows
            n_irrelevant_future_overlaps = n_future_on_path_cols
        else:
            raise ValueError(f"Unknown context: {self.context}")
        
        return n_relevant_future_overlaps, n_irrelevant_future_overlaps
        
