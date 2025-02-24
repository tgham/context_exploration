## imports
import numpy as np
from numpy import nan
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy
import scipy.stats as stats
from scipy.spatial.distance import cdist
import itertools
from tqdm.auto import tqdm
import seaborn as sns
import importlib
from scipy.stats import bernoulli
import warnings
from GP import *
from plotter import *
from scipy.special import softmax
from scipy.spatial.distance import cdist
import gymnasium as gym
from gymnasium.envs.registration import register, registry, make, spec
import pickle
import copy
from itertools import product

# from utils import make_env, Node, Tree, argm, value_iteration, data_keys
# from MCTS import MonteCarloTreeSearch, simulate_agent

from utils import make_env, Node, Tree, argm, data_keys, mountain_keys, parse_lists, KL_divergence, profile_func, KL_sim
from value_iteration import value_iteration
from MCTS import MonteCarloTreeSearch, MonteCarloTreeSearch_Free, MonteCarloTreeSearch_2AFC, simulate_agent

import IPython

import multiprocess as mp
import pingouin as pg
from scipy.special import expit

from agents import GPAgent, Farmer
from samplers import GridSampler


warnings.filterwarnings('ignore')



def count_turns(moves):
    """Count the number of turns (non-consecutive action changes) in a movement sequence."""
    turns = 0
    for i in range(1, len(moves)):
        if moves[i] != moves[i - 1]:  # A turn occurs when the direction changes
            turns += 1
    return turns

def generate_state_sequences(N, max_turns):
    """Generate state sequences from (0,0) with N moves (up/right) and at most max_turns."""
    
    # Generate all possible movement sequences
    move_sequences = product([(1, 0), (0, 1)], repeat=N)
    
    # Filter sequences that respect the max_turns constraint
    valid_sequences = [moves for moves in move_sequences if count_turns(moves) <= max_turns]

    # Convert movement sequences into state sequences as numpy arrays
    state_sequences = []
    for moves in valid_sequences:
        state = np.array([0, 0])  # Start at the origin
        path = [state.copy()]
        for move in moves:
            state += np.array(move)  # Update state
            path.append(state.copy())
        state_sequences.append(np.array(path))  # Convert path to numpy array
    
    return state_sequences

## callback function for saving results
def save_KLs(sim_out):
    KLs = sim_out[0]
    t = sim_out[1]

    for seq in range(len(KLs)):
        KL_dict['t'].append(t)
        KL_dict['sequence'].append(seq)
        KL_dict['KL'].append(KLs[seq])

    ## update progress bar
    master_pbar.update(1)


## init the env
N=9
n_episodes=2
beta_params = {
    'alpha_row': 5,
    'beta_row': 1,
    'alpha_col': 1,
    'beta_col': 1
    }
expt = '2AFC'
env = make_env(N, n_episodes,expt, beta_params, 'cityblock')
env.reset()

## plot true env
# fig, axs = plt.subplots(1,1, figsize = (5,5))
# plot_r(env.p_costs, ax = axs, title = 'True reward distribution')

## sampler init
n_iter = 10
lazy=False
n_samples = 50000

## initial set of root samples ('prior' samples)
farmer = Farmer(N)
farmer.get_env_info(env)
farmer.root_samples(farmer.obs, n_samples,n_iter, lazy=lazy,CE=False)
prior_p_samples = farmer.all_posterior_ps
prior_q_samples = farmer.all_posterior_qs
prior_samples = np.vstack([prior_p_samples.T, prior_q_samples.T])
# fig, axs = plt.subplots(2,6, figsize = (24,8))
# plot_r(farmer.posterior_mean_p_cost, axs[0,0], title = 'Posterior reward distribution\nmean root sample\nno obs')
# plot_r(farmer.posterior_mean_p_cost, axs[1,0], title = 'Posterior reward distribution\nmean root sample\nno obs')


### determine the states in which observations are made

## simple case: all in one row, or all in one column
n_obs = N
obs_coords_row = np.zeros((n_obs,3))
obs_coords_row[:,1] = np.arange(n_obs, dtype=int)
obs_coords_col = np.zeros((n_obs,3))
obs_coords_col[:,0] = np.arange(n_obs, dtype=int)
obs_set = [obs_coords_row, obs_coords_col]

## or, generate all sets of states for a given sequence length
max_turns = 1
obs_set = generate_state_sequences(n_obs-1,1)
for i, o in enumerate(obs_set):
    obs_set[i] = np.hstack((o, np.zeros((n_obs,1))))
obs_set = np.array(obs_set)

## set number of sims
n_tests = 120
KL_dict = {
    't': [],
    'sequence': [],
    'KL': [],
}
plotting=False

## repeat the obs_set for each test
obs_set = np.expand_dims(obs_set, 0)
obs_set = np.repeat(obs_set, n_tests, 0)
n_seqs = len(obs_set[0])


## parallel fitting
parallel = True
n_cores = 12
if __name__ == '__main__':
    master_pbar = tqdm(total=n_tests, desc='all_tests', position=0, leave=True, colour='green')
    if parallel:

        ## start pool
        n_cores = np.min([n_cores, n_tests])
        with mp.Pool(n_cores) as pool:
            print('Parallel uncertainty tests:')
            print('n samples:', n_samples)
            print('N:',N, ', n_seqs:', n_seqs)
            KLs = [pool.apply_async(KL_sim, args = (obs_set[t], t, farmer,n_samples, plotting),
                                    callback = save_KLs) for t in range(n_tests)
                   ]
            pool.close()
            pool.join()

    else:
        for t in range(n_tests):
            sim_out = KL_sim(obs_set[t], t, farmer, n_samples, plotting)
            save_KLs(sim_out)
print('parallel complete')

## dataframe of simulation results
df_KL = pd.DataFrame(KL_dict)

## save
df_KL.to_csv('KL_divergence_{}x{}_env_{}-{}-{}-{}_beta_{}_samples_{}_tests.csv'.format(N,N, 
                                                                    beta_params['alpha_row'], beta_params['beta_row'], beta_params['alpha_col'], beta_params['beta_col'],
                                                                    n_samples, n_tests))