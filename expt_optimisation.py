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

# from utils import make_env, Node, Tree, argm, value_iteration, data_keys
# from MCTS import MonteCarloTreeSearch, simulate_agent

from utils import make_env, Node, Tree, argm, data_keys, grid_keys, parse_lists, KL_divergence, profile_func, KL_sim, value_iteration
from MCTS import MonteCarloTreeSearch, MonteCarloTreeSearch_Free, MonteCarloTreeSearch_2AFC, simulate_agent

import IPython

import multiprocess as mp
import pingouin as pg
from scipy.special import expit

from agents import GPAgent, Farmer
from samplers import GridSampler


warnings.filterwarnings('ignore')



## callback function for saving simulation results
def save_results(sim):
    sim_out = sim[0]
    all_day_envs = sim[1]

    ## save simulation output
    for key in data_keys:
        sim_results[key].extend(sim_out[key])
    
    
    ## save the grid surfaces for each of the days
    for day in range(len(all_day_envs)):
        env = all_day_envs[day]
        all_grids['grid'].append(sim_out['grid'][0])
        all_grids['day'].append(day)
        # all_grids['env'].append(env) ## if pushed for space, comment this out
        for key in grid_keys:
            attribute = getattr(env, key)
            all_grids[key].append(attribute)

    ## update progress bar
    master_pbar.update(1)

    
## sim init
parallel=True
n_cores = 50
sim_results = {}
for key in data_keys:
    sim_results[key] = []
all_grids = {}
for key in grid_keys:
    all_grids[key] = []
all_grids['grid'] = [] 
all_grids['env'] = [] ## in case we want to save the whole thing
all_grids['day'] = [] ## in case we want to save the whole thing


### env inits
beta_params = {
    'alpha_row': 0.25,
    'beta_row': 0.25,
    'alpha_col': 0.25,
    'beta_col': 0.25
    # 'alpha_row': 1,
    # 'beta_row': 1,
    # 'alpha_col': 1,
    # 'beta_col': 1
}
N = 8
n_grids = 100
n_trials = 4
n_days = 5
expt = '2AFC'
expt_info = {
    'type': expt,
    'same_SGs': False,
    'context': 'column',
    # 'context': 'row',
}
env_params = {
    'N': N,
    'n_grids': n_grids,
    'n_trials': n_trials,
    'n_days': n_days,
    'expt_info': expt_info,
    'metric': 'cityblock',
    # 'expt': 'free',
    'beta_params': beta_params,
}
n_grids = env_params['n_grids']

## MCTS params
n_sims = 1000
MCTS_params = {
    'n_sims': n_sims,
    'n_futures': 0, 
    'exploration_constant': 1,
    'discount_factor': 1,
}

## sampler params
sampler_params = {
    'n_iter': 10,
    'lazy': False,
    'correct_prior': True,
}

## define agents to simulate
agents = [
    # 'GP',
           'BAMCP',
           'CE',
        #    'BAMCP_wrong',
        #    'BAMCP w/ CE',
        #    'CE w/ BAMCP'
          ]
progress=False

## loop through grid types
if __name__ == '__main__':
    master_pbar = tqdm(total=n_grids, desc='All_grids', position=0, leave=True, colour='green')

    ## begin parallelised simulations of grids
    if parallel:

        ## start pool
        n_cores = np.min([n_cores, n_grids])
        with mp.Pool(n_cores) as pool:
            print('Parallel simulation of ',expt,' expt, ', n_grids, ' grids, with ',n_trials,' trials, ',n_sims,' simulations per trial')
            sim_out = [pool.apply_async(simulate_agent, args=(m, env_params, MCTS_params, sampler_params, agents, progress),
                                            callback = save_results) for m in range(n_grids)]
            pool.close()
            pool.join()

    else:

        ## loop through grids
        for m in tqdm(range(n_grids)):
            sim_out = simulate_agent(m, env_params, MCTS_params, sampler_params, agents, progress)
            save_results(sim_out)

print('Parallel complete')

## remove empty keys from dict
del_keys = []
for key in sim_results.keys():
    if not bool(sim_results[key]):
        del_keys.append(key)
for dk in del_keys:
    sim_results.pop(dk)

## dataframe of simulation results
df_sim = pd.DataFrame(sim_results)


## save simulated grids + results
df_sim.to_csv('useful_saves/expt_optimisation/{}_{}x{}_env_{}_context_{}-{}-{}-{}_beta_{}_grids_{}_trials_{}_sims_{}_days_results.csv'.format(expt,N,N, expt_info['context'],
                                                                                       beta_params['alpha_row'], beta_params['beta_row'], beta_params['alpha_col'], beta_params['beta_col'],
                                                                                       n_grids, n_trials,n_sims, n_days))
with open('useful_saves/expt_optimisation/{}_{}x{}_env_{}_context_{}-{}-{}-{}_beta_{}_grids_{}_trials_{}_sims_{}_days_envs.pkl'.format(expt,N,N, expt_info['context'],
                                                                                                         beta_params['alpha_row'], beta_params['beta_row'], beta_params['alpha_col'], beta_params['beta_col'],
                                                                                                 n_grids, n_trials, n_sims, n_days), 'wb') as f:
    pickle.dump(all_grids, f)