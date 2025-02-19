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

from utils import make_env, Node, Tree, argm, data_keys, mountain_keys, parse_lists, KL_divergence, profile_func
from value_iteration import value_iteration
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
    env = sim[1]

    ## save simulation output
    for key in data_keys:
        sim_results[key].extend(sim_out[key])
    
    
    ## save the mountain surface
    all_mountains['mountain'].append(sim_out['mountain'][0])
    # all_mountains['env'].append(env) ## if pushed for space, comment this out
    for key in mountain_keys:
        attribute = getattr(env, key)
        all_mountains[key].append(attribute)

    ## update progress bar
    master_pbar.update(1)

    
## sim init
parallel=True
n_cores = 12
sim_results = {}
for key in data_keys:
    sim_results[key] = []
all_mountains = {}
for key in mountain_keys:
    all_mountains[key] = []
all_mountains['mountain'] = [] 
all_mountains['env'] = [] ## in case we want to save the whole thing

### env inits
beta_params = {
    'alpha_row': 5,
    'beta_row': 0.1,
    'alpha_col': 0.5,
    'beta_col': 0.5
}
N = 7
n_mountains = 12
n_episodes = 3
n_runs = 1
expt = '2AFC'
env_params = {
    'N': N,
    'n_mountains': n_mountains,
    'n_episodes': n_episodes,
    'n_runs': n_runs,
    'expt': expt,
    'metric': 'cityblock',
    # 'expt': 'free',
    'beta_params': beta_params,
}
n_mountains = env_params['n_mountains']

## MCTS params
n_sims = 50000
MCTS_params = {
    'n_sims': n_sims,
    'n_futures': 0,
    'exploration_constant': 25,
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
           'BAMCP_wrong',
        #    'CE',
        #    'BAMCP w/ CE',
        #    'CE w/ BAMCP'
          ]
progress=False

## loop through mountain types
if __name__ == '__main__':
    master_pbar = tqdm(total=n_mountains, desc='All_mountains', position=0, leave=True, colour='green')

    ## begin parallelised simulations of mountains
    if parallel:

        ## start pool
        n_cores = np.min([n_cores, n_mountains])
        with mp.Pool(n_cores) as pool:
            print('Parallel simulation of ',expt,' expt, ', n_mountains, ' mountains, with ',n_episodes,' episodes, ',n_sims,' simulations per episode')
            sim_out = [pool.apply_async(simulate_agent, args=(m, env_params, MCTS_params, sampler_params, agents, progress),
                                            callback = save_results) for m in range(n_mountains)]
            pool.close()
            pool.join()

    else:

        ## loop through mountains
        for m in tqdm(range(n_mountains)):
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


## save simulated mountains + results
df_sim.to_csv('{}_{}x{}_env_{}-{}-{}-{}_beta_{}_mountains_{}_episodes_{}_sims_results.csv'.format(expt,N,N, 
                                                                                       beta_params['alpha_row'], beta_params['beta_row'], beta_params['alpha_col'], beta_params['beta_col'],
                                                                                       n_mountains, n_episodes, n_sims))
with open('{}_{}x{}_env_{}_{}-{}-{}_beta_{}_mountains_{}_episodes_{}_sims_envs.pkl'.format(expt,N,N, 
                                                                                                         beta_params['alpha_row'], beta_params['beta_row'], beta_params['alpha_col'], beta_params['beta_col'],
                                                                                                 n_mountains, n_episodes, n_sims), 'wb') as f:
    pickle.dump(all_mountains, f)