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

from utils import make_env, Node, Tree, argm, data_keys, grid_keys, parse_lists, KL_divergence, profile_func, KL_sim, value_iteration
from MCTS import MonteCarloTreeSearch, MonteCarloTreeSearch_Free, MonteCarloTreeSearch_AFC, simulate_agent

import IPython

import multiprocess as mp
import pingouin as pg
from scipy.special import expit

from agents import Farmer
from samplers import GridSampler


warnings.filterwarnings('ignore')



## need this for paralellising because of the way the loops are structured...
def agent_loop(p, agent_params, hyperparams, agents):
    N = hyperparams['N']
    sim_outs = []

    ## load env objects
    with open('useful_saves/expt_optimisation/simulated_envs/ppt_'+str(p)+'_envs.pkl', 'rb') as f:
        env_objects = pickle.load(f)

    ## loop through agents
    for agent in agents:
        farmer = Farmer(N)
        sim_out = farmer.run(agent_params, hyperparams, agent=agent, df_trials=None, envs=env_objects, fit=False, progress=False)
        sim_outs.append(sim_out)
    
    ## join sim_outs together
    if len (sim_outs) == 0:
        return sim_out
    else:
        sim_out = {}
        for key in sim_outs[0].keys():
            sim_out[key] = []
            for sim in sim_outs:
                if key in sim:
                    sim_out[key].extend(sim[key])
    
    return sim_out

## callback for parallelised fitting
def save_sim(sim_out):
    for key in sim_out:
        all_sim_out[key].extend(sim_out[key])
    master_pbar.update(1)



## env inits
N = 11
metric = 'cityblock'
beta_params = {
    'alpha_row':0.5,
    'beta_row': 0.5,
    'alpha_col':0.5,
    'beta_col': 0.5
    # 'alpha_row': 0.5,
    # 'beta_row': 0.5,
    # 'alpha_col': 10,
    # 'beta_col': 0.1
    }

## trial info
n_sim_participants = 49
n_cities = 8
n_days = 5
n_trials = 4
expt = 'AFC'
n_afc = 3
expt_info = {
    'type': expt,
    'n_afc': n_afc,
}

## init df for saving expt info
df_expt = pd.DataFrame(columns=['participant', 'city', 'context', 'grid','trial', 
                                'better_path',
                                'start_A','start_B','goal_A','goal_B', 'path_A', 'path_B',
                                'path_A_expected_cost', 'path_B_expected_cost',
                                'path_A_actual_cost', 'path_B_actual_cost',
                                'path_A_future_overlap', 'path_B_future_overlap',
                                'abstract_sequence_A', 'abstract_sequence_B',
                                'dominant_axis_A','dominant_axis_B'
                                ])
if n_afc==3:
    df_expt = df_expt.join(pd.DataFrame(columns=['start_C', 'goal_C', 'path_C',
                        'path_C_expected_cost', 'path_C_actual_cost', 
                        'path_C_future_overlap', 'abstract_sequence_C', 
                        'dominant_axis_C']))
    
## init sim results
all_sim_out = {
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
        'Q_a':[],
        'Q_b':[],
        'Q_c':[],
        'leaf_visits_a':[],
        'leaf_visits_b':[],
        'leaf_visits_c':[]
    }
parallel = True
n_cores = 50

## init agent and expt
agent_params = [
    0.1, # temp
    0.1, # lapse
]
hyperparams = {
    'n_sims': 100000,
    'exploration_constant': 1,
    'discount_factor': 1,
    'n_iter': 10,
    'n_trials': n_trials,
    'n_afc': n_afc,
    'n_days': n_days,
    'n_cities': n_cities,
    'N': N,
    'participant': None, ## hacky
}
agents = ['BAMCP', 'CE']   

## generate dataset for each participant
ppt_envs = {}
for p in tqdm(range(1,n_sim_participants+1)):

    ## set contexts - i.e. half of the cities are row contexts, half are column contexts
    contexts = ['row']*int(n_cities/2) + ['column']*int(n_cities/2)
    np.random.shuffle(contexts)

    env_objects = {}


    ## loop through cities, where each city is a new context
    for c in range(n_cities):
        expt_info['context'] = contexts[c]

        ## create some envs
        envs = [make_env(N, n_trials, expt_info, beta_params, metric) for i in range(n_days)]

        ## save expt info
        for i, env in enumerate(envs):
            for e in range(n_trials):

                ## ensure that all tuples in the env.path_states list contain int, rather than int64
                if n_afc==2:
                    df_expt = pd.concat([df_expt, pd.DataFrame({'participant': p, 'city': int(c+1), 'context': expt_info['context'], 'grid': int(i+1), 'trial': int(e+1), 'start_A': [env.path_states[e][0][0]], 'start_B': [env.path_states[e][1][0]], 'goal_A': [env.path_states[e][0][-1]], 'goal_B': [env.path_states[e][1][-1]], 'path_A': [env.path_states[e][0]], 'path_B': [env.path_states[e][1]],
                                    'path_A_actual_cost': [env.path_actual_costs[e][0]], 'path_B_actual_cost': [env.path_actual_costs[e][1]],
                                    'path_A_expected_cost': env.path_expected_costs[e][0], 'path_B_expected_cost': env.path_expected_costs[e][1],
                                    'path_A_future_overlap': env.path_future_overlaps[e][0], 'path_B_future_overlap': env.path_future_overlaps[e][1],
                                    'abstract_sequence_A': [env.sampled_abstract_sequences[e][0]], 'abstract_sequence_B': [env.sampled_abstract_sequences[e][1]],
                                    'better_path': ['a','b'][np.argmax(env.path_actual_costs[e])],
                                    'dominant_axis_A': env.dominant_axis_A[e],
                                    'dominant_axis_B': env.dominant_axis_B[e]
                                    })], ignore_index=True)
                elif n_afc==3:
                    df_expt = pd.concat([df_expt, pd.DataFrame({'participant': p, 'city': int(c+1), 'context': expt_info['context'], 'grid': int(i+1), 'trial': int(e+1), 
                                    'start_A': [env.path_states[e][0][0]], 'goal_A': [env.path_states[e][0][-1]], 'path_A': [env.path_states[e][0]],
                                    'path_A_expected_cost': env.path_expected_costs[e][0], 'path_A_actual_cost': env.path_actual_costs[e][0],
                                    'path_A_future_overlap': env.path_future_overlaps[e][0], 'abstract_sequence_A': [env.sampled_abstract_sequences[e][0]],
                                    'dominant_axis_A': env.dominant_axis_A[e],
                                    'start_B': [env.path_states[e][1][0]], 'goal_B': [env.path_states[e][1][-1]], 'path_B': [env.path_states[e][1]],
                                    'path_B_expected_cost': env.path_expected_costs[e][1], 'path_B_actual_cost': env.path_actual_costs[e][1],
                                    'path_B_future_overlap': env.path_future_overlaps[e][1], 'abstract_sequence_B': [env.sampled_abstract_sequences[e][1]],
                                    'dominant_axis_B': env.dominant_axis_B[e],
                                    'start_C': [env.path_states[e][2][0]], 'goal_C': [env.path_states[e][2][-1]], 'path_C': [env.path_states[e][2]],
                                    'path_C_expected_cost': env.path_expected_costs[e][2], 'path_C_actual_cost': env.path_actual_costs[e][2],
                                    'path_C_future_overlap': env.path_future_overlaps[e][2], 'abstract_sequence_C': [env.sampled_abstract_sequences[e][2]],
                                    'better_path': ['a','b','c'][np.argmax(env.path_actual_costs[e])],
                                    })], ignore_index=True)

            env_key = 'city_'+str(c+1)+'_grid_'+str(i+1)

            ## just for safe-keeping, let's save the whole env too
            env_objects[env_key+'_env_object'] = [env]
            env_objects['participant'] = p

    
    ## save for that ppt
    ppt_envs[p] = env_objects
    with open('useful_saves/expt_optimisation/simulated_envs/ppt_'+str(p)+'_envs.pkl', 'wb') as f:
        pickle.dump(env_objects, f)

## save expt info
df_expt.to_csv('useful_saves/expt_optimisation/{}AFC_{}x{}_env_{}-{}-{}-{}_beta_{}_sim_ppts_{}_cities_{}_days_{}_trials_{}_sims_expt_info.csv'.format(n_afc,N,N,
                                                                                       beta_params['alpha_row'], beta_params['beta_row'], beta_params['alpha_col'], beta_params['beta_col'],
                                                                                       n_sim_participants, 
                                                                                       n_cities, n_days, n_trials,
                                                                                        hyperparams['n_sims']))
     

## loop through ppts
if __name__ == '__main__':
    master_pbar = tqdm(total=n_sim_participants, position=0, leave=True, colour='green')
    
    if not parallel:
        # for p in tqdm(range(1, n_participants+1)):
        for p in tqdm(range(1, n_sim_participants)):
            env_objects = ppt_envs[p]

            ## loop through agents
            sim_out = agent_loop(agent_params, hyperparams, agents, env_objects)
            save_sim(sim_out)

    elif parallel:

        ## start pool
        n_cores = np.min([n_cores, n_sim_participants])
        with mp.Pool(n_cores) as pool:
            print('Parallel simulation:', n_cities, ' cities, ', n_days,', days, ',n_trials,' trials, ',hyperparams['n_sims'],' simulations per trial')
            sim_out = [pool.apply_async(agent_loop, args=(p, agent_params, hyperparams, agents),
                                            callback = save_sim) for p in range(1, n_sim_participants+1)]
            pool.close()
            pool.join()

print()
print('Simulation complete')
print()

## convert dict to df
df_sim = pd.DataFrame(all_sim_out)


## save simulated grids + results
df_sim.to_csv('useful_saves/expt_optimisation/{}AFC_{}x{}_env_{}-{}-{}-{}_beta_{}_sim_ppts_{}_cities_{}_days_{}_trials_{}_sims_results.csv'.format(n_afc,N,N,
                                                                                       beta_params['alpha_row'], beta_params['beta_row'], beta_params['alpha_col'], beta_params['beta_col'],
                                                                                       n_sim_participants, 
                                                                                       n_cities, n_days, n_trials,hyperparams['n_sims']))