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

from utils import make_env, Node, Tree, argm, data_keys, grid_keys, parse_lists, KL_divergence, profile_func, KL_sim, value_iteration, generate_ppt_sequence
from MCTS import MonteCarloTreeSearch, MonteCarloTreeSearch_AFC

import IPython

import multiprocess as mp
import pingouin as pg
from scipy.special import expit

from agents import Farmer, BAMCP, CE
from samplers import GridSampler


warnings.filterwarnings('ignore')


## need this for paralellising because of the way the loops are structured...
def agent_loop(p, agent_params_list, hyperparams, agent_param_combos):
    """
    Run simulation for a single participant across all agent/parameter combinations.
    
    Args:
        p: participant index
        agent_params_list: list of parameter settings, each a list [temp, lapse, arm_weight, horizon, real_future]
        hyperparams: dictionary of hyperparameters
        agent_param_combos: list of (agent_name, param_idx) tuples
    """
    N = hyperparams['N']
    sim_outs = []


    ## load env objects
    # with open('useful_saves/expt_optimisation/simulated_envs/ppt_'+str(p)+'_envs.pkl', 'rb') as f:
    # with open('useful_saves/expt_optimisation/simulated_envs/env_objects/expt_2_env_objects_' + str(p) + '.pkl', 'rb') as f:
    with open('useful_saves/expt_optimisation/simulated_envs/expt_3/env_objects/expt_3_env_objects_' + str(p) + '.pkl', 'rb') as f: ## USE THIS IS GENERATING SEQS BEFORE SIMULATING
    # with open('expt/assets/trial_sequences/expt_3/env_objects/expt_3_env_objects_' + str(p) + '.pkl', 'rb') as f: ## USE THIS USING THE SEQS THAT WE ALREADY GENERATED FOR PPTS
        env_objects = pickle.load(f)

    ## loop through agent/parameter combinations
    for agent, param_idx in agent_param_combos:
        agent_params = agent_params_list[param_idx]

        ## unpack params 
        temp = agent_params[0]
        lapse = agent_params[1]
        arm_weight = agent_params[2]
        horizon = agent_params[3]
        real_future = agent_params[4]

        ## unpack hyperparams
        exploration_constant = hyperparams['exploration_constant']
        discount_factor = hyperparams['discount_factor']
        n_samples = hyperparams['n_samples']

        # farmer = Farmer(N, known_context = True, temp=temp, lapse=lapse, arm_weight=arm_weight, horizon=horizon, real_future_paths=real_future, exploration_constant=exploration_constant, discount_factor=discount_factor, n_samples=n_samples) ## known context if expt 3
        if agent == 'BAMCP':
            farmer = BAMCP(N, known_context = True, temp=temp, lapse=lapse, arm_weight=arm_weight, horizon=horizon, real_future_paths=real_future, exploration_constant=exploration_constant, discount_factor=discount_factor, n_samples=n_samples) ## known context if expt 3
        elif agent =='CE':
            farmer = CE(N, known_context = True, temp=temp, lapse=lapse, arm_weight=arm_weight, horizon=horizon, real_future_paths=real_future, exploration_constant=exploration_constant, discount_factor=discount_factor, n_samples=n_samples) ## known context if expt 3
        sim_out = farmer.run(hyperparams, agent=agent, df_trials=None, envs=env_objects, fit=False, progress=False)
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
N = 9
beta_params = {
    'alpha_row':0.25,
    'beta_row': 0.25,
    'alpha_col':0.25,
    'beta_col': 0.25
    # 'alpha_row': 0.5,
    # 'beta_row': 0.5,
    # 'alpha_col': 10,
    # 'beta_col': 0.1
    }

## trial info
n_sim_participants = 2
n_cities = 32
n_days = 1
n_trials = 4
expt = 'AFC'
n_afc = 2
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
                                'dominant_axis_A','dominant_axis_B',
                                'path_A_future_row_overlap', 'path_B_future_row_overlap',
                                'path_A_future_col_overlap', 'path_B_future_col_overlap',
                                'path_A_future_rel_overlap', 'path_B_future_rel_overlap',
                                'path_A_future_irrel_overlap', 'path_B_future_irrel_overlap',
                                'path_A_future_row_and_col_overlap', 'path_B_future_row_and_col_overlap'
                                ])
if n_afc==3:
    df_expt = df_expt.join(pd.DataFrame(columns=['start_C', 'goal_C', 'path_C',
                        'path_C_expected_cost', 'path_C_actual_cost', 
                        'path_C_future_overlap', 'abstract_sequence_C', 
                        'dominant_axis_C',
                        'path_C_future_row_overlap', 'path_C_future_col_overlap',
                        'path_C_future_rel_overlap', 'path_C_future_irrel_overlap',
                        'path_C_future_row_and_col_overlap'
                        ]))
    
    
## init sim results
all_sim_out = {
        'participant':[],
        'agent':[],
        'temp':[],
        'lapse':[],
        'arm_weight':[],
        'horizon':[],
        'real_future_paths':[],
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
        'leaf_visits_c':[],
        'CE_actions':[],
        'CE_Q_a':[],
        'CE_Q_b':[],
        'CE_Q_c':[],
        'CE_p_choice_A':[],
        'CE_p_choice_B':[],
        'CE_p_choice_C':[],
        'CE_p_correct':[],
        'distr_diff':[]
    }
parallel = False
n_cores = 128
create=False

## init agent and expt
## Define parameter variations to sweep over
param_settings = {
    'temp': [1],           # temperature values to try
    'lapse': [0],          # lapse rate values to try
    'arm_weight': [
                    0,
                    # 1
                    ],  
    'horizon': [
                # 1, 
                3
                ],
    'real_future': [
                    True
                    # ,False
                    ],   # whether to use real future paths
}

## Generate all combinations of parameter settings
param_combinations = list(itertools.product(
    param_settings['temp'],
    param_settings['lapse'],
    param_settings['arm_weight'],
    param_settings['horizon'],
    param_settings['real_future']
))

## Convert to list of lists for indexing
agent_params_list = [list(combo) for combo in param_combinations]
print(f"Running {len(agent_params_list)} parameter combinations:")
for i, params in enumerate(agent_params_list):
    print(f"  [{i}] temp={params[0]}, lapse={params[1]}, arm_weight={params[2]}, horizon={params[3]}, real_future={params[4]}")

hyperparams = {
    'n_samples': 1000,
    'exploration_constant': 1,
    'discount_factor': 1,
    'n_trials': n_trials,
    'n_afc': n_afc,
    'n_days': n_days,
    'n_cities': n_cities,
    'N': N,
    'participant': None, ## hacky
}
agents = [
    # 'BAMCP', 
    'CE', 
    # 'CE_one_arm'
    ]

## Generate all agent/parameter combinations (Each combo is (agent_name, param_idx))
agent_param_combos = [(agent, param_idx) 
                      for agent in agents 
                      for param_idx in range(len(agent_params_list))]
print(f"Total agent/parameter combinations per participant: {len(agent_param_combos)}")   

## generate dataset for each participant
if create:
    if parallel:
        if __name__ == '__main__':
            print('Generating {} experiment sequences in parallel'.format(n_sim_participants))
            n_cores = min(mp.cpu_count(), n_sim_participants)
            with mp.Pool(n_cores) as pool:
                results = list(tqdm(pool.starmap(
                    generate_ppt_sequence,
                    [(p, n_cities, n_days, n_trials, expt_info.copy(), beta_params, n_afc, N)
                    for p in range(1, n_sim_participants + 1)]
                ), total=n_sim_participants))

            # combine all participant-level DataFrames
            df_expt = pd.concat(results, ignore_index=True)

            ## save expt info
            df_expt.to_csv('useful_saves/expt_optimisation/sim_results/{}AFC_{}x{}_env_{}-{}-{}-{}_beta_{}_sim_ppts_{}_cities_{}_days_{}_trials_{}_sims_expt_info.csv'.format(n_afc,N,N,
                                                                                                beta_params['alpha_row'], beta_params['beta_row'], beta_params['alpha_col'], beta_params['beta_col'],
                                                                                                n_sim_participants, 
                                                                                                n_cities, n_days, n_trials,
                                                                                                    hyperparams['n_samples']))

    elif not parallel:
        print('Generating {} experiment sequences serially'.format(n_sim_participants))
        for p in tqdm(range(1,n_sim_participants+1)):

            out = generate_ppt_sequence(p, n_cities, n_days, n_trials, expt_info.copy(), beta_params, n_afc, N)
            df_expt = pd.concat([df_expt, out], ignore_index=True)

        ## save expt info
        df_expt.to_csv('useful_saves/expt_optimisation/sim_results/{}AFC_{}x{}_env_{}-{}-{}-{}_beta_{}_sim_ppts_{}_cities_{}_days_{}_trials_{}_sims_expt_info.csv'.format(n_afc,N,N,
                                                                                            beta_params['alpha_row'], beta_params['beta_row'], beta_params['alpha_col'], beta_params['beta_col'],
                                                                                            n_sim_participants, 
                                                                                            n_cities, n_days, n_trials,
                                                                                                hyperparams['n_samples']))
     

## loop through ppts
if __name__ == '__main__':
    master_pbar = tqdm(total=n_sim_participants, position=0, leave=True, colour='green')
    
    if not parallel:
        # for p in tqdm(range(1, n_participants+1)):
        for p in tqdm(range(1, n_sim_participants)):

            ## loop through agents and parameter combinations
            sim_out = agent_loop(p, agent_params_list, hyperparams, agent_param_combos)
            save_sim(sim_out)

    elif parallel:

        ## start pool
        n_cores = np.min([n_cores, n_sim_participants])
        with mp.Pool(n_cores) as pool:
            print('Parallel simulation:', n_cities, ' cities, ', n_days,', days, ',n_trials,' trials, ',hyperparams['n_samples'],' simulations per trial')
            print(f'Running {len(agents)} agents x {len(agent_params_list)} param settings = {len(agent_param_combos)} combinations per participant')
            sim_out = [pool.apply_async(agent_loop, args=(p, agent_params_list, hyperparams, agent_param_combos),
                                            callback = save_sim) for p in range(1, n_sim_participants+1)]
            pool.close()
            pool.join()

print()
print('Simulation complete')
print()

## convert dict to df
df_sim = pd.DataFrame(all_sim_out)


## save simulated grids + results
df_sim.to_csv('useful_saves/expt_optimisation/sim_results/{}AFC_{}x{}_env_{}_sim_ppts_{}_cities_{}_days_{}_trials_{}_sims_results.csv'.format(n_afc,N,N,
                                                                                       n_sim_participants, 
                                                                                       n_cities, n_days, n_trials,hyperparams['n_samples']))