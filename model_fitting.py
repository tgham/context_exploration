## imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import warnings
from plotter import *
from scipy.special import softmax
import gymnasium as gym
from gymnasium.envs.registration import register, registry, make, spec
import pickle
import copy
from functools import partial
from scipy.optimize import Bounds, minimize, differential_evolution
import multiprocess as mp
from pybads import BADS
from utils import make_env, Node, Tree, argm, data_keys, grid_keys, parse_lists, KL_divergence, profile_func, KL_sim, value_iteration, load_data
from MCTS import MonteCarloTreeSearch, MonteCarloTreeSearch_Free, MonteCarloTreeSearch_2AFC, simulate_agent
from agents import Farmer
from samplers import GridSampler

### load expt data

## first sample
df, df_q = load_data('expt/data/complete/pilot1')
df2, df_q = load_data('expt/data/complete/pilot1_rotated')
df = pd.concat([df, df2], ignore_index=True)
print('n_p:', df['pid'].nunique())


## define fitting function
def fit_model(pid, bounds, plausible_bounds, hyperparams, df_p, model_name='BAMCP'):
    """
    Fit model to a single participant.
    """
    if len(df_p) > 160:
        print(f'Participant {pid} has {len(df_p)} trials')

    # Load id mapping
    with open('expt/assets/trial_sequences/id_mapping.pkl', 'rb') as f:
        id_mapping = pickle.load(f)

    ## load p's corresponding env object
    try:
        id = id_mapping[pid][10:]
    except KeyError:
        raise KeyError(f'No id mapping for participant {pid}')

    try:
        try:
            with open('expt/assets/trial_sequences/env_objects/env_objects_{}.pkl'.format(id), 'rb') as f:
                envs = pickle.load(f)
        except:
            with open('expt/assets/trial_sequences/rotated_env_objects/env_objects_{}.pkl'.format(id), 'rb') as f:
                envs = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f'No env objects found for participant {pid} with id {id}')

    # Init agent
    agent = Farmer(hyperparams['N'], context_prior=0.5)
    kwargs = dict(
        hyperparams=hyperparams,
        df_trials=df_p,
        envs=envs,
        agent=model_name,
        fit=True
    )
    fun = partial(agent.run, **kwargs)

    ## some init
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    plb = np.array([b[0] for b in plausible_bounds])
    pub = np.array([b[1] for b in plausible_bounds])
    x0 = np.mean([lb, ub], axis=0)
    options = {
        'display': 'off'
    }
    if model_name == 'CE':
        options['uncertainty_handling'] = True
    elif model_name == 'BAMCP':
        options['uncertainty_handling'] = False
        # options['max_fun_evals'] = 20

    # Run PyBADS
    optimizer = BADS(
        fun, x0, lb, ub, plb, pub, options = options,
        )
    result = optimizer.optimize()

    return {
        'pid': pid,
        'temp': result['x'][0],
        'lapse': result['x'][1],
        'loss': result['fval'],
        'model': model_name,
        'success': result['success'],
        'nfev': result['func_count'],
        'nit': result['iterations'],
    }
    

## callback for parallelised fittig
def append_fit(fit_out):
    fitting_data[current_model]['pid'].append(fit_out['pid'])
    fitting_data[current_model]['temp'].append(fit_out['temp'])
    fitting_data[current_model]['lapse'].append(fit_out['lapse'])
    fitting_data[current_model]['loss'].append(fit_out['loss'])
    fitting_data[current_model]['model'].append(fit_out['model'])
    fitting_data[current_model]['success'].append(fit_out['success'])
    fitting_data[current_model]['nfev'].append(fit_out['nfev'])
    fitting_data[current_model]['nit'].append(fit_out['nit'])


    ## update progress bar
    pbar.update(1)


## init dict for saving fits
fitting_data = {}
df_fit = {}

## more init
bounds = [(0, 10)
          , (0, 1)
          ]
bounds = [(b[0] + np.finfo(float).tiny, b[1] - np.finfo(float).tiny) for b in bounds]
plausible_bounds = [(0.01, 3),
                     (0.01, 0.3)
                    ]
n_cores = 50
pids = df['pid'].unique()
n_participants = len(pids)
n_fit_participants = n_participants
# n_fit_participants = 12
pids_to_fit = pids[:n_fit_participants]  # Limit to first n_fit_participants
hyperparams = {
    'N': 8,
    'n_sims': 1000,
    'exploration_constant': 1,
    'discount_factor': 1,
    'n_iter': 10,
    'lazy': False,
}
parallel=True

## loop through models
models_to_fit = [
                'BAMCP', 
                 'CE'
                 ]
for model_type in models_to_fit:
    print(f"Fitting model: {model_type}")
    current_model = model_type
    fitting_data[current_model] = {
        'pid': [],
        'temp': [],
        'lapse': [],
        'loss': [],
        'model': [],
        'success': [],
        'nfev': [],
        'nit': []
    }

    ## parallel fit
    if parallel:
        if __name__ == '__main__':

            # Start pool
            pool = mp.Pool(min(n_cores, n_fit_participants))
            print(f"Beginning parallel fit with model: {current_model}, n_p = {n_fit_participants}")
            pbar = tqdm(total=n_fit_participants)

            # Dispatch all jobs
            fit_out = [pool.apply_async(
                    fit_model, args=(pid, bounds, plausible_bounds, hyperparams, df.loc[df['pid'] == pid], current_model), callback=append_fit
                ) for pid in pids_to_fit]
            pool.close()
            pool.join()
            pbar.close()

    ## serial fit
    else:
        print("Beginning serial fit with model:", current_model, "n_p =", n_fit_participants)
        for pid in tqdm(pids_to_fit):
            df_p = df.loc[df['pid'] == pid].copy()
            fit_out = fit_model(pid, bounds, plausible_bounds, hyperparams, df_p, model_name=current_model)
            fitting_data[current_model]['pid'].append(fit_out['pid'])
            fitting_data[current_model]['temp'].append(fit_out['temp'])
            fitting_data[current_model]['lapse'].append(fit_out['lapse'])
            fitting_data[current_model]['loss'].append(fit_out['loss'])
            fitting_data[current_model]['model'].append(fit_out['model'])
            fitting_data[current_model]['success'].append(fit_out['success'])
            fitting_data[current_model]['nfev'].append(fit_out['nfev'])
            fitting_data[current_model]['nit'].append(fit_out['nit'])



    # Convert to DataFrame
    df_fit[current_model] = pd.DataFrame(fitting_data[current_model])
    print(f"Fit complete: {len(df_fit[current_model])} participants successfully fit")
    # display(df_fit[current_model].head(10))

    ## save dict
    with open(f'useful_saves/fits/first_fits.pkl', 'wb') as f:
        pickle.dump(df_fit, f)
