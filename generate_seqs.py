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
from MCTS import MonteCarloTreeSearch, MonteCarloTreeSearch_Free, MonteCarloTreeSearch_AFC, simulate_agent

import IPython

import multiprocess as mp
import pingouin as pg
from scipy.special import expit

from agents import Farmer
from samplers import GridSampler


warnings.filterwarnings('ignore')

## env inits
N = 9
metric = 'cityblock'
known_costs = False
beta_params = {
    'alpha_row': 0.25,
    'beta_row': 0.25,
    'alpha_col': 0.25,
    'beta_col': 0.25
    # 'alpha_row': 0.5,
    # 'beta_row': 0.5,
    # 'alpha_col': 10,
    # 'beta_col': 0.1
    }

## trial info
n_participants = 100
n_cities = 6
n_days = 5
n_trials = 4
expt = 'AFC'
n_afc = 2
expt_info = {
    'type': expt,
    'n_afc': n_afc,
}
save_path = 'expt/assets/trial_sequences/expt_2'
parallel = True
n_max_cores = 100


## serial
if not parallel:
    print('Generating {} experiment sequences serially'.format(n_participants))
    for p in range(1,n_participants+1):
        generate_ppt_sequence(p, n_cities, n_days, n_trials, expt_info.copy(), beta_params, metric, n_afc, N, save_path)

## parallel
elif parallel:
    if __name__ == '__main__':
        print('Generating {} experiment sequences in parallel'.format(n_participants))
        n_cores = min(n_max_cores, n_participants)
        with mp.Pool(n_cores) as pool:
            results = list(tqdm(pool.starmap(
                generate_ppt_sequence,
                [(p, n_cities, n_days, n_trials, expt_info.copy(), beta_params, metric, n_afc, N, save_path)
                for p in range(1, n_participants + 1)]
            ), total=n_participants))
