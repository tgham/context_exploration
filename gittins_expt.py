"""
Parallelised bandit MCTS simulations.

Runs BAMCP over (sim, alpha, beta) combinations using multiprocessing,
collecting estimated Q-values and visit counts per arm, then saves to CSV.
"""

import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial

from utils import make_gittins_bandit_env
from MCTS import MonteCarloTreeSearch_Bandit
from agents import BAMCP
from runners import run_bandit


def run_single_sim(sim_idx, alpha, beta, n_arms, n_trials, gam, lam,
                   n_samples, exploration_constant, horizon):
    """Run one (sim, alpha, beta) combination and return a results dict."""
    env = make_gittins_bandit_env(
        n_arms=n_arms, alpha=alpha, beta=beta,
        n_trials=n_trials, gam=gam, lam=lam,
    )
    env.reset()

    agent = BAMCP(
        mcts_class=MonteCarloTreeSearch_Bandit, run_fn=run_bandit,
        n_samples=n_samples,
        exploration_constant=exploration_constant,
        discount_factor=gam,
        horizon=horizon,
        temp=1,
        lapse=0,
    )
    agent.init_mcts(env)
    env.set_sim(True)

    MCTS_Q = agent.compute_Q(env)

    # Collect per-arm visit counts
    visits = {}
    for action, leaf in agent.mcts.tree.root.action_leaves.items():
        visits[action] = leaf.n_action_visits

    # Build one row per arm
    rows = []
    n_afc = env.n_afc
    for arm in range(n_afc):
        rows.append({
            'sim': sim_idx,
            'alpha': alpha,
            'beta': beta,
            'arm': arm,
            'Q': MCTS_Q[arm] if arm < len(MCTS_Q) else np.nan,
            'n_visits': visits.get(arm, 0),
        })
    return rows


def _worker(args):
    """Unpack args tuple for Pool.map."""
    sim_idx, alpha, beta, kwargs = args
    return run_single_sim(sim_idx, alpha, beta, **kwargs)


def main():
    parser = argparse.ArgumentParser(description='Parallelised bandit MCTS simulations')
    parser.add_argument('--n_sims', type=int, default=10)
    parser.add_argument('--n_arms', type=int, default=1)
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--gam', type=float, default=0.9)
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--n_samples', type=int, default=250000)
    parser.add_argument('--exploration_constant', type=float, default=3.0)
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--n_workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--n_alphas', type=int, default=10,
                        help='Number of alpha values to simulate (default: 10)')
    args = parser.parse_args()

    
    ## a=b
    alphas = np.arange(1, args.n_alphas + 1) 
    betas = alphas 

    ## a=b+1
    # betas1 = alphas + 1
    # alphas = np.concatenate([alphas, alphas]) 
    # betas = np.concatenate([betas, betas1])   

    shared_kwargs = dict(
        n_arms=args.n_arms, n_trials=args.n_trials, gam=args.gam, lam=args.lam,
        n_samples=args.n_samples, exploration_constant=args.exploration_constant,
        horizon=args.horizon,
    )

    # Build task list: one entry per (sim, alpha, beta)
    tasks = []
    for sim_idx in range(args.n_sims):
        for alpha, beta in zip(alphas, betas):
            tasks.append((sim_idx, int(alpha), int(beta), shared_kwargs))

    print(f'Running {len(tasks)} tasks across {args.n_workers or "all"} workers...')

    with Pool(processes=args.n_workers) as pool:
        results = pool.map(_worker, tasks)

    # Flatten list of lists
    all_rows = [row for batch in results for row in batch]
    df = pd.DataFrame(all_rows)

    path = 'useful_saves/bandits/gittins_{}_sims_{}_samples_{}_discount_{}_expl_{}_lam.csv'.format(args.n_sims,
    args.n_samples, args.discount_factor, args.exploration_constant, args.lam)

    df.to_csv(path, index=False)
    print(f'Saved {len(df)} rows to {path}')

if __name__ == '__main__':
    main()
