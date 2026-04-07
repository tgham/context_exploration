"""
Parallelised bandit MCTS experiments with controlled posteriors.

For a 2-armed bandit:
- Arm 0: fixed Beta(1, 1) prior
- Arm 1: varied priors where alpha = beta, beta+1, or beta+2

Pulls arm 1 until observations match the desired Beta prior, then runs MCTS.
Parallelises over (sim_idx, alpha, beta) combinations.
"""

import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool

from utils import make_bandit_env
from MCTS import MonteCarloTreeSearch_Bandit
from agents import BAMCP
from runners import run_bandit


def get_target_observations(alpha, beta):
    """
    For a Beta(alpha, beta) prior, return (n_successes, n_failures) needed.
    Beta(alpha, beta) = Beta(1 + n_successes, 1 + n_failures)
    So: n_successes = alpha - 1, n_failures = beta - 1
    """
    return alpha - 1, beta - 1


def find_matching_env(alpha, beta, n_trials=100, max_retries=100):
    """
    Create environments and pull arm 1 until we get the desired observations.

    Returns:
        (env, sampler, n_pulls_needed) if successful
        (None, None, None) if max_retries exceeded
    """
    target_successes, target_failures = get_target_observations(alpha, beta)
    target_total = target_successes + target_failures

    for attempt in range(max_retries):
        env = make_bandit_env(n_arms=2, n_trials=n_trials, alpha=1, beta=1)
        env.reset()
        env.set_sim(False)

        successes, failures = 0, 0

        # Pull arm 1 until we get the target observations
        for pull in range(target_total):
            trial_obs, reward, terminated, truncated, info = env.step(1)  # pull arm 1
            if reward == 1:
                successes += 1
            else:
                failures += 1

            if terminated or truncated:
                break

        # Check if we got the right observations
        if successes == target_successes and failures == target_failures:
            sampler = env.make_sampler()
            return env, sampler, target_total

    return None, None, None


def run_single_experiment(sim_idx, alpha, beta, n_trials=100, n_samples=250000,
                         exploration_constant=3.0, discount_factor=0.9,
                         horizon=100, max_retries=100):
    """
    Run one MCTS experiment with arm 1 having Beta(alpha, beta) prior.

    Returns:
        dict with results, or None if initialization failed
    """
    # Set up environment with matching observations
    env, sampler, n_pulls = find_matching_env(alpha, beta, n_trials, max_retries)

    if env is None:
        return None

    # Set to sim mode and run MCTS
    env.set_sim(True)

    agent = BAMCP(
        mcts_class=MonteCarloTreeSearch_Bandit, run_fn=run_bandit,
        n_samples=n_samples,
        exploration_constant=exploration_constant,
        discount_factor=discount_factor,
        horizon=horizon,
        temp=1,
        lapse=0,
    )
    agent.init_mcts(env)
    MCTS_Q = agent.compute_Q(env)

    # Collect per-arm visit counts
    visits = {}
    for action, leaf in agent.mcts.tree.root.action_leaves.items():
        visits[action] = leaf.n_action_visits

    return {
        'sim': sim_idx,
        'alpha': alpha,
        'beta': beta,
        'offset': alpha - beta,  # 0, 1, or 2
        'Q_arm_0': MCTS_Q[0],
        'Q_arm_1': MCTS_Q[1],
        'n_visits_arm_0': visits.get(0, 0),
        'n_visits_arm_1': visits.get(1, 0),
    }


def _worker(args_tuple):
    """Unpack args tuple for Pool.map."""
    sim_idx, alpha, beta, kwargs = args_tuple
    return run_single_experiment(sim_idx, alpha, beta, **kwargs)


def main():
    parser = argparse.ArgumentParser(description='Bandit MCTS with controlled posteriors')
    parser.add_argument('--n_sims', type=int, default=10,
                        help='Number of simulations per (alpha, beta) pair')
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--max_beta', type=int, default=8)
    parser.add_argument('--n_samples', type=int, default=250000)
    parser.add_argument('--exploration_constant', type=float, default=3.0)
    parser.add_argument('--discount_factor', type=float, default=0.9)
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--max_retries', type=int, default=100,
                        help='Max retries to find matching env per experiment')
    parser.add_argument('--n_workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--output', type=str, default='bandit_single_search_results.csv')
    args = parser.parse_args()

    # Generate (alpha, beta) pairs
    # For each beta in [1, max_beta], test alpha = beta, beta+1, beta+2
    experiments = []
    for beta in range(1, args.max_beta + 1):
        for offset in [0, 1, 2]:
            alpha = beta + offset
            experiments.append((alpha, beta))

    shared_kwargs = dict(
        n_trials=args.n_trials,
        n_samples=args.n_samples,
        exploration_constant=args.exploration_constant,
        discount_factor=args.discount_factor,
        horizon=args.horizon,
        max_retries=args.max_retries,
    )

    # Build task list: one entry per (sim, alpha, beta)
    tasks = []
    for sim_idx in range(args.n_sims):
        for alpha, beta in experiments:
            tasks.append((sim_idx, alpha, beta, shared_kwargs))

    print(f'Running {len(tasks)} tasks across {args.n_workers or "all"} workers...')

    with Pool(processes=args.n_workers) as pool:
        results = pool.map(_worker, tasks)

    # Filter out None results (failed initializations)
    results = [r for r in results if r is not None]

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f'Saved {len(df)} experiments to {args.output}')


if __name__ == '__main__':
    main()
