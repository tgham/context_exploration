import numpy as np
import pandas as pd
import copy
import multiprocess as mp
from joblib import Parallel, delayed

from utils import make_bandit_env
from agents import BAMCP, CE
from runners import run_bandit
from MCTS import MonteCarloTreeSearch_Bandit


# --- Config ---
n_arms = 8
n_trials = 300
alpha = 1
beta = 1
n_sims = 100
parallel = False
n_workers = np.min([mp.cpu_count(), n_sims])
output_path = 'useful_saves/bandits/{}_arms_{}_sims.csv'.format(n_arms, n_sims)

# --- Agents ---
bamcp_agent = BAMCP(
    mcts_class=MonteCarloTreeSearch_Bandit, run_fn=run_bandit,
    n_samples=50000,
    exploration_constant=3,
    discount_factor=0.9,
    horizon=50,
    temp=1,
    lapse=0,
)
ce_agent = CE(
    mcts_class=None, run_fn=run_bandit,
    temp=1, lapse=0)

agents = {
    'BAMCP': bamcp_agent,
    'CE': ce_agent,
}


def run_one_sim(sim_idx, agents, n_arms, alpha, beta, n_trials):
    env = make_bandit_env(n_arms=n_arms, alpha=alpha, beta=beta, n_trials=n_trials)
    sim_results = {}
    for name, agent in agents.items():
        agent_copy = copy.deepcopy(agent)
        env_copy = copy.deepcopy(env)
        sim_results[name] = agent_copy.run(env_copy, greedy=True, verbose=True)
    return sim_results


def results_to_df(sim_outputs, agent_name, n_arms):
    rows = []
    for sim_idx, sim_result in enumerate(sim_outputs):
        r = sim_result[agent_name]
        n_t = len(r['actions'])
        cum_bayes_regret = np.cumsum(r['bayes_regret'])
        for t in range(n_t):
            row = {
                'sim': sim_idx,
                'trial': t + 1,
                'action': r['actions'][t],
                'reward': r['rewards'][t],
                'cumulative_reward': r['cumulative_reward'][t],
                'chose_optimal': r['chose_optimal'][t],
                'optimal_arm': r['optimal_arm'],
                'bayes_regret': r['bayes_regret'][t],
                'cum_bayes_regret': cum_bayes_regret[t],
                'CE_consistent': r['CE_actions'][t] == r['actions'][t],
            }
            for a in range(n_arms):
                row[f'Q_{a}'] = r['Q'][t, a]
                row[f'p_choice_{a}'] = r['p_choice'][t, a]
                row[f'post_prob_{a}'] = r['post_probs'][t, a]
                row[f'true_prob_{a}'] = r['true_probs'][a]
                row[f'CE_Q_{a}'] = r['CE_Q'][t, a]
            row['CE_action'] = r['CE_actions'][t]
            row['CE_chose_optimal'] = r['CE_chose_optimal'][t]
            row['CE_bayes_regret'] = r['CE_bayes_regret'][t]
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    import os

    if parallel:
        print(f"Running {n_sims} simulations with {n_workers} workers...")
        sim_outputs = Parallel(n_jobs=n_workers, verbose=10)(
            delayed(run_one_sim)(i, agents, n_arms, alpha, beta, n_trials)
            for i in range(n_sims)
        )
    else:
        print(f"Running {n_sims} simulations (serial)...")
        sim_outputs = [
            run_one_sim(i, agents, n_arms, alpha, beta, n_trials)
            for i in range(n_sims)
        ]

    all_dfs = []
    for name in agents:
        df = results_to_df(sim_outputs, name, n_arms)
        df.insert(0, 'agent', name)
        all_dfs.append(df)
    df_all = pd.concat(all_dfs, ignore_index=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_all.to_csv(output_path, index=False)
    print(f"Saved {len(df_all)} rows to {output_path}")
