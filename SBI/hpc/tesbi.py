"""
tesbi.py

Simulation-Based Inference (SBI) Pipeline for BAMCP parameter inference.
Uses SNPE with an MLP embedding net on binary choice vectors.

================================================================================
WORKFLOW
================================================================================
1) Simulate: Generate (params, choices) pairs by running BAMCP on grid envs.
2) Train SNPE: Train Neural Spline Flow (NSF) posterior with MLP embedding.
3) (Optional) Round-2: Active learning with proposal sampling.
4) Recovery: Validate posterior against known ground-truth cases.
5) Inference: Condition on observed participant data to sample the posterior.

================================================================================
USAGE EXAMPLES
================================================================================
--- A. LOCAL SMOKE TEST ---
    # 1. Train SNPE (Light: MAF density, small N)
    python tesbi.py --stage snpe --n1 5 --n2 0 --density maf

    # 2. Recovery Check (5 test cases)
    python tesbi.py --stage recover --K 5 --num_post 10

    # 3. Inference on participant data
    python tesbi.py --stage posterior --num_samples 10

--- B. HPC FULL RUN ---
    # 1. Train SNPE (Heavy: NSF density, 50k total sims)
    python tesbi.py --stage snpe --n1 30000 --n2 20000 --density nsf

    # 2. Full Recovery Sweep (200 test cases)
    python tesbi.py --stage recover --K 200 --num_post 4000

    # 3. Full Inference
    python tesbi.py --stage posterior --num_samples 4000
================================================================================
"""
import sys
import os
import random
import pickle
import argparse
import warnings
import multiprocessing
from typing import Dict, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed

from sbi.inference import SNPE
from sbi.utils import BoxUniform

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================
# Add project root to path so we can import MCTS, agents, runners
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from MCTS import MonteCarloTreeSearch_AFC
from agents import BAMCP
from runners import run_grid

# ==============================================================================
# DEVICE SETUP
# ==============================================================================
# --- CPU Device Setup ---
try:
    # SLURM Allocated CPUs
    N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK"))
except (ValueError, TypeError):
    # Default to all available CPUs locally
    N_JOBS = multiprocessing.cpu_count()

print(f"  [Auto-Config] Detected {N_JOBS} CPU cores available for joblib.")

# --- GPU Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("-" * 60)
print(f"Running on device: {str(device).upper()}")
if device.type == 'cuda':
    print(f"GPU Name:          {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory:        {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("WARNING: Running on CPU. This will be slow!")
print("-" * 60)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Parameter Ranges
PARAM_RANGES = {
    "temp": (0.0, 3.0),
    "lapse":   (0.0, 1.0),
    "aligned_weight": (0.0, 1.0),
    "orthogonal_weight":   (0.0, 1.0),
    "horizon": (1, 3),
    }

PARAM_ORDER = ["temp", "lapse", "aligned_weight", "orthogonal_weight", "horizon"]

FIXED_PARAMS = {
    "n_samples": 10000,
    "exploration_constant": 3,
    "discount_factor":   0.9,
    }

# Experiment Structure (expt 3: 32 cities × 1 day × 4 trials = 128 binary choices)
HYPERPARAMS = {
    "n_trials": 4,
    "n_days": 1,
    "n_cities": 32,
    "N": 9,
    "n_afc": 2,
    "greedy": False,
}

N_TRIALS_TOTAL = HYPERPARAMS["n_cities"] * HYPERPARAMS["n_days"] * HYPERPARAMS["n_trials"]  # 128

# Environment Objects
ENV_OBJECTS_DIR = PROJECT_ROOT / "expt/assets/trial_sequences/expt_3/env_objects"
PARTICIPANT_DATA_CSV = PROJECT_ROOT / "expt/data/complete/expt_3/df.csv"

def get_available_env_ids():
    """Get list of available env object IDs."""
    return sorted(
        int(p.stem.split("_")[-1])
        for p in ENV_OBJECTS_DIR.glob("expt_3_env_objects_*.pkl")
    )

def load_env_objects(env_id):
    """Load the env objects dict for a given sequence ID."""
    path = ENV_OBJECTS_DIR / f"expt_3_env_objects_{env_id}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)

# Artifact Paths
ART_DIR = Path("outputs/tesbi/")
POST1_PATH = ART_DIR / "posterior_round1.pkl"
POSTF_PATH = ART_DIR / "posterior_final.pkl"
RECOVERY_CSV = ART_DIR / "params_recovery.csv"
POST_SUMMARY_CSV = ART_DIR / "params_posteriors.csv"


# ==============================================================================
# 1. SIMULATOR WRAPPER
# ==============================================================================
def simulate_data(params: Dict[str, float], envs: Dict, seed: Optional[int] = None) -> np.ndarray:
    """
    Runs the BAMCP simulator for a single parameter set on the given envs.

    Args:
        params: Dict with keys matching PARAM_ORDER + FIXED_PARAMS
                (temp, lapse, aligned_weight, orthogonal_weight, horizon,
                 n_samples, exploration_constant, discount_factor).
        envs: Environment objects dict (from load_env_objects).
        seed: Random seed for reproducibility.

    Returns:
        Binary choice vector of shape (N_TRIALS_TOTAL,) as float32.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    bamcp = BAMCP(
        mcts_class=MonteCarloTreeSearch_AFC,
        run_fn=run_grid,
        temp=params["temp"],
        lapse=params["lapse"],
        horizon=int(round(params["horizon"])),
        exploration_constant=params["exploration_constant"],
        discount_factor=params["discount_factor"],
        n_samples=params["n_samples"],
        aligned_weight=params["aligned_weight"],
        orthogonal_weight=params["orthogonal_weight"],
    )

    sim_out = bamcp.run(
        HYPERPARAMS,
        agent_name="BAMCP",
        df_trials=None,
        envs=envs,
        fit=False,
        yoked=False,
        progress=False,
    )

    choices = np.array(sim_out["actions"], dtype=np.float32)
    return choices


# ==============================================================================
# 2. PARAMETER SETUP
# ==============================================================================
def make_box_prior() -> Tuple[BoxUniform, torch.Tensor, torch.Tensor]:
    """Constructs the SBI BoxUniform prior based on configured ranges."""
    low, high = [], []
    for k in PARAM_ORDER:
        lo, hi = PARAM_RANGES[k]
        low.append(lo)
        high.append(hi)
    
    low = torch.tensor(low, dtype=torch.float32).to(device)
    high = torch.tensor(high, dtype=torch.float32).to(device)
    prior = BoxUniform(low=low, high=high)
    return prior, low, high


def untransform(omega_vec: torch.Tensor) -> Dict[str, float]:
    """Converts transformed (log) parameters back to original space."""
    vals = omega_vec.detach().cpu().numpy().astype(float)
    out = {}
    for i, k in enumerate(PARAM_ORDER):
        v = vals[i]
        out[k] = float(v)
    out.update(FIXED_PARAMS)
    return out


# ==============================================================================
# 3. FEATURE ENGINEERING
# ==============================================================================
def build_features_from_sim(choices: np.ndarray) -> np.ndarray:
    """
    Convert simulated choice vector to feature vector.
    Input: flat array of 0/1 choices from simulate_data (length N_TRIALS_TOTAL).
    Output: same array as float32.
    """
    return choices.astype(np.float32)


def build_features_from_participant(df_participant: pd.DataFrame) -> np.ndarray:
    """
    Extract binary choice vector from a single participant's data.
    Sorted by city/day/trial. path_chosen='b' -> 1, 'a' -> 0, NaN -> 0.5.
    Output: float32 array of shape (N_TRIALS_TOTAL,).
    """
    df_sorted = df_participant.sort_values(["city", "day", "trial"])
    is_b = (df_sorted["path_chosen"] == "b").astype(float).values
    is_nan = df_sorted["path_chosen"].isna().values
    is_b[is_nan] = 0.5  # encode missed trials as uncertain
    return is_b.astype(np.float32)


# ==============================================================================
# 4. EMBEDDING NET
# ==============================================================================
def make_embedding_net(in_dim: int = N_TRIALS_TOTAL, embed_dim: int = 32) -> nn.Module:
    """
    Simple MLP to compress the 128-dim binary choice vector into a lower-dim
    embedding. Passed to SBI as embedding_net so it is trained jointly with the
    density estimator — no separate pretraining needed.
    """
    return nn.Sequential(
        nn.Linear(in_dim, 64),
        nn.ReLU(),
        nn.Linear(64, embed_dim),
        nn.ReLU(),
    )


# ==============================================================================
# 5. SIMULATION WORKER
# ==============================================================================

# Single shared env set, loaded once and used for ALL simulations.
# This ensures choice differences are driven by parameter variation, not grids.
_SIM_ENVS = None

def _load_sim_envs(env_id: int):
    """Load one env set to be shared across all simulation workers."""
    global _SIM_ENVS
    _SIM_ENVS = load_env_objects(env_id)
    print(f"  [Sim Envs] Loaded env set {env_id} "
          f"({HYPERPARAMS['n_cities']} cities x {HYPERPARAMS['n_trials']} trials).")


def worker_simulate(i, omega, seed_offset=0):
    """
    Simulate one (params, choices) pair using the shared env set.
    Called by joblib Parallel — must be a top-level function.
    """
    params = untransform(omega)
    choices = simulate_data(params, _SIM_ENVS, seed=seed_offset + i)
    return build_features_from_sim(choices)


def simulate_round(sampler_fn, n_samples, seed_offset=0):
    """
    Simulate n_samples (omega, x) pairs using the shared env set.
    Returns (omegas tensor [n, D], xs tensor [n, N_TRIALS_TOTAL]).
    """
    omegas = sampler_fn((n_samples,)).cpu()

    print(f"  Launching parallel simulation ({n_samples} sims, {N_JOBS} workers)...")
    X_list = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(worker_simulate)(i, omegas[i], seed_offset=seed_offset)
        for i in range(n_samples)
    )

    xs = torch.tensor(np.stack(X_list, axis=0), dtype=torch.float32)
    return omegas, xs


# ==============================================================================
# 6. PIPELINE STAGES
# ==============================================================================

def run_snpe(args, prior):
    """Simulates data and trains the SNPE posterior (with optional Round 2)."""
    _load_sim_envs(args.env_id)

    # --- Round 1 ---
    print(f"\n [Round1] Simulating {args.n1} pairs...")
    omegas_1, xs_1 = simulate_round(prior.sample, args.n1, seed_offset=0)

    embedding_net = make_embedding_net()
    print(f"  [Round1] Training SNPE ({args.density}) on {device}...")

    inference = SNPE(
        prior=prior,
        density_estimator=args.density,
        device=str(device),
        embedding_net=embedding_net,
    )
    inference.append_simulations(omegas_1, xs_1)

    density = inference.train(
        stop_after_epochs=20,
        max_num_epochs=500,
        show_train_summary=False,
    )
    posterior_1 = inference.build_posterior(density)

    with open(POST1_PATH, "wb") as f:
        pickle.dump(posterior_1, f)

    # --- Round 2 (optional active learning) ---
    if args.n2 > 0:
        print(f"\n [Round2] Active learning with {args.n2} simulations...")

        # Reference observation for proposal sampling (mean of Round 1 xs)
        ref_x = xs_1.mean(dim=0, keepdim=True).to(device)

        def proposal_sampler(shape):
            n = shape[0]
            n_prior = int(args.mix_prior_frac * n)
            n_post = n - n_prior
            post_samples = posterior_1.sample((n_post,), x=ref_x).reshape(n_post, -1)
            return torch.cat([
                prior.sample((n_prior,)).to(device),
                post_samples.to(device),
            ], dim=0)

        omegas_2, xs_2 = simulate_round(proposal_sampler, args.n2, seed_offset=1_000_000)

        omegas_all = torch.cat([omegas_1, omegas_2], dim=0)
        xs_all = torch.cat([xs_1, xs_2], dim=0)

        print(f"  [Round2] Training final posterior on {device}...")
        embedding_net_final = make_embedding_net()
        inference = SNPE(
            prior=prior,
            density_estimator=args.density,
            device=str(device),
            embedding_net=embedding_net_final,
        )
        inference.append_simulations(omegas_all, xs_all.float())

        density = inference.train(
            stop_after_epochs=20,
            max_num_epochs=500,
            show_train_summary=False,
        )
        posterior_final = inference.build_posterior(density)
    else:
        posterior_final = posterior_1

    with open(POSTF_PATH, "wb") as f:
        pickle.dump(posterior_final, f)
    print(f"  [SNPE] Posterior saved to {POSTF_PATH}")


def run_recovery(args, prior, posterior):
    """Validates the posterior against known ground-truth simulated cases."""
    _load_sim_envs(args.env_id)

    print(f"\n [Recovery] Checking {args.K} ground-truth cases...")
    omegas_true = prior.sample((args.K,)).cpu()
    details = []

    for i in range(args.K):
        gt_params = untransform(omegas_true[i])
        choices = simulate_data(gt_params, _SIM_ENVS, seed=4242 + i)
        x_obs = torch.tensor(
            build_features_from_sim(choices), dtype=torch.float32
        ).unsqueeze(0).to(device)

        samples = posterior.sample((args.num_post,), x=x_obs).cpu()

        row = {"case": i}
        samples_np = samples.numpy()
        for k_idx, k in enumerate(PARAM_ORDER):
            vals = samples_np[:, k_idx]
            mu = np.mean(vals)
            lo, hi = np.percentile(vals, 5), np.percentile(vals, 95)
            gt_val = gt_params[k]
            row[f"gt_{k}"] = gt_val
            row[f"mu_{k}"] = mu
            row[f"hit90_{k}"] = 1.0 if lo <= gt_val <= hi else 0.0
        details.append(row)

    pd.DataFrame(details).to_csv(RECOVERY_CSV, index=False)
    print(f"  [Recovery] Saved to {RECOVERY_CSV}")


def run_inference(args, posterior):
    """Runs inference on real participant data from expt 3."""
    out_root = ART_DIR / "subjects"
    out_root.mkdir(parents=True, exist_ok=True)

    # Load participant data
    df_all = pd.read_csv(str(PARTICIPANT_DATA_CSV), low_memory=False)
    pids = sorted(df_all["pid"].unique())

    print(f"\n [Inference] Processing {len(pids)} participants...")
    summaries = []

    for i, pid in enumerate(pids):
        if i % 10 == 0:
            print(f"   ... Progress: {i}/{len(pids)} participants", flush=True)
        try:
            df_sub = df_all[df_all["pid"] == pid]
            x_obs = build_features_from_participant(df_sub)
            x_tensor = torch.tensor(x_obs, dtype=torch.float32).unsqueeze(0).to(device)

            samples = posterior.sample(
                (args.num_samples,), x=x_tensor
            ).cpu()

            # Build results DataFrame with named columns
            rows = [untransform(s) for s in samples]
            post_df = pd.DataFrame(rows)
            summ = post_df.describe(
                percentiles=[0.05, 0.5, 0.95]
            ).T[["mean", "std", "5%", "50%", "95%"]]

            subj_dir = out_root / pid
            subj_dir.mkdir(exist_ok=True)
            post_df.to_csv(subj_dir / "posterior_samples.csv", index=False)
            summ.to_csv(subj_dir / "posterior_summary.csv")

            s_row = summ.copy()
            s_row["pid"] = pid
            summaries.append(s_row.reset_index())

        except Exception as e:
            print(f"\n [Error] {pid}: {e}")

    if summaries:
        pd.concat(summaries).to_csv(POST_SUMMARY_CSV, index=False)
        print(f"  [Inference] All summaries saved to {POST_SUMMARY_CSV}")


# ==============================================================================
# 7. MAIN
# ==============================================================================
def main():
    warnings.filterwarnings("ignore")
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    SEED = 137
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

    parser = argparse.ArgumentParser(description="SBI Pipeline for BAMCP")
    parser.add_argument("--stage", choices=["all", "snpe", "recover", "posterior"], default="all")
    parser.add_argument("--n1", type=int, default=20000, help="Round 1 simulations")
    parser.add_argument("--n2", type=int, default=15000, help="Round 2 simulations (0 to skip)")
    parser.add_argument("--mix_prior_frac", type=float, default=0.2, help="Prior fraction in Round 2 proposal")
    parser.add_argument("--density", choices=["nsf", "maf", "mdn"], default="nsf", help="Density estimator")
    parser.add_argument("--env_id", type=int, default=1,
                        help="Env object ID to use for all simulations (same grids for every agent)")

    # Recovery args
    parser.add_argument("--K", type=int, default=20, help="Recovery test cases")
    parser.add_argument("--num_post", type=int, default=1000, help="Posterior samples per recovery case")

    # Inference args
    parser.add_argument("--num_samples", type=int, default=4000, help="Posterior samples per participant")

    args = parser.parse_args()
    ART_DIR.mkdir(parents=True, exist_ok=True)

    prior, _, _ = make_box_prior()

    # --- Pipeline ---

    # 1. SNPE (simulate + train)
    if args.stage in ["all", "snpe"]:
        run_snpe(args, prior)

    # 2. Load posterior for downstream stages
    if args.stage in ["all", "recover", "posterior"]:
        with open(POSTF_PATH, "rb") as f:
            posterior = pickle.load(f)

    # 3. Recovery validation
    if args.stage in ["all", "recover"]:
        run_recovery(args, prior, posterior)

    # 4. Inference on participant data
    if args.stage in ["all", "posterior"]:
        run_inference(args, posterior)

if __name__ == "__main__":
    main()