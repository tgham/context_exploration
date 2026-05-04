"""
tesbi_e2e.py 

End-to-End Transformer-encoded Simulation-Based Inference (TeSBI) Pipeline.
Re-architected to jointly train the Transformer Encoder and Normalizing Flow (Zuko).
Safe for single-node execution (CPU generation -> GPU training) without CUDA deadlocks.

================================================================================
USAGE EXAMPLES
================================================================================
    # RUN EVERYTHING ON A SINGLE NODE (Viper-GPU)
    python tesbi_e2e.py --stage all --n_sims 30000 --epochs 100

    # OR RUN STAGES INDIVIDUALLY:
    python tesbi_e2e.py --stage simulate --n_sims 30000
    python tesbi_e2e.py --stage train --epochs 100
    python tesbi_e2e.py --stage recover --K 20 --num_post 1000
    python tesbi_e2e.py --stage posterior --df_path expt/data/complete/expt_3/df.csv
================================================================================
"""
import sys
import os
import math
import random
import argparse
import warnings
import multiprocessing
import gc
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from joblib import Parallel, delayed
import zuko


# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent.parent))

from MCTS import MonteCarloTreeSearch_AFC
from agents import BAMCP
from runners import run_grid
# ==============================================================================
# CPU CORES SETUP (No global GPU init to prevent multi-processing deadlock)
# ==============================================================================
try:
    N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK"))
except (ValueError, TypeError):
    N_JOBS = multiprocessing.cpu_count()


# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Parameter Ranges
PARAM_RANGES = {
    "temp": (0.0, 3.0),
    "lapse":   (0.0, 1.0),
    "aligned_weight": (0.0, 1.0),
    "orthogonal_weight":   (0.0, 1.0),
    "horizon": (0, 3),
    }

PARAM_ORDER = ["temp", "lapse", "aligned_weight", "orthogonal_weight", "horizon"]

FIXED_PARAMS = {
    "n_samples": 50000,
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
    "greedy": True,
}
LOG_PARAMS = [] ## empty for now

ART_DIR = Path("../outputs/")
RUN_DIR = ART_DIR  # overridden in __main__ once n_sims / n_samples are known
DATASET_PATH = RUN_DIR / "simulated_dataset.pt"
MODEL_PATH = RUN_DIR / "amortized_inference_net.pth"
RECOVERY_CSV = RUN_DIR / "params_recovery.csv"
POST_SUMMARY_CSV = RUN_DIR / "params_posteriors.csv"

# Shared env file used for all simulations (single trial sequence)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / "expt/assets/trial_sequences/expt_3/env_objects/expt_3_env_objects_1.pkl"
ENV_OBJECTS_DIR = ENV_PATH.parent
DEFAULT_DF_PATH = PROJECT_ROOT / "expt/data/complete/expt_3/df.csv"
ID_MAPPING_PATH = PROJECT_ROOT / "expt/data/complete/expt_3/id_mapping_expt_3.pkl"

with open(ENV_PATH, "rb") as f:
    ENV_OBJECTS = pickle.load(f)

def load_env_objects(env_id):
    """Load the env objects dict for a given sequence ID."""
    path = ENV_OBJECTS_DIR / f"expt_3_env_objects_{env_id}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)

def _load_id_mapping():
    with open(ID_MAPPING_PATH, "rb") as f:
        return pickle.load(f)

def _env_id_for_pid(id_mapping, pid):
    """expt_3 branch of utils.load_data: id_mapping[pid][12:] is the env-object id."""
    return id_mapping[pid][12:]

# ==============================================================================
# PARAMETER SCALING (Min-Max + Log)
# ==============================================================================
_param_min = []
_param_max = []
for k in PARAM_ORDER:
    lo, hi = PARAM_RANGES[k]
    if k in LOG_PARAMS:
        lo, hi = math.log(lo), math.log(hi)
    _param_min.append(lo)
    _param_max.append(hi)

PARAM_TRANSFORMED_MIN = np.array(_param_min, dtype=np.float32)
PARAM_TRANSFORMED_MAX = np.array(_param_max, dtype=np.float32)

def sample_prior(n_samples: int) -> np.ndarray:
    samples = []
    for k in PARAM_ORDER:
        lo, hi = PARAM_RANGES[k]
        if k in LOG_PARAMS:
            samples.append(np.random.uniform(math.log(lo), math.log(hi), n_samples))
        else:
            samples.append(np.random.uniform(lo, hi, n_samples))
    return np.stack(samples, axis=1)

def scale_params(raw_array: np.ndarray) -> np.ndarray:
    return (raw_array - PARAM_TRANSFORMED_MIN) / (PARAM_TRANSFORMED_MAX - PARAM_TRANSFORMED_MIN + 1e-8)

def unscale_params(scaled_array: np.ndarray) -> np.ndarray:
    if isinstance(scaled_array, torch.Tensor):
        scaled_array = scaled_array.detach().cpu().numpy()
    transformed = scaled_array * (PARAM_TRANSFORMED_MAX - PARAM_TRANSFORMED_MIN) + PARAM_TRANSFORMED_MIN
    return transformed

def scaled_to_dict(scaled_vec: np.ndarray) -> Dict[str, float]:
    unscaled = unscale_params(scaled_vec)
    out = {}
    for i, k in enumerate(PARAM_ORDER):
        val = unscaled[i]
        out[k] = float(np.exp(val)) if k in LOG_PARAMS else float(val)
    out.update(FIXED_PARAMS)
    return out

# ==============================================================================
# SIMULATOR & FEATURE ENGINEERING
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

def worker_simulate(i, omega, seed_offset=0):
    omega_scaled = scale_params(omega)
    param_dict = scaled_to_dict(omega_scaled)
    choices = simulate_data(param_dict, ENV_OBJECTS, seed=seed_offset + i)
    return choices, omega_scaled

# ==============================================================================
# END-TO-END MODEL: TRANSFORMER + ZUKO FLOW
# ==============================================================================
class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        N = x.size(1)
        return x + self.pe[:N].unsqueeze(0)

class EndToEndTeSBI(nn.Module):
    def __init__(self, in_dim=1, d_model=64, n_heads=4, n_layers=2, param_dim=5):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pe = SinusoidalPE(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.context_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Zuko Neural Spline Flow natively integrates with PyTorch
        self.flow = zuko.flows.NSF(features=param_dim, context=d_model, hidden_features=[128, 128])

    def forward(self, x_seq, true_params_scaled=None):
        h = self.proj(x_seq)
        h = self.pe(h)
        h = self.encoder(h)
        
        # Mean pool across trials
        context = h.mean(dim=1)  
        context = self.context_head(context)
        
        dist = self.flow(context)
        if true_params_scaled is not None:
            return dist.log_prob(true_params_scaled)
        return dist

# ==============================================================================
# PIPELINE STAGES
# ==============================================================================
def stage_simulate(n_sims: int, parallel: bool):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    
    omegas = sample_prior(n_sims)
    if parallel:
        print(f"\n [Simulate] Generating {n_sims} datasets from BAMCP with {args.n_samples} samples in parallel using {N_JOBS} CPUs...")
        results = Parallel(n_jobs=N_JOBS, verbose=1)(
            delayed(worker_simulate)(i, omegas[i]) for i in range(n_sims)
        )
    else:
        print(f"\n [Simulate] Generating {n_sims} datasets from BAMCP with {args.n_samples} samples sequentially on CPU...")
        results = [worker_simulate(i, omegas[i]) for i in range(n_sims)]
    
    X_data = [r[0] for r in results]
    Y_data = [r[1] for r in results]
    
    X_tensor = torch.tensor(np.array(X_data), dtype=torch.float32).unsqueeze(-1)  # (n_sims, n_trials, 1)
    Y_tensor = torch.tensor(np.array(Y_data), dtype=torch.float32)
    
    # print(f" example X[0]: {X_tensor[0].squeeze().numpy()}")
    # print(f" example Y[0]: {Y_tensor[0].numpy()}")

    dataset = TensorDataset(X_tensor, Y_tensor)
    torch.save(dataset, DATASET_PATH)
    print(f"[Simulate] Dataset successfully saved to {DATASET_PATH}")

def stage_train(epochs: int, batch_size=128, lr=1e-3, patience=15):
    # Safe to initialize CUDA here since multiprocessing is done
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n [Train] Starting E2E Joint Training on {str(device).upper()}...")
    if device.type == 'cuda': print(f" [Train] GPU Name: {torch.cuda.get_device_name(0)}")
    
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset missing at {DATASET_PATH}. Run '--stage simulate' first.")
    
    dataset = torch.load(DATASET_PATH, weights_only=False)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model = EndToEndTeSBI(in_dim=1, param_dim=len(PARAM_ORDER)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_loss = float('inf')
    wait = 0
    
    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for Xb, Yb in train_loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            optimizer.zero_grad()
            log_probs = model(Xb, true_params_scaled=Yb)
            loss = -log_probs.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item() * Xb.size(0)
            
        train_loss /= len(train_ds)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, Yb in val_loader:
                Xb, Yb = Xb.to(device), Yb.to(device)
                log_probs = model(Xb, true_params_scaled=Yb)
                loss = -log_probs.mean()
                val_loss += loss.item() * Xb.size(0)
        val_loss /= len(val_ds)
        
        print(f"Epoch {ep+1:03d}/{epochs} | Train NLL: {train_loss:.4f} | Val NLL: {val_loss:.4f}", end="")
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            wait = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(" * [Saved Best]")
        else:
            wait += 1
            print(f" (Wait {wait})")
            if wait >= patience:
                print("Early stopping triggered.")
                break

def stage_recover(K: int, num_post: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n [Recovery] Checking {K} ground-truth cases on {str(device).upper()}...")
    
    model = EndToEndTeSBI(in_dim=1, param_dim=len(PARAM_ORDER)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    omegas_raw = sample_prior(K)
    details = []
    
    for i in range(K):
        gt_params = scaled_to_dict(scale_params(omegas_raw[i]))
        X = simulate_data(gt_params, ENV_OBJECTS, seed=4242 + i)

        tensor_x = torch.tensor(X[None, :, None], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            posterior_dist = model(tensor_x)
            samples_scaled = posterior_dist.sample((num_post,))[:, 0, :].cpu().numpy()
            
        row = {"case": i}
        for k_idx, k in enumerate(PARAM_ORDER):
            s_unscaled = unscale_params(samples_scaled)[:, k_idx]
            vals = np.exp(s_unscaled) if k in LOG_PARAMS else s_unscaled
            
            mu, lo, hi = np.mean(vals), np.percentile(vals, 5), np.percentile(vals, 95)
            gt_val = gt_params[k]
            row[f"gt_{k}"] = gt_val
            row[f"mu_{k}"] = mu
            row[f"hit90_{k}"] = 1.0 if lo <= gt_val <= hi else 0.0
        details.append(row)

    pd.DataFrame(details).to_csv(RECOVERY_CSV, index=False)
    print(f"\n [Recovery] Results saved to {RECOVERY_CSV}")

def stage_inference(df_path: str, num_samples: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = RUN_DIR / "subjects"
    out_root.mkdir(parents=True, exist_ok=True)

    n_trials_total = HYPERPARAMS["n_cities"] * HYPERPARAMS["n_days"] * HYPERPARAMS["n_trials"]

    df = pd.read_csv(df_path, low_memory=False)
    pids = df["pid"].unique()
    print(f"\n [Inference] Processing {len(pids)} participants from {df_path} on {str(device).upper()}...")

    model = EndToEndTeSBI(in_dim=1, param_dim=len(PARAM_ORDER)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    summaries = []
    for pid in pids:
        try:
            df_ppt = df.loc[df["pid"] == pid].sort_values(["city", "day", "trial"])
            choices = (df_ppt["path_chosen"] == "b").values.astype(np.float32)

            if len(choices) != n_trials_total:
                print(f" [Skip] {pid}: {len(choices)} trials != expected {n_trials_total}")
                continue

            tensor_x = torch.tensor(choices[None, :, None], dtype=torch.float32).to(device)

            with torch.no_grad():
                posterior_dist = model(tensor_x)
                samples_scaled = posterior_dist.sample((num_samples,))[:, 0, :].cpu().numpy()

            rows = [scaled_to_dict(s) for s in samples_scaled]
            post_df = pd.DataFrame(rows)
            summ = post_df.describe(percentiles=[0.05, 0.5, 0.95]).T[["mean", "std", "5%", "50%", "95%"]]

            subj_dir = out_root / str(pid)
            subj_dir.mkdir(exist_ok=True)
            post_df.to_csv(subj_dir / "posterior_samples.csv", index=False)
            summ.to_csv(subj_dir / "posterior_summary.csv")

            s_row = summ.copy()
            s_row["pid"] = pid
            summaries.append(s_row.reset_index())
        except Exception as e:
            print(f" [Error] {pid}: {e}")

    if summaries:
        pd.concat(summaries).to_csv(POST_SUMMARY_CSV, index=False)
        print(f"\n [Inference] All summaries saved to {POST_SUMMARY_CSV}")
    else:
        print("\n [Inference] No summaries generated.")


PPC_FIELDS = ["p_correct", "p_chose_orthogonal", "p_chose_more_future_rel_overlap", 'aligned_path_aligned_arm_len', 'aligned_path_orthogonal_arm_len', 'orthogonal_path_aligned_arm_len', 'orthogonal_path_orthogonal_arm_len', 'objective','context']


def worker_ppc(pid, env_id, params, seed):
    """Load this pid's envs, run BAMCP at `params`, return a long-format DataFrame
    with city/day/trial + the three PPC fields, tagged with pid."""
    if seed is not None:
        np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)

    envs = load_env_objects(env_id)
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
        HYPERPARAMS, agent_name="BAMCP", df_trials=None,
        envs=envs, fit=False, yoked=False, progress=False,
    )

    df = pd.DataFrame({k: sim_out[k] for k in ["city", "day", "trial"] + PPC_FIELDS})
    df["pid"] = pid
    return df


def stage_ppc(df_path: str, post_csv: Optional[str] = None):
    """Posterior predictive check: simulate BAMCP at each pid's posterior-mean params
    on that pid's own env objects. Parallelised across pids."""
    csv_path = Path(post_csv) if post_csv else POST_SUMMARY_CSV
    post_df = pd.read_csv(csv_path)
    means = post_df.pivot_table(index="pid", columns="index", values="mean")

    df_all = pd.read_csv(df_path, low_memory=False)
    pids = sorted(df_all["pid"].unique())
    id_mapping = _load_id_mapping()

    tasks = []
    for i, pid in enumerate(pids):
        if pid not in means.index:
            print(f"   [skip] {pid}: no posterior row"); continue
        if pid not in id_mapping:
            print(f"   [skip] {pid}: not in id_mapping"); continue
        env_id = _env_id_for_pid(id_mapping, pid)
        row = means.loc[pid]
        params = {k: float(row[k]) for k in PARAM_ORDER}
        params.update(FIXED_PARAMS)
        
        ## debugging: change n_samples to 10
        # params["n_samples"] = 2

        tasks.append((pid, env_id, params, 4242 + i))

    print(f"\n [PPC] Simulating {len(tasks)} participants in parallel ({N_JOBS} workers)...")
    results = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(worker_ppc)(pid, env_id, params, seed)
        for (pid, env_id, params, seed) in tasks
    )

    df_ppc = pd.concat(results, ignore_index=True)
    out_path = RUN_DIR / "ppc.csv"
    df_ppc.to_csv(out_path, index=False)
    print(f"  [PPC] Saved to {out_path}")
    return df_ppc


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    
    SEED = 137
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

    parser = argparse.ArgumentParser(description="End-to-End TeSBI Pipeline (Transformer + Zuko)")
    parser.add_argument("--stage", choices=["all", "simulate", "train", "recover", "posterior", "ppc"], required=True)
    parser.add_argument("--n_sims", type=int, default=30000, help="Number of simulated datasets")
    parser.add_argument("--n_samples", type=int, default=FIXED_PARAMS["n_samples"],
                        help="BAMCP MCTS rollouts per decision (samples per simulated model)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--parallel", type=int, default=1, help="Use parallel CPU simulation")
    parser.add_argument("--K", type=int, default=20, help="Recovery test cases")
    parser.add_argument("--num_post", type=int, default=1000, help="Samples per recovery/inference")
    parser.add_argument("--df_path", type=str, default=str(DEFAULT_DF_PATH),
                        help="Path to participant choice CSV (expt_3 df.csv)")
    parser.add_argument("--post_csv", type=str, default=None,
                        help="Path to params_posteriors.csv for PPC (default: POST_SUMMARY_CSV)")

    args = parser.parse_args()

    FIXED_PARAMS["n_samples"] = args.n_samples
    RUN_DIR = ART_DIR / f"{args.n_sims}_sims_{args.n_samples}_samples"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_PATH = RUN_DIR / "simulated_dataset.pt"
    MODEL_PATH = RUN_DIR / "amortized_inference_net.pth"
    RECOVERY_CSV = RUN_DIR / "params_recovery.csv"
    POST_SUMMARY_CSV = RUN_DIR / "params_posteriors.csv"
    print(f"[Setup] Run directory: {RUN_DIR}")

    # 1. GENERATE DATA (PURE CPU)
    if args.stage in ["all", "simulate"]:
        stage_simulate(args.n_sims, args.parallel==1)
        
    # 2. CLEAR MEMORY BEFORE GPU TASKS
    if args.stage == "all":
        print("\n [System] Simulating complete. Running Garbage Collection before loading CUDA...")
        gc.collect()
        
    # 3. TRAIN (GPU INITIALIZED HERE)
    if args.stage in ["all", "train"]:
        stage_train(args.epochs, args.batch_size)
        
    # 4. RECOVER (GPU)
    if args.stage in ["all", "recover"]:
        stage_recover(args.K, args.num_post)
        
    # 5. INFERENCE (GPU)
    if args.stage in ["all", "posterior"]:
        stage_inference(args.df_path, args.num_post)

    # 6. POSTERIOR PREDICTIVE CHECK (CPU, parallel over pids)
    if args.stage in ["all", "ppc"]:
        stage_ppc(args.df_path, args.post_csv)