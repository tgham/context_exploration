"""
tesbi.py

Two-stage Transformer-encoded Simulation-Based Inference (TeSBI) for BAMCP.

================================================================================
WORKFLOW
================================================================================
1) Pretrain: Simulate (omega, choices) and train a Transformer encoder under
   MSE loss to predict omega from the binary choice sequence.
2) SNPE Round 1: Freeze the encoder, simulate fresh (omega, choices), embed +
   z-score, fit a Neural Spline Flow (NSF) posterior on (omega, embedding_z).
3) SNPE Round 2 (active learning, optional): Sample from a prior+posterior
   mixture proposal, simulate, embed + z-score (same standardizer), refit.
4) Recovery: Validate the posterior against known ground-truth cases (via
   the frozen encoder).
5) Inference: Condition on observed participant data to sample the posterior.
6) PPC: Posterior predictive check at each pid's posterior-mean params.

================================================================================
USAGE EXAMPLES
================================================================================
--- A. LOCAL SMOKE TEST ---
    # 1. Pretrain encoder (small N, few epochs)
    python tesbi.py --stage pretrain --n1_pre 5 --n2 2 --epochs 2

    # 2. Train SNPE on frozen encoder (light)
    python tesbi.py --stage snpe --n1_pre 5 --n2 2 --density maf

    # 3. Recovery + inference + PPC
    python tesbi.py --stage recover --n1_pre 5 --n2 2 --K 5 --num_post 50
    python tesbi.py --stage posterior --n1_pre 5 --n2 2 --num_samples 50
    python tesbi.py --stage ppc --n1_pre 5 --n2 2

--- B. HPC FULL RUN ---
    python tesbi.py --stage pretrain --n1_pre 30000 --epochs 8
    python tesbi.py --stage snpe --n1_pre 30000 --n2 20000 --density nsf
    python tesbi.py --stage recover --K 200 --num_post 4000
    python tesbi.py --stage posterior --num_samples 4000
    python tesbi.py --stage ppc

--- or, run all in one line
    python tesbi.py --stage all --n1_pre 30000 --n2 20000 --density nsf
================================================================================
"""
import sys
import os
import gc
import random
import pickle
import argparse
import warnings
import multiprocessing
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
    "aligned_weight": (0.0, 1.0),
    "orthogonal_weight":   (0.0, 1.0),
    "horizon": (0, 3),
    }

PARAM_ORDER = ["temp", "aligned_weight", "orthogonal_weight", "horizon"]

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
    "greedy": True,
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

ID_MAPPING_PATH = PROJECT_ROOT / "expt/data/complete/expt_3/id_mapping_expt_3.pkl"

def _load_id_mapping():
    with open(ID_MAPPING_PATH, "rb") as f:
        return pickle.load(f)

def _env_id_for_pid(id_mapping, pid):
    """expt_3 branch of utils.load_data: id_mapping[pid][12:] is the env-object id."""
    return id_mapping[pid][12:]

# Artifact Paths
ART_DIR = Path("SBI/outputs/2_step/")
RUN_DIR = ART_DIR  # overridden in main() once n_pre/n1/n2/n_samples are known
ENC_PATH = RUN_DIR / "encoder.pt"
STD_MEAN_PATH = RUN_DIR / "embeds_mean.npy"
STD_STD_PATH = RUN_DIR / "embeds_std.npy"
POST1_PATH = RUN_DIR / "posterior_round1.pkl"
POSTF_PATH = RUN_DIR / "posterior_final.pkl"
RECOVERY_CSV = RUN_DIR / "params_recovery.csv"
POST_SUMMARY_CSV = RUN_DIR / "params_posteriors.csv"


# ==============================================================================
# 1. SIMULATOR WRAPPER
# ==============================================================================
def simulate_data(params: Dict[str, float], envs: Dict, seed: Optional[int] = None) -> np.ndarray:
    """
    Runs the BAMCP simulator for a single parameter set on the given envs.

    Args:
        params: Dict with keys matching PARAM_ORDER + FIXED_PARAMS
                (temp, aligned_weight, orthogonal_weight, horizon,
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
    Convert simulated choice vector to per-trial feature matrix.
    Input: flat array of 0/1 choices from simulate_data (length N_TRIALS_TOTAL).
    Output: float32 array of shape (N_TRIALS_TOTAL, 1) — single feature per trial,
    shaped for the transformer encoder (sequence of single-channel tokens).
    """
    return choices.astype(np.float32).reshape(-1, 1)


def build_features_from_participant(df_participant: pd.DataFrame) -> np.ndarray:
    """
    Extract binary choice vector from a single participant's data.
    Sorted by city/day/trial. path_chosen='b' -> 1, else (incl. NaN) -> 0.
    Output: float32 array of shape (N_TRIALS_TOTAL, 1).
    """
    df_sorted = df_participant.sort_values(["city", "day", "trial"])
    choices = (df_sorted["path_chosen"] == "b").values.astype(np.float32)
    return choices.reshape(-1, 1)


# ==============================================================================
# 4. TRANSFORMER ENCODER (Stage A: pretrained, then frozen)
# ==============================================================================
class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding."""
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


class TrialTransformer(nn.Module):
    """Transformer encoder + mean pooling that maps (B, N_trials, in_dim) -> (B, out_dim)."""
    def __init__(self, in_dim: int, model_dim: int = 64, nhead: int = 4, nlayers: int = 2,
                 dropout: float = 0.1, out_dim: int = 64, use_pos_enc: bool = True):
        super().__init__()
        self.proj = nn.Linear(in_dim, model_dim)
        self.use_pos = use_pos_enc
        if use_pos_enc:
            self.pe = SinusoidalPE(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nhead, dim_feedforward=model_dim * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, out_dim),
        )

    def forward(self, x):
        h = self.proj(x)
        if self.use_pos:
            h = self.pe(h)
        h = self.encoder(h)
        h = h.mean(dim=1)  # mean pool across trials
        return self.head(h)


# Encoder hyperparameters fixed across pretrain/load (so state_dict matches).
ENCODER_IN_DIM = 1
ENCODER_OUT_DIM = 64


def make_encoder() -> TrialTransformer:
    """Construct a fresh TrialTransformer with the standard architecture."""
    return TrialTransformer(in_dim=ENCODER_IN_DIM, out_dim=ENCODER_OUT_DIM, use_pos_enc=True)


# ==============================================================================
# 4b. TRAINING UTILS (encoder pretraining)
# ==============================================================================
class OmegaDataset(Dataset):
    """List-backed dataset for paired (X[N_trials, F], omega[D]) tensors."""
    def __init__(self, X_list: List[np.ndarray], omega_list: List[np.ndarray]):
        self.X = [torch.tensor(x, dtype=torch.float32) for x in X_list]
        self.Y = [torch.tensor(y, dtype=torch.float32) for y in omega_list]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


@dataclass
class SimulationBlock:
    """Pairs of (parameters, encoder embeddings) used for SNPE rounds."""
    omegas: torch.Tensor
    embeds: torch.Tensor


class SummaryStandardizer:
    """Per-dim z-score standardiser fit on Round 1 embeddings, reused on Round 2."""
    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def train_encoder_on_simulations(
    X_list: List[np.ndarray],
    omega_list: List[np.ndarray],
    in_dim: int,
    omega_dim: int,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 137,
    patience: int = 10,
):
    """
    Pretrain the TrialTransformer with a tiny MLP regression head against
    z-scored omegas under MSE loss. Returns (frozen-ready encoder, reg_head).
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        print(f"  [Auto-Scale] Detected {n_gpu} GPUs!")
        print(f"   Batch size scaled from {batch_size} -> {batch_size * n_gpu}")
        batch_size = batch_size * n_gpu

    torch.manual_seed(seed); np.random.seed(seed)
    total_len = len(X_list)
    val_len = int(0.2 * total_len)
    train_len = total_len - val_len

    perm = np.random.permutation(total_len)
    train_idx, val_idx = perm[:train_len], perm[train_len:]

    X_train = [X_list[i] for i in train_idx]
    X_val = [X_list[i] for i in val_idx]

    Y_all = np.stack(omega_list, axis=0).astype(np.float32)
    y_mean = Y_all[train_idx].mean(axis=0, keepdims=True)
    y_std = Y_all[train_idx].std(axis=0, keepdims=True)
    y_std[y_std < 1e-6] = 1.0

    def scale_y(indices):
        return [(Y_all[i] - y_mean[0]) / y_std[0] for i in indices]

    train_ds = OmegaDataset(X_train, scale_y(train_idx))
    val_ds = OmegaDataset(X_val, scale_y(val_idx))

    n_workers = min(8, max(1, N_JOBS // 2))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=n_workers, pin_memory=(device.type == "cuda"))
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=n_workers, pin_memory=(device.type == "cuda"))

    encoder = TrialTransformer(in_dim=in_dim, out_dim=ENCODER_OUT_DIM, use_pos_enc=True).to(device)
    if n_gpu > 1:
        encoder = nn.DataParallel(encoder)
    reg_head = nn.Sequential(
        nn.Linear(ENCODER_OUT_DIM, 64),
        nn.ReLU(),
        nn.Linear(64, omega_dim),
    ).to(device)

    params = list(encoder.parameters()) + list(reg_head.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    print(f"\n [Encoder] Training on {device}...")
    for ep in range(epochs):
        encoder.train(); reg_head.train()
        train_loss = 0.0
        for Xb, Yb in train_dl:
            Xb, Yb = Xb.to(device), Yb.to(device)
            pred = reg_head(encoder(Xb))
            loss = loss_fn(pred, Yb)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item() * Xb.size(0)
        train_loss /= len(train_ds)

        encoder.eval(); reg_head.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, Yb in val_dl:
                Xb, Yb = Xb.to(device), Yb.to(device)
                pred = reg_head(encoder(Xb))
                loss = loss_fn(pred, Yb)
                val_loss += loss.item() * Xb.size(0)
        val_loss /= len(val_ds)

        print(f"Epoch {ep+1:03d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}", end="")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            raw_encoder = encoder.module if isinstance(encoder, nn.DataParallel) else encoder
            best_state = {
                "enc": {k: v.cpu().clone() for k, v in raw_encoder.state_dict().items()},
                "head": {k: v.cpu().clone() for k, v in reg_head.state_dict().items()},
            }
            print(" *")
        else:
            patience_counter += 1
            print(f" (Wait {patience_counter})")
            if patience_counter >= patience:
                print("Early stopping.")
                break

    if isinstance(encoder, nn.DataParallel):
        encoder = encoder.module
    if best_state:
        encoder.load_state_dict(best_state["enc"])
        reg_head.load_state_dict(best_state["head"])
    encoder.eval()
    return encoder, reg_head


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


def _parallel_simulate(omegas: torch.Tensor, seed_offset: int = 0) -> List[np.ndarray]:
    """Run the BAMCP simulator in parallel for a batch of omegas. Returns list of
    per-trial feature arrays of shape (N_TRIALS_TOTAL, 1)."""
    n = omegas.shape[0]
    print(f"  Launching parallel simulation ({n} sims, {N_JOBS} workers)...")
    return Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(worker_simulate)(i, omegas[i], seed_offset=seed_offset)
        for i in range(n)
    )


def simulate_round(sampler_fn, n_samples, encoder: TrialTransformer,
                   seed_offset: int = 0) -> SimulationBlock:
    """
    Simulate n_samples (omega, choices) pairs, then push each through the (frozen)
    encoder to produce embeddings. Returns SimulationBlock(omegas, embeds).

    The encoder is parked on CPU during the joblib parallel sim to avoid CUDA /
    multiprocessing conflicts, then moved back to GPU for embedding.
    """
    omegas = sampler_fn((n_samples,)).cpu()

    encoder.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    X_list = _parallel_simulate(omegas, seed_offset=seed_offset)

    encoder.to(device)
    encoder.eval()
    embeds = []
    with torch.no_grad():
        for x in X_list:
            tensor_x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, N, 1)
            emb = encoder(tensor_x.to(device)).cpu()                      # (1, out_dim)
            embeds.append(emb)
    embeds_tensor = torch.cat(embeds, dim=0)
    return SimulationBlock(omegas=omegas, embeds=embeds_tensor)


# ==============================================================================
# 6. PIPELINE STAGES
# ==============================================================================

def run_pretrain(args, prior):
    """
    Stage A: simulate fresh (omega, choices) and pretrain the TrialTransformer
    encoder under MSE loss to predict omega. Saves the frozen-ready encoder
    state_dict to ENC_PATH.
    """
    _load_sim_envs(args.env_id)

    print(f"\n [Pretrain] Simulating {args.n1_pre} pairs for encoder pretraining...")
    omegas = prior.sample((args.n1_pre,)).cpu()
    X_list = _parallel_simulate(omegas, seed_offset=0)
    Omega_list = [omegas[i].numpy() for i in range(args.n1_pre)]

    encoder, _ = train_encoder_on_simulations(
        X_list, Omega_list,
        in_dim=ENCODER_IN_DIM,
        omega_dim=len(PARAM_ORDER),
        epochs=args.epochs,
        patience=args.patience,
    )
    torch.save(encoder.state_dict(), ENC_PATH)
    print(f"  [Pretrain] Encoder saved to {ENC_PATH}")


def run_snpe(args, prior, encoder):
    """
    Stage B+C: with the frozen encoder, simulate Round 1 sims, embed, fit a
    per-dim z-score standardiser, then train sbi's SNPE flow on (omega,
    embedding_z). If args.n2 > 0, run a second active-learning round using
    a prior+posterior mixture proposal, transform via the SAME standardiser,
    concat, and retrain.

    The encoder is NOT passed to sbi as embedding_net — sbi sees the
    precomputed standardised embeddings as `x`.
    """
    _load_sim_envs(args.env_id)
    encoder = encoder.to(device)

    # --- Round 1 ---
    print(f"\n [Round1] Simulating {args.n1_pre} pairs...")
    block1 = simulate_round(prior.sample, args.n1_pre, encoder, seed_offset=0)

    stdzr = SummaryStandardizer()
    embeds_1_z = stdzr.fit_transform(block1.embeds.numpy())
    np.save(STD_MEAN_PATH, stdzr.mean_)
    np.save(STD_STD_PATH, stdzr.std_)

    print(f"  [Round1] Training SNPE ({args.density}) on {device}...")
    inference = SNPE(prior=prior, density_estimator=args.density, device=str(device))
    inference.append_simulations(block1.omegas, torch.tensor(embeds_1_z, dtype=torch.float32))
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

        # Reference observation for proposal sampling: mean of R1 z-scored embeddings.
        ref_x = torch.tensor(
            embeds_1_z.mean(axis=0, keepdims=True), dtype=torch.float32
        ).to(device)

        def proposal_sampler(shape):
            n = shape[0]
            n_prior = int(args.mix_prior_frac * n)
            n_post = n - n_prior
            post_samples = posterior_1.sample((n_post,), x=ref_x).reshape(n_post, -1)
            return torch.cat([
                prior.sample((n_prior,)).to(device),
                post_samples.to(device),
            ], dim=0)

        block2 = simulate_round(proposal_sampler, args.n2, encoder, seed_offset=1_000_000)
        embeds_2_z = stdzr.transform(block2.embeds.numpy())  # reuse R1 standardiser

        omegas_all = torch.cat([block1.omegas, block2.omegas], dim=0)
        embeds_all = torch.cat([
            torch.tensor(embeds_1_z, dtype=torch.float32),
            torch.tensor(embeds_2_z, dtype=torch.float32),
        ], dim=0)

        print(f"  [Round2] Training final posterior on {device}...")
        inference = SNPE(prior=prior, density_estimator=args.density, device=str(device))
        inference.append_simulations(omegas_all, embeds_all.float())
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


def _embed_observation(x_raw: np.ndarray, encoder: TrialTransformer,
                       stdzr: SummaryStandardizer) -> torch.Tensor:
    """Encode a single (N_trials, 1) observation through the frozen encoder and
    z-score with the Round 1 standardiser. Returns a (1, out_dim) tensor on `device`."""
    encoder.eval()
    with torch.no_grad():
        tensor_x = torch.tensor(x_raw[None, ...], dtype=torch.float32).to(device)  # (1, N, 1)
        z = encoder(tensor_x).cpu().numpy()                                        # (1, out_dim)
    z_z = stdzr.transform(z)
    return torch.tensor(z_z, dtype=torch.float32).to(device)


def run_recovery(args, prior, posterior, encoder, stdzr):
    """Validates the posterior against known ground-truth simulated cases."""
    _load_sim_envs(args.env_id)

    print(f"\n [Recovery] Checking {args.K} ground-truth cases...")
    omegas_true = prior.sample((args.K,)).cpu()

    encoder.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    X_list = _parallel_simulate(omegas_true, seed_offset=4242)

    encoder.to(device)
    encoder.eval()

    details = []
    for i in range(args.K):
        gt_params = untransform(omegas_true[i])
        x_obs = _embed_observation(X_list[i], encoder, stdzr)  # (1, out_dim) z-scored

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


def run_inference(args, posterior, encoder, stdzr):
    """Runs inference on real participant data from expt 3."""
    out_root = RUN_DIR / "subjects"
    out_root.mkdir(parents=True, exist_ok=True)
    encoder = encoder.to(device)

    df_all = pd.read_csv(str(PARTICIPANT_DATA_CSV), low_memory=False)
    pids = sorted(df_all["pid"].unique())

    print(f"\n [Inference] Processing {len(pids)} participants...")
    summaries = []

    for i, pid in enumerate(pids):
        if i % 10 == 0:
            print(f"   ... Progress: {i}/{len(pids)} participants", flush=True)
        try:
            df_sub = df_all[df_all["pid"] == pid]
            x_raw = build_features_from_participant(df_sub)        # (N, 1)
            if x_raw.shape[0] != N_TRIALS_TOTAL:
                print(f" [Skip] {pid}: {x_raw.shape[0]} trials != expected {N_TRIALS_TOTAL}")
                continue
            x_obs = _embed_observation(x_raw, encoder, stdzr)      # (1, out_dim) z-scored

            samples = posterior.sample((args.num_samples,), x=x_obs).cpu()

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


PPC_FIELDS = ["p_correct", "p_chose_orthogonal", "p_chose_more_future_rel_overlap"]


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


def run_ppc(args):
    """Posterior predictive check: simulate BAMCP at each pid's posterior-mean params
    on that pid's own env objects. Parallelised across pids."""
    post_csv = Path(args.post_csv) if args.post_csv else POST_SUMMARY_CSV
    post_df = pd.read_csv(post_csv)
    means = post_df.pivot_table(index="pid", columns="index", values="mean")

    df_all = pd.read_csv(str(PARTICIPANT_DATA_CSV), low_memory=False)
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
# 7. MAIN
# ==============================================================================
def main():
    warnings.filterwarnings("ignore")
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    SEED = 137
    np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

    parser = argparse.ArgumentParser(description="2-stage Transformer SNPE Pipeline for BAMCP")
    parser.add_argument("--stage", choices=["all", "pretrain", "snpe", "recover", "posterior", "ppc"], default="all")

    # Pretrain (encoder) args
    parser.add_argument("--n1_pre", type=int, default=20000, help="Simulations for pretraining/Round 1")
    parser.add_argument("--epochs", type=int, default=5, help="Encoder pretraining epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience")
    parser.add_argument("--n2", type=int, default=15000, help="SNPE Round 2 simulations (0 to skip)")

    # SNPE args
    parser.add_argument("--n_samples", type=int, default=FIXED_PARAMS["n_samples"],
                        help="BAMCP MCTS rollouts per decision (samples per simulated model)")
    parser.add_argument("--mix_prior_frac", type=float, default=0.2, help="Prior fraction in Round 2 proposal")
    parser.add_argument("--density", choices=["nsf", "maf", "mdn"], default="nsf", help="Density estimator")
    parser.add_argument("--env_id", type=int, default=1,
                        help="Env object ID to use for all simulations (same grids for every agent)")

    # Recovery args
    parser.add_argument("--K", type=int, default=20, help="Recovery test cases")
    parser.add_argument("--num_post", type=int, default=1000, help="Posterior samples per recovery case")

    # Inference args
    parser.add_argument("--num_samples", type=int, default=4000, help="Posterior samples per participant")

    # PPC args
    parser.add_argument("--post_csv", type=str, default=None,
                        help="Path to params_posteriors.csv for PPC (default: POST_SUMMARY_CSV)")

    args = parser.parse_args()

    global RUN_DIR, ENC_PATH, STD_MEAN_PATH, STD_STD_PATH
    global POST1_PATH, POSTF_PATH, RECOVERY_CSV, POST_SUMMARY_CSV
    FIXED_PARAMS["n_samples"] = args.n_samples
    RUN_DIR = ART_DIR / (
        f"{args.n1_pre}_n1sims_{args.n2}_n2sims_{args.n_samples}_samples"
    )
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    ENC_PATH = RUN_DIR / "encoder.pt"
    STD_MEAN_PATH = RUN_DIR / "embeds_mean.npy"
    STD_STD_PATH = RUN_DIR / "embeds_std.npy"
    POST1_PATH = RUN_DIR / "posterior_round1.pkl"
    POSTF_PATH = RUN_DIR / "posterior_final.pkl"
    RECOVERY_CSV = RUN_DIR / "params_recovery.csv"
    POST_SUMMARY_CSV = RUN_DIR / "params_posteriors.csv"
    print(f"[Setup] Run directory: {RUN_DIR}")

    prior, _, _ = make_box_prior()

    # --- Pipeline ---

    # 1. Stage A: pretrain encoder
    if args.stage in ["all", "pretrain"]:
        run_pretrain(args, prior)
        if args.stage == "pretrain":
            return

    # Load frozen encoder for any downstream stage that needs it.
    encoder = None
    if args.stage in ["all", "snpe", "recover", "posterior"]:
        if not ENC_PATH.exists():
            raise FileNotFoundError(
                f"Encoder weights not found at {ENC_PATH}. Run '--stage pretrain' first."
            )
        encoder = make_encoder()
        encoder.load_state_dict(torch.load(ENC_PATH, map_location=device))
        encoder.eval()
        encoder.to(device)
        for p in encoder.parameters():
            p.requires_grad = False

    # 2. Stage B+C: SNPE on frozen embeddings
    if args.stage in ["all", "snpe"]:
        run_snpe(args, prior, encoder)

    # 3. Load posterior + standardiser for downstream stages
    if args.stage in ["all", "recover", "posterior"]:
        with open(POSTF_PATH, "rb") as f:
            posterior = pickle.load(f)
        stdzr = SummaryStandardizer()
        stdzr.mean_ = np.load(STD_MEAN_PATH)
        stdzr.std_ = np.load(STD_STD_PATH)

    # 4. Recovery validation
    if args.stage in ["all", "recover"]:
        run_recovery(args, prior, posterior, encoder, stdzr)

    # 5. Inference on participant data
    if args.stage in ["all", "posterior"]:
        run_inference(args, posterior, encoder, stdzr)

    # 6. Posterior predictive check
    if args.stage in ["all", "ppc"]:
        run_ppc(args)

if __name__ == "__main__":
    main()