"""
tesbi_e2e.py

End-to-End Transformer-encoded Simulation-Based Inference (TeSBI) for BAMCP.

The Transformer encoder is passed to sbi as the `embedding_net`. sbi hands that *same* encoder to both the categorical
mass net (for the discrete `horizon`) and the conditional normalising flow (for the
continuous params), so one `inference.train()` call backprops into the encoder from
BOTH losses. The result is the correct factorised mixed posterior

    p(theta_c, theta_d | x) = p(theta_d | x) . p(theta_c | x, theta_d)

with the encoder's summary statistics learned for the posterior objective directly,
rather than a separate regression objective.

================================================================================
WORKFLOW
================================================================================
1) Simulate: draw omega from the prior, run BAMCP, save per-trial raw features
   (cached to disk).
2) Train: build the encoder + MNPE, feed flattened feature sequences as x, and
   jointly train encoder + categorical mass net + conditional flow. Save posterior.
3) Recovery: validate the posterior against known ground-truth simulated cases.
4) Inference: condition on observed participant data to sample the posterior.
5) PPC: posterior predictive check at each pid's posterior point estimate.

================================================================================
USAGE EXAMPLES
================================================================================
--- LOCAL SMOKE TEST ---
    python SBI/hpc/tesbi_e2e.py --stage simulate  --n_sims 30 --n_samples 50
    python SBI/hpc/tesbi_e2e.py --stage train     --n_sims 30 --n_samples 50 --epochs 3
    python SBI/hpc/tesbi_e2e.py --stage recover   --n_sims 30 --n_samples 50 --K 5 --num_post 50
    python SBI/hpc/tesbi_e2e.py --stage posterior --n_sims 30 --n_samples 50 --num_samples 50

--- HPC FULL RUN ---
    python SBI/hpc/tesbi_e2e.py --stage all --n_sims 30000 --n_samples 10000
================================================================================
"""
import sys
import os
import gc
import json
import random
import pickle
import argparse
import warnings
import multiprocessing
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Distribution, biject_to, constraints
from torch.distributions.transforms import CatTransform, IndependentTransform, identity_transform
from joblib import Parallel, delayed

import sbi
from sbi.inference import SNPE, MNPE
from sbi.neural_nets import posterior_nn
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
    # "temp": (0.0, 0.5), # ignore for now, fixed to 0.25
    "lapse": (0.0, 1.0),
    "aligned_weight": (0.0, 3.0),
    "orthogonal_weight":   (0.0, 3.0),
    "horizon": (0, 3),  # discrete: samples uniformly from {0, 1, 2, 3}
    }

PARAM_ORDER = [
                # "temp",
               "lapse",
                "aligned_weight",
               "orthogonal_weight",
               "horizon"
               ]

# Parameters that are discrete integer categoricals rather than continuous.
# When any of these appear in PARAM_ORDER, inference switches from SNPE to
# sbi's Mixed NPE (MNPE), which models the discrete dims with a categorical
# mass network and the continuous dims with a normalising flow.
DISCRETE_PARAMS = {"horizon"}


def discrete_param_names() -> List[str]:
    """Discrete entries of PARAM_ORDER, in order."""
    return [p for p in PARAM_ORDER if p in DISCRETE_PARAMS]


# MNPE requires the discrete columns to be the *trailing* columns of theta
# (continuous first, discrete last)
_disc_positions = [i for i, p in enumerate(PARAM_ORDER) if p in DISCRETE_PARAMS]
if _disc_positions:
    assert _disc_positions == list(range(len(PARAM_ORDER) - len(_disc_positions), len(PARAM_ORDER))), (
        f"MNPE requires discrete params {discrete_param_names()} to be the last entries of "
        f"PARAM_ORDER (continuous first, discrete last); got PARAM_ORDER={PARAM_ORDER}."
    )

# True when MNPE (mixed continuous/discrete inference) should be used.
USE_MNPE = len(discrete_param_names()) > 0

## check version can handle discrete params
if USE_MNPE and tuple(int(p) for p in sbi.__version__.split(".")[:2]) < (0, 26):
    raise ImportError(
        f"MNPE needs sbi>=0.26.0 to map discrete values to categories; found sbi {sbi.__version__}."
    )

# Restrict  discrete parameters to specific subsets of values
DISCRETE_PARAM_VALUES = {
    "horizon": [0, 3],
}


def discrete_values(name: str) -> List[int]:
    """Allowed values of a discrete param: its DISCRETE_PARAM_VALUES subset if
    given, otherwise every integer in its inclusive PARAM_RANGES range."""
    if name in DISCRETE_PARAM_VALUES:
        return [int(v) for v in DISCRETE_PARAM_VALUES[name]]
    lo, hi = PARAM_RANGES[name]
    return list(range(int(lo), int(hi) + 1))

FIXED_PARAMS = {
    "n_samples": 10000,
    "exploration_constant": 3,
    "discount_factor":   0.9,
    "temp": 0.25, ## override for now
    # "horizon": 3, ## override for now
    # 'orthogonal_weight': 1,  # override for now
    # 'lapse': 0,  # override for now
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

# ==============================================================================
# FEATURE SCHEMA
# ==============================================================================
# SAVED_FIELDS: the per-trial schema persisted to disk after each simulation.
# We actually save all features the simulator emits, but the encoder only sees a subset (FEATURES).
SAVED_FIELDS: Optional[List[str]] = None

# FEATURES: subset of SAVED_FIELDS that the encoder actually sees
FEATURES = [
    "chose_orthogonal",
    "trial",

    # "gen_net_costs_diff",
    # "actual_net_costs_diff",

    # 'aligned_path_actual_net_costs',
    # 'orthogonal_path_actual_net_costs',
    # 'aligned_path_gen_net_costs',
    # 'orthogonal_path_gen_net_costs',

    # 'aligned_path_aligned_arm_actual_net_costs',
    # 'orthogonal_path_aligned_arm_actual_net_costs',
    # 'aligned_path_aligned_arm_gen_net_costs',
    # 'orthogonal_path_aligned_arm_gen_net_costs',

    'aligned_path_aligned_arm_len',
    'orthogonal_path_aligned_arm_len',

    "objective"
]

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
ART_DIR = Path("SBI/outputs/e2e/")
RUN_DIR = ART_DIR  # overridden in main() once n_sims/n_samples are known
DATA_PATH = RUN_DIR / "sim_data.npy"
COLUMNS_PATH = RUN_DIR / "sim_columns.json"
OMEGA_PATH = RUN_DIR / "sim_omega.npy"
POSTF_PATH = RUN_DIR / "posterior_final.pkl"
RECOVERY_CSV = RUN_DIR / "params_recovery.csv"
POST_SUMMARY_CSV = RUN_DIR / "params_posteriors.csv"


# ==============================================================================
# 1. SIMULATOR WRAPPER
# ==============================================================================
def _to_trial_column(values) -> Optional[np.ndarray]:
    """Coerce a single sim_out value to a 1-D float32 column of length
    N_TRIALS_TOTAL, or return None if it isn't a per-trial numeric scalar
    (e.g. strings like `agent`, the `context` vector, or action paths)."""
    try:
        arr = np.asarray(values, dtype=np.float32)
    except (ValueError, TypeError):
        return None
    if arr.ndim != 1 or arr.shape[0] != N_TRIALS_TOTAL:
        return None
    return arr


def simulate_data(params: Dict[str, float], envs: Dict, seed: Optional[int] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Runs the BAMCP simulator for a single parameter set on the given envs.

    Saves *every* per-trial numeric scalar the simulator emits. Returns (raw, fields)
    where `raw` is a float32 matrix of shape (N_TRIALS_TOTAL, len(fields)) and
    `fields` names its columns
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
        horizon=params["horizon"],
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

    # Build the raw matrix and the list of column names
    columns = {}
    for k, v in sim_out.items():
        col = _to_trial_column(v)
        if col is not None:
            columns[k] = col

    fields = list(columns.keys())
    raw = np.stack([columns[k] for k in fields], axis=1).astype(np.float32)
    return raw, fields


# ==============================================================================
# 2. PARAMETER SETUP
# ==============================================================================
class _MixedSupport(constraints.Constraint):
    """Support of MixedPrior: the continuous box, times each discrete dim's
    allowed value set."""
    event_dim = 1

    def __init__(self, low: torch.Tensor, high: torch.Tensor, disc_values: List[torch.Tensor]):
        super().__init__()
        self.low, self.high, self.disc_values = low, high, disc_values

    def check(self, value):
        n_cont = self.low.numel()
        cont = value[..., :n_cont]
        ok = ((cont >= self.low) & (cont <= self.high)).all(dim=-1)
        for j, vals in enumerate(self.disc_values):
            ok = ok & (value[..., n_cont + j].unsqueeze(-1) == vals).any(dim=-1)
        return ok


@biject_to.register(_MixedSupport)
def _mixed_support_bijection(constraint: _MixedSupport):
    """sbi's DirectPosterior builds an unconstrained theta transform from the
    prior support (used for MAP / potentials, not for sampling): sigmoid-to-box
    for the continuous dims, identity for the discrete dims."""
    cont = biject_to(constraints.interval(constraint.low, constraint.high))
    return IndependentTransform(
        CatTransform([cont, identity_transform], dim=-1,
                     lengths=[constraint.low.numel(), len(constraint.disc_values)]),
        reinterpreted_batch_ndims=1,
    )


class MixedPrior(Distribution):
    """The prior omegas are actually simulated from: BoxUniform over the
    continuous entries of PARAM_ORDER, times an independent uniform categorical
    over each discrete entry's allowed values (see `discrete_values`).

    A plain BoxUniform would give the discrete dims a continuous density over
    [lo, hi], i.e. put mass on values (e.g. horizon 1, 2) that are never
    simulated, and assign log_prob -inf at the upper bound (e.g. horizon 3).
    Discrete dims must be trailing (asserted at the top of the file).
    """
    arg_constraints = {}

    def __init__(self, low: torch.Tensor, high: torch.Tensor, disc_values: List[torch.Tensor]):
        self.cont = BoxUniform(low=low, high=high)
        self.low, self.high, self.disc_values = low, high, disc_values
        dim = low.numel() + len(disc_values)
        super().__init__(batch_shape=torch.Size(), event_shape=torch.Size([dim]), validate_args=False)

    @property
    def support(self):
        return _MixedSupport(self.low, self.high, self.disc_values)

    @property
    def mean(self):
        return torch.cat([self.cont.mean] + [v.mean().reshape(1) for v in self.disc_values])

    @property
    def stddev(self):
        return torch.cat([self.cont.stddev] + [v.std(unbiased=False).reshape(1) for v in self.disc_values])

    @property
    def variance(self):
        return self.stddev ** 2

    def sample(self, sample_shape=torch.Size()):
        sample_shape = torch.Size(sample_shape)
        cols = [self.cont.sample(sample_shape)]
        for vals in self.disc_values:
            idx = torch.randint(len(vals), sample_shape, device=vals.device)
            cols.append(vals[idx].unsqueeze(-1))
        return torch.cat(cols, dim=-1)

    def log_prob(self, value):
        n_cont = self.low.numel()
        lp = self.cont.log_prob(value[..., :n_cont])
        for j, vals in enumerate(self.disc_values):
            hit = (value[..., n_cont + j].unsqueeze(-1) == vals).any(dim=-1)
            lp = lp + torch.where(hit, -np.log(len(vals)), -np.inf)
        return lp


def make_prior() -> MixedPrior:
    """Constructs the prior from PARAM_RANGES (continuous params) and
    `discrete_values` (discrete params)."""
    cont = [k for k in PARAM_ORDER if k not in DISCRETE_PARAMS]
    low = torch.tensor([PARAM_RANGES[k][0] for k in cont], dtype=torch.float32, device=device)
    high = torch.tensor([PARAM_RANGES[k][1] for k in cont], dtype=torch.float32, device=device)
    disc_values = [torch.tensor(discrete_values(k), dtype=torch.float32, device=device)
                   for k in discrete_param_names()]
    return MixedPrior(low, high, disc_values)


def sample_prior(prior: MixedPrior, shape) -> torch.Tensor:
    """Draw omegas from the prior (takes a shape tuple, e.g. ``(n,)``).
    Returns a CPU tensor."""
    return prior.sample(shape).cpu()


def untransform(omega_vec: torch.Tensor) -> Dict[str, float]:
    """Converts a theta vector back to a simulator-ready parameter dict."""
    vals = omega_vec.detach().cpu().numpy().astype(float)
    out = {}
    for i, k in enumerate(PARAM_ORDER):
        v = vals[i]
        out[k] = int(round(v)) if k in DISCRETE_PARAMS else float(v)
    out.update(FIXED_PARAMS)
    return out


# ==============================================================================
# 3. FEATURE ENGINEERING
# ==============================================================================
def _set_saved_fields(fields: List[str]) -> None:
    """Record the column schema discovered at simulate/load time so the
    `_saved_field_index` lookup below resolves against the on-disk layout."""
    global SAVED_FIELDS
    SAVED_FIELDS = list(fields)


def _saved_field_index(name: str) -> int:
    """Column index of `name` in the saved raw matrix."""
    if SAVED_FIELDS is None:
        raise RuntimeError(
            "SAVED_FIELDS is not populated yet — simulate or load a dataset "
            "(which records the column schema) before building features."
        )
    return SAVED_FIELDS.index(name)


def build_features_from_sim(raw: np.ndarray) -> np.ndarray:
    """
    Materialise the encoder input X from a per-simulation raw matrix.
    Input:  raw of shape (N_TRIALS_TOTAL, len(SAVED_FIELDS)).
    Output: float32 array of shape (N_TRIALS_TOTAL, len(FEATURES)).
    """
    cols = [raw[:, _saved_field_index(name)] for name in FEATURES]
    return np.stack(cols, axis=1).astype(np.float32)


# Fill value marking a missed trial in x (every feature of that trial set to it).
MISSED_TRIAL = -999.0


def build_features_from_participant(df_participant: pd.DataFrame) -> np.ndarray:
    """
    Build the encoder input X from a single participant's preprocessed CSV.
    Pulls each FEATURES column straight from the DataFrame (utils.py already
    computes all *_diff columns for the participant pipeline).
    Output: float32 array of shape (N_TRIALS_TOTAL, len(FEATURES)).

    Missed trials (no `path_chosen`) are returned as rows filled with
    MISSED_TRIAL (sbi rejects non-finite x, so NaN can't be used); SeqEmbedding
    masks them out of attention and pooling.
    """
    df_sorted = df_participant.sort_values(["city", "day", "trial"])
    cols = []
    for name in FEATURES:
        if name == "chose_orthogonal":
            cols.append((df_sorted["chose_orthogonal"] == True).values.astype(np.float32))
        elif name == 'objective':
            cols.append(df_sorted[name].map({'costs': -1, 'rewards': 1}).values.astype(np.float32))
        else:
            cols.append(df_sorted[name].values.astype(np.float32))
    X = np.stack(cols, axis=1).astype(np.float32)
    X[df_sorted["path_chosen"].isna().values] = MISSED_TRIAL
    return X


# ==============================================================================
# 4. TRANSFORMER ENCODER (trained jointly as the sbi embedding_net)
# ==============================================================================

## positional encoding used to give the transformer a sense of trial order
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

    def forward(self, x, pad_mask: Optional[torch.Tensor] = None):
        """pad_mask: optional (B, N_trials) bool, True for trials to ignore
        (missed trials). They are excluded from attention and from pooling."""
        h = self.proj(x)
        if self.use_pos:
            h = self.pe(h)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        if pad_mask is None:
            h = h.mean(dim=1)  # mean pool across trials
        else:
            keep = (~pad_mask).unsqueeze(-1).to(h.dtype)
            h = (h * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)  # mean over kept trials
        return self.head(h)


# Encoder hyperparameters. in_dim is derived from FEATURES; change FEATURES (top
# of file) to alter it.
ENCODER_IN_DIM = len(FEATURES)
ENCODER_OUT_DIM = 64


class SeqEmbedding(nn.Module):
    """sbi `embedding_net` wrapper around TrialTransformer.

    sbi conditions on x as a 2-D batch, so we feed it flattened per-trial
    features of shape (B, N_TRIALS_TOTAL * F) and reshape back to
    (B, N_TRIALS_TOTAL, F) here before running the transformer, so sbi's shape
    handling sees a plain 2-D tensor while the encoder still sees the trial
    sequence. Output: (B, ENCODER_OUT_DIM).

    Standardisation happens here, per *feature* (stats pooled over sims x trials,
    stored as buffers so they travel with the pickled posterior), so sbi's own
    x z-scoring must be off (z_score_x="none"). sbi's "structured" would use one
    scalar for all features; "independent" would use one per (trial, feature).

    Trials whose features all equal MISSED_TRIAL are masked out of attention
    and pooling (see build_features_from_participant).
    """
    def __init__(self, n_trials: int, n_features: int, out_dim: int = ENCODER_OUT_DIM,
                 feat_mean: Optional[torch.Tensor] = None, feat_std: Optional[torch.Tensor] = None):
        super().__init__()
        self.n_trials = n_trials
        self.n_features = n_features
        self.register_buffer("feat_mean", torch.zeros(n_features) if feat_mean is None else feat_mean.float().clone())
        self.register_buffer("feat_std", torch.ones(n_features) if feat_std is None else feat_std.float().clamp_min(1e-6).clone())
        self.encoder = TrialTransformer(in_dim=n_features, out_dim=out_dim, use_pos_enc=True)

    def forward(self, x):
        # sbi calls the embedding net with 2-D (batch, N*F) during
        # training/log_prob, but the MNPE categorical sampler passes a 3-D
        # (sample_dim, batch, N*F) context. Preserve all leading dims and only
        # unflatten the trailing feature axis into (N_trials, F).
        lead = x.shape[:-1]
        x = x.reshape(-1, self.n_trials, self.n_features)
        missed = (x == MISSED_TRIAL).all(dim=-1)  # (B, N_trials)
        x = (x - self.feat_mean) / self.feat_std
        x = x.masked_fill(missed.unsqueeze(-1), 0.0)
        out = self.encoder(x, pad_mask=missed if missed.any() else None)
        return out.reshape(*lead, out.shape[-1])


def make_embedding_net(x_train: torch.Tensor) -> SeqEmbedding:
    """Construct a fresh flatten-aware encoder for use as the sbi embedding_net,
    with per-feature standardisation stats taken from the flattened training x
    of shape (n_sims, N_TRIALS_TOTAL * F). x_train must be finite (checked in
    run_train)."""
    seq = x_train.reshape(-1, ENCODER_IN_DIM)
    seq = seq[~(seq == MISSED_TRIAL).all(dim=-1)]
    return SeqEmbedding(N_TRIALS_TOTAL, ENCODER_IN_DIM, ENCODER_OUT_DIM,
                        feat_mean=seq.mean(dim=0), feat_std=seq.std(dim=0))


# ==============================================================================
# 5. SIMULATION WORKER
# ==============================================================================

_ENV_OBJECTS_CACHE: Dict[int, object] = {}

def _load_env_objects_cached(env_id: int):
    """Per-worker cache so each process pays the disk cost at most once per env."""
    cached = _ENV_OBJECTS_CACHE.get(env_id)
    if cached is None:
        cached = load_env_objects(env_id)
        _ENV_OBJECTS_CACHE[env_id] = cached
    return cached


def _sample_env_ids(n: int, seed: int = 0) -> List[int]:
    """Assign env_ids to n simulations by cycling through a shuffled list of
    participant pids and mapping each via _env_id_for_pid. Lets the encoder see
    a variety of environments rather than one fixed grid."""
    df_all = pd.read_csv(str(PARTICIPANT_DATA_CSV), low_memory=False)
    id_mapping = _load_id_mapping()
    pids = [p for p in df_all["pid"].unique().tolist() if p in id_mapping]
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)
    return [int(_env_id_for_pid(id_mapping, pids[i % len(pids)])) for i in range(n)]


def worker_simulate(i, omega, env_id, seed_offset=0):
    """
    Simulate one parameter set on the assigned env set.
    Returns (raw, fields): the per-trial raw matrix and its column names.
    Called by joblib Parallel — must be a top-level function.
    """
    params = untransform(omega)
    envs = _load_env_objects_cached(int(env_id))
    return simulate_data(params, envs, seed=seed_offset + i)


def _parallel_simulate(omegas: torch.Tensor, seed_offset: int = 0) -> Tuple[List[np.ndarray], List[str]]:
    """Run the BAMCP simulator in parallel for a batch of omegas. Returns
    (raw_list, fields): a list of per-trial raw arrays and the shared column
    names (identical across sims, taken from the first result)."""
    n = omegas.shape[0]
    env_ids = _sample_env_ids(n, seed=seed_offset)
    print(f"  Launching parallel simulation ({n} sims, {N_JOBS} workers, "
          f"{len(set(env_ids))} unique env_ids)...")
    tasks = [
        delayed(worker_simulate)(int(i), omegas[i].clone(), int(env_ids[i]), int(seed_offset))
        for i in range(n)
    ]
    results = Parallel(n_jobs=N_JOBS, verbose=1)(tasks)
    raw_list = [raw for raw, _ in results]
    fields = results[0][1]
    return raw_list, fields




def _sim_config(n_sims: int) -> Dict:
    """Everything that determines what a cached simulation dataset contains."""
    return {
        "n_sims": int(n_sims),
        "param_order": PARAM_ORDER,
        "param_ranges": {k: list(v) for k, v in PARAM_RANGES.items()},
        "discrete_param_values": DISCRETE_PARAM_VALUES,
        "fixed_params": FIXED_PARAMS,
        "hyperparams": HYPERPARAMS,
    }


def _simulate_or_load(
    omegas: torch.Tensor, seed_offset: int,
    data_path: Optional[Path], columns_path: Optional[Path], omega_path: Optional[Path],
    force: bool = False,
) -> Tuple[List[np.ndarray], torch.Tensor]:
    """
    Return (raw_list, omegas) where each entry of raw_list is the per-sim raw
    matrix of shape (N_TRIALS_TOTAL, len(SAVED_FIELDS)). The column schema is
    discovered from the simulator (or read back from the cached sidecar) and
    recorded in the SAVED_FIELDS global.

    If the cached data, columns, and omega files all exist (and force is False)
    AND the cached schema still contains every column FEATURES needs, load from
    disk and skip simulation. Otherwise resim, save the new (N, T, F_full)
    array plus a JSON sidecar naming the columns.

    The cache must also carry a `sim_config.json` matching `_sim_config` (prior,
    fixed params, ...). A missing or mismatched config raises
    rather than silently reusing (or silently re-running hours of) simulations;
    pass --resim to overwrite it.
    """
    sim_config_path = data_path.with_name("sim_config.json") if data_path is not None else None
    expected_cfg = json.loads(json.dumps(_sim_config(omegas.shape[0])))
    cache_ok = (
        not force
        and data_path is not None
        and columns_path is not None
        and omega_path is not None
        and data_path.exists()
        and columns_path.exists()
        and omega_path.exists()
    )
    if cache_ok:
        cached_cfg = None
        if sim_config_path.exists():
            with open(sim_config_path, "r") as f:
                cached_cfg = json.load(f)
        if cached_cfg != expected_cfg:
            diff = (
                "no sim_config.json (dataset predates cache validation)" if cached_cfg is None
                else {k: (cached_cfg.get(k), expected_cfg[k]) for k in expected_cfg if cached_cfg.get(k) != expected_cfg[k]}
            )
            raise RuntimeError(
                f"Cached simulations in {data_path.parent} do not match the current "
                f"simulation config: {diff}. Re-run with --resim (or use a fresh run dir)."
            )
        with open(columns_path, "r") as f:
            cached_cols = json.load(f)
        missing = [f for f in FEATURES if f not in cached_cols]
        if not missing:
            print(f"  [Cache] Loading simulations from {data_path.name} / {omega_path.name}")
            _set_saved_fields(cached_cols)
            raw_arr = np.load(data_path)
            raw_list = [raw_arr[i] for i in range(len(raw_arr))]
            omegas = torch.tensor(np.load(omega_path), dtype=torch.float32)
            return raw_list, omegas
        print(f"  [Cache] Cached schema in {columns_path.name} is missing columns "
              f"required by FEATURES {missing}; resimulating.")

    raw_list, fields = _parallel_simulate(omegas, seed_offset=seed_offset)
    _set_saved_fields(fields)
    if data_path is not None and columns_path is not None and omega_path is not None:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(data_path, np.stack(raw_list))
        np.save(omega_path, omegas.numpy())
        with open(columns_path, "w") as f:
            json.dump(fields, f, indent=2)
        with open(sim_config_path, "w") as f:
            json.dump(expected_cfg, f, indent=2)
        print(f"  [Cache] Saved data -> {data_path.name}, omega -> {omega_path.name}, "
              f"columns -> {columns_path.name}")
    return raw_list, omegas


# ==============================================================================
# 6. PIPELINE STAGES
# ==============================================================================

def _features_matrix(raw_list: List[np.ndarray]) -> torch.Tensor:
    """Build the flattened feature tensor x of shape (n_sims, N_TRIALS_TOTAL * F)
    that sbi conditions on (SeqEmbedding reshapes it back to a trial sequence)."""
    X = np.stack([build_features_from_sim(r).reshape(-1) for r in raw_list])
    return torch.tensor(X, dtype=torch.float32)


def run_simulate(args, prior):
    """Stage 1: draw omegas, run BAMCP, cache the raw per-trial features."""
    print(f"\n [Simulate] Simulating {args.n_sims} (omega, choices) pairs...")
    omegas = sample_prior(prior, (args.n_sims,))
    _simulate_or_load(
        omegas, 0, DATA_PATH, COLUMNS_PATH, OMEGA_PATH, force=args.resim,
    )
    print("  [Simulate] Done.")


def run_train(args, prior):
    """
    Stage 2: build the encoder + MNPE and train END-TO-END.

    The Transformer encoder is passed to sbi as `embedding_net`. sbi wires the
    same encoder into both the categorical mass net (discrete `horizon`) and the
    conditional flow (continuous params), so one `inference.train()` call jointly
    optimises encoder + categorical + flow under the posterior NLL.
    """
    omegas = sample_prior(prior, (args.n_sims,))  # only used if cache is cold
    # With --stage all, run_simulate has already honoured --resim; forcing again
    # here would throw away that dataset and simulate everything a second time.
    raw_list, omegas = _simulate_or_load(
        omegas, 0, DATA_PATH, COLUMNS_PATH, OMEGA_PATH,
        force=args.resim and args.stage == "train",
    )
    x = _features_matrix(raw_list)  # (n_sims, N_TRIALS_TOTAL * F)
    print(f"  [Train] x shape {tuple(x.shape)} (flattened {N_TRIALS_TOTAL} trials x {len(FEATURES)} features)")

    # Unyoked simulations never produce NaN features, so a non-finite x means a
    # simulator/feature bug. Fail here rather than let sbi silently drop those
    # sims (exclude_invalid_x), which would hide the bug and could bias training.
    bad = ~torch.isfinite(x.reshape(x.shape[0], N_TRIALS_TOTAL, len(FEATURES)))
    if bad.any():
        per_feat = {f: int(bad[..., j].any(dim=1).sum()) for j, f in enumerate(FEATURES) if bad[..., j].any()}
        raise ValueError(
            f"{int(bad.any(dim=(1, 2)).sum())}/{x.shape[0]} simulations have non-finite features "
            f"(sims affected per feature: {per_feat}). Simulated data should never contain NaN/inf."
        )

    embedding_net = make_embedding_net(x)
    n_enc_params = sum(p.numel() for p in embedding_net.parameters())
    print(f"  [Train] Encoder embedding_net params: {n_enc_params:,}")

    trainer_name = "MNPE" if USE_MNPE else f"SNPE ({args.density})"
    density_estimator = posterior_nn(
        model="mnpe" if USE_MNPE else args.density,
        embedding_net=embedding_net,
        z_score_x="none",             # x is standardised per feature inside SeqEmbedding
        z_score_theta="independent",
    )
    inference = (MNPE if USE_MNPE else SNPE)(
        prior=prior, density_estimator=density_estimator, device=str(device)
    )

    print(f"  [Train] Training {trainer_name} end-to-end on {device}...")
    inference.append_simulations(omegas, x)
    estimator = inference.train(
        stop_after_epochs=args.stop_after_epochs,
        max_num_epochs=args.epochs,
        show_train_summary=True,
    )
    posterior = inference.build_posterior(estimator)

    POSTF_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(POSTF_PATH, "wb") as f:
        pickle.dump(posterior, f)
    print(f"  [Train] Posterior saved to {POSTF_PATH}")


def run_recovery(args, prior, posterior):
    """Validates the posterior against known ground-truth simulated cases."""
    print(f"\n [Recovery] Checking {args.K} ground-truth cases...")
    omegas_true = sample_prior(prior, (args.K,))

    raw_list, fields = _parallel_simulate(omegas_true, seed_offset=4242)
    _set_saved_fields(fields)

    details = []
    for i in range(args.K):
        gt_params = untransform(omegas_true[i])
        x_feat = build_features_from_sim(raw_list[i]).reshape(-1)          # (N*F,)
        x_obs = torch.tensor(x_feat[None, :], dtype=torch.float32).to(device)

        samples = posterior.sample((args.num_post,), x=x_obs).cpu()

        row = {"case": i}
        samples_np = samples.numpy()
        for k_idx, k in enumerate(PARAM_ORDER):
            vals = samples_np[:, k_idx]
            gt_val = gt_params[k]
            row[f"gt_{k}"] = gt_val
            if k in DISCRETE_PARAMS:
                # Categorical recovery: mean/percentile-CI are meaningless for a
                # discrete param. Report the modal estimate, whether it matches
                # the true category (hard accuracy), and the posterior mass the
                # inference put on the true category (soft / calibration metric).
                cats = np.array(discrete_values(k))
                pmf = np.array([(np.round(vals) == c).mean() for c in cats])
                gt_cat = int(round(gt_val))
                row[f"mode_{k}"] = int(cats[np.argmax(pmf)])
                row[f"correct_{k}"] = 1.0 if row[f"mode_{k}"] == gt_cat else 0.0
                row[f"p_true_{k}"] = float(pmf[cats == gt_cat].sum())
            else:
                mu = np.mean(vals)
                lo_q, hi_q = np.percentile(vals, 5), np.percentile(vals, 95)
                row[f"mu_{k}"] = mu
                row[f"hit90_{k}"] = 1.0 if lo_q <= gt_val <= hi_q else 0.0
        details.append(row)

    df_rec = pd.DataFrame(details)
    df_rec.to_csv(RECOVERY_CSV, index=False)
    print(f"  [Recovery] Saved to {RECOVERY_CSV}")
    for k in discrete_param_names():
        if f"correct_{k}" in df_rec.columns:
            print(f"  [Recovery] {k}: modal accuracy = {df_rec[f'correct_{k}'].mean():.3f}, "
                  f"mean P(true category) = {df_rec[f'p_true_{k}'].mean():.3f}")


def run_inference(args, posterior):
    """Runs inference on real participant data from expt 3."""
    out_root = RUN_DIR / "subjects"
    out_root.mkdir(parents=True, exist_ok=True)

    df_all = pd.read_csv(str(PARTICIPANT_DATA_CSV), low_memory=False)
    pids = sorted(df_all["pid"].unique())

    print(f"\n [Inference] Processing {len(pids)} participants...")
    summaries = []

    for i, pid in enumerate(pids):
        if i % 10 == 0:
            print(f"   ... Progress: {i}/{len(pids)} participants", flush=True)
        try:
            df_sub = df_all[df_all["pid"] == pid]
            x_feat = build_features_from_participant(df_sub)       # (N, F)
            if x_feat.shape[0] != N_TRIALS_TOTAL:
                print(f" [Skip] {pid}: {x_feat.shape[0]} trials != expected {N_TRIALS_TOTAL}")
                continue
            x_obs = torch.tensor(x_feat.reshape(-1)[None, :], dtype=torch.float32).to(device)

            samples = posterior.sample((args.num_samples,), x=x_obs).cpu()

            rows = [untransform(s) for s in samples]
            post_df = pd.DataFrame(rows)
            summ = post_df.describe(
                percentiles=[0.05, 0.5, 0.95]
            ).T[["mean", "std", "5%", "50%", "95%"]]

            # For discrete params, record the full categorical posterior, not
            # just summary stats (mean/median are not valid categories). `mode`
            # is the most probable category (the point estimate used by PPC);
            # `p_{c}` columns hold the inferred probability of each category c
            # over its allowed values (zero-count categories -> 0.0).
            # All NaN for continuous params, which use the mean downstream.
            summ["mode"] = np.nan
            for k in discrete_param_names():
                if k not in post_df.columns:
                    continue
                summ.loc[k, "mode"] = int(post_df[k].mode().iloc[0])
                probs = post_df[k].value_counts(normalize=True)
                for c in discrete_values(k):
                    summ.loc[k, f"p_{c}"] = float(probs.get(c, 0.0))

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


def worker_ppc(pid, env_id, params, seed, draw=0):
    """Load this pid's envs, run BAMCP at `params`, return a long-format DataFrame
    with all simulated trial fields, tagged with pid and posterior draw index."""
    if seed is not None:
        np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)

    envs = load_env_objects(env_id)

    bamcp = BAMCP(
        mcts_class=MonteCarloTreeSearch_AFC,
        run_fn=run_grid,
        temp=params["temp"],
        lapse=params["lapse"],
        horizon=params["horizon"],
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

    df = pd.DataFrame(dict(sim_out))
    # The simulated (realised) choice would otherwise collide with the human
    # `chose_orthogonal` when ppc.csv is merged with the participant df.
    df = df.rename(columns={"chose_orthogonal": "sim_chose_orthogonal"})
    df["pid"] = pid
    df["draw"] = draw
    return df


def _ppc_param_sets(samples: pd.DataFrame, n_draws: int, rng: np.random.Generator) -> List[Dict]:
    """Parameter sets to simulate for one pid, from its joint posterior samples.

    n_draws > 0: n_draws joint draws (a proper posterior predictive).
    n_draws == 0: a single point estimate that respects the mixed structure:
        the modal discrete configuration, and the continuous posterior mean
        *conditional on* that configuration (the unconditional mean would
        average over horizons and may sit where the posterior has little mass).
    """
    disc = discrete_param_names()
    cont = [k for k in PARAM_ORDER if k not in DISCRETE_PARAMS]

    def _as_params(row) -> Dict:
        out = {k: int(round(row[k])) for k in disc}
        out.update({k: float(row[k]) for k in cont})
        out.update(FIXED_PARAMS)
        return out

    if n_draws > 0:
        idx = rng.choice(len(samples), size=n_draws, replace=len(samples) < n_draws)
        return [_as_params(samples.iloc[j]) for j in idx]

    point = {}
    sub = samples
    if disc:
        modal = samples.groupby(disc).size().idxmax()
        modal = modal if isinstance(modal, tuple) else (modal,)
        point.update(dict(zip(disc, modal)))
        sub = samples[(samples[disc] == pd.Series(point)).all(axis=1)]
    point.update(sub[cont].mean().to_dict())
    return [_as_params(point)]


def run_ppc(args):
    """Posterior predictive check on each pid's own env objects, simulated from
    the joint posterior samples saved by run_inference (see _ppc_param_sets for
    point-estimate vs. joint-draw modes). Parallelised across pids x draws."""
    subjects_dir = Path(args.subjects_dir) if args.subjects_dir else RUN_DIR / "subjects"
    df_all = pd.read_csv(str(PARTICIPANT_DATA_CSV), low_memory=False)
    pids = sorted(df_all["pid"].unique())
    id_mapping = _load_id_mapping()
    rng = np.random.default_rng(4242)

    tasks = []
    for i, pid in enumerate(pids):
        samples_path = subjects_dir / pid / "posterior_samples.csv"
        if not samples_path.exists():
            print(f"   [skip] {pid}: no {samples_path}"); continue
        if pid not in id_mapping:
            print(f"   [skip] {pid}: not in id_mapping"); continue
        env_id = _env_id_for_pid(id_mapping, pid)
        param_sets = _ppc_param_sets(pd.read_csv(samples_path), args.ppc_draws, rng)
        for d, params in enumerate(param_sets):
            tasks.append((pid, env_id, params, 4242 + i * len(param_sets) + d, d))

    mode = f"{args.ppc_draws} joint posterior draws" if args.ppc_draws > 0 else "conditional point estimate"
    print(f"\n [PPC] Simulating {len(tasks)} runs ({mode}) in parallel ({N_JOBS} workers)...")
    results = Parallel(n_jobs=N_JOBS, verbose=1)(
        delayed(worker_ppc)(pid, env_id, params, seed, draw)
        for (pid, env_id, params, seed, draw) in tasks
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

    parser = argparse.ArgumentParser(description="End-to-End Transformer MNPE Pipeline for BAMCP")
    parser.add_argument("--stage", choices=["all", "simulate", "train", "recover", "posterior", "ppc"], default="all")

    # Simulate / train args
    parser.add_argument("--n_sims", type=int, default=30000, help="Number of simulated (omega, x) pairs")
    parser.add_argument("--epochs", type=int, default=500, help="Max training epochs")
    parser.add_argument("--stop_after_epochs", type=int, default=20, help="Early-stopping patience (epochs)")
    parser.add_argument("--resim", action="store_true", help="Re-simulate even if cached data/omega files exist")

    parser.add_argument("--n_samples", type=int, default=FIXED_PARAMS["n_samples"],
                        help="BAMCP MCTS rollouts per decision (samples per simulated model)")
    parser.add_argument("--density", choices=["nsf", "maf", "mdn"], default="nsf",
                        help="Density estimator when there are no discrete params (SNPE)")

    # Recovery args
    parser.add_argument("--K", type=int, default=500, help="Recovery test cases")
    parser.add_argument("--num_post", type=int, default=1000, help="Posterior samples per recovery case")

    # Inference args
    parser.add_argument("--num_samples", type=int, default=4000, help="Posterior samples per participant")

    # PPC args
    parser.add_argument("--subjects_dir", type=str, default=None,
                        help="Dir of per-pid posterior_samples.csv for PPC (default: RUN_DIR/subjects)")
    parser.add_argument("--ppc_draws", type=int, default=0,
                        help="Joint posterior draws simulated per pid; 0 = single conditional point estimate")

    args = parser.parse_args()

    global RUN_DIR, DATA_PATH, COLUMNS_PATH, OMEGA_PATH
    global POSTF_PATH, RECOVERY_CSV, POST_SUMMARY_CSV
    FIXED_PARAMS["n_samples"] = args.n_samples
    RUN_DIR = ART_DIR / f"{args.n_sims}_sims_{args.n_samples}_samples"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PATH = RUN_DIR / "sim_data.npy"
    COLUMNS_PATH = RUN_DIR / "sim_columns.json"
    OMEGA_PATH = RUN_DIR / "sim_omega.npy"
    POSTF_PATH = RUN_DIR / "posterior_final.pkl"
    RECOVERY_CSV = RUN_DIR / "params_recovery.csv"
    POST_SUMMARY_CSV = RUN_DIR / "params_posteriors.csv"
    print(f"[Setup] Run directory: {RUN_DIR}")

    run_config = {
        "param_order": PARAM_ORDER,
        "param_ranges": PARAM_RANGES,
        "fixed_params": FIXED_PARAMS,
        "discrete_params": discrete_param_names(),
        "trainer": "mnpe" if USE_MNPE else "snpe",
        "features": FEATURES,
        "encoder_out_dim": ENCODER_OUT_DIM,
        "n_sims": args.n_sims,
        "n_samples": args.n_samples,
        "density": args.density,
        "pipeline": "end_to_end",
    }
    with open(RUN_DIR / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"[Setup] Run config saved to {RUN_DIR / 'run_config.json'}")

    prior = make_prior()

    # --- Pipeline ---

    # 1. Simulate (pure CPU)
    if args.stage in ["all", "simulate"]:
        run_simulate(args, prior)

    # Clear memory before GPU tasks (mirrors the reference e2e ordering).
    if args.stage == "all":
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 2. Train end-to-end (GPU)
    if args.stage in ["all", "train"]:
        run_train(args, prior)

    # 3. Load posterior for downstream stages
    posterior = None
    if args.stage in ["all", "recover", "posterior"]:
        if not POSTF_PATH.exists():
            raise FileNotFoundError(
                f"Posterior not found at {POSTF_PATH}. Run '--stage train' first."
            )
        with open(POSTF_PATH, "rb") as f:
            posterior = pickle.load(f)

    # 4. Recovery validation
    if args.stage in ["all", "recover"]:
        run_recovery(args, prior, posterior)

    # 5. Inference on participant data
    if args.stage in ["all", "posterior"]:
        run_inference(args, posterior)

    # 6. Posterior predictive check
    if args.stage in ["all", "ppc"]:
        run_ppc(args)


if __name__ == "__main__":
    main()
