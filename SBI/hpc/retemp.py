"""
retemp.py

Re-derive SBI synthetic choices at a *new softmax temperature* without re-running
the (expensive) BAMCP simulator.

The cached datasets produced by tesbi.py (`pretrain_data.npy`, `snpe_r1_data.npy`,
optionally `snpe_r2_data.npy`) store every per-trial scalar the simulator emitted,
including the per-option Q-values (`Q_a`, `Q_b`), the choice probabilities
(`p_choice_A`, `p_choice_B`, `p_chose_orthogonal`) and the per-trial `lapse`/`temp`.

Since the agent's choice rule is purely a function of the Q-values, temp and lapse
(agents.Farmer.softmax):

    CP = (1 - lapse) * softmax([Q_a, Q_b] / temp) + lapse / n_afc      (n_afc = 2)

we can analytically recompute the choice probabilities at a new temperature and
re-sample the binary choices, then write a fresh set of `*_data.npy` into a new
run dir. Everything else (Q-values, costs, omega) is temperature-independent and
copied through unchanged.

NOTE: the new choices change the encoder input (FEATURES includes
`chose_orthogonal`), so the encoder and posteriors are intentionally NOT copied —
re-run the tesbi.py pretrain/snpe stages on the new dir to retrain them.

USAGE
-----
    python retemp.py --run_dir 30000_n1sims_0_n2sims_10001_samples --temp 0.5
    python retemp.py --run_dir <name> --temp 0.5 --out_dir <name> --seed 137 --force
"""
import sys
import json
import shutil
import argparse
from pathlib import Path

import numpy as np
from scipy.special import softmax

# Base artifact directory, mirroring tesbi.py's ART_DIR (relative to project root).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ART_DIR = PROJECT_ROOT / "SBI/outputs/2_step"

# Dataset prefixes to convert (skipped if `<prefix>_data.npy` is absent).
PREFIXES = ["pretrain", "snpe_r1", "snpe_r2"]

# Columns required to recompute choices, and the ones we overwrite.
REQUIRED_COLS = [
    "Q_a", "Q_b", "lapse",
    "p_chose_orthogonal", "p_choice_A", "p_choice_B", "chose_orthogonal",
]


def _load_columns(columns_path: Path):
    with open(columns_path, "r") as f:
        cols = json.load(f)
    return cols, {c: i for i, c in enumerate(cols)}


def retemp_dataset(prefix: str, run_dir: Path, out_dir: Path, temp_new: float,
                   rng: np.random.Generator) -> bool:
    """Convert one dataset prefix. Returns True if converted, False if skipped."""
    data_path = run_dir / f"{prefix}_data.npy"
    columns_path = run_dir / f"{prefix}_columns.json"
    omega_path = run_dir / f"{prefix}_omega.npy"

    if not data_path.exists():
        return False
    if not (columns_path.exists() and omega_path.exists()):
        print(f"  [skip] {prefix}: missing columns/omega sidecar next to {data_path.name}")
        return False

    cols, idx = _load_columns(columns_path)
    missing = [c for c in REQUIRED_COLS if c not in idx]
    if missing:
        raise KeyError(
            f"{prefix}_columns.json is missing required columns {missing}; "
            f"this dataset cannot be re-tempered (was it simulated before Q_a/Q_b "
            f"were saved?)."
        )

    data = np.load(data_path)  # (n_sims, n_trials, n_cols), float32
    n_sims, n_trials, _ = data.shape

    Q_a = data[:, :, idx["Q_a"]].astype(np.float64)
    Q_b = data[:, :, idx["Q_b"]].astype(np.float64)
    lapse = data[:, :, idx["lapse"]].astype(np.float64)            # (n_sims, n_trials)
    p_orth_old = data[:, :, idx["p_chose_orthogonal"]].astype(np.float64)
    pA_old = data[:, :, idx["p_choice_A"]].astype(np.float64)

    # New choice probabilities under the agent's exact choice rule
    # (agents.Farmer.softmax: (1-lapse)*softmax(Q/temp) + lapse/n_afc, n_afc=2).
    # This vectorised form was verified bitwise-identical to calling the agent's
    # own softmax method per trial, so we keep it for speed.
    sm = softmax(np.stack([Q_a, Q_b], axis=-1) / temp_new, axis=-1)  # (n_sims, n_trials, 2)
    cp = (1.0 - lapse[..., None]) * sm + lapse[..., None] / 2.0
    pA_new, pB_new = cp[..., 0], cp[..., 1]

    # Recover which option is the orthogonal path from the *old* columns: the saved
    # p_chose_orthogonal always equals exactly one of p_choice_A / p_choice_B.
    orth_is_A = np.isclose(p_orth_old, pA_old, equal_nan=True)
    p_orth_new = np.where(orth_is_A, pA_new, pB_new)

    # NaN safety: where the original orthogonal prob was undefined (e.g. L-shaped
    # trials), leave probs and choice untouched rather than sampling on NaN.
    finite = np.isfinite(p_orth_old)
    p_orth_new = np.where(finite, p_orth_new, p_orth_old)
    pA_new = np.where(finite, pA_new, data[:, :, idx["p_choice_A"]].astype(np.float64))
    pB_new = np.where(finite, pB_new, data[:, :, idx["p_choice_B"]].astype(np.float64))

    chose_old = data[:, :, idx["chose_orthogonal"]]
    chose_new = chose_old.astype(np.float64).copy()
    chose_new[finite] = rng.binomial(1, p_orth_new[finite]).astype(np.float64)

    # Write the recomputed columns back into a copy; everything else is unchanged.
    out = data.copy()
    out[:, :, idx["chose_orthogonal"]] = chose_new.astype(np.float32)
    out[:, :, idx["p_chose_orthogonal"]] = p_orth_new.astype(np.float32)
    out[:, :, idx["p_choice_A"]] = pA_new.astype(np.float32)
    out[:, :, idx["p_choice_B"]] = pB_new.astype(np.float32)
    if "temp" in idx:  # keep the per-trial temp column self-consistent
        out[:, :, idx["temp"]] = np.float32(temp_new)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{prefix}_data.npy", out)
    shutil.copy2(omega_path, out_dir / f"{prefix}_omega.npy")       # temp not in omega here
    shutil.copy2(columns_path, out_dir / f"{prefix}_columns.json")

    flipped = (chose_new[finite] != chose_old[finite]).mean() if finite.any() else 0.0
    dp = np.abs(p_orth_new[finite] - p_orth_old[finite]).mean() if finite.any() else 0.0
    print(f"  [ok] {prefix}: {n_sims} sims x {n_trials} trials | "
          f"mean |Δp_orth| = {dp:.4f} | choices flipped = {flipped:.3f}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Re-derive SBI synthetic choices at a new softmax temperature.")
    parser.add_argument("--run_dir", required=True,
                        help="Run dir name under SBI/outputs/2_step/ (e.g. "
                             "30000_n1sims_0_n2sims_10001_samples)")
    parser.add_argument("--temp", type=float, required=True,
                        help="New softmax temperature (must differ from 1).")
    parser.add_argument("--out_dir", default=None,
                        help="Output dir name (default: <run_dir>_temp<temp>).")
    parser.add_argument("--seed", type=int, default=137,
                        help="RNG seed for re-sampling choices.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite out_dir if it already exists.")
    args = parser.parse_args()

    run_dir = ART_DIR / args.run_dir
    if not run_dir.is_dir():
        sys.exit(f"Run dir not found: {run_dir}")
    if args.temp == 1:
        sys.exit("--temp == 1 is a no-op (source data is at temp=1). Choose a different value.")

    out_name = args.out_dir or f"{args.run_dir}_temp{args.temp}"
    out_dir = ART_DIR / out_name
    if out_dir.exists() and not args.force:
        sys.exit(f"Output dir already exists: {out_dir} (use --force to overwrite).")

    rng = np.random.default_rng(args.seed)
    print(f"[retemp] {run_dir.name} -> {out_dir.name} | temp 1 -> {args.temp} | seed {args.seed}")

    converted = [p for p in PREFIXES if retemp_dataset(p, run_dir, out_dir, args.temp, rng)]
    if not converted:
        sys.exit("No datasets converted (no *_data.npy found in run dir).")

    # Copy run_config.json with the new fixed temperature.
    rc_path = run_dir / "run_config.json"
    if rc_path.exists():
        with open(rc_path, "r") as f:
            rc = json.load(f)
        rc.setdefault("fixed_params", {})["temp"] = args.temp
        with open(out_dir / "run_config.json", "w") as f:
            json.dump(rc, f, indent=2)
        print(f"  [ok] run_config.json copied with fixed_params.temp = {args.temp}")

    print(f"[retemp] Done. New datasets: {', '.join(converted)} -> {out_dir}")
    print("[retemp] NOTE: encoder.pt / posteriors were NOT copied — re-run the tesbi.py "
          "pretrain/snpe stages on the new dir to retrain on the new choices.")


if __name__ == "__main__":
    main()
