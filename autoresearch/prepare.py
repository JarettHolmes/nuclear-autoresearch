"""prepare.py (IMMUTABLE)

Data loading, splitting, scaling, and evaluation for nuclear emulator research.
This file is fixed — the agent only modifies train.py.

Data: oxygen isotope nuclear observables from BANNANE paper (arXiv:2502.20363v2).
Task: predict binding energy (E_B) and charge radius (R_ch) from 17 LEC parameters
      across 4 fidelity levels (emax=4,6,8,10).
Metric: combined_score = eb_rmse + 100 * rch_rmse (lower is better).
"""

import glob
import math
import os
import re

import mlx.core as mx
import numpy as np

# ============================================================
# CONSTANTS (do not modify)
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "oxygen")
TIME_BUDGET = 300  # seconds
SEED = 42
FIDELITY_LEVELS = [4, 6, 8, 10]

INPUT_COLS = [
    "Ct1S0pp", "Ct1S0np", "Ct1S0nn", "Ct3S1", "C1S0", "C3P0",
    "C1P1", "C3P1", "C3S1", "CE1", "C3P2", "c1", "c2", "c3",
    "c4", "cD", "cE",
]
TARGET_COLS = ["Energy ket", "Rch"]
FIDELITY_COL = "emax"
SAMPLE_COL = "Sample"

N_EMBEDDING_DIM = 20
FIDELITY_EMBEDDING_DIM = 8
D_INPUT = len(INPUT_COLS) + N_EMBEDDING_DIM + FIDELITY_EMBEDDING_DIM  # 45


# ============================================================
# DATA LOADING (no pandas dependency)
# ============================================================
_NEEDED_COLS = [SAMPLE_COL, FIDELITY_COL] + INPUT_COLS + TARGET_COLS


def load_data():
    """Load all oxygen isotope CSVs. Returns (data, col_idx)."""
    pattern = os.path.join(DATA_DIR, "O*_radii.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")

    all_data = []
    out_header = _NEEDED_COLS + ["Z", "N", "A"]
    out_col_idx = {name: i for i, name in enumerate(out_header)}

    for fpath in files:
        basename = os.path.basename(fpath)
        match = re.match(r"O(\d+)_radii\.csv", basename)
        if not match:
            continue
        A = int(match.group(1))
        Z = 8
        N = A - Z

        with open(fpath, "r") as f:
            header = f.readline().strip().split(",")
            col_map = {name: i for i, name in enumerate(header)}
            needed_idxs = [col_map[c] for c in _NEEDED_COLS]

            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                row = [float(parts[i]) for i in needed_idxs]
                row.extend([float(Z), float(N), float(A)])
                all_data.append(row)

    data = np.array(all_data, dtype=np.float64)
    print(f"Loaded {len(data)} rows from {len(files)} isotope files")
    return data, out_col_idx


# ============================================================
# SPLITTING AND SCALING
# ============================================================
def split_and_scale(data, col_idx):
    """Split by sample ID (60/20/20), fit scalers on train."""
    sample_col_i = col_idx[SAMPLE_COL]
    samples = np.unique(data[:, sample_col_i])

    rng = np.random.RandomState(SEED)
    rng.shuffle(samples)
    n = len(samples)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)

    train_ids = set(samples[:n_train].tolist())
    val_ids = set(samples[n_train:n_train + n_val].tolist())
    test_ids = set(samples[n_train + n_val:].tolist())

    train_mask = np.array([data[i, sample_col_i] in train_ids for i in range(len(data))])
    val_mask = np.array([data[i, sample_col_i] in val_ids for i in range(len(data))])
    test_mask = np.array([data[i, sample_col_i] in test_ids for i in range(len(data))])

    input_idxs = [col_idx[c] for c in INPUT_COLS]
    target_idxs = [col_idx[c] for c in TARGET_COLS]
    fidelity_idx = col_idx[FIDELITY_COL]
    n_idx = col_idx["N"]

    train_X = data[train_mask][:, input_idxs]
    train_y = data[train_mask][:, target_idxs]
    X_mean, X_std = train_X.mean(axis=0), train_X.std(axis=0)
    y_mean, y_std = train_y.mean(axis=0), train_y.std(axis=0)
    X_std[X_std == 0] = 1.0
    y_std[y_std == 0] = 1.0

    def build_split(mask):
        subset = data[mask]
        result = {}
        for emax in FIDELITY_LEVELS:
            fmask = subset[:, fidelity_idx] == emax
            if fmask.sum() == 0:
                continue
            X = (subset[fmask][:, input_idxs] - X_mean) / X_std
            y = (subset[fmask][:, target_idxs] - y_mean) / y_std
            N = subset[fmask][:, n_idx].astype(np.int32)
            result[emax] = {
                "X": mx.array(X.astype(np.float32)),
                "y": mx.array(y.astype(np.float32)),
                "N": mx.array(N),
            }
        return result

    train_data = build_split(train_mask)
    val_data = build_split(val_mask)
    test_data = build_split(test_mask)

    scaler = {"X_mean": X_mean, "X_std": X_std, "y_mean": y_mean, "y_std": y_std}

    for name, split in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        print(f"  {name}: {split.sum()} rows")

    return train_data, val_data, test_data, scaler


# ============================================================
# ENCODING HELPERS
# ============================================================
def sinusoidal_encoding(n_values, dim=N_EMBEDDING_DIM):
    """Sinusoidal positional encoding for neutron number N."""
    n = n_values.astype(mx.float32).reshape(-1, 1)
    i = mx.arange(dim // 2).astype(mx.float32)
    phi = mx.exp(-2.0 * i / dim * math.log(10000.0))
    angles = n * phi
    return mx.concatenate([mx.sin(angles), mx.cos(angles)], axis=-1)


def fidelity_index(emax):
    """Map emax value to index: {4:0, 6:1, 8:2, 10:3}."""
    return {4: 0, 6: 1, 8: 2, 10: 3}[emax]


# ============================================================
# EVALUATION
# ============================================================
def evaluate(model, test_data, scaler, mc_samples=100):
    """Evaluate model on test set at emax=10 with MC dropout."""
    y_mean = scaler["y_mean"]
    y_std = scaler["y_std"]

    ds = test_data[10]
    X, N, y_scaled = ds["X"], ds["N"], ds["y"]

    all_preds = []
    for _ in range(mc_samples):
        pred_scaled = model(X, N, 10)
        mx.eval(pred_scaled)
        all_preds.append(np.array(pred_scaled))

    preds = np.stack(all_preds, axis=0)
    mean_scaled = preds.mean(axis=0)

    y_pred = mean_scaled * y_std + y_mean
    y_true = np.array(y_scaled) * y_std + y_mean

    residuals = y_pred - y_true
    eb_rmse = float(np.sqrt(np.mean(residuals[:, 0] ** 2)))
    rch_rmse = float(np.sqrt(np.mean(residuals[:, 1] ** 2)))
    combined_score = eb_rmse + 100 * rch_rmse

    return {"eb_rmse": eb_rmse, "rch_rmse": rch_rmse, "combined_score": combined_score}


def check_physics(model, test_data, scaler):
    """Check for physics violations."""
    y_mean = scaler["y_mean"]
    y_std = scaler["y_std"]

    pos_eb = 0
    neg_rch = 0
    for emax in FIDELITY_LEVELS:
        if emax not in test_data:
            continue
        ds = test_data[emax]
        pred_scaled = model(ds["X"], ds["N"], emax)
        mx.eval(pred_scaled)
        pred = np.array(pred_scaled) * y_std + y_mean
        pos_eb += int((pred[:, 0] > 0).sum())
        neg_rch += int((pred[:, 1] < 0).sum())

    return {"positive_eb": pos_eb, "negative_rch": neg_rch}


# ============================================================
# LOAD ON IMPORT
# ============================================================
print("Loading nuclear data...")
data, col_idx = load_data()
train_data, val_data, test_data, scaler = split_and_scale(data, col_idx)
print("Data ready.\n")
