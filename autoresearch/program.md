# autoresearch-nuclear

An autonomous research loop for nuclear physics emulation, built on the autoresearch pattern. Trains multi-fidelity neural network emulators for nuclear structure calculations (binding energies and charge radii) using MLX on Apple Silicon.

**Paper reference:** arXiv:2502.20363v2 — "Global Framework for Emulation of Nuclear Calculations" (BANNANE).

**Monorepo note:** This project lives inside a larger repo. Always stage only `autoresearch-nuclear/` paths. Never use blind `git add -A`.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar17`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `prepare.py` — fixed: data loading, splitting, scaling, evaluation, physics checks. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop, hyperparameters.
4. **Verify data exists**: Check that `autoresearch-nuclear/../jason-holt/paper_o_bannane/DATA/all_o/` contains O*_radii.csv files.
5. **Initialize results.tsv**: Create `results.tsv` with header row and baseline entry. Run `uv run train.py` once to establish YOUR baseline.
6. **Confirm and go**.

## The Task

You are training a neural network emulator that predicts two nuclear observables:
- **E_B** (binding energy) — should always be negative (bound nuclei)
- **R_ch** (charge radius) — should always be positive

From 17 Low Energy Constants (LECs) + neutron number N + fidelity level (emax=4,6,8,10).

The model uses a **multi-fidelity hierarchical structure**: base prediction at emax=4, then cumulative delta corrections for emax=6,8,10. This encodes the physics that higher-fidelity calculations build on lower-fidelity ones.

**Paper targets:** E_B RMSE ≈ 0.8 MeV, R_ch RMSE ≈ 0.01 fm, combined_score ≈ 1.8.

## Metric

**combined_score = eb_rmse + 100 × rch_rmse** (lower is better).

This is evaluated at emax=10 on the test set using MC dropout (100 forward passes with dropout active, predictions averaged). The `evaluate()` function in prepare.py is the ground truth.

## Physics Constraints

**HARD CONSTRAINTS** — a model that violates these is broken:
- E_B must be negative for all predictions (positive E_B = unbound nucleus = unphysical)
- R_ch must be positive for all predictions (negative charge radius = unphysical)

The `check_physics()` function reports violation counts. Any violations should be treated as a crash/failure.

## Experimentation

Each experiment trains on Apple Silicon via MLX for a **fixed time budget of 5 minutes** (300s wall clock). Run: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — everything is fair game: model architecture, optimizer, hyperparameters, loss functions, training strategy, MC inference, batch size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It contains fixed data loading, evaluation, and constants.
- Install new packages beyond what's in `pyproject.toml`.
- Change the evaluation metric or data splits.

**The goal: get the lowest combined_score** while maintaining zero physics violations.

**Ideas to explore (in roughly increasing boldness):**
- Hyperparameter tuning: learning rate schedules, batch sizes, loss weights
- Architecture modifications: wider/deeper networks, different activation functions, residual connections, layer normalization
- Attention variants: different attention mechanisms, more heads, cross-fidelity attention
- Training strategies: curriculum learning (easy fidelities first), gradient clipping, weight decay
- Loss modifications: separate E_B and R_ch losses with different weights, Huber loss, physics-informed loss penalties
- Delta head structure: share parameters across deltas, add skip connections
- Encoding changes: different neutron number encodings, learned fidelity embeddings vs fixed

## Output format

```
---
combined_score:   2.364300
eb_rmse:          1.033000
rch_rmse:         0.013310
physics_ok:       True
positive_eb:      0
negative_rch:     0
training_seconds: 259.0
total_seconds:    285.3
num_params:       74792
num_epochs:       20000
```

Extract with: `grep "^combined_score:\|^physics_ok:" run.log`

## Logging results

Log to `results.tsv` (tab-separated):

```
commit	combined_score	eb_rmse	rch_rmse	physics_ok	status	description
```

1. git commit hash (short, 7 chars)
2. combined_score (e.g. 2.364300) — use 0.000000 for crashes
3. eb_rmse (e.g. 1.033000)
4. rch_rmse (e.g. 0.013310)
5. physics_ok: True/False
6. status: `keep`, `discard`, or `crash`
7. short text description

## The experiment loop

LOOP FOREVER:

1. Look at the git state
2. Modify `train.py` with an experimental idea
3. `git add autoresearch-nuclear/train.py && git commit -m "experiment: <description>"`
4. Run: `uv run train.py > run.log 2>&1`
5. Read results: `grep "^combined_score:\|^physics_ok:" run.log`
6. If grep is empty → crash. Run `tail -n 50 run.log` for the traceback.
7. **REJECT if physics_ok is False** — physics violations are never acceptable, even if combined_score improves. Treat as a crash/discard.
8. Record in results.tsv
9. If combined_score improved AND physics_ok is True: `git add autoresearch-nuclear/results.tsv && git commit --amend --no-edit`
10. If worse or physics violated: record discard hash, then `git reset --hard <previous kept commit>`

**Timeout**: Each experiment should take ~6 minutes. If it exceeds 10 minutes, kill and treat as failure.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. You are autonomous. If you run out of ideas, re-read the paper reference, try combining previous near-misses, try radical changes. The loop runs until manually interrupted.
