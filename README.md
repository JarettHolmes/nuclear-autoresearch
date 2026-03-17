# nuclear-autoresearch

Autonomous neural network research for nuclear physics emulation, applied to the BANNANE framework ([arXiv:2502.20363v2](https://arxiv.org/abs/2502.20363v2)).

An AI agent iteratively modifies a neural network emulator for nuclear structure calculations — binding energies and charge radii of oxygen isotopes — keeping changes that improve accuracy and reverting those that don't. Runs natively on Apple Silicon via [MLX](https://github.com/ml-explore/mlx). Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

## Results

Starting from a faithful reimplementation of the BANNANE architecture, 24 autonomous experiments identified three architectural improvements that together outperform the published results:

| Metric | Published (Table I) | This work | Change |
|--------|----|----|---|
| E_B RMSE (emax=10) | 0.8 MeV | **0.44 MeV** | -45% |
| R_ch RMSE (emax=10) | 0.01 fm | **0.0096 fm** | -4% |
| Combined score | ~1.8 | **1.40** | -22% |
| Physics violations | 0 | 0 | -- |

The three changes that mattered, found in sequence:

| Experiment | Change | Combined score |
|---|---|---|
| Baseline | BANNANE reimplementation on MLX | 2.21 |
| +Residual connection | Skip connection in shared latent network | 1.82 |
| +SiLU activation | Replace LeakyReLU with SiLU throughout | 1.54 |
| +Wider heads | Prediction head hidden dim 64 → 128 | **1.40** |

19 other experiments (layer norm, deeper networks, Huber loss, gradient clipping, weight decay, larger batches, GELU, etc.) were tried and discarded. Full history in `autoresearch/results.tsv`.

### Caveats

These results should be interpreted carefully:

- **Evaluation methodology may differ.** The published paper uses a full Bayesian Neural Network with variational inference. This work uses deterministic training with MC dropout — a simpler but different approach. The comparison is on test-set RMSE using the same dataset and splitting strategy, but exact data splits may differ due to random seed differences.
- **No cross-validation.** Results are from a single train/val/test split. Performance could vary with different splits.
- **Uncertainty calibration not validated.** MC dropout provides uncertainty estimates, but whether they're well-calibrated (e.g., 68% of predictions within 1-sigma) has not been tested.
- **Not tested on out-of-distribution data.** The model has only been evaluated on held-out samples from the same LEC distribution used for training.

## How it works

### The emulator

Nuclear structure calculations (IMSRG) take hours per parameter set. The emulator predicts results in milliseconds:

- **Input:** 17 Low Energy Constants (LECs) that parameterize nuclear forces, plus neutron number N and fidelity level (emax)
- **Output:** Binding energy (E_B) and charge radius (R_ch)
- **Architecture:** Shared latent network → fidelity-specific attention → base prediction (emax=4) + cumulative delta corrections (emax=6,8,10)
- **Physics constraints:** E_B must be negative (bound nuclei), R_ch must be positive

### The autoresearch loop

Adapted from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) pattern:

1. `prepare.py` — Fixed: loads data, defines evaluation metric. Never modified.
2. `train.py` — Mutable: model architecture and training. The agent modifies this.
3. `program.md` — Instructions for the autonomous agent.

Each experiment: edit `train.py` → commit → train for 5 minutes → evaluate → keep if improved, revert if not. The agent runs indefinitely until interrupted.

## Quick start

Requirements: Apple Silicon Mac, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
cd autoresearch
uv sync
uv run train.py          # single 5-minute experiment
```

To run the autonomous loop, point [Claude Code](https://claude.ai/claude-code) at `autoresearch/program.md`.

## Repository structure

```
nuclear-autoresearch/
├── README.md                  # This file
├── data/
│   └── oxygen/                # O12-O24 isotope observables (from BANNANE public data)
│       ├── O12_radii.csv
│       ├── ...
│       └── O24_radii.csv
└── autoresearch/
    ├── prepare.py             # Data pipeline + evaluation (immutable)
    ├── train.py               # Model + training (agent modifies this)
    ├── program.md             # Autonomous experiment protocol
    ├── results.tsv            # Experiment history
    └── pyproject.toml         # Dependencies (mlx, numpy)
```

## Data

Oxygen isotope nuclear observables from the [BANNANE public dataset](https://github.com/munozariasjm/paper_o_bannane). 13 isotopes (O12-O24), 17,234 total samples across 4 fidelity levels, 17 LEC input parameters, 2 target observables.

## References

- A. Belley, J. M. Munoz, R. F. Garcia Ruiz. "Global Framework for Emulation of Nuclear Calculations." [arXiv:2502.20363v2](https://arxiv.org/abs/2502.20363v2) (2025).
- A. Karpathy. [autoresearch](https://github.com/karpathy/autoresearch) — autonomous ML research framework.
- T. Creator. [autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) — Apple Silicon port.

## License

MIT
