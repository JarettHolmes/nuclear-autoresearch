"""autoresearch-nuclear: train.py (MUTABLE)

Nuclear emulator training script. Apple Silicon MLX.
The agent modifies this file to improve combined_score.
Usage: uv run train.py
"""

import math
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from prepare import (
    TIME_BUDGET,
    FIDELITY_LEVELS,
    D_INPUT,
    N_EMBEDDING_DIM,
    FIDELITY_EMBEDDING_DIM,
    train_data,
    val_data,
    test_data,
    scaler,
    sinusoidal_encoding,
    fidelity_index,
    evaluate,
    check_physics,
)


# ============================================================
# HYPERPARAMETERS — modify these to experiment
# ============================================================
SHARED_LATENT_DIM = 128
HIDDEN_DIM = 64
HEAD_HIDDEN_DIM = 128
NUM_HEADS = 2
DROPOUT = 0.05
LEARNING_RATE = 1e-3
MAX_EPOCHS = 20000
BATCH_SIZE = 256
PATIENCE = 200
LR_PATIENCE = 25
LR_DECAY = 0.5
NUM_MC_SAMPLES = 100
LOSS_WEIGHTS = {4: 1.0, 6: 1.5, 8: 2.0, 10: 2.5}
SEED = 42


# ============================================================
# MODEL ARCHITECTURE — modify to experiment with structure
# ============================================================
class FidelityAttention(nn.Module):
    """Multi-head attention with fidelity-specific query projections."""

    def __init__(self, d_h, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_h // num_heads
        self.d_h = d_h
        self.W_k = nn.Linear(d_h, d_h)
        self.W_v = nn.Linear(d_h, d_h)
        # One query projection per fidelity level
        self.W_q_4 = nn.Linear(d_h, d_h)
        self.W_q_6 = nn.Linear(d_h, d_h)
        self.W_q_8 = nn.Linear(d_h, d_h)
        self.W_q_10 = nn.Linear(d_h, d_h)
        self.W_o = nn.Linear(d_h, d_h)

    def __call__(self, h, fidelity):
        batch = h.shape[0]
        q_proj = {4: self.W_q_4, 6: self.W_q_6, 8: self.W_q_8, 10: self.W_q_10}[fidelity]

        Q = q_proj(h).reshape(batch, self.num_heads, 1, self.d_k)
        K = self.W_k(h).reshape(batch, self.num_heads, 1, self.d_k)
        V = self.W_v(h).reshape(batch, self.num_heads, 1, self.d_k)

        scale = 1.0 / math.sqrt(self.d_k)
        scores = (Q @ mx.transpose(K, (0, 1, 3, 2))) * scale
        attn = mx.softmax(scores, axis=-1)
        out = (attn @ V).reshape(batch, self.d_h)
        return self.W_o(out)


class NuclearEmulator(nn.Module):
    """Hierarchical multi-fidelity nuclear emulator.

    Architecture: shared latent -> fidelity attention -> base + cumulative delta heads.
    Predicts [E_B, R_ch] for a given fidelity level (emax).
    """

    def __init__(self):
        super().__init__()
        self.fidelity_levels = sorted(FIDELITY_LEVELS)
        self.base_fidelity = self.fidelity_levels[0]
        self.delta_fidelities = self.fidelity_levels[1:]

        # Fidelity embedding (learned, 4 levels)
        self.fidelity_emb = nn.Embedding(len(FIDELITY_LEVELS), FIDELITY_EMBEDDING_DIM)

        # Shared latent network with residual projection
        self.shared_1 = nn.Linear(D_INPUT, HIDDEN_DIM)
        self.shared_2 = nn.Linear(HIDDEN_DIM, SHARED_LATENT_DIM)
        self.shared_skip = nn.Linear(D_INPUT, SHARED_LATENT_DIM)

        # Fidelity attention
        self.attention = FidelityAttention(SHARED_LATENT_DIM, NUM_HEADS)

        # Base head (emax=4)
        self.base_h = nn.Linear(SHARED_LATENT_DIM, HEAD_HIDDEN_DIM)
        self.base_out = nn.Linear(HEAD_HIDDEN_DIM, 2)

        # Delta heads (emax=6, 8, 10)
        self.delta_6_h = nn.Linear(SHARED_LATENT_DIM, HEAD_HIDDEN_DIM)
        self.delta_6_out = nn.Linear(HEAD_HIDDEN_DIM, 2)
        self.delta_8_h = nn.Linear(SHARED_LATENT_DIM, HEAD_HIDDEN_DIM)
        self.delta_8_out = nn.Linear(HEAD_HIDDEN_DIM, 2)
        self.delta_10_h = nn.Linear(SHARED_LATENT_DIM, HEAD_HIDDEN_DIM)
        self.delta_10_out = nn.Linear(HEAD_HIDDEN_DIM, 2)

    def _build_input(self, x_lec, n_vals, emax):
        """Concatenate [scaled LECs, sinusoidal(N), fidelity_embedding]."""
        n_enc = sinusoidal_encoding(n_vals, N_EMBEDDING_DIM)
        f_idx = mx.full((x_lec.shape[0],), fidelity_index(emax), dtype=mx.int32)
        f_emb = self.fidelity_emb(f_idx)
        return mx.concatenate([x_lec, n_enc, f_emb], axis=-1)

    def _shared_forward(self, x_input):
        """Shared latent with residual: input -> hidden -> latent + skip."""
        h = nn.silu(self.shared_1(x_input))
        h = nn.Dropout(DROPOUT)(h)
        h = nn.silu(self.shared_2(h))
        h = h + self.shared_skip(x_input)  # residual connection
        h = nn.Dropout(DROPOUT)(h)
        return h

    def _delta_head(self, fidelity, o):
        """Apply the appropriate delta head."""
        heads = {
            6: (self.delta_6_h, self.delta_6_out),
            8: (self.delta_8_h, self.delta_8_out),
            10: (self.delta_10_h, self.delta_10_out),
        }
        h_layer, out_layer = heads[fidelity]
        return out_layer(nn.silu(h_layer(o)))

    def __call__(self, x_lec, n_vals, target_emax):
        """Forward pass: predict [E_B, R_ch] at target_emax.

        Uses cumulative structure: y = base(emax=4) + sum(delta(f) for f <= target_emax).
        """
        # Base prediction at emax=4
        z_base = self._build_input(x_lec, n_vals, self.base_fidelity)
        h_base = self._shared_forward(z_base)
        o_base = self.attention(h_base, self.base_fidelity)
        y = self.base_out(nn.silu(self.base_h(o_base)))

        # Cumulative deltas
        for f in self.delta_fidelities:
            if f > target_emax:
                break
            z_f = self._build_input(x_lec, n_vals, f)
            h_f = self._shared_forward(z_f)
            o_f = self.attention(h_f, f)
            y = y + self._delta_head(f, o_f)

        return y


# ============================================================
# TRAINING
# ============================================================
def main():
    t_start = time.time()
    mx.random.seed(SEED)
    np.random.seed(SEED)

    model = NuclearEmulator()
    mx.eval(model.parameters())
    num_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Parameters: {num_params:,}")

    # Loss function with gradient
    def loss_fn(model, x, n, emax, y_true, weight):
        pred = model(x, n, emax)
        return weight * mx.mean((pred - y_true) ** 2)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Optimizer
    optimizer = optim.Adam(learning_rate=LEARNING_RATE)
    current_lr = LEARNING_RATE

    best_val = float("inf")
    best_state = None
    patience_counter = 0
    lr_patience_counter = 0
    total_training_time = 0.0

    for epoch in range(1, MAX_EPOCHS + 1):
        t_epoch = time.time()

        # Train one epoch: sample batch from each fidelity
        total_loss = 0.0
        for emax in FIDELITY_LEVELS:
            ds = train_data.get(emax)
            if ds is None:
                continue
            n = ds["X"].shape[0]
            batch_n = min(BATCH_SIZE, n)
            idx = mx.array(np.random.randint(0, n, size=batch_n))

            x_batch = ds["X"][idx]
            n_batch = ds["N"][idx]
            y_batch = ds["y"][idx]
            weight = LOSS_WEIGHTS.get(emax, 1.0)

            loss, grads = loss_and_grad(model, x_batch, n_batch, emax, y_batch, weight)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            total_loss += loss.item()

        dt = time.time() - t_epoch
        total_training_time += dt

        # Validate every 100 epochs
        if epoch % 100 == 0:
            val_loss = 0.0
            for emax, ds in val_data.items():
                pred = model(ds["X"], ds["N"], emax)
                w = LOSS_WEIGHTS.get(emax, 1.0)
                val_loss += w * mx.mean((pred - ds["y"]) ** 2).item()

            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
                lr_patience_counter = 0
                # Save best state
                best_state = [(k, mx.array(np.array(v))) for k, v in tree_flatten(model.parameters())]
            else:
                patience_counter += 1
                lr_patience_counter += 1

                if lr_patience_counter >= LR_PATIENCE:
                    current_lr *= LR_DECAY
                    optimizer = optim.Adam(learning_rate=current_lr)
                    lr_patience_counter = 0

                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 2000 == 0:
                marker = " *" if patience_counter == 0 else ""
                remaining = max(0.0, TIME_BUDGET - total_training_time)
                print(
                    f"[{epoch:>6}] train={total_loss:.4f} val={val_loss:.4f} "
                    f"lr={current_lr:.2e} ({total_training_time:.0f}s) "
                    f"remaining={remaining:.0f}s{marker}"
                )

        # Time budget check
        if total_training_time >= TIME_BUDGET:
            print(f"Time budget reached at epoch {epoch}")
            break

    # Restore best state
    if best_state:
        model.load_weights(best_state)
        mx.eval(model.parameters())

    training_seconds = time.time() - t_start

    # === MC DROPOUT EVALUATION ===
    print("\nStarting MC dropout evaluation...")
    metrics = evaluate(model, test_data, scaler, mc_samples=NUM_MC_SAMPLES)
    physics = check_physics(model, test_data, scaler)

    total_seconds = time.time() - t_start

    # === RESULTS ===
    print("\n---")
    print(f"combined_score:   {metrics['combined_score']:.6f}")
    print(f"eb_rmse:          {metrics['eb_rmse']:.6f}")
    print(f"rch_rmse:         {metrics['rch_rmse']:.6f}")
    print(f"physics_ok:       {physics['positive_eb'] == 0 and physics['negative_rch'] == 0}")
    print(f"positive_eb:      {physics['positive_eb']}")
    print(f"negative_rch:     {physics['negative_rch']}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"num_params:       {num_params}")
    print(f"num_epochs:       {epoch}")


if __name__ == "__main__":
    main()
