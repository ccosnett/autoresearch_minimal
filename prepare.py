"""
Data preparation and evaluation for the linear regression toy problem.

Generates synthetic data: y = 3*x1 + 7*x2 - 2 + noise
Fixed random seed ensures reproducibility across experiments.

DO NOT MODIFY THIS FILE — it is the fixed evaluation harness.

Usage:
    python prepare.py           # generates data and prints summary
    (also imported by train.py)
"""

import os
import numpy as np

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

SEED = 42
N_FEATURES = 5
N_TRAIN = 500
N_VAL = 200
TIME_BUDGET = 10  # seconds — generous for this problem

# Ground truth coefficients: y = W @ x + b + noise
TRUE_WEIGHTS = np.array([3.0, 7.0, -2.0, 0.5, 4.0])
TRUE_BIAS = -2.0
NOISE_STD = 0.5

# ---------------------------------------------------------------------------
# Data generation (deterministic)
# ---------------------------------------------------------------------------

def _generate_data():
    """Generate fixed train/val splits. Always returns the same data."""
    rng = np.random.RandomState(SEED)

    X_all = rng.randn(N_TRAIN + N_VAL, N_FEATURES)
    y_all = X_all @ TRUE_WEIGHTS + TRUE_BIAS + rng.randn(N_TRAIN + N_VAL) * NOISE_STD

    X_train, X_val = X_all[:N_TRAIN], X_all[N_TRAIN:]
    y_train, y_val = y_all[:N_TRAIN], y_all[N_TRAIN:]

    return X_train, y_train, X_val, y_val


# Cache so repeated imports don't regenerate
_data_cache = None

def get_data():
    """Returns (X_train, y_train, X_val, y_val) as numpy arrays."""
    global _data_cache
    if _data_cache is None:
        _data_cache = _generate_data()
    return _data_cache


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate_mse(predictions, y_true=None):
    """
    Compute mean squared error on the validation set.

    Args:
        predictions: numpy array of shape (N_VAL,) — predicted y values
        y_true: if None, uses the fixed validation targets

    Returns:
        float: MSE (lower is better)
    """
    if y_true is None:
        _, _, _, y_val = get_data()
        y_true = y_val

    predictions = np.asarray(predictions, dtype=np.float64).ravel()
    y_true = np.asarray(y_true, dtype=np.float64).ravel()

    assert predictions.shape == y_true.shape, (
        f"Shape mismatch: predictions {predictions.shape} vs targets {y_true.shape}"
    )

    return float(np.mean((predictions - y_true) ** 2))


# ---------------------------------------------------------------------------
# Main — run to verify data generation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    X_train, y_train, X_val, y_val = get_data()

    print(f"Linear regression toy problem")
    print(f"  Features:    {N_FEATURES}")
    print(f"  Train size:  {N_TRAIN}")
    print(f"  Val size:    {N_VAL}")
    print(f"  Noise std:   {NOISE_STD}")
    print(f"  Time budget: {TIME_BUDGET}s")
    print()
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    print(f"  X_val shape:   {X_val.shape}")
    print(f"  y_val range:   [{y_val.min():.2f}, {y_val.max():.2f}]")
    print()

    # Baseline: predict mean of training targets
    mean_pred = np.full(N_VAL, y_train.mean())
    baseline_mse = evaluate_mse(mean_pred)
    print(f"  Baseline MSE (predict mean): {baseline_mse:.6f}")
    print()
    print("Ready to train. Run: python train.py")
