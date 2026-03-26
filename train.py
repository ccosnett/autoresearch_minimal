"""
Linear regression training script.
The agent modifies this file to improve val_mse.

Usage: uv run train.py
"""

import time
import numpy as np

from prepare import TIME_BUDGET, get_data, evaluate_mse

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

X_train, y_train, X_val, y_val = get_data()
n_samples, n_features = X_train.shape

t_start = time.time()

# Leverage-weighted OLS: downweight high-leverage points
X_bias = np.column_stack([X_train, np.ones(n_samples)])

# Compute hat matrix diagonal (leverage)
H = X_bias @ np.linalg.solve(X_bias.T @ X_bias, X_bias.T)
h = np.diag(H)

# Weight inversely proportional to leverage
weights = 1.0 / h
weights /= weights.sum()  # normalize

W = np.diag(weights)
theta = np.linalg.solve(X_bias.T @ W @ X_bias, X_bias.T @ W @ y_train)
w = theta[:n_features]
b = theta[n_features]

print(f"Leverage-weighted OLS (leverage range: [{h.min():.4f}, {h.max():.4f}])")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

t_end = time.time()
training_time = t_end - t_start

val_predictions = X_val @ w + b
val_mse = evaluate_mse(val_predictions)

print("---")
print(f"val_mse:          {val_mse:.6f}")
print(f"training_seconds: {training_time:.1f}")
print(f"num_iterations:   1")
print(f"weights:          {w}")
print(f"bias:             {b:.6f}")
