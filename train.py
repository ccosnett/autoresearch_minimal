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

# Locally weighted linear regression (LOWESS-style)
# For each val point, fit a local weighted regression using training points
# Weight by Gaussian kernel on distance

X_bias = np.column_stack([X_train, np.ones(n_samples)])

# Also do global OLS for comparison
theta_global = np.linalg.lstsq(X_bias, y_train, rcond=None)[0]
w = theta_global[:n_features]
b = theta_global[n_features]

# Local predictions for val set
BANDWIDTH = 2.0  # kernel bandwidth
local_preds = np.zeros(X_val.shape[0])

for i in range(X_val.shape[0]):
    dists = np.sum((X_train - X_val[i])**2, axis=1)
    kernel_weights = np.exp(-dists / (2 * BANDWIDTH**2))

    Wk = np.diag(kernel_weights)
    try:
        theta_local = np.linalg.solve(X_bias.T @ Wk @ X_bias, X_bias.T @ (Wk @ y_train))
        local_preds[i] = np.append(X_val[i], 1.0) @ theta_local
    except np.linalg.LinAlgError:
        local_preds[i] = X_val[i] @ w + b

print(f"Locally weighted regression (bandwidth={BANDWIDTH})")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

t_end = time.time()
training_time = t_end - t_start

val_predictions = local_preds
val_mse = evaluate_mse(val_predictions)

print("---")
print(f"val_mse:          {val_mse:.6f}")
print(f"training_seconds: {training_time:.1f}")
print(f"num_iterations:   1")
print(f"weights:          {w}")
print(f"bias:             {b:.6f}")
