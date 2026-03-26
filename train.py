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

# Closed-form OLS: append bias column, solve normal equations
X_bias = np.column_stack([X_train, np.ones(n_samples)])
theta = np.linalg.lstsq(X_bias, y_train, rcond=None)[0]
w = theta[:n_features]
b = theta[n_features]

print(f"OLS solved in one step")

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
