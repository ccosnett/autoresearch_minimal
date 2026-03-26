"""
Linear regression training script.
The agent modifies this file to improve val_mse.

Usage: python train.py
"""

import time
import numpy as np

from prepare import TIME_BUDGET, get_data, evaluate_mse

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

LEARNING_RATE = 0.01
NUM_ITERATIONS = 1000

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

X_train, y_train, X_val, y_val = get_data()
n_samples, n_features = X_train.shape

# Initialize weights and bias
w = np.zeros(n_features)
b = 0.0

t_start = time.time()

for i in range(NUM_ITERATIONS):
    # Check time budget
    if time.time() - t_start > TIME_BUDGET:
        break

    # Forward pass
    y_pred = X_train @ w + b
    error = y_pred - y_train

    # Gradients
    grad_w = (2.0 / n_samples) * (X_train.T @ error)
    grad_b = (2.0 / n_samples) * np.sum(error)

    # Update
    w -= LEARNING_RATE * grad_w
    b -= LEARNING_RATE * grad_b

    # Log every 100 steps
    if (i + 1) % 100 == 0:
        train_mse = float(np.mean(error ** 2))
        print(f"step {i+1:05d} | train_mse: {train_mse:.6f}")

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
print(f"num_iterations:   {min(i + 1, NUM_ITERATIONS)}")
print(f"weights:          {w}")
print(f"bias:             {b:.6f}")
