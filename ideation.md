# Minimal Toy Problem Ideas for Autoresearch

## Context

The autoresearch loop works as follows: **agent edits `train.py` → runs it → checks metric → keeps or discards**. The key structural elements are:

1. **`prepare.py`** — fixed data prep + evaluation (agent can't touch)
2. **`train.py`** — the single file the agent modifies
3. **`program.md`** — instructions for the agent
4. A **single scalar metric** (lower is better) that determines keep/discard

To make a minimal toy version, we need the simplest possible problem that still preserves this loop. Here are ideas ranked from simplest to most interesting:

---

## Idea 1: Linear Regression (Simplest Possible)

**The problem:** Fit `y = Wx + b` on synthetic data (e.g., `y = 3x₁ + 7x₂ - 2 + noise`).

**`prepare.py`:** Generates fixed synthetic data, provides `evaluate_mse(model, X_val, y_val)` function.

**`train.py`:** Agent starts with vanilla gradient descent. Can experiment with:
- Learning rate, momentum, number of iterations
- Adding regularization (L1, L2)
- Trying closed-form solution (`(X^T X)^{-1} X^T y`)
- Feature normalization
- Different optimizers (SGD, Adam-like manual implementations)

**Metric:** Validation MSE (lower is better).

**Why it's good:** Dead simple, runs in milliseconds, zero dependencies beyond numpy/torch. The agent can "discover" the closed-form solution, which is a fun outcome.

**Why it might be _too_ simple:** The search space is tiny. The agent will solve it optimally in 1-2 experiments (closed-form → done). Not much room for iterative discovery.

---

## Idea 2: Polynomial Regression / Curve Fitting (Sweet Spot) ⭐

**The problem:** Fit a function `y = f(x)` where `f` is a non-trivial curve (e.g., `sin(x) + 0.5*x²` or a real dataset like CO2 concentrations).

**`prepare.py`:** Generates/loads data, provides `evaluate_mse()`.

**`train.py`:** Agent starts with a simple linear model. Can experiment with:
- Polynomial degree (underfitting vs overfitting tradeoff!)
- Basis functions (polynomial, Fourier, RBF)
- Regularization strength
- Train/val split awareness
- Feature engineering
- Small neural net (1-2 hidden layers)

**Metric:** Validation MSE.

**Why it's good:** Simple enough to run instantly, but has a real **overfitting vs underfitting tradeoff** the agent must navigate. Increasing polynomial degree helps... until it doesn't. This mirrors the real autoresearch dynamic where "more complex" isn't always better. The agent can make dozens of meaningful experiments.

---

## Idea 3: 2D Classification (Moons/Circles)

**The problem:** Classify points in 2D using sklearn's `make_moons` or `make_circles`.

**`prepare.py`:** Generates fixed dataset, provides `evaluate_accuracy()` or `evaluate_loss()`.

**`train.py`:** Agent starts with logistic regression. Can experiment with:
- Adding hidden layers (inventing a small neural net)
- Activation functions
- Learning rate schedules
- Feature engineering (adding `x²`, `xy`, etc.)
- Decision boundary complexity

**Metric:** Validation cross-entropy loss.

**Why it's good:** Non-linearly separable data forces the agent to discover that a linear model isn't enough. It must "invent" non-linear features or a neural net. Visually compelling — you can plot decision boundaries.

---

## Idea 4: Tiny Sequence Prediction (Mini Version of Current Problem)

**The problem:** Predict the next element in simple synthetic sequences (e.g., `1,2,3,4,?` or `a,b,a,b,?` or Fibonacci).

**`prepare.py`:** Generates sequence data with a few pattern types.

**`train.py`:** Agent starts with a simple lookup table or linear model. Can experiment with:
- RNNs, attention, MLPs
- Sequence length
- Embedding size
- Hidden dimensions

**Metric:** Validation accuracy or cross-entropy.

**Why it's good:** It's a micro version of the actual autoresearch problem (language modeling), which makes the lessons directly transferable. But it runs in seconds, not minutes.

---

## Recommendation

**Go with Idea 2 (Polynomial/Curve Fitting)** as the sweet spot:

| Criteria | Linear Reg | Curve Fitting | Classification | Sequences |
|---|---|---|---|---|
| Simplicity | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| Experiment diversity | ⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Overfitting tradeoff | ❌ | ✅ | ✅ | ✅ |
| Runs instantly | ✅ | ✅ | ✅ | ✅ |
| Dependencies needed | numpy only | numpy only | numpy/sklearn | torch |
| "Aha" moments | 1 (closed form) | Many | Some | Many |

The curve fitting problem has the right balance: **simple enough to understand in seconds, rich enough that the agent can run 50+ meaningful experiments**. The overfitting tradeoff is the key — it gives the agent a real problem to navigate, not just a knob to turn.

## Concrete Implementation Sketch

```
prepare.py:
  - Generate y = sin(2πx) + 0.3*noise, x ∈ [0, 1], 200 train / 100 val points
  - Fixed random seed
  - evaluate_mse(predictions, y_val) → single float
  - TIME_BUDGET = 10 seconds (or even 5)

train.py (starting point):
  - Simple linear regression: y = w*x + b
  - SGD for TIME_BUDGET seconds
  - Print val_mse at the end

program.md:
  - Same loop structure as current autoresearch
  - Metric: val_mse (lower is better)
  - Agent modifies train.py only
```

The agent would then discover it needs polynomial features, figure out the right degree, maybe try regularization, maybe try a small neural net — all in seconds per experiment instead of 5 minutes.
