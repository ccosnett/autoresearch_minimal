# README Update Plan

## Goal

Rewrite `README.md` to clearly explain what **autoresearch_minimal** is: a stripped-down, CPU-friendly fork of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) that replaces the GPU-heavy LLM training loop with a toy linear regression problem — same autonomous experiment loop, zero GPU required.

## Key Differences from Upstream to Highlight

| Aspect | Karpathy's autoresearch | This fork (autoresearch_minimal) |
|---|---|---|
| **Problem** | GPT language model training | Linear regression (5 features, synthetic data) |
| **Hardware** | Single NVIDIA GPU (H100) | CPU only — runs anywhere |
| **Time per experiment** | 5 minutes | ~seconds |
| **Metric** | `val_bpb` (bits per byte) | `val_mse` (mean squared error) |
| **Dependencies** | PyTorch + GPU packages | NumPy only (torch in deps but not used by default) |
| **Agent search space** | Architecture, optimizer, batch size, etc. | Learning rate, iterations, regularization, closed-form solutions, feature engineering |

## Proposed README Structure

### 1. Title & One-Liner
- `# autoresearch_minimal`
- One sentence: "A minimal, CPU-friendly fork of Karpathy's autoresearch — same autonomous AI experiment loop, but on a toy problem you can run on any laptop."

### 2. What Is This?
- 2–3 sentences explaining the fork relationship.
- Link to upstream repo.
- Explain why: lower the barrier to entry, let people experiment with the autoresearch loop without needing a GPU.

### 3. How It Works
- Same three-file structure as upstream:
  - `prepare.py` — data generation + evaluation (fixed, do not modify)
  - `train.py` — the file the agent edits
  - `program.md` — agent instructions
- The problem: fit `y = Wx + b` on synthetic data (5 features, 500 train / 200 val samples).
- Metric: `val_mse` (lower is better).
- Time budget: 10 seconds per experiment.

### 4. Quick Start
```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run prepare.py    # verify data generation
uv run train.py      # run baseline (~seconds)
```

### 5. Running the Agent
- Same instructions as upstream: point Claude/Codex at `program.md`.
- Mention that experiments run in seconds, so you can run hundreds in minutes instead of overnight.

### 6. Project Structure
```
prepare.py      — constants, synthetic data generation, evaluation (do not modify)
train.py        — linear regression training (agent modifies this)
program.md      — agent instructions / experiment loop
ideation.md     — notes on toy problem selection
pyproject.toml  — dependencies
```

### 7. What Can the Agent Explore?
Brief list of the kinds of experiments the agent can try:
- Learning rate, momentum, number of iterations
- Regularization (L1, L2)
- Closed-form solution (normal equation)
- Feature normalization
- Different optimizers

### 8. License
- MIT (same as upstream)

## Sections to Remove (from current README)

- The Karpathy quote (not relevant to this fork)
- GPU/platform support discussion (this fork is CPU-only)
- Notable forks section (those are forks of upstream, not this repo)
- Detailed tuning recommendations for smaller compute (not needed — this already *is* the small-compute version)
- References to nanochat, GPT model, BPE tokenizer, etc.

## Writing Guidelines

- Keep it under 80 lines if possible — this is the *minimal* version, the README should be minimal too.
- Link to upstream for people who want the full LLM version.
- Don't repeat information that's already in `program.md`.
- Use the same tone as upstream (direct, technical, concise).
