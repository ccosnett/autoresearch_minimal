# autoresearch_minimal

A minimal, CPU-friendly fork of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — same autonomous AI experiment loop, but on a toy problem you can run on any laptop.

## What is this?

The original autoresearch lets an AI agent autonomously improve a GPT training setup on a GPU. This fork replaces the LLM with a **toy linear regression problem** (5 features, synthetic data) so you can explore the same experiment loop with **no GPU required**. Experiments run in seconds instead of minutes.

| | Upstream | This fork |
|---|---|---|
| **Problem** | GPT training | Linear regression |
| **Hardware** | NVIDIA GPU | CPU only |
| **Time/experiment** | ~5 min | ~seconds |
| **Metric** | `val_bpb` | `val_mse` |
| **Dependencies** | PyTorch + GPU | NumPy |

## Quick start

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run prepare.py    # generate synthetic data
uv run train.py      # run baseline
```

## Running the agent

Point Claude/Codex at `program.md` and let it go. Experiments take seconds, so you can run hundreds in minutes.

## Project structure

```
prepare.py      — data generation + evaluation (do not modify)
train.py        — linear regression training (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## What can the agent explore?

- Learning rate, momentum, iterations
- Regularization (L1, L2)
- Closed-form solution (normal equation)
- Feature normalization
- Different optimizers

## License

MIT
