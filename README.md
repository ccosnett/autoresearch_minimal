# autoresearch_minimal

A fork of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) that solves a 2-variable linear regression problem on CPU.

| | Upstream | This fork |
|---|---|---|
| **Problem** | GPT training | Linear regression |
| **Hardware** | NVIDIA GPU | CPU only |
| **Time/experiment** | ~5 min | ~seconds |
| **Metric** | `val_bpb` | `val_mse` |
| **Dependencies** | PyTorch + GPU | NumPy |

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  AI agent   │────>│ modify      │────>│ evaluate     │
│  reads      │     │ train.py    │     │ val_mse      │
│  program.md │<────│             │<────│              │
└─────────────┘     └─────────────┘     └─────────────┘
```

## Quick start

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run prepare.py    # generate synthetic data
uv run train.py      # run baseline
```

## License

MIT
