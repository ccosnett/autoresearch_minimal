# Autoresearch: A Bedtime Story About Autonomous AI Research

*Settle in, grab your tea, and let us walk you through one of the most fascinating open-source experiments of 2026: teaching an AI agent to do its own machine learning research while you sleep.*

---

## The Big Idea

Imagine this: you go to bed at midnight. While you dream, an AI agent is sitting at your GPU, running experiment after experiment on a small language model. It tweaks the architecture, adjusts hyperparameters, tries wild ideas, keeps what works, throws away what doesn't. By the time you wake up, there's a neat log of ~100 experiments and -- hopefully -- a measurably better model than the one you left it with.

That's **autoresearch**.

It was born from a simple question by [Andrej Karpathy](https://github.com/karpathy): *What happens if you give an LLM coding agent a real training setup and tell it to optimize freely?* The answer turned out to be surprisingly compelling. The agent doesn't just randomly search -- it reasons about what to try, learns from failures, and iterates intelligently.

As the README's opening quote (attributed to Karpathy, March 2026) imagines:

> *One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone.*

This repo is, as the quote says, "the story of how it all began."

---

## What's in the Box: Repository Map

The repo is deliberately tiny. Every file earns its place:

```
autoresearch_minimal/
  prepare.py          -- Data download, tokenizer training, and runtime utilities
  train.py            -- The GPT model, optimizer, and training loop (the ONE file the agent edits)
  program.md          -- Instructions for the AI agent (the ONE file the human edits)
  analysis.ipynb      -- Jupyter notebook for visualizing experiment results
  progress.png        -- Generated chart of experiment progress over time
  pyproject.toml      -- Python project config and dependencies
  .python-version     -- Pins Python 3.10
  .gitignore          -- Keeps things tidy
  .github/workflows/
    claude.yml        -- GitHub Actions workflow for Claude Code integration
  README.md           -- Project overview and quick start guide
```

Let's go through each piece in detail.

---

## Chapter 1: The Foundation -- `prepare.py`

**Role:** The bedrock. Downloads data, trains a tokenizer, and provides runtime utilities. This file is **read-only** -- the agent must never touch it. It defines the rules of the game.

### The Constants (The Rules)

Three sacred numbers define every experiment:

| Constant | Value | Meaning |
|----------|-------|---------|
| `MAX_SEQ_LEN` | 2048 | Context window length (how many tokens the model "sees" at once) |
| `TIME_BUDGET` | 300 | Training time in seconds (exactly 5 minutes, wall clock) |
| `EVAL_TOKENS` | ~21M | How many tokens are used to evaluate the model's quality |

The 5-minute time budget is a masterstroke of experimental design. It doesn't matter if the agent makes the model huge or tiny, uses a clever optimizer or a naive one -- every experiment gets exactly 5 minutes of training. This makes results directly comparable and means the agent is searching for *the best model you can train in 5 minutes on your specific hardware*.

### Data: The ClimbMix-400B Dataset

The training data comes from Karpathy's [climbmix-400b-shuffle](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) dataset on HuggingFace. It's stored as thousands of Parquet shards (up to shard 6542). By default, 10 training shards are downloaded, plus one pinned validation shard (shard 6542, always the same, so evaluation is consistent).

The download system is robust:
- Retries up to 5 times with exponential backoff
- Uses temporary files to avoid corruption from interrupted downloads
- Parallelizes across 8 workers by default
- Skips shards that are already downloaded

### The Tokenizer: BPE with a Rust Engine

The tokenizer is a Byte-Pair Encoding (BPE) tokenizer -- the same family used by GPT-4. It's trained using `rustbpe` (a Rust-based BPE implementation for speed) and then wrapped in `tiktoken` (OpenAI's tokenizer library) for compatibility.

Key specs:
- **Vocabulary size:** 8,192 tokens (deliberately small -- this is a minimal setup)
- **Split pattern:** GPT-4 style regex, handling contractions, Unicode, numbers, and whitespace
- **Special tokens:** 4 reserved tokens, with `<|reserved_0|>` used as BOS (Beginning of Sequence)

After training, a `token_bytes.pt` lookup table is saved. This maps each token ID to its byte length in UTF-8 -- critical for the evaluation metric (more on that shortly).

### The Dataloader: Best-Fit Packing

The dataloader (`make_dataloader`) is surprisingly sophisticated for a "minimal" repo. It uses **BOS-aligned best-fit packing**:

1. Every row in the batch starts with a BOS token
2. Documents are packed into fixed-length rows using a best-fit algorithm (like bin packing)
3. The largest document that fits the remaining space is chosen
4. If nothing fits, the shortest document is cropped to fill exactly
5. Result: **100% utilization** -- no padding tokens wasted

This is an infinite iterator that cycles through shards, tracking epochs. It pre-allocates pinned CPU buffers and GPU buffers for efficient async data transfer.

### The Metric: Bits Per Byte (BPB)

The evaluation function `evaluate_bpb` computes **bits per byte** -- a vocabulary-size-independent metric. This is crucial because it means the agent could change the vocab size and results would still be comparable.

How it works:
1. Run the model on validation data
2. For each predicted token, compute the cross-entropy loss (in nats)
3. Multiply by a mask that excludes special tokens
4. Sum the loss and the byte lengths of all target tokens
5. Convert: `BPB = total_nats / (ln(2) * total_bytes)`

Lower is better. A BPB of 1.0 means the model uses 1 bit per byte on average to encode the text -- quite compressed!

---

## Chapter 2: The Arena -- `train.py`

**Role:** This is where the magic happens. The complete GPT model, a custom dual optimizer, and the training loop -- all in one file. This is the **only file the agent modifies**.

### The Model: A Modern GPT

The model is a simplified version of [nanochat](https://github.com/karpathy/nanochat), incorporating several recent advances in transformer architecture:

#### Architecture Config (Defaults)

```python
DEPTH = 8                # Number of transformer layers
ASPECT_RATIO = 64        # model_dim = depth * aspect_ratio = 512
HEAD_DIM = 128           # Attention head dimension
WINDOW_PATTERN = "SSSL"  # Sliding window pattern
```

With these defaults, the model has:
- **8 layers** of transformer blocks
- **512-dimensional** embeddings (8 * 64, rounded up to nearest HEAD_DIM multiple)
- **4 attention heads** (512 / 128)
- Roughly **50M parameters**

#### Modern Tricks Inside

**1. RMS Normalization**
Instead of LayerNorm, the model uses RMS normalization (`F.rms_norm`) -- simpler, no mean subtraction, no learnable parameters. Applied before attention and MLP (Pre-Norm style).

**2. Rotary Position Embeddings (RoPE)**
Positions are encoded using rotary embeddings rather than learned position embeddings. RoPE encodes relative position information directly into the attention computation by rotating query and key vectors. The implementation pre-computes sin/cos tables for up to 10x the sequence length.

**3. Sliding Window Attention**
The `WINDOW_PATTERN = "SSSL"` means: three layers of **S**hort (half-context) attention windows, then one layer of **L**ong (full-context) attention. This alternating pattern is cheaper than full attention everywhere while still allowing information to propagate across the full context. The last layer always gets full attention regardless of the pattern.

**4. Value Embeddings (ResFormer / Value Residual)**
Alternating layers have a "value embedding" -- an additional embedding lookup that gets mixed into the attention values with an input-dependent gate. The gate uses `2 * sigmoid(linear(x[:32]))`, initialized to produce a neutral scale of 1.0. This is inspired by the ResFormer paper and helps with training stability.

**5. Residual Lambdas (Per-Layer Scalable Residual)**
Instead of a plain residual connection `x = x + block(x)`, the model uses:
```python
x = resid_lambda[i] * x + x0_lambda[i] * x0
```
where `x0` is the original embedding. This gives the model per-layer control over how much to weight the residual stream vs. the original input -- a technique that helps with deep network training.

**6. Logit Soft-Capping**
Output logits are capped using `15 * tanh(logits / 15)`. This prevents logit explosion during training, acting as a smooth clamp that keeps gradients flowing.

**7. Squared ReLU Activation**
The MLP uses `ReLU(x)^2` instead of the more common GELU or SwiGLU. Squared ReLU tends to produce sparser activations.

**8. Flash Attention 3**
Attention is computed using Flash Attention 3 (via the `kernels` package). On Hopper GPUs (H100), it uses `varunneal/flash-attention-3`; on other architectures, it falls back to `kernels-community/flash-attn3`.

#### Weight Initialization

The initialization is carefully tuned:
- Embedding weights: normal distribution with std=1.0
- LM head (unembedding): normal with std=0.001 (near-zero, learns from gradients)
- Query, key, value projections: uniform in `[-s, s]` where `s = sqrt(3) / sqrt(n_embd)`
- Output projections: zeros (so residual connections start as identity)
- Residual lambdas: 1.0 (identity), x0 lambdas: 0.1 (small initial mixing)
- VE gate weights: zeros (neutral 1.0 scaling via sigmoid)

### The Optimizer: MuonAdamW

This is not your garden-variety optimizer. It's a **hybrid** that uses two different algorithms for different parameter types:

#### AdamW (for 1D parameters)
Used for: embeddings, unembedding (lm_head), scalar parameters (residual lambdas)

Standard AdamW with bias correction, compiled with `torch.compile` for maximum performance. Different learning rates for different parameter groups:
- Embedding LR: 0.6
- Unembedding LR: 0.004
- Scalar LR: 0.5

All learning rates are scaled by `1/sqrt(model_dim/768)` -- a width-dependent scaling rule that keeps things stable as the model grows.

#### Muon (for 2D matrix parameters)
Used for: all weight matrices in attention and MLP layers

Muon is a matrix-aware optimizer that uses **polar decomposition** for orthogonalization. Instead of traditional SGD with momentum, it:

1. **Nesterov momentum** on the gradient
2. **Polar Express orthogonalization**: an iterative approximation to polar decomposition using polynomial coefficients (the mysterious `polar_express_coeffs`). This projects the gradient onto the nearest orthogonal matrix, which acts as a natural preconditioner
3. **NorMuon variance reduction**: normalizes the update using a running second moment, similar to Adam's adaptive learning rate but operating on the matrix level
4. **Cautious weight decay**: only applies weight decay where the gradient and parameter have the same sign (a recent trick that prevents decay from fighting the gradient)

The Muon parameters are grouped by shape and processed as stacked tensors for efficiency. Everything is `torch.compile`d.

### Learning Rate Schedule

The schedule is time-based (not step-based), using `progress = training_time / TIME_BUDGET`:

```
|--- warmup ---|--- constant ---|--- warmdown ---|
0%             WARMUP_RATIO    (1-WARMDOWN_RATIO)  100%
```

With defaults: 0% warmup, 50% warmdown to 0.0 final LR. So the model trains at full LR for the first half, then linearly decays to zero.

Muon momentum also warms up from 0.85 to 0.95 over the first 300 steps. Weight decay linearly decays to zero by the end of training.

### The Training Loop

The loop is beautifully simple:

```
while training_time < 5 minutes:
    for each micro_step in gradient_accumulation:
        forward pass (with autocast to bfloat16)
        backward pass
        load next batch
    update learning rates based on wall-clock progress
    optimizer step
    zero gradients
    log metrics
```

Key details:
- **Gradient accumulation**: with `TOTAL_BATCH_SIZE = 2^19` (~524K tokens) and `DEVICE_BATCH_SIZE = 128`, there are `524288 / (128 * 2048) = 2` accumulation steps per optimizer step
- **First 10 steps are warmup**: training time doesn't count until step 11 (to exclude torch.compile overhead)
- **Garbage collection**: Python's GC is frozen after step 0 and only unfrozen every 5000 steps to avoid ~500ms stalls
- **Fast fail**: if loss goes NaN or exceeds 100, the script exits immediately
- **EMA smoothing**: training loss is displayed as an exponential moving average (beta=0.9) for readability

After training completes, the model is evaluated on the validation set and a summary is printed:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

---

## Chapter 3: The Director -- `program.md`

**Role:** This is the "soul" of the autonomous agent. It's a Markdown file that tells the AI what to do, how to do it, and what the rules are. The human writes this; the agent follows it.

### The Setup Phase

Before experiments begin, the agent must:
1. Agree on a **run tag** (e.g., `mar5`) and create a branch `autoresearch/<tag>`
2. Read all in-scope files for context
3. Verify data exists (check `~/.cache/autoresearch/`)
4. Create a `results.tsv` file with a header row
5. Run the **baseline** -- the unmodified `train.py` -- to establish a starting val_bpb

### The Experiment Loop

Then the agent enters an infinite loop:

```
LOOP FOREVER:
  1. Look at git state
  2. Edit train.py with an experimental idea
  3. git commit
  4. Run: uv run train.py > run.log 2>&1
  5. Read results: grep "^val_bpb:" run.log
  6. If crash: read traceback, maybe fix, maybe skip
  7. Log to results.tsv
  8. If improved: KEEP (advance branch)
  9. If not: DISCARD (git reset to previous state)
```

The key insight: **the agent never stops**. The `program.md` explicitly says:

> *NEVER STOP. The human might be asleep. You are autonomous. If you run out of ideas, think harder.*

### The Results Log

Results are tracked in `results.tsv` (tab-separated, intentionally not committed to git):

| commit | val_bpb | memory_gb | status | description |
|--------|---------|-----------|--------|-------------|
| a1b2c3d | 0.997900 | 44.0 | keep | baseline |
| b2c3d4e | 0.993200 | 44.2 | keep | increase LR to 0.04 |
| c3d4e5f | 1.005000 | 44.0 | discard | switch to GeLU activation |

### The Simplicity Criterion

One of the most interesting design choices: complexity has a cost. From `program.md`:

> *A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep.*

This is an elegant constraint that prevents the agent from turning `train.py` into an unreadable mess of micro-optimizations.

---

## Chapter 4: The Analysis -- `analysis.ipynb`

**Role:** A Jupyter notebook for visualizing experiment results after a run. It reads `results.tsv` and generates charts.

The notebook does four things:

1. **Load and parse** `results.tsv`, converting to numeric types
2. **Count outcomes**: how many experiments were kept, discarded, or crashed (and the keep rate)
3. **Plot progress**: a scatter plot showing val_bpb over time, with:
   - Green dots for kept experiments (with labeled descriptions)
   - Gray dots for discarded experiments
   - A step-line showing the running best
4. **Rank improvements**: sort kept experiments by their delta improvement over the previous best

The output is saved as `progress.png`, which is displayed in the README as the teaser image.

---

## Chapter 5: The Infrastructure

### Dependencies (`pyproject.toml`)

The project uses [uv](https://docs.astral.sh/uv/) as the package manager. Dependencies:

| Package | Purpose |
|---------|---------|
| `torch==2.9.1` | PyTorch (pinned, from CUDA 12.8 index) |
| `kernels>=0.11.7` | Flash Attention 3 kernel loading |
| `rustbpe>=0.1.0` | Rust-based BPE tokenizer training |
| `tiktoken>=0.11.0` | Tokenizer runtime (OpenAI's library) |
| `pyarrow>=21.0.0` | Reading Parquet data files |
| `requests>=2.32.0` | Downloading data shards from HuggingFace |
| `numpy`, `pandas`, `matplotlib` | Analysis and visualization |

### CI/CD: Claude Code Integration (`.github/workflows/claude.yml`)

The repo includes a GitHub Actions workflow that triggers Claude Code when someone mentions `@claude` in an issue or PR comment. This lets the AI agent respond to requests directly on GitHub -- like the one that created this very document!

### Git Hygiene (`.gitignore`)

The `.gitignore` keeps the repo clean by excluding:
- Python artifacts (`__pycache__`, `.pyc`, etc.)
- Virtual environments (`.venv`)
- Generated agent files (`CLAUDE.md`, `AGENTS.md`)
- Experiment results (`results.tsv`, `results/`)
- Work-in-progress directories (`dev/`, `worktrees/`, `queue/`)

---

## Chapter 6: The Design Philosophy

### Why One File?

The agent only touches `train.py`. This is a deliberate constraint:
- **Reviewable diffs**: you can `git log` and see exactly what changed
- **Manageable scope**: the agent can't accidentally break data loading or evaluation
- **Fair comparison**: the evaluation harness is fixed, so improvements are real

### Why 5 Minutes?

Fixed wall-clock time is clever for several reasons:
- **Hardware-agnostic comparison**: whether you have an H100 or an RTX 4090, every experiment takes the same time, and the metric reflects what's optimal *for your hardware*
- **Predictable throughput**: ~12 experiments per hour, ~100 overnight
- **Architectural freedom**: the agent can make the model bigger or smaller without worrying about training time -- the time budget automatically adapts

### Why Bits Per Byte?

BPB is vocab-size-independent. If the agent decides to change the vocabulary size (or if different forks use different tokenizers), results are still comparable. It measures how efficiently the model compresses text, measured in bits per byte of the original UTF-8 text.

### The Human-Agent Division of Labor

This is perhaps the most profound design choice:

| | Human writes | Agent writes |
|---|---|---|
| **What** | `program.md` (agent instructions) | `train.py` (model code) |
| **Like** | A research director writing a grant proposal | A researcher running experiments |
| **Iterates on** | The "research org code" -- how to direct AI research | The actual ML code -- architectures, hyperparams, optimizers |

The human doesn't write Python. The human writes *instructions in English*. The Python is entirely the agent's domain. This is a fundamentally different paradigm from traditional ML research.

---

## Chapter 7: A Deeper Look at the Math

For the mathematically curious reader who isn't quite asleep yet...

### Polar Express Orthogonalization

The Muon optimizer's secret weapon. Given a gradient matrix G, we want to find the nearest orthogonal matrix (in Frobenius norm). The exact solution is `U @ V^T` from the SVD `G = U S V^T`, but SVD is expensive.

The "Polar Express" is a polynomial iteration that converges to this solution:

```python
X = G / (||G|| * 1.02)  # normalize
for a, b, c in coefficients:
    A = X^T @ X           # (or X @ X^T for tall matrices)
    B = b*A + c*(A @ A)   # cubic polynomial
    X = a*X + X @ B       # update
```

After 5 iterations (the default `ns_steps`), X is approximately the orthogonal polar factor of G. The coefficients are pre-computed to maximize convergence speed.

Why orthogonalize? It's a form of natural gradient -- it accounts for the geometry of the parameter space, producing updates that are more "uniform" across directions. This tends to work much better than plain SGD for matrix-valued parameters.

### The NorMuon Variance Reduction

After orthogonalization, the update is further refined:
1. Compute per-row (or per-column) variance of the update
2. Maintain a running EMA of this variance (like Adam's second moment)
3. Scale the update by `1/sqrt(running_variance)` per row/column
4. Renormalize to preserve the original update magnitude

This combines the benefits of orthogonal updates with per-dimension adaptive learning rates.

### Cautious Weight Decay

Standard weight decay: `p = p - lr * wd * p` (always shrinks parameters toward zero).

Cautious weight decay: `p = p - lr * wd * p * mask` where `mask = (grad * p) >= 0`.

This only decays a parameter when the gradient is "pushing in the same direction" as the parameter value. If the gradient says "make this bigger" and the parameter is positive, decay is applied. If the gradient says "make this smaller" but the parameter is positive, no decay -- don't fight the gradient.

---

## Chapter 8: How to Run It Yourself

### Prerequisites
- An NVIDIA GPU (H100 ideal, but see forks for other hardware)
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

### Quick Start

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Download data and train tokenizer (~2 min)
uv run prepare.py

# Run a single training experiment (~5 min)
uv run train.py
```

### Going Autonomous

Point your favorite AI coding agent (Claude, Codex, etc.) at the repo and say:

> Hi, have a look at program.md and let's kick off a new experiment! Let's do the setup first.

Then walk away. Come back to a log of experiments and a (hopefully) improved model.

### For Smaller Hardware

If you don't have an H100, consider:
1. Use a simpler dataset (e.g., [TinyStories](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean))
2. Decrease `vocab_size` (down to 4096, 2048, or even 256 for byte-level)
3. Lower `MAX_SEQ_LEN` (down to 256 if needed)
4. Decrease `DEPTH` (from 8 to 4)
5. Use `WINDOW_PATTERN = "L"` (skip sliding window)
6. Lower `TOTAL_BATCH_SIZE` (down to `2^14`)

Or check out the community forks for macOS, Windows, and AMD support.

---

## Epilogue: Why This Matters

Autoresearch sits at a fascinating intersection. It's not just an ML training script -- it's an experiment in **meta-research**: can we automate the process of doing research itself?

The traditional ML workflow is: human has idea, human writes code, human runs experiment, human analyzes results, human has next idea. Autoresearch compresses this into: human sets direction (via `program.md`), agent does everything else.

The results so far suggest that LLM agents are surprisingly competent at this. They try sensible things, learn from failures, and make genuine improvements. They don't match a top ML researcher's intuition, but they can explore a vast search space tirelessly and systematically -- which, for hyperparameter optimization at least, is exactly what you want.

And the beauty of the setup is its simplicity. Three files. One metric. One GPU. Five minutes. Everything else is up to the agent.

Sweet dreams.

---

*Generated with [Claude Code](https://claude.ai/code) for issue #2.*
