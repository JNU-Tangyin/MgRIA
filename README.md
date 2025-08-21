# What to Recommend is When to Recommend: Modeling Multi-Granularity Repurchase Cycles

State-of-the-art sequential recommenders excel at predicting *what* a user will buy next, yet often fail to predict *when*. This paper argues this is due to a flawed assumption: that repurchase intervals follow a single, simple pattern. We empirically demonstrate that the distribution of repeated-purchase intervals (DRPI) is, in fact, a complex mixture: a dominant power-law trend overlaid with multiple periodic spikes at weekly, monthly, and other granularities. We formalize this as the **Principle of Multi-Granularity Repurchase Cycles**. Ignoring this multi-modal reality introduces systematic timing bias, especially for frequently repurchased items.
To solve this, we propose MgRIA, a novel recommendation paradigm that explicitly models these cycles. MgRIA uses a multi-granularity timestamp embedding to disentangle coexisting periodicities and a distribution-aware scoring mechanism to predict repurchase likelihood over time. Across three real-world datasets, MgRIA significantly outperforms strong baselines on both standard (Recall/nDCG) and time-aware (Time-MRR) metrics. The model also provides interpretability by revealing the specific repurchase cycles driving its predictions. By operationalizing our discovered principle, MgRIA bridges the gap between predicting *what* and *when*.

recommendation system; repurchase time interval; multi-granularity; mixture distribution; attention mechanism

# Repository Structure (refactored)

- **configs/** YAML configs (e.g., `configs/default.yaml`)
- **src/** core library

  - `src/config/`: dataclass configs and constants
  - `src/data/`: dataset, dataloader, real data loader
  - `src/models/`: model factory, MgRIA model and loss, outputs
  - `src/training/`: `MgRIATrainer` (train/eval, checkpointing)
  - `src/utils/`: helpers (logging, seeds, device)
- **scripts/**

  - `scripts/examples/experiment_synthetic.py`: synthetic quick-check of the refactored stack
- **results/** auto-created; each run saves history/metrics/checkpoints
- **figures/** paper plots
- **tables/** LaTeX tables
- **README.md**, **requirements.txt**

## Usage

1. Install Python 3.9 and dependencies

```bash
pip install -r requirements.txt
```

2. Prepare data

Follow `configs/default.yaml` for dataset settings. Real datasets are loaded via `src/data/real_data.py` and `src/data/dataloader.py`.

3. Train and evaluate (primary entry)

```bash
# Tafeng (default), 5 epochs cap
python main.py --dataset tafeng

# Equity
python main.py --dataset equity

# Taobao with custom epochs
python main.py --dataset taobao --epochs 10
```

Each run creates a timestamped folder under `results/<dataset>/YYYYMMDD_HHMMSS/` containing:

- `history.json` with per-epoch train/eval loss
- `metrics.json` with top-k metrics
- `loss_curve.png` automatic loss plot (requires matplotlib)
- `../<dataset>_checkpoints/` with `best_model.pt` and periodic checkpoints

4. Quick synthetic example (sanity check)

```bash
python scripts/examples/experiment_synthetic.py
```

This script instantiates the model, runs a short synthetic training loop, tests predict/ensemble, and logs outputs.
