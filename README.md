# Autonomous AI Grid Optimizer

A carbon-aware, RL-controlled microgrid with multi-agent orchestration using LangGraph.

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. (Optional) Regenerate datasets: `python data/build_datasets.py --train-days 21 --eval-days 7`
3. Run training: `python train.py`
4. Evaluate: `python evaluate.py`
5. Dashboard: `streamlit run dashboard/app.py`

The dataset builder synthesizes multi-day solar/load/price/carbon profiles and writes `data/train_data.csv` and `data/eval_data.csv`. Point the scripts to your own telemetry by overriding `--data-path` (Ray launcher) or the `data_path` argument in `MicrogridEnv`.

## Environment Knobs

`env.microgrid_env.MicrogridEnv` now supports:

- `episode_length`: roll fixed-width windows (default = entire CSV).
- `reward_weights`: dict with `cost`, `carbon`, `battery_penalty`, `unmet_demand`, `export_credit`.
- `grid_import_limit`: cap imports per step to induce unmet demand penalties.
- `log_path`: CSV file capturing every step (`time, action, solar, demand, grid_import, …`).

Example:

```python
env = MicrogridEnv(
    data_path="data/train_data.csv",
    episode_length=168,
    reward_weights={"cost": 1.0, "carbon": 0.2, "export_credit": 0.4},
    log_path="logs/train_steps.csv",
    grid_import_limit=3.0,
)
```

## Live Carbon Intensity (NESO API)

Great Britain’s National Energy System Operator exposes half-hourly national and regional carbon intensity endpoints such as `/intensity/{from}/{to}` and `/regional/intensity/{from}/{to}/regionid/{regionid}` (region ids 1–17 cover each DNO plus GB aggregates).citeturn4view0turn2view0

Run `python data/fetch_carbon_intensity.py --hours 48 --region-id 13 --merge-dataset data/eval_data.csv` to:

1. Download the latest 48 h forecast for London (region 13) into `data/carbon_intensity_live.csv`.
2. Replace any overlapping `carbon_intensity` values in `data/eval_data.csv` with the chosen column (`--merge-column forecast|actual`).

The script accepts outward postcodes (e.g., `--postcode SW1`), merges via nearest-timestamp matching (±30 min), and can also leave the live CSV standalone if you want the model to learn from historical values later.

## High-Performance / Distributed Training

Ray + RLlib integration enables large-scale PPO runs across many CPU or GPU workers.

### Local multi-core

```
python -m hpc.train_distributed \
  --num-workers 8 \
  --envs-per-worker 4 \
  --data-path data/train_data.csv \
  --episode-length 168 \
  --reward-weight-carbon 0.2 \
  --reward-weight-battery 2.0 \
  --reward-weight-unmet 150.0 \
  --reward-weight-export 0.4 \
  --step-log-dir logs/ray-steps \
  --train-batch-size 65536 \
  --checkpoint-freq 2 \
  --output-dir checkpoints/local-ray
```

Add `--ray-address auto` to attach to an existing Ray head (e.g., `ray start --head --port 6379`). Use `--local-mode` for quick debugging without spinning up workers.

### SLURM / HPC clusters

1. Edit `hpc/slurm_job.sbatch` with your module/conda commands and scratch paths.
2. Submit via `sbatch hpc/slurm_job.sbatch`.
3. Let additional nodes join the Ray head (`ray start --address <head-ip>:6379`) to scale horizontally.

The script exposes extra knobs (`--episode-length`, `--reward-weight-*`, `--step-log-dir`, `--checkpoint-freq`, `--target-reward`, etc.) so you can integrate with monitoring stacks or stop automatically once a quality bar is hit. Each worker writes its own CSV under `--step-log-dir`, making it easy to inspect how PPO behaves over time.

## Per-step Diagnostics & Dashboard

Training/eval scripts append to `logs/train_steps.csv` and `logs/eval_steps.csv` (one row per environment step). The Streamlit dashboard visualizes:

- Raw solar vs. demand curves from either dataset split.
- Aggregated reward/cost/emission summaries derived from the per-step logs (when available).

To inspect logs manually:

```
python -c "import pandas as pd; print(pd.read_csv('logs/train_steps.csv').head())"
```

## Project Structure

- `data/`: Dataset builder + generated CSVs (`train_data.csv`, `eval_data.csv`)
- `data/fetch_carbon_intensity.py`: NESO Carbon Intensity fetch/merge helper
- `env/`: Gymnasium environment
- `agents/`: Agent modules
- `graph/`: LangGraph orchestration
- `baselines/`: Baseline strategies
- `dashboard/`: Streamlit UI
- `hpc/`: Ray-powered training entry point and SLURM template
- `train.py`: RL training script
- `evaluate.py`: Evaluation script
