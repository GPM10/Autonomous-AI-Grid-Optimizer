# Autonomous AI Grid Optimizer

A carbon-aware, RL-controlled microgrid with multi-agent orchestration using LangGraph.

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. (Optional) Regenerate datasets: `python data/build_datasets.py --train-days 21 --eval-days 7`
3. OPSD SQLite slice (optional): `python data/export_opsd_sqlite.py --country GB_GBN --start 2020-01-01T00:00:00Z --end 2020-01-07T23:00:00Z --resolution 60 --output data/train_data.csv`
4. Run training: `python train.py`
5. Evaluate: `python evaluate.py`
6. Dashboard: `streamlit run dashboard/app.py`

The dataset builder synthesizes multi-day solar/load/price/carbon profiles and writes `data/train_data.csv` and `data/eval_data.csv`. Point the scripts to your own telemetry by overriding `--data-path` (Ray launcher) or the `data_path` argument in `MicrogridEnv`.

### Training & Evaluation Tips

- When PyPI access is restricted, install the core dependencies manually. At minimum you'll need `torch`, `pandas`, and `stable-baselines3` (Ray/RLlib is only required for the distributed launcher).
- Run `python train.py` to (re)generate `artifacts/ppo_microgrid.zip` and `logs/train_steps.csv`. This script now honors the PINN-related kwargs set inside the env factory.
- Run `python evaluate.py` to compare the PPO checkpoint against a random baseline. Results land in `logs/eval_steps.csv`, which the Streamlit dashboard can visualize alongside LangGraph baselines.


### Using the OPSD SQLite database

- Download `time_series.sqlite` from [Open Power System Data](https://data.open-power-system-data.org/time_series/?utm_source=openai) and place it in `data/`.
- Export a country slice to CSV:

```
python data/export_opsd_sqlite.py \
  --country GB_GBN \
  --start 2020-01-01T00:00:00Z \
  --end 2020-01-07T23:00:00Z \
  --resolution 60 \
  --output data/gb_eval.csv
```

- Or bypass CSVs entirely inside Python:

```python
from data.export_opsd_sqlite import extract_opsd_slice
from env.microgrid_env import MicrogridEnv

df = extract_opsd_slice(
    db_path="data/time_series.sqlite",
    country="GB_GBN",
    start="2020-01-01T00:00:00Z",
    end="2020-01-07T23:00:00Z",
    resolution=60,
)
env = MicrogridEnv(data_frame=df)
```

This keeps your training/eval datasets synced with the upstream OPSD snapshot.

### Physics-Informed Battery Thermal Model

- Enable the optional PINN-driven thermal constraints by passing `use_pinn=True` (and optionally `ambient_temperature`, `max_battery_temp`) when constructing `MicrogridEnv` or when wiring up the LangGraph baseline.
- The model lives in `pinn/battery_temperature.py` and learns a correction term on top of a simple heat-transfer ODE, penalizing the RL agent whenever battery temperatures exceed safe limits.

Example:

```python
from env.microgrid_env import MicrogridEnv

env = MicrogridEnv(
    data_path="data/train_data.csv",
    use_pinn=True,
    ambient_temperature=24.0,
    max_battery_temp=45.0,
)
```

Each step logs `battery_temp` to the per-step CSV and applies `thermal_penalty` weights when overheating occurs, encouraging physics-consistent behavior.

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

Already downloaded the dataset from the NESO portal? Pass `--local-csv data/regional_carbon_intensity.csv` to reuse it offline:

```
python data/fetch_carbon_intensity.py --local-csv data/regional_carbon_intensity.csv \
  --merge-dataset data/eval_data.csv --merge-column forecast
```

## LangGraph Multi-Agent Baseline

The `agents/` package now includes lightweight forecasting and control agents:

- `SolarForecaster` and `DemandForecaster` compute short-horizon estimates using rolling means + diurnal heuristics.
- `HybridController` picks actions (idle, charge, discharge, import) that balance carbon intensity, price, and battery state.

`graph/langgraph_flow.py` wires these agents into a LangGraph pipeline. To run the controller in the environment:

```
python baselines/langgraph_agent.py --episodes 5 --data-path data/eval_data.csv
```

This script loops through the Gymnasium env, feeds observations + history into the compiled graph, and logs outcomes to `logs/langgraph_steps.csv`, giving you a deterministic multi-agent baseline to compare against PPO/RL runs.

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
- `data/export_opsd_sqlite.py`: Extract OPSD SQLite slices (CSV or direct DataFrame)
- `agents/`: Forecasting + control agents for LangGraph
- `env/`: Gymnasium environment
- `baselines/`: Baseline strategies
- `dashboard/`: Streamlit UI
- `hpc/`: Ray-powered training entry point and SLURM template
- `pinn/`: Physics-informed models (battery thermal PINN)
- `train.py`: RL training script
- `evaluate.py`: Evaluation script
