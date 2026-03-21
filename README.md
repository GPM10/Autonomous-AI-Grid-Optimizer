# Autonomous AI Grid Optimizer

A carbon-aware, RL-controlled microgrid with multi-agent orchestration using LangGraph.

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Run training: `python train.py`
3. Evaluate: `python evaluate.py`
4. Dashboard: `streamlit run dashboard/app.py`

## High-Performance / Distributed Training

Ray + RLlib integration enables large-scale PPO runs across many CPU or GPU workers.

### Local multi-core

```
python -m hpc.train_distributed \
  --num-workers 8 \
  --envs-per-worker 4 \
  --train-batch-size 65536 \
  --checkpoint-freq 2 \
  --output-dir checkpoints/local-ray
```

Add `--ray-address auto` to attach to an existing Ray head (e.g., `ray start --head --port 6379`). Use `--local-mode` for quick debugging without spinning up workers.

### SLURM / HPC clusters

1. Edit `hpc/slurm_job.sbatch` with your module/conda commands and scratch paths.
2. Submit via `sbatch hpc/slurm_job.sbatch`.
3. Let additional nodes join the Ray head (`ray start --address <head-ip>:6379`) to scale horizontally.

The script exposes extra knobs (`--checkpoint-freq`, `--target-reward`, `--metrics-path`, etc.) so you can integrate with monitoring stacks or stop automatically once a quality bar is hit.

## Project Structure

- `data/`: Sample dataset
- `env/`: Gymnasium environment
- `agents/`: Agent modules
- `graph/`: LangGraph orchestration
- `baselines/`: Baseline strategies
- `dashboard/`: Streamlit UI
- `hpc/`: Ray-powered training entry point and SLURM template
- `train.py`: RL training script
- `evaluate.py`: Evaluation script
