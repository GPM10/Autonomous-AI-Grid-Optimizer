# Autonomous AI Grid Optimizer

A carbon-aware, RL-controlled microgrid with multi-agent orchestration using LangGraph.

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Run training: `python train.py`
3. Evaluate: `python evaluate.py`
4. Dashboard: `streamlit run dashboard/app.py`

## Project Structure

- `data/`: Sample dataset
- `env/`: Gymnasium environment
- `agents/`: Agent modules
- `graph/`: LangGraph orchestration
- `baselines/`: Baseline strategies
- `dashboard/`: Streamlit UI
- `train.py`: RL training script
- `evaluate.py`: Evaluation script