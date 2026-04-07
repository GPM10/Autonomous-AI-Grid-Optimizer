from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.title("Autonomous AI Grid Optimizer Dashboard")

DATASETS = {
    "Training (data/train_data.csv)": Path("data/train_data.csv"),
    "Evaluation (data/eval_data.csv)": Path("data/eval_data.csv"),
}
LOG_FILES = {
    "Training (logs/train_steps.csv)": Path("logs/train_steps.csv"),
    "Evaluation (logs/eval_steps.csv)": Path("logs/eval_steps.csv"),
}
BASELINE_LOG = Path("logs/langgraph_steps.csv")

st.sidebar.header("Data Sources")
dataset_label = st.sidebar.selectbox("Dataset split", list(DATASETS.keys()))
data_path = DATASETS[dataset_label]
log_label = st.sidebar.selectbox("PPO per-step log", list(LOG_FILES.keys()))
log_path = LOG_FILES[log_label]
compare_baseline = st.sidebar.checkbox("Compare with LangGraph baseline", value=True)

baseline_label = f"LangGraph ({BASELINE_LOG})"
if compare_baseline:
    st.caption(f"Dataset: `{data_path}` | PPO log: `{log_path}` | Baseline log: `{BASELINE_LOG}`")
else:
    st.caption(f"Dataset: `{data_path}` | PPO log: `{log_path}`")

if not data_path.exists():
    st.error(f"Dataset '{data_path}' not found. Run `python data/build_datasets.py`.")
    st.stop()

data = pd.read_csv(data_path)
data["time"] = pd.to_datetime(data["time"])

st.subheader("Data Overview")
st.dataframe(data.head())

st.subheader("Solar vs Demand")
fig, ax = plt.subplots()
ax.plot(data["time"], data["solar"], label="Solar (kW)")
ax.plot(data["time"], data["demand"], label="Demand (kW)")
ax.set_xlabel("Time")
ax.set_ylabel("kW")
ax.legend()
st.pyplot(fig)

def load_log(path: Path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")
    return df


def summarize_log(label: str, df: pd.DataFrame):
    st.markdown(f"### {label}")
    numeric_cols = ["reward", "cost", "emissions", "grid_import", "grid_export", "unmet_demand"]
    available_cols = [c for c in numeric_cols if c in df.columns]
    if available_cols:
        agg = df[available_cols].agg(["mean", "sum"]).T
        agg.columns = ["Mean", "Sum"]
        st.table(agg)
    if "reward" in df.columns and "time" in df.columns:
        st.line_chart(df.set_index("time")[["reward"]])
    else:
        st.line_chart(df[available_cols] if available_cols else df)


st.subheader("Per-step Diagnostics")
log_df = load_log(log_path)
if log_df is None:
    st.info(f"No log file at '{log_path}'. Enable per-step logging via `log_path` or `--step-log-dir`.")
else:
    summarize_log("PPO Metrics", log_df)

if compare_baseline:
    baseline_df = load_log(BASELINE_LOG)
    if baseline_df is None:
        st.warning(f"Baseline log '{BASELINE_LOG}' not found. Run `python baselines/langgraph_agent.py` to generate it.")
    else:
        summarize_log("LangGraph Baseline Metrics", baseline_df)
        if "time" in log_df.columns and "reward" in log_df.columns and "reward" in baseline_df.columns:
            merged = pd.merge_asof(
                log_df[["time", "reward"]].rename(columns={"reward": "ppo_reward"}),
                baseline_df[["time", "reward"]].rename(columns={"reward": "langgraph_reward"}),
                on="time",
                tolerance=pd.Timedelta("30min"),
                direction="nearest",
            ).dropna()
            if not merged.empty:
                st.markdown("#### Reward Comparison (Aligned)")
                chart_df = merged.set_index("time")
                st.line_chart(chart_df)
