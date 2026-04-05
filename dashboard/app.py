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

st.sidebar.header("Data Sources")
dataset_label = st.sidebar.selectbox("Dataset split", list(DATASETS.keys()))
data_path = DATASETS[dataset_label]
log_label = st.sidebar.selectbox("Per-step log", list(LOG_FILES.keys()))
log_path = LOG_FILES[log_label]

st.caption(f"Dataset: `{data_path}` | Step log: `{log_path}`")

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

st.subheader("Per-step Diagnostics")
if log_path.exists():
    log_df = pd.read_csv(log_path)
    if "time" in log_df.columns:
        log_df["time"] = pd.to_datetime(log_df["time"])
    numeric_cols = ["reward", "cost", "emissions", "grid_import", "grid_export", "unmet_demand"]
    available_cols = [c for c in numeric_cols if c in log_df.columns]
    if available_cols:
        agg = log_df[available_cols].agg(["mean", "sum"]).T
        agg.columns = ["Mean", "Sum"]
        st.table(agg)
    st.line_chart(log_df.set_index("time")[["reward"]] if "reward" in log_df.columns else log_df)
else:
    st.info(f"No log file at '{log_path}'. Enable per-step logging via `log_path` or `--step-log-dir`.")
