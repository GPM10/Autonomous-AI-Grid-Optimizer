import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Autonomous AI Grid Optimizer Dashboard")

# Load data
data = pd.read_csv('data/sample_data.csv')
data['time'] = pd.to_datetime(data['time'])

st.subheader("Data Overview")
st.dataframe(data.head())

st.subheader("Solar vs Demand")
fig, ax = plt.subplots()
ax.plot(data['time'], data['solar'], label='Solar')
ax.plot(data['time'], data['demand'], label='Demand')
ax.legend()
st.pyplot(fig)

# Placeholder for results
st.subheader("Performance Metrics")
st.write("RL Reward: TBD")
st.write("Baseline Reward: TBD")