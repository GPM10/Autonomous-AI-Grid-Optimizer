"""Run the LangGraph-based multi-agent controller inside the microgrid env."""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List

import numpy as np

from env.microgrid_env import MicrogridEnv
from graph.langgraph_flow import Observation, compiled_graph


def obs_to_dict(obs: np.ndarray) -> Observation:
    return Observation(
        solar=float(obs[0]),
        demand=float(obs[1]),
        battery=float(obs[2]),
        price=float(obs[3]),
        carbon=float(obs[4]),
        hour=float(obs[5]),
    )


def run_episode(env: MicrogridEnv, history: Deque[Observation], maxlen: int) -> Dict[str, float]:
    obs, _ = env.reset()
    history.clear()
    history.append(obs_to_dict(obs))
    total_reward = 0.0
    steps = 0
    while True:
        state = {
            "observation": history[-1],
            "history": list(history),
        }
        result = compiled_graph.invoke(state)
        action = int(result["action"])
        obs, reward, done, _, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if done:
            break
        history.append(obs_to_dict(obs))
        if len(history) > maxlen:
            history.popleft()
    return {"reward": total_reward, "steps": steps}


def main():
    parser = argparse.ArgumentParser(description="Evaluate LangGraph controller baseline.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--data-path", default="data/eval_data.csv")
    parser.add_argument("--log-path", default="logs/langgraph_steps.csv")
    args = parser.parse_args()

    env = MicrogridEnv(
        data_path=args.data_path,
        episode_length=168,
        log_path=args.log_path,
        reward_weights={
            "cost": 1.0,
            "carbon": 0.2,
            "battery_penalty": 1.5,
            "unmet_demand": 120.0,
            "export_credit": 0.3,
        },
    )
    history: Deque[Observation] = deque(maxlen=48)
    results: List[Dict[str, float]] = []
    for ep in range(args.episodes):
        stats = run_episode(env, history, maxlen=48)
        results.append(stats)
        print(f"[Episode {ep+1}] reward={stats['reward']:.2f} steps={stats['steps']}")
    avg_reward = sum(r["reward"] for r in results) / len(results)
    print(f"Average reward over {len(results)} episodes: {avg_reward:.2f}")
    print(f"Per-step log saved to {args.log_path}")


if __name__ == "__main__":
    main()
