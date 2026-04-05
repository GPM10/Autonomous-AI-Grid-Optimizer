from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from env.microgrid_env import MicrogridEnv

EVAL_DATA = "data/eval_data.csv"
LOG_DIR = Path("logs")
MODEL_BASENAME = Path("artifacts/ppo_microgrid")
MODEL_PATH = MODEL_BASENAME.with_suffix(".zip")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_policy(env, model=None, n_episodes=1):
    total_rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            if model is not None:
                action, _ = model.predict(obs)
            else:
                action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
        total_rewards.append(episode_reward)
    return float(np.mean(total_rewards))


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Trained model not found. Run train.py first to create artifacts/ppo_microgrid.zip"
    )

model = PPO.load(str(MODEL_BASENAME))

env = MicrogridEnv(
    data_path=EVAL_DATA,
    episode_length=120,
    reward_weights={
        "cost": 1.0,
        "carbon": 0.2,
        "battery_penalty": 2.0,
        "unmet_demand": 150.0,
        "export_credit": 0.4,
    },
    log_path=str(LOG_DIR / "eval_steps.csv"),
)

rl_reward = evaluate_policy(env, model, n_episodes=3)
random_reward = evaluate_policy(env, None, n_episodes=3)

print(f"RL Agent Average Reward: {rl_reward:.2f}")
print(f"Random Baseline Average Reward: {random_reward:.2f}")
