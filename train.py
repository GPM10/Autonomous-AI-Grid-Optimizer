from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from env.microgrid_env import MicrogridEnv

TRAIN_DATA = "data/train_data.csv"
LOG_DIR = Path("logs")
MODEL_DIR = Path("artifacts")
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

env_kwargs = dict(
    data_path=TRAIN_DATA,
    episode_length=168,
    reward_weights={
        "cost": 1.0,
        "carbon": 0.2,
        "battery_penalty": 2.0,
        "unmet_demand": 150.0,
        "export_credit": 0.4,
    },
    log_path=str(LOG_DIR / "train_steps.csv"),
)

# Create environment
env = make_vec_env(MicrogridEnv, n_envs=1, env_kwargs=env_kwargs)

# Create PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=50000)

# Save the model
model.save(str(MODEL_DIR / "ppo_microgrid"))

print("Training complete. Model saved to artifacts/ppo_microgrid.zip")
