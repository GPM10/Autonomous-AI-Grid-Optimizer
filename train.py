from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env.microgrid_env import MicrogridEnv

# Create environment
env = make_vec_env(MicrogridEnv, n_envs=1)

# Create PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_microgrid")

print("Training complete. Model saved as ppo_microgrid.zip")