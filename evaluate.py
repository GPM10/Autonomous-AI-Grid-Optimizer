import numpy as np
from stable_baselines3 import PPO
from env.microgrid_env import MicrogridEnv

def evaluate_policy(env, model=None, n_episodes=1):
    total_rewards = []
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            if model:
                action, _ = model.predict(obs)
            else:
                action = env.action_space.sample()  # random for baseline
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
        total_rewards.append(episode_reward)
    return np.mean(total_rewards)

# Load trained model
model = PPO.load("ppo_microgrid")

# Create environment
env = MicrogridEnv()

# Evaluate RL agent
rl_reward = evaluate_policy(env, model)

# Evaluate random baseline
random_reward = evaluate_policy(env, None)

print(f"RL Agent Average Reward: {rl_reward}")
print(f"Random Baseline Average Reward: {random_reward}")