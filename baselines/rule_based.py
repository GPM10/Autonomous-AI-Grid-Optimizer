import numpy as np
from env.microgrid_env import MicrogridEnv

class RuleBasedAgent:
    def __init__(self, env):
        self.env = env
    
    def predict(self, obs):
        solar, demand, battery_level, price, carbon, time = obs
        net = solar - demand
        
        if net > 0 and battery_level < self.env.battery_capacity * 0.9:
            return 1  # charge
        elif net < 0 and battery_level > 0.1 * self.env.battery_capacity:
            return 2  # discharge
        else:
            return 0  # idle

# Example usage
env = MicrogridEnv()
agent = RuleBasedAgent(env)

obs, _ = env.reset()
total_reward = 0
while True:
    action = agent.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    total_reward += reward
    if done:
        break

print(f"Rule-based total reward: {total_reward}")