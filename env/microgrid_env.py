import gymnasium as gym
import numpy as np
from gymnasium import spaces

class MicrogridEnv(gym.Env):
    def __init__(self, data_path='data/sample_data.csv'):
        super(MicrogridEnv, self).__init__()
        
        # Load data
        import pandas as pd
        self.data = pd.read_csv(data_path)
        self.data['time'] = pd.to_datetime(self.data['time'])
        self.max_steps = len(self.data)
        self.current_step = 0
        
        # Battery parameters
        self.battery_capacity = 10.0  # kWh
        self.battery_level = 5.0  # start at 50%
        self.charge_rate = 2.0  # kW
        self.discharge_rate = 2.0  # kW
        self.efficiency = 0.9
        
        # State space: [solar, demand, battery_level, price, carbon_intensity, time_of_day]
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, 0, 0]), 
                                            high=np.array([20, 10, self.battery_capacity, 1, 500, 23]), 
                                            dtype=np.float32)
        
        # Action space: 0=idle, 1=charge, 2=discharge, 3=import from grid
        self.action_space = spaces.Discrete(4)
        
        # Reward parameters
        self.alpha = 0.1  # weight for emissions
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.battery_level = 5.0
        return self._get_obs(), {}
    
    def _get_obs(self):
        row = self.data.iloc[self.current_step]
        time_of_day = row['time'].hour
        return np.array([
            row['solar'],
            row['demand'],
            self.battery_level,
            row['price'],
            row['carbon_intensity'],
            time_of_day
        ], dtype=np.float32)
    
    def step(self, action):
        row = self.data.iloc[self.current_step]
        solar = row['solar']
        demand = row['demand']
        price = row['price']
        carbon_intensity = row['carbon_intensity']
        
        # Calculate net energy
        net_energy = solar - demand
        
        # Action effects
        grid_import = 0.0
        if action == 0:  # idle
            pass
        elif action == 1:  # charge battery
            charge_amount = min(self.charge_rate, self.battery_capacity - self.battery_level)
            self.battery_level += charge_amount * self.efficiency
            net_energy -= charge_amount
        elif action == 2:  # discharge battery
            discharge_amount = min(self.discharge_rate, self.battery_level)
            self.battery_level -= discharge_amount
            net_energy += discharge_amount * self.efficiency
        elif action == 3:  # import from grid
            grid_import = max(0, -net_energy)
            net_energy += grid_import
        
        # If net_energy < 0, import from grid
        if net_energy < 0:
            grid_import += -net_energy
            net_energy = 0
        
        # Cost and emissions
        cost = grid_import * price
        emissions = grid_import * carbon_intensity
        
        # Reward
        reward = - (cost + self.alpha * emissions)
        
        # Penalties
        if net_energy < 0:  # unmet demand
            reward -= 100  # large penalty
        
        # Battery abuse penalty (if level too low or high)
        if self.battery_level < 0.1 * self.battery_capacity or self.battery_level > 0.9 * self.battery_capacity:
            reward -= 1
        
        # Next step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return self._get_obs(), reward, done, False, {}
    
    def render(self, mode='human'):
        pass