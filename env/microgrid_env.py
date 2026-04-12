import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pinn import BatteryThermalPINN


class MicrogridEnv(gym.Env):
    def __init__(
        self,
        data_path: str = 'data/train_data.csv',
        episode_length: Optional[int] = None,
        reward_weights: Optional[Dict[str, float]] = None,
        log_path: Optional[str] = None,
        grid_import_limit: Optional[float] = None,
        data_frame=None,
        use_pinn: bool = False,
        ambient_temperature: float = 25.0,
        max_battery_temp: float = 45.0,
        pinn_model_path: str = 'artifacts/pinn_battery.pt',
    ):
        super(MicrogridEnv, self).__init__()
        
        # Load data
        import pandas as pd
        if data_frame is not None:
            self.data = data_frame.copy()
        else:
            self.data = pd.read_csv(data_path)
        self.data['time'] = pd.to_datetime(self.data['time'])
        self.max_steps = len(self.data)
        self.current_step = 0
        self.episode_length = episode_length or self.max_steps
        self.start_index = 0
        self.grid_import_limit = grid_import_limit
        self.log_path = Path(log_path) if log_path else None
        self.episode_log: List[Dict[str, Any]] = []
        self.use_pinn = use_pinn
        self.ambient_temperature = ambient_temperature
        self.max_battery_temp = max_battery_temp
        self.pinn = BatteryThermalPINN(model_path=pinn_model_path) if use_pinn else None
        self.battery_temp = ambient_temperature

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
        default_weights = {
            'cost': 1.0,
            'carbon': 0.1,
            'battery_penalty': 1.0,
            'unmet_demand': 100.0,
            'export_credit': 0.2,
            'thermal_penalty': 2.0,
        }
        if reward_weights:
            default_weights.update(reward_weights)
        self.reward_weights = default_weights

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.battery_level = 5.0
        self.battery_temp = self.ambient_temperature
        self.episode_log = []
        self._set_start_index()
        return self._get_obs(), {}

    def _get_obs(self):
        idx = min(self.start_index + self.current_step, self.max_steps - 1)
        row = self.data.iloc[idx]
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
        idx = self.start_index + self.current_step
        row = self.data.iloc[idx]
        solar = row['solar']
        demand = row['demand']
        price = row['price']
        carbon_intensity = row['carbon_intensity']
        timestamp = row['time']
        
        # Calculate net energy (positive = surplus)
        net_energy = solar - demand
        
        # Action effects
        grid_import = 0.0
        grid_export = 0.0
        unmet_demand = 0.0
        battery_power = 0.0
        if action == 0:  # idle
            pass
        elif action == 1:  # charge battery
            charge_amount = min(self.charge_rate, self.battery_capacity - self.battery_level)
            self.battery_level += charge_amount * self.efficiency
            net_energy -= charge_amount
            battery_power = charge_amount
        elif action == 2:  # discharge battery
            discharge_amount = min(self.discharge_rate, self.battery_level)
            self.battery_level -= discharge_amount
            net_energy += discharge_amount * self.efficiency
            battery_power = -discharge_amount
        elif action == 3:  # import from grid
            grid_import = max(0, -net_energy)
            net_energy += grid_import
        
        if net_energy > 0:
            grid_export = net_energy
            net_energy = 0
        elif net_energy < 0:
            deficit = -net_energy
            if self.grid_import_limit is not None:
                allowed = min(deficit, self.grid_import_limit)
                unmet_demand = deficit - allowed
            else:
                allowed = deficit
            grid_import += allowed
            net_energy = 0

        # Cost and emissions
        cost = grid_import * price
        export_credit = grid_export * price
        emissions = grid_import * carbon_intensity

        # Battery abuse penalty (if level too low or high)
        lower_bound = 0.1 * self.battery_capacity
        upper_bound = 0.9 * self.battery_capacity
        battery_penalty = 0.0
        if self.battery_level < lower_bound:
            battery_penalty = lower_bound - self.battery_level
        elif self.battery_level > upper_bound:
            battery_penalty = self.battery_level - upper_bound

        reward = (
            - self.reward_weights['cost'] * cost
            - self.reward_weights['carbon'] * emissions
            - self.reward_weights['battery_penalty'] * battery_penalty
            - self.reward_weights['unmet_demand'] * unmet_demand
            + self.reward_weights['export_credit'] * export_credit
        )

        if self.pinn is not None:
            self.battery_temp = self.pinn.predict_next(self.battery_temp, battery_power, self.ambient_temperature)
            if self.battery_temp > self.max_battery_temp:
                reward -= self.reward_weights['thermal_penalty'] * (self.battery_temp - self.max_battery_temp)

        # Next step
        self.current_step += 1
        done = self.current_step >= self.episode_length

        info = {
            'time': str(timestamp),
            'action': int(action),
            'solar': solar,
            'demand': demand,
            'grid_import': grid_import,
            'grid_export': grid_export,
            'battery_level': self.battery_level,
            'battery_temp': self.battery_temp,
            'cost': cost,
            'emissions': emissions,
            'unmet_demand': unmet_demand,
            'reward': reward,
        }

        self._log_step(info, done)
        
        return self._get_obs(), reward, done, False, info

    def render(self, mode='human'):
        pass

    def _set_start_index(self):
        if self.episode_length > self.max_steps:
            self.episode_length = self.max_steps
        if self.episode_length == self.max_steps:
            self.start_index = 0
            return
        max_start = self.max_steps - self.episode_length
        self.start_index = int(self.np_random.integers(0, max_start + 1))

    def _log_step(self, info: Dict[str, Any], done: bool):
        if self.log_path is None:
            return
        self.episode_log.append(info)
        if done:
            self._flush_log()

    def _flush_log(self):
        if not self.episode_log:
            return
        fieldnames = list(self.episode_log[0].keys())
        write_header = not self.log_path.exists()
        with self.log_path.open('a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(self.episode_log)
        self.episode_log = []
