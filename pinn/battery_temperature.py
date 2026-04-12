"""Physics-informed neural network for battery thermal dynamics."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _CorrectionNet(nn.Module):
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)


class BatteryThermalPINN:
    """Learns a correction to simple battery thermal dynamics."""

    def __init__(
        self,
        alpha: float = 0.12,
        beta: float = 0.05,
        dt_hours: float = 1.0,
        model_path: str = "artifacts/pinn_battery.pt",
        train_steps: int = 1500,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.dt = dt_hours
        self.model_path = Path(model_path)
        self.net = _CorrectionNet().to(_device())
        if self.model_path.exists():
            self.net.load_state_dict(torch.load(self.model_path, map_location=_device()))
            self.net.eval()
        else:
            self._train(train_steps)
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.net.state_dict(), self.model_path)
            self.net.eval()

    def _physics_rate(self, temp: torch.Tensor, power: torch.Tensor, ambient: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.abs(power) - self.beta * (temp - ambient)

    def _train(self, steps: int) -> None:
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        for step in range(steps):
            temp = torch.rand(256, 1, device=_device()) * 20 + 20  # 20-40 C
            power = torch.rand(256, 1, device=_device()) * 4 - 2    # -2 to 2 kW
            ambient = torch.rand(256, 1, device=_device()) * 15 + 10  # 10-25 C

            physics_rate = self._physics_rate(temp, power, ambient)
            noisy_obs = physics_rate + 0.02 * torch.randn_like(physics_rate)

            inputs = torch.cat([(temp - 30) / 10, power / 2, (ambient - 20) / 10], dim=1)
            correction = self.net(inputs)
            pred_rate = physics_rate + correction

            physics_penalty = correction.pow(2).mean()
            data_loss = (pred_rate - noisy_obs).pow(2).mean()
            loss = data_loss + 0.001 * physics_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict_next(self, temp: float, power: float, ambient: float) -> float:
        self.net.eval()
        with torch.no_grad():
            feat = torch.tensor([
                (temp - 30) / 10,
                power / 2,
                (ambient - 20) / 10,
            ], dtype=torch.float32, device=_device()).unsqueeze(0)
            correction = self.net(feat).item()
        physics_rate = self.alpha * abs(power) - self.beta * (temp - ambient)
        delta = (physics_rate + correction) * self.dt
        return temp + delta
