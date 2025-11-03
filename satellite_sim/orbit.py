"""Orbit state helpers."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np


@dataclass
class OrbitParameters:
    radius: float
    angular_velocity: float


class OrbitSimulator:
    """Maintains the orbital position of the satellite."""

    def __init__(self, params: OrbitParameters):
        self.params = params
        self._angle = 0.0

    def update(self, dt: float) -> None:
        self._angle = (self._angle + self.params.angular_velocity * dt) % (2 * math.pi)

    def set_radius(self, radius: float) -> None:
        self.params.radius = max(radius, 1.5)

    def set_speed(self, angular_velocity: float) -> None:
        self.params.angular_velocity = angular_velocity

    def set_angle(self, angle: float) -> None:
        self._angle = angle % (2 * math.pi)

    def position(self) -> np.ndarray:
        x = self.params.radius * math.cos(self._angle)
        y = self.params.radius * math.sin(self._angle)
        return np.array([x, y, 0.0])

    def velocity(self) -> np.ndarray:
        vx = -self.params.radius * math.sin(self._angle) * self.params.angular_velocity
        vy = self.params.radius * math.cos(self._angle) * self.params.angular_velocity
        return np.array([vx, vy, 0.0])

    def parameters(self) -> Tuple[float, float, float]:
        return self.params.radius, self.params.angular_velocity, self._angle
