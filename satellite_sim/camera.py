"""Camera management utilities."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np


@dataclass
class Camera:
    """Represents a simple turntable camera."""

    distance: float = 6.0
    yaw: float = 0.0
    pitch: float = math.radians(20)
    target: np.ndarray = np.zeros(3)

    def copy(self) -> "Camera":
        return Camera(self.distance, self.yaw, self.pitch, self.target.copy())

    def forward(self) -> np.ndarray:
        cos_pitch = math.cos(self.pitch)
        return np.array(
            [
                math.cos(self.yaw) * cos_pitch,
                math.sin(self.yaw) * cos_pitch,
                math.sin(self.pitch),
            ]
        )

    def right(self) -> np.ndarray:
        fwd = self.forward()
        up = np.array([0.0, 0.0, 1.0])
        return np.cross(fwd, up)

    def up(self) -> np.ndarray:
        right = self.right()
        return np.cross(right, self.forward())

    def position(self) -> np.ndarray:
        return self.target - self.forward() * self.distance

    def orbit(self, d_yaw: float, d_pitch: float) -> None:
        self.yaw = (self.yaw + d_yaw) % (2 * math.pi)
        self.pitch = float(np.clip(self.pitch + d_pitch, math.radians(-80), math.radians(80)))

    def zoom(self, factor: float) -> None:
        self.distance = float(np.clip(self.distance * factor, 1.5, 50.0))

    def pan(self, dx: float, dy: float) -> None:
        right = self.right()
        up = np.array([0.0, 0.0, 1.0])
        self.target = self.target + right * dx + up * dy

    def view_matrix(self) -> np.ndarray:
        pos = self.position()
        fwd = self.forward()
        right = self.right()
        up = self.up()
        mat = np.identity(4, dtype=np.float32)
        mat[0, :3] = right
        mat[1, :3] = up
        mat[2, :3] = -fwd
        mat[:3, 3] = -mat[:3, :3] @ pos
        return mat

    def view_parameters(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.position(), self.forward(), self.up()
