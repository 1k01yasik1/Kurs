"""Procedural planet generation."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np


@dataclass
class PlanetConfig:
    base_radius: float = 2.0
    lat_steps: int = 180
    lon_steps: int = 360
    height_scale: float = 0.25


class Planet:
    """Planet surface generated in polar coordinates."""

    def __init__(self, config: PlanetConfig | None = None):
        self.config = config or PlanetConfig()
        self.heights = np.zeros((self.config.lat_steps, self.config.lon_steps), dtype=np.float32)

    def generate(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        lat = np.linspace(-math.pi / 2, math.pi / 2, self.config.lat_steps)
        lon = np.linspace(0.0, 2 * math.pi, self.config.lon_steps, endpoint=False)
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")

        base_noise = self._fbm_noise(lat_grid, lon_grid, rng)
        crater_noise = self._crater_field(lat_grid, lon_grid, rng)
        self.heights = (base_noise * 0.7 + crater_noise * 0.3).astype(np.float32)
        self.heights = np.clip(self.heights, -1.0, 1.0)

    def _fbm_noise(self, lat_grid: np.ndarray, lon_grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        freq = 1.0
        amplitude = 1.0
        total = np.zeros_like(lat_grid)
        for _ in range(4):
            phase_lat = rng.uniform(0, 2 * math.pi)
            phase_lon = rng.uniform(0, 2 * math.pi)
            total += amplitude * (
                np.sin(lat_grid * freq + phase_lat) * np.cos(lon_grid * freq + phase_lon)
            )
            freq *= 2.0
            amplitude *= 0.5
        total /= 1.0 + 0.5 + 0.25 + 0.125
        return total

    def _crater_field(self, lat_grid: np.ndarray, lon_grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        craters = np.zeros_like(lat_grid)
        count = rng.integers(12, 24)
        for _ in range(count):
            center_lat = rng.uniform(-math.pi / 2, math.pi / 2)
            center_lon = rng.uniform(0, 2 * math.pi)
            radius = rng.uniform(0.05, 0.18)
            depth = rng.uniform(-0.5, -0.1)
            dist = self._angular_distance(lat_grid, lon_grid, center_lat, center_lon)
            crater = depth * np.exp(-(dist ** 2) / (2 * radius ** 2))
            craters = np.minimum(craters, crater)
        return craters

    def _angular_distance(
        self, lat_grid: np.ndarray, lon_grid: np.ndarray, center_lat: float, center_lon: float
    ) -> np.ndarray:
        d_lat = lat_grid - center_lat
        d_lon = lon_grid - center_lon
        a = (
            np.sin(d_lat / 2) ** 2
            + np.cos(lat_grid) * np.cos(center_lat) * np.sin(d_lon / 2) ** 2
        )
        return 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a + 1e-6))

    def radius_at_direction(self, direction: np.ndarray) -> float:
        theta = math.acos(np.clip(direction[2] / np.linalg.norm(direction), -1.0, 1.0))
        phi = math.atan2(direction[1], direction[0]) % (2 * math.pi)
        lat_idx = int(theta / math.pi * (self.config.lat_steps - 1))
        lon_idx = int(phi / (2 * math.pi) * self.config.lon_steps) % self.config.lon_steps
        height = self.heights[lat_idx, lon_idx]
        return self.config.base_radius * (1.0 + self.config.height_scale * height)

    def normal_at_direction(self, direction: np.ndarray) -> np.ndarray:
        theta = math.acos(np.clip(direction[2] / np.linalg.norm(direction), -1.0, 1.0))
        phi = math.atan2(direction[1], direction[0]) % (2 * math.pi)
        lat = theta / math.pi * (self.config.lat_steps - 1)
        lon = phi / (2 * math.pi) * self.config.lon_steps
        lat0 = int(math.floor(lat))
        lon0 = int(math.floor(lon))
        lat1 = min(lat0 + 1, self.config.lat_steps - 1)
        lon1 = (lon0 + 1) % self.config.lon_steps
        u = lat - lat0
        v = lon - lon0
        h00 = self.heights[lat0, lon0]
        h10 = self.heights[lat1, lon0]
        h01 = self.heights[lat0, lon1]
        h11 = self.heights[lat1, lon1]
        dh_dlat = (1 - v) * (h10 - h00) + v * (h11 - h01)
        dh_dlon = (1 - u) * (h01 - h00) + u * (h11 - h10)
        normal = direction / np.linalg.norm(direction)
        tangent_lat = np.array([normal[0], normal[1], 0.0])
        if np.linalg.norm(tangent_lat) < 1e-5:
            tangent_lat = np.array([1.0, 0.0, 0.0])
        tangent_lat = tangent_lat / np.linalg.norm(tangent_lat)
        tangent_lon = np.cross(normal, tangent_lat)
        normal = normal - tangent_lat * dh_dlat * self.config.height_scale
        normal = normal - tangent_lon * dh_dlon * self.config.height_scale
        return normal / np.linalg.norm(normal)

    def height_map(self) -> np.ndarray:
        return self.heights.copy()

    def polar_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        lat = np.linspace(-math.pi / 2, math.pi / 2, self.config.lat_steps)
        lon = np.linspace(0.0, 2 * math.pi, self.config.lon_steps, endpoint=False)
        return lat, lon
