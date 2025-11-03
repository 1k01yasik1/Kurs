"""Simulation settings and parameter helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import uniform
from typing import Tuple

from pygame.math import Vector2


@dataclass
class OrbitSettings:
    """Parameters that describe the orbit setup."""

    mu: float = 398600.4418  # km^3 / s^2 for Earth
    body_radius_km: float = 6378.137
    body_name: str = "Earth"
    satellite_name: str = "Satellite"
    distance_km: float = 42164.0
    velocity_factor: float = 1.0
    plane_rotation_deg: float = 0.0

    def initial_state(self) -> Tuple[Vector2, Vector2]:
        position = Vector2(self.distance_km, 0).rotate(self.plane_rotation_deg)
        circular_velocity = (self.mu / self.distance_km) ** 0.5
        velocity = Vector2(0, self.velocity_factor * circular_velocity).rotate(
            self.plane_rotation_deg
        )
        return position, velocity

    def circular_velocity(self) -> float:
        return (self.mu / self.distance_km) ** 0.5

    def randomize(self) -> None:
        self.distance_km = uniform(self.body_radius_km + 500.0, 70000.0)
        self.velocity_factor = uniform(0.6, 1.4)
        self.plane_rotation_deg = uniform(0.0, 360.0)


@dataclass
class ViewSettings:
    width: int = 1024
    height: int = 768
    background_color: Tuple[int, int, int] = (5, 6, 20)
    body_color: Tuple[int, int, int] = (45, 140, 255)
    satellite_color: Tuple[int, int, int] = (255, 230, 150)
    trail_color: Tuple[int, int, int] = (120, 200, 255)
    grid_color: Tuple[int, int, int] = (40, 40, 60)
    label_color: Tuple[int, int, int] = (220, 220, 230)
    pixels_per_km: float = 0.005
    min_scale: float = 0.0005
    max_scale: float = 0.05


@dataclass
class SimulationToggles:
    show_trail: bool = True
    show_grid: bool = True
    paused: bool = False


@dataclass
class SimulationConfig:
    orbit: OrbitSettings = field(default_factory=OrbitSettings)
    view: ViewSettings = field(default_factory=ViewSettings)
    toggles: SimulationToggles = field(default_factory=SimulationToggles)
    base_dt: float = 2.0
    time_scale: float = 60.0
    max_trail_points: int = 2000
    trail_sample_interval: float = 60.0

    def reset_defaults(self) -> None:
        self.__dict__.update(SimulationConfig().__dict__)
