"""CPU ray tracer for the satellite visualisation."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np

from .camera import Camera
from .orbit import OrbitSimulator
from .planet import Planet


@dataclass
class Light:
    direction: np.ndarray
    color: np.ndarray
    intensity: float = 1.2


@dataclass
class RenderSettings:
    width: int = 640
    height: int = 480
    fov: float = math.radians(60)
    max_distance: float = 100.0
    planet_tolerance: float = 0.005
    planet_steps: int = 128


@dataclass
class Material:
    base_color: np.ndarray
    specular: float
    shininess: float


class Scene:
    def __init__(
        self,
        planet: Planet,
        orbit: OrbitSimulator,
        light: Light,
        satellite_radius: float = 0.25,
    ) -> None:
        self.planet = planet
        self.orbit = orbit
        self.light = light
        self.satellite_radius = satellite_radius
        self.planet_material = Material(np.array([0.4, 0.55, 0.35]), 0.2, 32.0)
        self.satellite_material = Material(np.array([0.8, 0.8, 0.85]), 0.4, 64.0)
        self.space_color = np.array([0.02, 0.02, 0.05])

    def render(self, camera: Camera, settings: RenderSettings, dt: float) -> np.ndarray:
        aspect = settings.width / settings.height
        tan_half_fov = math.tan(settings.fov / 2)
        origin = camera.position()
        forward = camera.forward()
        right = camera.right()
        up = camera.up()
        image = np.zeros((settings.height, settings.width, 3), dtype=np.float32)

        sat_position = self.orbit.position()
        self.orbit.update(dt)

        for y in range(settings.height):
            ndc_y = (1 - 2 * (y + 0.5) / settings.height) * tan_half_fov
            for x in range(settings.width):
                ndc_x = (2 * (x + 0.5) / settings.width - 1) * tan_half_fov * aspect
                direction = (forward + ndc_x * right + ndc_y * up)
                direction = direction / np.linalg.norm(direction)
                color = self._trace_ray(origin, direction, sat_position, settings)
                image[y, x] = color

        image = np.clip(image, 0.0, 1.0)
        return (image * 255).astype(np.uint8)

    def _trace_ray(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        sat_position: np.ndarray,
        settings: RenderSettings,
    ) -> np.ndarray:
        sat_hit = self._intersect_sphere(origin, direction, sat_position, self.satellite_radius)
        planet_hit = self._march_planet(origin, direction, settings)

        if planet_hit is None and sat_hit is None:
            return self.space_color

        if sat_hit is not None and (planet_hit is None or sat_hit[0] < planet_hit[0]):
            return self._shade_sphere_hit(origin, direction, sat_hit, sat_position)

        if planet_hit is not None:
            return self._shade_planet_hit(origin, direction, planet_hit, sat_position)

        return self.space_color

    def _shade_planet_hit(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        hit: Tuple[float, np.ndarray],
        sat_position: np.ndarray,
    ) -> np.ndarray:
        distance, position = hit
        normal = self.planet.normal_at_direction(position)
        view_dir = -direction
        light_dir = -self.light.direction / np.linalg.norm(self.light.direction)
        color = self.planet_material.base_color.copy()
        if self._is_shadowed_by_satellite(position, sat_position, light_dir):
            diffuse = 0.0
            specular = 0.0
        else:
            diffuse = max(np.dot(normal, light_dir), 0.0)
            reflect = 2 * np.dot(normal, light_dir) * normal - light_dir
            specular = max(np.dot(reflect, view_dir), 0.0) ** self.planet_material.shininess
        ambient = 0.08
        final_color = (
            color * (ambient + diffuse * self.light.intensity)
            + self.planet_material.specular * specular * self.light.intensity
        )
        final_color = self._apply_atmosphere(position, final_color)
        return final_color

    def _shade_sphere_hit(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        hit: Tuple[float, np.ndarray],
        sat_position: np.ndarray,
    ) -> np.ndarray:
        _, position = hit
        normal = (position - sat_position) / self.satellite_radius
        view_dir = -direction
        light_dir = -self.light.direction / np.linalg.norm(self.light.direction)
        if self._is_shadowed_by_planet(position, light_dir):
            diffuse = 0.0
            specular = 0.0
        else:
            diffuse = max(np.dot(normal, light_dir), 0.0)
            reflect = 2 * np.dot(normal, light_dir) * normal - light_dir
            specular = max(np.dot(reflect, view_dir), 0.0) ** self.satellite_material.shininess
        ambient = 0.05
        final_color = (
            self.satellite_material.base_color * (ambient + diffuse * self.light.intensity)
            + self.satellite_material.specular * specular * self.light.intensity
        )
        return final_color

    def _march_planet(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        settings: RenderSettings,
    ) -> Optional[Tuple[float, np.ndarray]]:
        t = 0.0
        for _ in range(settings.planet_steps):
            position = origin + direction * t
            radius = np.linalg.norm(position)
            if radius < 1e-5:
                t += 0.1
                continue
            surface_radius = self.planet.radius_at_direction(position)
            distance = radius - surface_radius
            if distance < settings.planet_tolerance:
                return t, position
            step = max(distance * 0.7, settings.planet_tolerance * 0.5)
            t += step
            if t > settings.max_distance:
                break
        return None

    def _intersect_sphere(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        center: np.ndarray,
        radius: float,
    ) -> Optional[Tuple[float, np.ndarray]]:
        oc = origin - center
        b = np.dot(oc, direction)
        c = np.dot(oc, oc) - radius ** 2
        discriminant = b * b - c
        if discriminant < 0.0:
            return None
        sqrt_disc = math.sqrt(discriminant)
        t = -b - sqrt_disc
        if t <= 1e-4:
            t = -b + sqrt_disc
        if t <= 1e-4:
            return None
        position = origin + direction * t
        return t, position

    def _is_shadowed_by_satellite(
        self, point: np.ndarray, sat_position: np.ndarray, light_dir: np.ndarray
    ) -> bool:
        hit = self._intersect_sphere(point + light_dir * 0.01, light_dir, sat_position, self.satellite_radius)
        return hit is not None

    def _is_shadowed_by_planet(self, point: np.ndarray, light_dir: np.ndarray) -> bool:
        settings = RenderSettings()
        hit = self._march_planet(point + light_dir * 0.01, light_dir, settings)
        return hit is not None

    def _apply_atmosphere(self, position: np.ndarray, color: np.ndarray) -> np.ndarray:
        altitude = np.linalg.norm(position) - self.planet.config.base_radius
        haze = np.clip(altitude / (self.planet.config.base_radius * 0.5), 0.0, 1.0)
        sky_color = np.array([0.1, 0.2, 0.4])
        return color * (1 - 0.3 * haze) + sky_color * (0.15 * haze)
