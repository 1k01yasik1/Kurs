"""Physics helpers for orbit simulation."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from typing import Tuple

from pygame.math import Vector2


@dataclass
class OrbitState:
    """Snapshot of an orbiting object's state."""

    position: Vector2
    velocity: Vector2
    time: float = 0.0

    def copy(self) -> "OrbitState":
        return OrbitState(self.position.copy(), self.velocity.copy(), self.time)


class OrbitSimulator:
    """Integrates the motion of a satellite around a central body."""

    def __init__(self, mu: float, position: Vector2, velocity: Vector2) -> None:
        self.mu = mu
        self.state = OrbitState(position.copy(), velocity.copy(), 0.0)

    def reset(self, mu: float, position: Vector2, velocity: Vector2) -> None:
        self.mu = mu
        self.state = OrbitState(position.copy(), velocity.copy(), 0.0)

    def acceleration(self) -> Vector2:
        r = self.state.position
        distance = r.length()
        if distance == 0:
            return Vector2()
        return -self.mu * r / (distance ** 3)

    def step(self, dt: float) -> None:
        """Advance the simulation by ``dt`` seconds using leapfrog integration."""

        if dt <= 0:
            return

        position = self.state.position
        velocity = self.state.velocity

        accel_initial = self.acceleration()
        velocity += accel_initial * (dt / 2.0)
        position += velocity * dt
        accel_final = self.acceleration()
        velocity += accel_final * (dt / 2.0)

        self.state.position = position
        self.state.velocity = velocity
        self.state.time += dt

    def current_state(self) -> OrbitState:
        return self.state.copy()


def specific_energy(mu: float, position: Vector2, velocity: Vector2) -> float:
    distance = position.length()
    if distance == 0:
        return float("inf")
    speed = velocity.length()
    return 0.5 * speed * speed - mu / distance


def angular_momentum_z(position: Vector2, velocity: Vector2) -> float:
    return position.x * velocity.y - position.y * velocity.x


def eccentricity_vector(mu: float, position: Vector2, velocity: Vector2) -> Vector2:
    distance = position.length()
    if distance == 0:
        return Vector2()

    h = angular_momentum_z(position, velocity)
    vx, vy = velocity.x, velocity.y
    e_x = vy * h / mu - position.x / distance
    e_y = -vx * h / mu - position.y / distance
    return Vector2(e_x, e_y)


def semi_major_axis(mu: float, position: Vector2, velocity: Vector2) -> float | None:
    energy = specific_energy(mu, position, velocity)
    if energy >= 0:
        return None
    return -mu / (2.0 * energy)


def orbital_period(mu: float, position: Vector2, velocity: Vector2) -> float | None:
    a = semi_major_axis(mu, position, velocity)
    if a is None:
        return None
    return 2.0 * pi * sqrt(a ** 3 / mu)


def periapsis(mu: float, position: Vector2, velocity: Vector2) -> float | None:
    e = eccentricity_vector(mu, position, velocity).length()
    a = semi_major_axis(mu, position, velocity)
    if a is None:
        return None
    return a * (1.0 - e)


def apoapsis(mu: float, position: Vector2, velocity: Vector2) -> float | None:
    e = eccentricity_vector(mu, position, velocity).length()
    a = semi_major_axis(mu, position, velocity)
    if a is None:
        return None
    if e >= 1.0:
        return None
    return a * (1.0 + e)


def format_seconds(seconds: float) -> str:
    if seconds == float("inf") or seconds != seconds:
        return "∞"
    if seconds < 0:
        seconds = 0.0
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    parts = []
    if days:
        parts.append(f"{days}d")
    if days or hours:
        parts.append(f"{hours:02d}h")
    parts.append(f"{minutes:02d}m")
    parts.append(f"{sec:02d}s")
    return " ".join(parts)


def state_summary(mu: float, position: Vector2, velocity: Vector2) -> Tuple[str, ...]:
    distance = position.length()
    speed = velocity.length()
    energy = specific_energy(mu, position, velocity)
    period = orbital_period(mu, position, velocity)
    e = eccentricity_vector(mu, position, velocity).length()
    per = periapsis(mu, position, velocity)
    apo = apoapsis(mu, position, velocity)

    lines = [
        f"Distance: {distance:,.0f} km",
        f"Speed: {speed:,.2f} km/s",
        f"Specific energy: {energy:,.2f} km^2/s^2",
        f"Eccentricity: {e:.4f}",
    ]
    if period is not None:
        lines.append(f"Period: {format_seconds(period)}")
    else:
        lines.append("Period: —")
    if per is not None:
        lines.append(f"Periapsis: {per:,.0f} km")
    else:
        lines.append("Periapsis: —")
    if apo is not None:
        lines.append(f"Apoapsis: {apo:,.0f} km")
    else:
        lines.append("Apoapsis: —")
    return tuple(lines)
