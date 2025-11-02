"""Pygame application for visualising satellite motion."""

from __future__ import annotations

from typing import Iterable, List

import pygame
from pygame.math import Vector2

from .physics import OrbitSimulator, state_summary
from .settings import SimulationConfig


class SimulationApp:
    """Main application class encapsulating the event loop and rendering."""

    def __init__(self, config: SimulationConfig | None = None) -> None:
        pygame.init()
        pygame.display.set_caption("Satellite orbit simulator")
        self.config = config or SimulationConfig()
        self.font = pygame.font.Font(None, 22)
        self.screen = pygame.display.set_mode(
            (self.config.view.width, self.config.view.height)
        )
        self.clock = pygame.time.Clock()
        self._init_simulator()
        self.trail: List[Vector2] = []
        self.trail_timer = 0.0
        self.running = True
        self.time_scale = self.config.time_scale
        self.base_dt = self.config.base_dt
        self.scale = self.config.view.pixels_per_km

    def _init_simulator(self) -> None:
        position, velocity = self.config.orbit.initial_state()
        self.simulator = OrbitSimulator(
            self.config.orbit.mu, position, velocity
        )
        self.trail_timer = 0.0

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------
    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self._handle_key(event.key)

    def _handle_key(self, key: int) -> None:
        toggles = self.config.toggles
        orbit = self.config.orbit
        view = self.config.view

        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_SPACE:
            toggles.paused = not toggles.paused
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.time_scale = min(self.time_scale * 1.2, 86400.0)
        elif key in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_KP_MINUS):
            self.time_scale = max(self.time_scale / 1.2, 0.1)
        elif key == pygame.K_t:
            toggles.show_trail = not toggles.show_trail
        elif key == pygame.K_g:
            toggles.show_grid = not toggles.show_grid
        elif key == pygame.K_c:
            self.trail.clear()
        elif key == pygame.K_r:
            self.config = SimulationConfig()
            self._init_simulator()
            self.trail.clear()
            self.time_scale = self.config.time_scale
            self.scale = self.config.view.pixels_per_km
        elif key == pygame.K_n:
            orbit.randomize()
            self._init_simulator()
            self.trail.clear()
        elif key == pygame.K_UP:
            orbit.velocity_factor = min(orbit.velocity_factor * 1.05, 5.0)
            self._init_simulator()
            self.trail.clear()
        elif key == pygame.K_DOWN:
            orbit.velocity_factor = max(orbit.velocity_factor / 1.05, 0.2)
            self._init_simulator()
            self.trail.clear()
        elif key == pygame.K_RIGHT:
            orbit.distance_km = min(orbit.distance_km * 1.05, 120000.0)
            self._init_simulator()
            self.trail.clear()
        elif key == pygame.K_LEFT:
            orbit.distance_km = max(
                max(orbit.distance_km / 1.05, orbit.body_radius_km + 200.0),
                orbit.body_radius_km + 1.0,
            )
            self._init_simulator()
            self.trail.clear()
        elif key == pygame.K_LEFTBRACKET:
            orbit.mu = max(orbit.mu * 0.95, 1000.0)
            self._init_simulator()
        elif key == pygame.K_RIGHTBRACKET:
            orbit.mu = min(orbit.mu * 1.05, 1000000.0)
            self._init_simulator()
        elif key == pygame.K_z:
            self.scale = max(self.scale / 1.2, view.min_scale)
        elif key == pygame.K_x:
            self.scale = min(self.scale * 1.2, view.max_scale)

    # ------------------------------------------------------------------
    # Simulation update
    # ------------------------------------------------------------------
    def update(self, dt_real: float) -> None:
        if self.config.toggles.paused:
            return
        dt_simulation = self.time_scale * dt_real
        max_step = self.base_dt
        remaining = dt_simulation
        while remaining > 0:
            current_dt = min(max_step, remaining)
            self.simulator.step(current_dt)
            remaining -= current_dt
            if self.config.toggles.show_trail:
                self._update_trail(current_dt)

    def _update_trail(self, dt: float) -> None:
        self.trail_timer += dt
        if self.trail_timer < self.config.trail_sample_interval:
            return
        self.trail_timer = 0.0
        self.trail.append(self.simulator.current_state().position)
        if len(self.trail) > self.config.max_trail_points:
            self.trail.pop(0)

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def draw(self) -> None:
        view = self.config.view
        self.screen.fill(view.background_color)
        if self.config.toggles.show_grid:
            self._draw_grid()
        self._draw_central_body()
        if self.config.toggles.show_trail and len(self.trail) > 1:
            self._draw_trail(self.trail, view.trail_color)
        self._draw_satellite()
        self._draw_panel()
        pygame.display.flip()

    def _world_to_screen(self, point: Vector2) -> Vector2:
        cx = self.config.view.width / 2
        cy = self.config.view.height / 2
        return Vector2(cx, cy) + point * self.scale

    def _draw_grid(self) -> None:
        view = self.config.view
        width, height = view.width, view.height
        center = Vector2(width / 2, height / 2)
        spacing = max(50, int(200 * self.scale))
        color = view.grid_color
        for offset in range(0, width // spacing + 2):
            x = center.x + (offset - width // (2 * spacing)) * spacing
            pygame.draw.line(self.screen, color, (x, 0), (x, height), 1)
        for offset in range(0, height // spacing + 2):
            y = center.y + (offset - height // (2 * spacing)) * spacing
            pygame.draw.line(self.screen, color, (0, y), (width, y), 1)
        pygame.draw.line(
            self.screen, color, (center.x, 0), (center.x, height), 2
        )
        pygame.draw.line(
            self.screen, color, (0, center.y), (width, center.y), 2
        )

    def _draw_central_body(self) -> None:
        view = self.config.view
        radius_pixels = max(
            6, int(self.config.orbit.body_radius_km * self.scale)
        )
        pygame.draw.circle(
            self.screen,
            view.body_color,
            (view.width // 2, view.height // 2),
            radius_pixels,
        )

    def _draw_trail(self, points: Iterable[Vector2], color: tuple[int, int, int]) -> None:
        screen_points = [self._world_to_screen(p) for p in points]
        if len(screen_points) > 1:
            pygame.draw.lines(self.screen, color, False, screen_points, 2)

    def _draw_satellite(self) -> None:
        position = self.simulator.current_state().position
        screen_pos = self._world_to_screen(position)
        pygame.draw.circle(
            self.screen,
            self.config.view.satellite_color,
            (int(screen_pos.x), int(screen_pos.y)),
            6,
        )

    def _draw_panel(self) -> None:
        margin = 12
        lines = self._info_lines()
        for index, text in enumerate(lines):
            surface = self.font.render(text, True, self.config.view.label_color)
            self.screen.blit(surface, (margin, margin + index * 22))

    def _info_lines(self) -> List[str]:
        state = self.simulator.current_state()
        orbit = self.config.orbit
        summary = state_summary(orbit.mu, state.position, state.velocity)
        status = "Paused" if self.config.toggles.paused else "Running"
        info = [
            f"Status: {status}",
            f"Central body: {orbit.body_name}",
            f"Gravitational parameter μ: {orbit.mu:,.1f} km³/s²",
            f"Satellite: {orbit.satellite_name}",
            f"Time scale: x{self.time_scale:.1f}",
            f"Sim time: {state.time:,.0f} s",
            f"Zoom: {self.scale:.5f} px/km",
            f"Velocity factor: {orbit.velocity_factor:.3f}",
            f"Circular velocity: {orbit.circular_velocity():.2f} km/s",
            f"Distance setting: {orbit.distance_km:,.0f} km",
            "",
        ]
        info.extend(summary)
        info.append("")
        info.extend(self._control_help())
        return info

    def _control_help(self) -> List[str]:
        return [
            "Space – pause/resume",
            "+/- – change time scale",
            "Arrows – change orbit distance / velocity",
            "[ ] – adjust gravitational parameter",
            "T – toggle trail, C – clear trail",
            "G – toggle grid, R – reset, N – randomise",
            "Z/X – zoom out/in",
            "Esc – quit",
        ]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        while self.running:
            dt_ms = self.clock.tick(60)
            dt_real = dt_ms / 1000.0
            self.handle_events()
            self.update(dt_real)
            self.draw()
        pygame.quit()


__all__ = ["SimulationApp"]
