"""PyQt application that visualises the satellite orbit."""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from .camera import Camera
from .orbit import OrbitParameters, OrbitSimulator
from .planet import Planet, PlanetConfig
from .renderer import Light, RenderSettings, Scene


@dataclass
class UiState:
    seed: int = 1
    recording: bool = False
    last_frame_time: float = time.time()
    frame_index: int = 0


class RenderWidget(QtWidgets.QLabel):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setAlignment(QtCore.Qt.AlignCenter)

    def update_image(self, image: np.ndarray) -> None:
        height, width, _ = image.shape
        qimage = QtGui.QImage(image.data, width, height, 3 * width, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.setPixmap(pixmap)


class ControlPanel(QtWidgets.QWidget):
    parameters_changed = QtCore.pyqtSignal()
    seed_changed = QtCore.pyqtSignal(int)
    camera_changed = QtCore.pyqtSignal()
    record_toggled = QtCore.pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._create_translation_group())
        layout.addWidget(self._create_rotation_group())
        layout.addWidget(self._create_scaling_group())
        layout.addWidget(self._create_orbit_group())
        layout.addStretch(1)

    def _create_translation_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Перенос")
        form = QtWidgets.QGridLayout(group)
        self.dx = self._create_spin_box()
        self.dy = self._create_spin_box()
        self.dz = self._create_spin_box()
        form.addWidget(QtWidgets.QLabel("dx"), 0, 0)
        form.addWidget(self.dx, 0, 1)
        form.addWidget(QtWidgets.QLabel("dy"), 1, 0)
        form.addWidget(self.dy, 1, 1)
        form.addWidget(QtWidgets.QLabel("dz"), 2, 0)
        form.addWidget(self.dz, 2, 1)
        button = QtWidgets.QPushButton("Перенести")
        button.clicked.connect(self._on_translate)
        form.addWidget(button, 3, 0, 1, 2)
        return group

    def _create_rotation_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Поворот")
        form = QtWidgets.QGridLayout(group)
        self.angle_x = self._create_spin_box(maximum=360.0)
        self.angle_y = self._create_spin_box(maximum=360.0)
        self.angle_z = self._create_spin_box(maximum=360.0)
        form.addWidget(QtWidgets.QLabel("angle x"), 0, 0)
        form.addWidget(self.angle_x, 0, 1)
        form.addWidget(QtWidgets.QLabel("angle y"), 1, 0)
        form.addWidget(self.angle_y, 1, 1)
        form.addWidget(QtWidgets.QLabel("angle z"), 2, 0)
        form.addWidget(self.angle_z, 2, 1)
        button = QtWidgets.QPushButton("Повернуть")
        button.clicked.connect(self._on_rotate)
        form.addWidget(button, 3, 0, 1, 2)
        return group

    def _create_scaling_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Масштабирование")
        form = QtWidgets.QGridLayout(group)
        self.kx = self._create_spin_box(value=1.0, minimum=0.1, maximum=10.0)
        self.ky = self._create_spin_box(value=1.0, minimum=0.1, maximum=10.0)
        self.kz = self._create_spin_box(value=1.0, minimum=0.1, maximum=10.0)
        form.addWidget(QtWidgets.QLabel("kx"), 0, 0)
        form.addWidget(self.kx, 0, 1)
        form.addWidget(QtWidgets.QLabel("ky"), 1, 0)
        form.addWidget(self.ky, 1, 1)
        form.addWidget(QtWidgets.QLabel("kz"), 2, 0)
        form.addWidget(self.kz, 2, 1)
        button = QtWidgets.QPushButton("Промасштабировать")
        button.clicked.connect(self._on_scale)
        form.addWidget(button, 3, 0, 1, 2)
        return group

    def _create_orbit_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Параметры орбиты")
        form = QtWidgets.QGridLayout(group)
        self.orbit_radius = self._create_spin_box(value=4.0, minimum=1.5, maximum=12.0)
        self.orbit_speed = self._create_spin_box(value=0.8, minimum=0.0, maximum=5.0)
        self.seed_box = QtWidgets.QSpinBox()
        self.seed_box.setRange(0, 10_000)
        self.seed_box.setValue(1)
        form.addWidget(QtWidgets.QLabel("Радиус"), 0, 0)
        form.addWidget(self.orbit_radius, 0, 1)
        form.addWidget(QtWidgets.QLabel("Скорость"), 1, 0)
        form.addWidget(self.orbit_speed, 1, 1)
        form.addWidget(QtWidgets.QLabel("Seed"), 2, 0)
        form.addWidget(self.seed_box, 2, 1)
        button_apply = QtWidgets.QPushButton("Применить")
        button_apply.clicked.connect(self._on_apply)
        form.addWidget(button_apply, 3, 0, 1, 2)
        self.record_button = QtWidgets.QPushButton("Начать запись")
        self.record_button.setCheckable(True)
        self.record_button.toggled.connect(self.record_toggled)
        form.addWidget(self.record_button, 4, 0, 1, 2)
        return group

    def _create_spin_box(
        self,
        value: float = 0.0,
        minimum: float = -10.0,
        maximum: float = 10.0,
        step: float = 0.1,
    ) -> QtWidgets.QDoubleSpinBox:
        box = QtWidgets.QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setDecimals(2)
        box.setSingleStep(step)
        box.setValue(value)
        return box

    def _on_translate(self) -> None:
        self.parameters_changed.emit()

    def _on_rotate(self) -> None:
        self.camera_changed.emit()

    def _on_scale(self) -> None:
        self.parameters_changed.emit()

    def _on_apply(self) -> None:
        self.parameters_changed.emit()
        self.seed_changed.emit(self.seed_box.value())


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Satellite Visualiser")
        self.ui_state = UiState()

        self.render_widget = RenderWidget()
        self.panel = ControlPanel()

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.addWidget(self.panel)
        layout.addWidget(self.render_widget, stretch=1)
        self.setCentralWidget(central)

        self.camera = Camera()
        self.planet = Planet(PlanetConfig())
        self.planet.generate(self.ui_state.seed)
        self.orbit = OrbitSimulator(OrbitParameters(radius=4.0, angular_velocity=0.8))
        light_dir = np.array([0.6, -0.8, -0.4])
        self.scene = Scene(self.planet, self.orbit, Light(light_dir, np.array([1.0, 0.95, 0.9])))
        self.settings = RenderSettings()

        self.panel.parameters_changed.connect(self._apply_controls)
        self.panel.seed_changed.connect(self._regenerate_planet)
        self.panel.camera_changed.connect(self._apply_rotation)
        self.panel.record_toggled.connect(self._toggle_recording)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(30)

        self._update_frame()

    def _apply_controls(self) -> None:
        dx = self.panel.dx.value()
        dy = self.panel.dy.value()
        dz = self.panel.dz.value()
        self.camera.pan(dx * 0.1, dy * 0.1)
        self.camera.target[2] += dz * 0.1
        sx = self.panel.kx.value()
        sy = self.panel.ky.value()
        sz = self.panel.kz.value()
        scale_factor = (sx + sy + sz) / 3.0
        self.camera.zoom(1.0 / max(scale_factor, 0.1))
        radius = self.panel.orbit_radius.value()
        speed = self.panel.orbit_speed.value()
        self.orbit.set_radius(radius)
        self.orbit.set_speed(speed)

    def _apply_rotation(self) -> None:
        yaw = math.radians(self.panel.angle_y.value())
        pitch = math.radians(self.panel.angle_x.value())
        self.camera.orbit(yaw, pitch)

    def _regenerate_planet(self, seed: int) -> None:
        self.planet.generate(seed)

    def _toggle_recording(self, active: bool) -> None:
        self.ui_state.recording = active
        self.panel.record_button.setText("Остановить запись" if active else "Начать запись")
        if active:
            self.ui_state.frame_index = 0
            output = Path("frames")
            output.mkdir(exist_ok=True)

    def _update_frame(self) -> None:
        current_time = time.time()
        dt = min(current_time - self.ui_state.last_frame_time, 0.05)
        self.ui_state.last_frame_time = current_time
        image = self.scene.render(self.camera, self.settings, dt)
        self.render_widget.update_image(image)
        if self.ui_state.recording:
            self._save_frame(image)

    def _save_frame(self, image: np.ndarray) -> None:
        output = Path("frames")
        output.mkdir(exist_ok=True)
        filename = output / f"frame_{self.ui_state.frame_index:05d}.png"
        QtGui.QImage(image.data, image.shape[1], image.shape[0], 3 * image.shape[1], QtGui.QImage.Format_RGB888).save(str(filename))
        self.ui_state.frame_index += 1


def run() -> None:
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.resize(960, 520)
    window.show()
    app.exec_()
