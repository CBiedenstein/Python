"""
3D Coverage visualization display for Site Survey tab.
Provides interactive 3D terrain surface with coverage overlay.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox,
    QComboBox, QPushButton
)
from PyQt6.QtCore import pyqtSignal

try:
    import pyqtgraph.opengl as gl
    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False

from ui.styles import PLOT_COLORS


class CoverageDisplay3D(QWidget):
    """
    3D surface plot with coverage overlay.

    Shows:
    - Terrain surface mesh
    - Color-coded coverage/detection overlay
    - Interactive camera controls
    - Vertical exaggeration adjustment
    """

    def __init__(self, theme: str = 'dark', parent=None):
        super().__init__(parent)
        self._theme = theme
        self._colors = PLOT_COLORS[theme]
        self._coverage_result = None
        self._surface = None
        self._radar_marker = None
        self._vertical_exag = 5.0
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        if not HAS_OPENGL:
            # Show message if OpenGL not available
            error_label = QLabel(
                "3D visualization requires PyOpenGL.\n"
                "Install with: pip install PyOpenGL PyOpenGL_accelerate"
            )
            error_label.setStyleSheet("color: #f85149; padding: 20px;")
            layout.addWidget(error_label)
            return

        # Control bar
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setContentsMargins(4, 4, 4, 4)

        # Layer selection
        ctrl_layout.addWidget(QLabel("Color by:"))
        self.layer_combo = QComboBox()
        self.layer_combo.addItems([
            "SNR (dB)",
            "Detection",
            "Elevation (m)",
            "Path Loss (dB)"
        ])
        self.layer_combo.currentIndexChanged.connect(self._on_layer_changed)
        ctrl_layout.addWidget(self.layer_combo)

        ctrl_layout.addStretch()

        # Vertical exaggeration
        ctrl_layout.addWidget(QLabel("Vert. Exag:"))
        self.exag_spin = QDoubleSpinBox()
        self.exag_spin.setRange(1.0, 50.0)
        self.exag_spin.setValue(5.0)
        self.exag_spin.setDecimals(1)
        self.exag_spin.setSuffix("x")
        self.exag_spin.valueChanged.connect(self._on_exag_changed)
        ctrl_layout.addWidget(self.exag_spin)

        # Reset view button
        self.reset_btn = QPushButton("Reset View")
        self.reset_btn.clicked.connect(self._reset_camera)
        ctrl_layout.addWidget(self.reset_btn)

        layout.addLayout(ctrl_layout)

        # Create 3D view widget
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setCameraPosition(distance=100, elevation=30, azimuth=45)

        # Set background color
        bg = self._colors['background']
        # Convert hex to RGB tuple
        if bg.startswith('#'):
            r = int(bg[1:3], 16) / 255
            g = int(bg[3:5], 16) / 255
            b = int(bg[5:7], 16) / 255
            self.gl_widget.setBackgroundColor((r, g, b, 1.0))

        # Add grid for reference
        self.grid = gl.GLGridItem()
        self.grid.setSize(100, 100)
        self.grid.setSpacing(10, 10)
        self.gl_widget.addItem(self.grid)

        layout.addWidget(self.gl_widget, 1)

    def _on_layer_changed(self, index):
        """Handle layer selection change."""
        self._update_surface_colors()

    def _on_exag_changed(self, value):
        """Handle vertical exaggeration change."""
        self._vertical_exag = value
        self._update_display()

    def _reset_camera(self):
        """Reset camera to default position."""
        self.gl_widget.setCameraPosition(distance=100, elevation=30, azimuth=45)

    def update_coverage(self, coverage_result):
        """
        Update display with new coverage results.

        Args:
            coverage_result: CoverageResult object
        """
        if not HAS_OPENGL:
            return

        self._coverage_result = coverage_result
        self._update_display()

    def _update_display(self):
        """Update the 3D visualization."""
        if not HAS_OPENGL or self._coverage_result is None:
            return

        result = self._coverage_result

        # Remove existing surface if any
        if self._surface is not None:
            self.gl_widget.removeItem(self._surface)
            self._surface = None

        if self._radar_marker is not None:
            self.gl_widget.removeItem(self._radar_marker)
            self._radar_marker = None

        # Downsample for performance if needed
        max_points = 200
        step_lat = max(1, len(result.lat_array) // max_points)
        step_lon = max(1, len(result.lon_array) // max_points)

        lat_arr = result.lat_array[::step_lat]
        lon_arr = result.lon_array[::step_lon]
        elev = result.elevation_m[::step_lat, ::step_lon]

        # Create coordinate grids
        # Normalize coordinates to reasonable range for display
        lat_center = (lat_arr[0] + lat_arr[-1]) / 2
        lon_center = (lon_arr[0] + lon_arr[-1]) / 2

        # Convert to local coordinates (km from center)
        import math
        lat_scale = 111.0  # km per degree latitude
        lon_scale = 111.0 * math.cos(math.radians(lat_center))

        x = (lon_arr - lon_center) * lon_scale
        y = (lat_arr - lat_center) * lat_scale

        # Create meshgrid
        X, Y = np.meshgrid(x, y)

        # Apply vertical exaggeration
        # Normalize elevation to similar scale as horizontal
        elev_normalized = elev / 1000.0  # Convert to km
        Z = elev_normalized * self._vertical_exag

        # Get colors for surface
        colors = self._get_surface_colors()
        if colors is not None:
            colors = colors[::step_lat, ::step_lon]

        # Create surface plot
        self._surface = gl.GLSurfacePlotItem(
            x=x, y=y, z=Z,
            colors=colors,
            shader='shaded',
            smooth=True
        )
        self.gl_widget.addItem(self._surface)

        # Add radar marker
        radar_x = (result.radar_lon - lon_center) * lon_scale
        radar_y = (result.radar_lat - lat_center) * lat_scale
        radar_elev = result.radar_elevation_m / 1000.0 * self._vertical_exag
        radar_z = radar_elev + (result.radar_height_m / 1000.0 * self._vertical_exag)

        marker_pos = np.array([[radar_x, radar_y, radar_z]])
        self._radar_marker = gl.GLScatterPlotItem(
            pos=marker_pos,
            color=(1, 0, 0, 1),  # Red
            size=15
        )
        self.gl_widget.addItem(self._radar_marker)

        # Update grid size to match terrain
        self.grid.setSize(abs(x[-1] - x[0]), abs(y[-1] - y[0]))

        # Adjust camera distance based on terrain size
        terrain_size = max(abs(x[-1] - x[0]), abs(y[-1] - y[0]))
        self.gl_widget.setCameraPosition(distance=terrain_size * 1.5)

    def _get_surface_colors(self):
        """Get RGBA colors for surface based on selected layer."""
        if self._coverage_result is None:
            return None

        result = self._coverage_result
        layer_text = self.layer_combo.currentText()

        # Select data layer
        if "SNR" in layer_text:
            data = result.snr_db.copy()
            data = np.clip(data, -20, 50)
            data = np.where(np.isinf(data), np.nan, data)
            cmap = 'viridis'
        elif "Detection" in layer_text:
            data = result.detected.astype(float)
            cmap = 'detection'
        elif "Elevation" in layer_text:
            data = result.elevation_m.copy()
            cmap = 'terrain'
        elif "Path Loss" in layer_text:
            data = result.path_loss_db.copy()
            data = np.clip(data, 0, 200)
            data = np.where(np.isinf(data), np.nan, data)
            cmap = 'viridis'
        else:
            data = result.snr_db.copy()
            cmap = 'viridis'

        # Normalize data
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            vmin, vmax = np.nanmin(valid_data), np.nanmax(valid_data)
            if vmax > vmin:
                norm_data = (data - vmin) / (vmax - vmin)
            else:
                norm_data = np.zeros_like(data)
        else:
            norm_data = np.zeros_like(data)

        norm_data = np.nan_to_num(norm_data, nan=0.5)

        # Create RGBA colors
        colors = np.zeros((*data.shape, 4), dtype=np.float32)

        if cmap == 'viridis':
            # Viridis-like colormap
            colors[:, :, 0] = 0.267 + 0.733 * norm_data  # R
            colors[:, :, 1] = norm_data  # G
            colors[:, :, 2] = 0.329 + 0.341 * (1 - norm_data)  # B
        elif cmap == 'detection':
            # Binary red/green
            colors[:, :, 0] = 1 - norm_data  # R (red when 0)
            colors[:, :, 1] = norm_data  # G (green when 1)
            colors[:, :, 2] = 0.2  # B
        elif cmap == 'terrain':
            # Terrain colormap (blue-green-brown-white)
            # Low = blue/green, high = brown/white
            colors[:, :, 0] = np.clip(norm_data * 1.5, 0, 1)
            colors[:, :, 1] = np.clip(0.5 + norm_data * 0.5, 0, 1)
            colors[:, :, 2] = np.clip(0.3 - norm_data * 0.3, 0, 1)
        else:
            # Grayscale fallback
            colors[:, :, 0] = norm_data
            colors[:, :, 1] = norm_data
            colors[:, :, 2] = norm_data

        colors[:, :, 3] = 1.0  # Alpha

        return colors

    def _update_surface_colors(self):
        """Update surface colors without rebuilding geometry."""
        if not HAS_OPENGL or self._surface is None:
            return

        # For now, just rebuild the whole display
        # A more efficient implementation would update colors in place
        self._update_display()

    def set_theme(self, theme: str):
        """Update display theme."""
        if not HAS_OPENGL:
            return

        self._theme = theme
        self._colors = PLOT_COLORS[theme]

        bg = self._colors['background']
        if bg.startswith('#'):
            r = int(bg[1:3], 16) / 255
            g = int(bg[3:5], 16) / 255
            b = int(bg[5:7], 16) / 255
            self.gl_widget.setBackgroundColor((r, g, b, 1.0))

    def clear(self):
        """Clear the display."""
        if not HAS_OPENGL:
            return

        self._coverage_result = None

        if self._surface is not None:
            self.gl_widget.removeItem(self._surface)
            self._surface = None

        if self._radar_marker is not None:
            self.gl_widget.removeItem(self._radar_marker)
            self._radar_marker = None
