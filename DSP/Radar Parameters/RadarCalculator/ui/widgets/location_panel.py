"""
Location input panel for Site Survey tab.
Provides radar position input and terrain data source selection.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QDoubleSpinBox, QGroupBox, QComboBox,
    QPushButton, QFileDialog, QProgressBar, QFrame
)
from PyQt6.QtCore import pyqtSignal, Qt, QThread, pyqtSlot, QTimer
from PyQt6.QtGui import QIcon


class TerrainLoadThread(QThread):
    """Background thread for loading terrain data."""

    finished = pyqtSignal(object)  # TerrainData or None
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    progress_percent = pyqtSignal(int)  # 0-100 progress percentage

    def __init__(self, loader, source_type: str, lat: float = None,
                 lon: float = None, range_km: float = None, filepath: str = None):
        super().__init__()
        self._loader = loader
        self._source_type = source_type
        self._lat = lat
        self._lon = lon
        self._range_km = range_km
        self._filepath = filepath
        self._current_percent = 0

    def _progress_callback(self, message: str):
        """Thread-safe progress callback with percentage parsing."""
        self.progress.emit(message)

        # Parse progress from message and update percentage
        msg_lower = message.lower()
        if "downloading" in msg_lower or "requesting" in msg_lower:
            self._update_percent(10)
        elif "fetching" in msg_lower:
            self._update_percent(20)
        elif "tile" in msg_lower:
            # Try to parse "tile X/Y" format
            import re
            match = re.search(r'tile\s+(\d+)/(\d+)', msg_lower)
            if match:
                current = int(match.group(1))
                total = int(match.group(2))
                # Scale 20-80 for tile downloads
                pct = 20 + int(60 * current / total)
                self._update_percent(pct)
        elif "downloaded" in msg_lower:
            self._update_percent(80)
        elif "merging" in msg_lower:
            self._update_percent(90)
        elif "loading" in msg_lower:
            self._update_percent(30)

    def _update_percent(self, percent: int):
        """Update progress percentage (only increases)."""
        if percent > self._current_percent:
            self._current_percent = percent
            self.progress_percent.emit(percent)

    def run(self):
        try:
            if self._source_type == "srtm":
                self.progress.emit("Downloading SRTM data...")
                terrain = self._loader.load_srtm(
                    self._lat, self._lon, self._range_km,
                    progress_callback=self._progress_callback
                )
            elif self._source_type == "geotiff":
                self.progress.emit("Loading GeoTIFF...")
                terrain = self._loader.load_geotiff(self._filepath)
            elif self._source_type == "dted":
                self.progress.emit("Loading DTED...")
                terrain = self._loader.load_dted(self._filepath)
            else:
                raise ValueError(f"Unknown source type: {self._source_type}")

            self.finished.emit(terrain)

        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit(None)


class LocationPanel(QWidget):
    """
    Panel for specifying radar location and terrain data source.

    Emits signals when location changes or terrain is loaded.
    """

    location_changed = pyqtSignal(float, float, float)  # lat, lon, height
    terrain_loaded = pyqtSignal(object)  # TerrainData
    terrain_load_started = pyqtSignal()
    terrain_load_error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._terrain_loader = None
        self._terrain = None
        self._load_thread = None
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Radar location group
        location_group = QGroupBox("Radar Location")
        location_layout = QGridLayout(location_group)

        # Latitude input
        location_layout.addWidget(QLabel("Latitude:"), 0, 0)
        self.lat_spin = QDoubleSpinBox()
        self.lat_spin.setRange(-90.0, 90.0)
        self.lat_spin.setDecimals(6)
        self.lat_spin.setValue(39.0)
        self.lat_spin.setSuffix("°")
        self.lat_spin.setToolTip("Radar site latitude in degrees (-90 to 90)")
        location_layout.addWidget(self.lat_spin, 0, 1)

        # Longitude input
        location_layout.addWidget(QLabel("Longitude:"), 1, 0)
        self.lon_spin = QDoubleSpinBox()
        self.lon_spin.setRange(-180.0, 180.0)
        self.lon_spin.setDecimals(6)
        self.lon_spin.setValue(-105.0)
        self.lon_spin.setSuffix("°")
        self.lon_spin.setToolTip("Radar site longitude in degrees (-180 to 180)")
        location_layout.addWidget(self.lon_spin, 1, 1)

        # Radar height AGL
        location_layout.addWidget(QLabel("Antenna Height:"), 2, 0)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0.0, 1000.0)
        self.height_spin.setDecimals(1)
        self.height_spin.setValue(30.0)
        self.height_spin.setSuffix(" m AGL")
        self.height_spin.setToolTip("Radar antenna height Above Ground Level")
        location_layout.addWidget(self.height_spin, 2, 1)

        # Coverage range
        location_layout.addWidget(QLabel("Coverage Range:"), 3, 0)
        self.range_spin = QDoubleSpinBox()
        self.range_spin.setRange(1.0, 500.0)
        self.range_spin.setDecimals(1)
        self.range_spin.setValue(150.0)
        self.range_spin.setSuffix(" km")
        self.range_spin.setToolTip("Maximum range for coverage calculation")
        location_layout.addWidget(self.range_spin, 3, 1)

        layout.addWidget(location_group)

        # Terrain data group
        terrain_group = QGroupBox("Terrain Data")
        terrain_layout = QVBoxLayout(terrain_group)

        # Source selection
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Source:"))
        self.source_combo = QComboBox()
        self.source_combo.addItems([
            "Auto-download SRTM",
            "Load GeoTIFF File",
            "Load DTED File"
        ])
        self.source_combo.setToolTip(
            "SRTM: Automatically download elevation data from NASA\n"
            "GeoTIFF: Load a GeoTIFF elevation file\n"
            "DTED: Load a DTED elevation file"
        )
        source_layout.addWidget(self.source_combo, 1)
        terrain_layout.addLayout(source_layout)

        # Load button
        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Terrain")
        self.load_btn.setObjectName("primaryButton")
        self.load_btn.setToolTip("Load terrain data from selected source")
        btn_layout.addWidget(self.load_btn)
        terrain_layout.addLayout(btn_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                background-color: #2d2d2d;
                text-align: center;
                color: #fff;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #00aa55, stop: 1 #00ff88
                );
                border-radius: 2px;
            }
        """)
        self.progress_bar.hide()  # Hidden until download starts
        terrain_layout.addWidget(self.progress_bar)

        # Progress text indicator
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #888; font-style: italic;")
        self.progress_label.setWordWrap(True)
        terrain_layout.addWidget(self.progress_label)

        # Status label
        self.status_label = QLabel("No terrain loaded")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #888;")
        terrain_layout.addWidget(self.status_label)

        layout.addWidget(terrain_group)

    def _connect_signals(self):
        self.lat_spin.valueChanged.connect(self._on_location_changed)
        self.lon_spin.valueChanged.connect(self._on_location_changed)
        self.height_spin.valueChanged.connect(self._on_location_changed)
        self.range_spin.valueChanged.connect(self._on_location_changed)
        self.load_btn.clicked.connect(self._on_load_clicked)

    def _on_location_changed(self):
        self.location_changed.emit(
            self.lat_spin.value(),
            self.lon_spin.value(),
            self.height_spin.value()
        )

    def _on_load_clicked(self):
        """Handle load terrain button click."""
        source_text = self.source_combo.currentText()

        # Lazy import terrain loader
        if self._terrain_loader is None:
            try:
                from core.terrain import TerrainLoader
                self._terrain_loader = TerrainLoader()
            except ImportError as e:
                self.status_label.setText(f"Error: {str(e)}")
                self.status_label.setStyleSheet("color: #ff6b6b;")
                self.terrain_load_error.emit(str(e))
                return

        if "SRTM" in source_text:
            self._load_srtm()
        elif "GeoTIFF" in source_text:
            self._load_file("geotiff")
        elif "DTED" in source_text:
            self._load_file("dted")

    def _load_srtm(self):
        """Load SRTM data for current location."""
        lat = self.lat_spin.value()
        lon = self.lon_spin.value()
        range_km = self.range_spin.value()

        self.load_btn.setEnabled(False)
        self.progress_label.setText("Downloading SRTM data...")
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.terrain_load_started.emit()

        self._load_thread = TerrainLoadThread(
            self._terrain_loader, "srtm",
            lat=lat, lon=lon, range_km=range_km
        )
        self._load_thread.finished.connect(self._on_terrain_loaded)
        self._load_thread.error.connect(self._on_terrain_error)
        self._load_thread.progress.connect(self._on_progress)
        self._load_thread.progress_percent.connect(self._on_progress_percent)
        self._load_thread.start()

    def _load_file(self, file_type: str):
        """Load terrain from file."""
        if file_type == "geotiff":
            filter_str = "GeoTIFF Files (*.tif *.tiff);;All Files (*)"
            title = "Open GeoTIFF Elevation File"
        else:
            filter_str = "DTED Files (*.dt0 *.dt1 *.dt2);;All Files (*)"
            title = "Open DTED Elevation File"

        filepath, _ = QFileDialog.getOpenFileName(
            self, title, "", filter_str
        )

        if not filepath:
            return

        self.load_btn.setEnabled(False)
        self.progress_label.setText(f"Loading {file_type.upper()}...")
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.terrain_load_started.emit()

        self._load_thread = TerrainLoadThread(
            self._terrain_loader, file_type, filepath=filepath
        )
        self._load_thread.finished.connect(self._on_terrain_loaded)
        self._load_thread.error.connect(self._on_terrain_error)
        self._load_thread.progress.connect(self._on_progress)
        self._load_thread.progress_percent.connect(self._on_progress_percent)
        self._load_thread.start()

    @pyqtSlot(str)
    def _on_progress(self, message: str):
        self.progress_label.setText(message)

    @pyqtSlot(int)
    def _on_progress_percent(self, percent: int):
        """Update progress bar with percentage."""
        self.progress_bar.setValue(percent)

    @pyqtSlot(object)
    def _on_terrain_loaded(self, terrain):
        self.load_btn.setEnabled(True)
        self.progress_label.setText("")
        self.progress_bar.setValue(100)
        # Hide progress bar after a brief delay
        QTimer.singleShot(500, self.progress_bar.hide)

        if terrain is not None:
            self._terrain = terrain
            bounds = terrain.bounds
            shape = terrain.shape
            res = terrain.resolution_m

            self.status_label.setText(
                f"Loaded: {shape[0]}×{shape[1]} grid\n"
                f"Resolution: ~{res:.0f}m\n"
                f"Bounds: {bounds.lat_min:.4f}° to {bounds.lat_max:.4f}°N\n"
                f"        {bounds.lon_min:.4f}° to {bounds.lon_max:.4f}°E"
            )
            self.status_label.setStyleSheet("color: #00ff88;")
            self.terrain_loaded.emit(terrain)
        else:
            self.status_label.setText("Failed to load terrain")
            self.status_label.setStyleSheet("color: #ff6b6b;")

    @pyqtSlot(str)
    def _on_terrain_error(self, error_msg: str):
        self.load_btn.setEnabled(True)
        self.progress_label.setText("")
        self.progress_bar.hide()
        self.status_label.setText(f"Error: {error_msg}")
        self.status_label.setStyleSheet("color: #ff6b6b;")
        self.terrain_load_error.emit(error_msg)

    # Public API

    def get_latitude(self) -> float:
        """Get radar latitude in degrees."""
        return self.lat_spin.value()

    def get_longitude(self) -> float:
        """Get radar longitude in degrees."""
        return self.lon_spin.value()

    def get_height_m(self) -> float:
        """Get radar antenna height AGL in meters."""
        return self.height_spin.value()

    def get_range_km(self) -> float:
        """Get coverage range in km."""
        return self.range_spin.value()

    def get_terrain(self):
        """Get loaded TerrainData or None."""
        return self._terrain

    def set_location(self, lat: float, lon: float, height_m: float = None):
        """Set radar location."""
        self.lat_spin.blockSignals(True)
        self.lon_spin.blockSignals(True)

        self.lat_spin.setValue(lat)
        self.lon_spin.setValue(lon)
        if height_m is not None:
            self.height_spin.setValue(height_m)

        self.lat_spin.blockSignals(False)
        self.lon_spin.blockSignals(False)
        self._on_location_changed()

    def set_range_km(self, range_km: float):
        """Set coverage range in km."""
        self.range_spin.setValue(range_km)

    def has_terrain(self) -> bool:
        """Check if terrain data is loaded."""
        return self._terrain is not None
