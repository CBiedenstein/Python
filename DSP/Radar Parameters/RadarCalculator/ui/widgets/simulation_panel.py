"""
Simulation control panel for track generator.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSpinBox, QDoubleSpinBox, QGroupBox,
    QPushButton, QComboBox, QCheckBox, QSlider,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor

from core.track_simulator import (
    TrackSimulator, TargetLibrary, TargetCategory, Track
)


class SimulationControlPanel(QWidget):
    """
    Control panel for managing track simulation.
    """

    simulation_started = pyqtSignal()
    simulation_stopped = pyqtSignal()
    simulation_updated = pyqtSignal()
    tracks_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._simulator = TrackSimulator()
        self._timer = QTimer()
        self._timer.timeout.connect(self._on_timer_tick)
        self._update_rate_ms = 1000  # 1 second default
        self._time_scale = 1.0  # Real-time default

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Simulation Controls
        sim_group = QGroupBox("Simulation Control")
        sim_layout = QVBoxLayout(sim_group)

        # Start/Stop buttons
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("▶ Start")
        self.start_btn.setObjectName("primaryButton")
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setEnabled(False)
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.setObjectName("dangerButton")

        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.clear_btn)
        sim_layout.addLayout(btn_layout)

        # Time controls
        time_layout = QGridLayout()

        time_layout.addWidget(QLabel("Update Rate:"), 0, 0)
        self.update_rate_spin = QDoubleSpinBox()
        self.update_rate_spin.setRange(0.1, 10.0)
        self.update_rate_spin.setValue(1.0)
        self.update_rate_spin.setSuffix(" sec")
        self.update_rate_spin.setDecimals(1)
        time_layout.addWidget(self.update_rate_spin, 0, 1)

        time_layout.addWidget(QLabel("Time Scale:"), 1, 0)
        self.time_scale_spin = QDoubleSpinBox()
        self.time_scale_spin.setRange(0.1, 100.0)
        self.time_scale_spin.setValue(1.0)
        self.time_scale_spin.setSuffix("x")
        self.time_scale_spin.setDecimals(1)
        time_layout.addWidget(self.time_scale_spin, 1, 1)

        sim_layout.addLayout(time_layout)

        layout.addWidget(sim_group)

        # Track Generation
        gen_group = QGroupBox("Generate Tracks")
        gen_layout = QVBoxLayout(gen_group)

        # Quick populate
        quick_layout = QGridLayout()
        quick_layout.addWidget(QLabel("Aircraft:"), 0, 0)
        self.num_aircraft_spin = QSpinBox()
        self.num_aircraft_spin.setRange(0, 50)
        self.num_aircraft_spin.setValue(10)
        quick_layout.addWidget(self.num_aircraft_spin, 0, 1)

        quick_layout.addWidget(QLabel("Vessels:"), 0, 2)
        self.num_vessels_spin = QSpinBox()
        self.num_vessels_spin.setRange(0, 50)
        self.num_vessels_spin.setValue(5)
        quick_layout.addWidget(self.num_vessels_spin, 0, 3)

        self.populate_btn = QPushButton("Populate Random")
        quick_layout.addWidget(self.populate_btn, 1, 0, 1, 4)

        gen_layout.addLayout(quick_layout)

        # Add specific target
        add_layout = QGridLayout()
        add_layout.addWidget(QLabel("Add Specific:"), 0, 0)

        self.category_combo = QComboBox()
        for cat in TargetCategory:
            self.category_combo.addItem(cat.value, cat)
        add_layout.addWidget(self.category_combo, 0, 1)

        self.target_combo = QComboBox()
        self._update_target_combo()
        add_layout.addWidget(self.target_combo, 1, 0, 1, 2)

        self.add_target_btn = QPushButton("Add Target")
        add_layout.addWidget(self.add_target_btn, 2, 0, 1, 2)

        gen_layout.addLayout(add_layout)

        layout.addWidget(gen_group)

        # Display Options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)

        self.show_labels_check = QCheckBox("Show Track Labels")
        self.show_labels_check.setChecked(True)
        display_layout.addWidget(self.show_labels_check)

        self.show_history_check = QCheckBox("Show Track History")
        self.show_history_check.setChecked(True)
        display_layout.addWidget(self.show_history_check)

        self.show_vectors_check = QCheckBox("Show Velocity Vectors")
        self.show_vectors_check.setChecked(True)
        display_layout.addWidget(self.show_vectors_check)

        # Range setting
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Max Range:"))
        self.max_range_spin = QSpinBox()
        self.max_range_spin.setRange(10, 500)
        self.max_range_spin.setValue(240)
        self.max_range_spin.setSuffix(" nmi")
        range_layout.addWidget(self.max_range_spin)
        display_layout.addLayout(range_layout)

        layout.addWidget(display_group)

        # Track count indicator
        self.track_count_label = QLabel("Active Tracks: 0")
        self.track_count_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        layout.addWidget(self.track_count_label)

        layout.addStretch()

    def _connect_signals(self):
        self.start_btn.clicked.connect(self.start_simulation)
        self.stop_btn.clicked.connect(self.stop_simulation)
        self.clear_btn.clicked.connect(self.clear_tracks)

        self.update_rate_spin.valueChanged.connect(self._on_update_rate_changed)
        self.time_scale_spin.valueChanged.connect(self._on_time_scale_changed)

        self.populate_btn.clicked.connect(self._on_populate_clicked)
        self.add_target_btn.clicked.connect(self._on_add_target_clicked)

        self.category_combo.currentIndexChanged.connect(self._update_target_combo)

        self.max_range_spin.valueChanged.connect(self._on_max_range_changed)

    def _update_target_combo(self):
        """Update target combo based on selected category."""
        self.target_combo.clear()
        category = self.category_combo.currentData()
        if category:
            targets = self._simulator.library.get_targets_by_category(category)
            for target in targets:
                self.target_combo.addItem(
                    f"{target.name} ({target.rcs_m2:.2f} m²)",
                    target
                )

    def _on_update_rate_changed(self, value):
        self._update_rate_ms = int(value * 1000)
        if self._timer.isActive():
            self._timer.setInterval(self._update_rate_ms)

    def _on_time_scale_changed(self, value):
        self._time_scale = value

    def _on_max_range_changed(self, value):
        self._simulator.max_range_m = value * 1852

    def _on_populate_clicked(self):
        num_aircraft = self.num_aircraft_spin.value()
        num_vessels = self.num_vessels_spin.value()
        self._simulator.populate_random_tracks(num_aircraft, num_vessels)
        self._update_track_count()
        self.tracks_changed.emit()

    def _on_add_target_clicked(self):
        profile = self.target_combo.currentData()
        if profile:
            self._simulator.create_track(profile=profile)
            self._update_track_count()
            self.tracks_changed.emit()

    def _on_timer_tick(self):
        # Calculate simulation time step
        dt_seconds = (self._update_rate_ms / 1000.0) * self._time_scale
        self._simulator.update(dt_seconds)
        self._update_track_count()
        self.simulation_updated.emit()

    def _update_track_count(self):
        count = len(self._simulator.tracks)
        self.track_count_label.setText(f"Active Tracks: {count}")

    def start_simulation(self):
        self._timer.start(self._update_rate_ms)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.simulation_started.emit()

    def stop_simulation(self):
        self._timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.simulation_stopped.emit()

    def clear_tracks(self):
        self._simulator.clear_all_tracks()
        self._update_track_count()
        self.tracks_changed.emit()

    def get_simulator(self) -> TrackSimulator:
        """Get the track simulator instance."""
        return self._simulator

    def get_track_data(self) -> list:
        """Get current track data for display."""
        return self._simulator.get_track_data()

    def get_show_labels(self) -> bool:
        return self.show_labels_check.isChecked()

    def get_show_history(self) -> bool:
        return self.show_history_check.isChecked()

    def get_show_vectors(self) -> bool:
        return self.show_vectors_check.isChecked()

    def get_max_range(self) -> int:
        return self.max_range_spin.value()


class TrackTablePanel(QWidget):
    """
    Table showing details of all active tracks.
    """

    track_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Active Tracks")
        group_layout = QVBoxLayout(group)

        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "ID", "Type", "Category", "Range\n(nmi)", "Az\n(°)",
            "Speed\n(kts)", "Alt\n(ft)", "RCS\n(dBsm)"
        ])

        # Configure table
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)

        self.table.itemSelectionChanged.connect(self._on_selection_changed)

        group_layout.addWidget(self.table)
        layout.addWidget(group)

    def update_tracks(self, tracks_data: list):
        """Update table with track data."""
        self.table.setRowCount(len(tracks_data))

        for row, track in enumerate(tracks_data):
            self.table.setItem(row, 0, QTableWidgetItem(str(track['id'])))
            self.table.setItem(row, 1, QTableWidgetItem(track['name']))
            self.table.setItem(row, 2, QTableWidgetItem(track['category']))
            self.table.setItem(row, 3, QTableWidgetItem(f"{track['range_nmi']:.1f}"))
            self.table.setItem(row, 4, QTableWidgetItem(f"{track['azimuth_deg']:.1f}"))
            self.table.setItem(row, 5, QTableWidgetItem(f"{track['speed_kts']:.0f}"))
            self.table.setItem(row, 6, QTableWidgetItem(f"{track['altitude_ft']:.0f}"))
            self.table.setItem(row, 7, QTableWidgetItem(f"{track['rcs_dbsm']:.1f}"))

            # Color code by category
            color = self._get_category_color(track['category'])
            for col in range(8):
                item = self.table.item(row, col)
                if item:
                    item.setBackground(QColor(color).darker(300))

    def _get_category_color(self, category_str: str) -> str:
        """Get color for category."""
        category_colors = {
            "Commercial Aircraft": '#00BFFF',
            "General Aviation": '#87CEEB',
            "Military Fighter": '#FF6347',
            "Military Stealth": '#9400D3',
            "Military Heavy": '#FFA500',
            "Military Helicopter": '#FFD700',
            "Civilian Helicopter": '#98FB98',
            "UAV/Drone": '#FF69B4',
            "Cargo Ship": '#4682B4',
            "Tanker": '#8B4513',
            "Container Ship": '#2E8B57',
            "Cruise Ship": '#FFFFFF',
            "Naval Combatant": '#DC143C',
            "Naval Carrier": '#FF0000',
            "Patrol Boat": '#00CED1',
            "Fishing Vessel": '#808080',
            "Yacht": '#E6E6FA',
            "Small Boat": '#A9A9A9',
        }
        return category_colors.get(category_str, '#FFFFFF')

    def _on_selection_changed(self):
        """Handle row selection."""
        selected = self.table.selectedItems()
        if selected:
            row = selected[0].row()
            track_id_item = self.table.item(row, 0)
            if track_id_item:
                track_id = int(track_id_item.text())
                self.track_selected.emit(track_id)
