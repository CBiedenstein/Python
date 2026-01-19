"""
Main application window for the Radar Parameter Calculator.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QScrollArea, QMenuBar, QMenu, QStatusBar,
    QFileDialog, QMessageBox, QApplication, QTabWidget
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QIcon, QScreen

from ui.styles import DARK_THEME, LIGHT_THEME
from ui.widgets.parameter_panel import ParameterPanel
from ui.widgets.radar_selector import RadarSelector
from ui.widgets.target_panel import TargetPanel
from ui.widgets.results_panel import ResultsPanel
from ui.widgets.unit_toggle import UnitToggleBar
from ui.widgets.waveform_panel import WaveformPanel
from ui.plots.plot_manager import PlotManager
from ui.simulation_tab import SimulationTab
from ui.chirp_generator_tab import ChirpGeneratorTab

from core.radar_equations import RadarCalculator, RadarParameters, TargetScenario
from core.waveforms import calculate_waveform_performance
from profiles import RadarProfile


class CalculatorTab(QWidget):
    """
    The calculator tab containing radar parameter calculations and plots.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._calculator = None
        self._splitter_initialized = False
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(0)

        # Main content splitter
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)

        # Left panel (scrollable)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setMinimumWidth(270)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        # Radar selector
        self.radar_selector = RadarSelector()
        left_layout.addWidget(self.radar_selector)

        # Parameter panel
        self.param_panel = ParameterPanel()
        left_layout.addWidget(self.param_panel)

        # Target panel
        self.target_panel = TargetPanel()
        left_layout.addWidget(self.target_panel)

        # Waveform panel
        self.waveform_panel = WaveformPanel()
        left_layout.addWidget(self.waveform_panel)

        left_layout.addStretch()
        left_scroll.setWidget(left_widget)
        self.splitter.addWidget(left_scroll)

        # Center panel (plots) - make it stretch
        self.plot_manager = PlotManager()
        self.plot_manager.setMinimumWidth(400)
        self.splitter.addWidget(self.plot_manager)

        # Right panel (results)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setMinimumWidth(250)

        self.results_panel = ResultsPanel()
        right_scroll.setWidget(self.results_panel)
        self.splitter.addWidget(right_scroll)

        # Set stretch factors (left: 0, center: 1, right: 0) - center stretches
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)

        main_layout.addWidget(self.splitter, 1)

        # Unit toggle bar at bottom
        self.unit_toggle = UnitToggleBar()
        main_layout.addWidget(self.unit_toggle)

    def showEvent(self, event):
        """Handle widget show event to initialize splitter sizes."""
        super().showEvent(event)
        if not self._splitter_initialized:
            # Defer splitter sizing to after the event loop processes the show
            QTimer.singleShot(0, self._initialize_splitter_sizes)

    def _initialize_splitter_sizes(self):
        """Set initial splitter sizes based on widget width."""
        if self._splitter_initialized:
            return
        self._splitter_initialized = True

        # Calculate sizes based on available width
        total_width = self.width()
        if total_width < 100:
            # Widget not properly sized yet, use defaults
            total_width = 1200

        # Left panel: fixed ~300px, Right panel: fixed ~280px, Center: remainder
        left_width = 300
        right_width = 280
        center_width = max(400, total_width - left_width - right_width - 20)

        self.splitter.setSizes([left_width, center_width, right_width])

    def _connect_signals(self):
        """Connect widget signals."""
        # Profile changes
        self.radar_selector.profile_changed.connect(self._on_profile_changed)

        # Parameter changes
        self.param_panel.parameter_changed.connect(self._on_parameters_changed)

        # Target changes
        self.target_panel.target_changed.connect(self._on_target_changed)

        # Waveform changes
        self.waveform_panel.waveform_changed.connect(self._on_waveform_changed)

        # Unit changes
        self.unit_toggle.units_changed.connect(self._on_units_changed)

    def initial_setup(self):
        """Perform initial setup after window is shown."""
        # Get the first profile and set it
        profile = self.radar_selector.get_current_profile()
        if profile:
            self._on_profile_changed(profile)

    def _on_profile_changed(self, profile: RadarProfile):
        """Handle profile selection change."""
        # Update parameter panel with profile values
        self.param_panel.set_from_profile(profile)

        # Create calculator with profile parameters
        self._update_calculator()

    def _on_parameters_changed(self):
        """Handle parameter value changes."""
        self._update_calculator()

    def _on_target_changed(self):
        """Handle target parameter changes."""
        self._update_results()

    def _on_waveform_changed(self):
        """Handle waveform parameter changes."""
        self._update_results()

    def _on_units_changed(self):
        """Handle unit selection changes."""
        self._update_results()
        self.plot_manager.set_range_unit(self.unit_toggle.get_range_unit())

    def _update_calculator(self):
        """Update the radar calculator with current parameters."""
        params = RadarParameters(
            frequency_hz=self.param_panel.get_frequency_hz(),
            peak_power_w=self.param_panel.get_power_watts(),
            tx_gain_dbi=self.param_panel.get_tx_gain_dbi(),
            rx_gain_dbi=self.param_panel.get_rx_gain_dbi(),
            bandwidth_hz=self.param_panel.get_bandwidth_hz(),
            noise_figure_db=self.param_panel.get_noise_figure_db(),
            system_loss_db=self.param_panel.get_system_loss_db(),
            required_snr_db=self.param_panel.get_required_snr_db()
        )

        self._calculator = RadarCalculator(params)
        self.plot_manager.set_calculator(self._calculator)
        self._update_results()

    def _update_results(self):
        """Update results panel and plots."""
        if self._calculator is None:
            return

        # Get target parameters
        rcs_m2 = self.target_panel.get_rcs_m2()
        range_m = self.target_panel.get_range_m()

        # Calculate results
        target = TargetScenario(
            rcs_m2=rcs_m2,
            range_m=range_m,
            name=self.target_panel.get_target_name()
        )

        results = self._calculator.calculate_performance(target)

        # Get processing gain from waveform panel
        processing_gain_db = self.waveform_panel.get_processing_gain_db()

        # Update results panel with processing gain
        self.results_panel.update_results(
            results,
            self._calculator.params,
            self.unit_toggle.get_range_unit(),
            processing_gain_db
        )

        # Update plots with processing gain
        self.plot_manager.set_target(rcs_m2, range_m)
        self.plot_manager.set_processing_gain(processing_gain_db)

    def get_calculator(self):
        return self._calculator

    def get_current_target(self):
        return TargetScenario(
            rcs_m2=self.target_panel.get_rcs_m2(),
            range_m=self.target_panel.get_range_m(),
            name=self.target_panel.get_target_name()
        )


class MainWindow(QMainWindow):
    """
    Main application window for the Radar Parameter Calculator.
    """

    def __init__(self):
        super().__init__()
        self._dark_mode = True
        self._setup_ui()
        self._setup_menu()
        self._apply_theme()

        # Initialize with default profile
        QTimer.singleShot(100, self._initial_setup)

    def _setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("Radar Parameter Calculator")
        self.setMinimumSize(1100, 700)

        # Get screen geometry to size appropriately
        screen = QApplication.primaryScreen()
        if screen:
            screen_geometry = screen.availableGeometry()
            # Use 85% of screen width and 90% of screen height
            width = min(int(screen_geometry.width() * 0.85), 1600)
            height = min(int(screen_geometry.height() * 0.90), 1000)
            self.resize(width, height)
            # Center on screen
            x = (screen_geometry.width() - width) // 2
            y = (screen_geometry.height() - height) // 2
            self.move(x, y)
        else:
            self.resize(1400, 900)

        # Central widget with tabs
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)

        # Calculator tab
        self.calculator_tab = CalculatorTab()
        self.tab_widget.addTab(self.calculator_tab, "Calculator")

        # Simulation tab
        self.simulation_tab = SimulationTab()
        self.tab_widget.addTab(self.simulation_tab, "Track Simulator")

        # Chirp Generator tab
        self.chirp_tab = ChirpGeneratorTab()
        self.tab_widget.addTab(self.chirp_tab, "Chirp Generator")

        main_layout.addWidget(self.tab_widget)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _setup_menu(self):
        """Set up the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        export_csv_action = QAction("Export Calculator Results to CSV...", self)
        export_csv_action.triggered.connect(self._export_csv)
        file_menu.addAction(export_csv_action)

        export_tracks_action = QAction("Export Simulation Tracks to CSV...", self)
        export_tracks_action.triggered.connect(self._export_tracks)
        file_menu.addAction(export_tracks_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        theme_action = QAction("Toggle Dark/Light Theme", self)
        theme_action.triggered.connect(self._toggle_theme)
        view_menu.addAction(theme_action)

        view_menu.addSeparator()

        calc_tab_action = QAction("Calculator Tab", self)
        calc_tab_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(0))
        view_menu.addAction(calc_tab_action)

        sim_tab_action = QAction("Simulation Tab", self)
        sim_tab_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(1))
        view_menu.addAction(sim_tab_action)

        chirp_tab_action = QAction("Chirp Generator Tab", self)
        chirp_tab_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(2))
        view_menu.addAction(chirp_tab_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _initial_setup(self):
        """Perform initial setup after window is shown."""
        self.calculator_tab.initial_setup()
        self.chirp_tab.initial_setup()

    def _toggle_theme(self):
        """Toggle between dark and light themes."""
        self._dark_mode = not self._dark_mode
        self._apply_theme()

    def _apply_theme(self):
        """Apply the current theme."""
        if self._dark_mode:
            self.setStyleSheet(DARK_THEME)
        else:
            self.setStyleSheet(LIGHT_THEME)

    def _export_csv(self):
        """Export current calculator results to CSV."""
        calculator = self.calculator_tab.get_calculator()
        if calculator is None:
            QMessageBox.warning(self, "Export Error", "No calculations to export.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "radar_results.csv",
            "CSV Files (*.csv);;All Files (*)"
        )

        if filename:
            try:
                self._write_calculator_csv(filename)
                self.status_bar.showMessage(f"Exported to {filename}")
            except Exception as e:
                QMessageBox.warning(self, "Export Error", f"Could not export: {e}")

    def _write_calculator_csv(self, filename: str):
        """Write calculator results to CSV file."""
        import csv
        from datetime import datetime

        calculator = self.calculator_tab.get_calculator()
        params = calculator.params
        target = self.calculator_tab.get_current_target()
        results = calculator.calculate_performance(target)

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(['Radar Parameter Calculator Export'])
            writer.writerow(['Timestamp', datetime.now().isoformat()])
            writer.writerow([])

            # Radar parameters
            writer.writerow(['RADAR PARAMETERS'])
            writer.writerow(['Frequency (Hz)', params.frequency_hz])
            writer.writerow(['Peak Power (W)', params.peak_power_w])
            writer.writerow(['Tx Gain (dBi)', params.tx_gain_dbi])
            writer.writerow(['Rx Gain (dBi)', params.rx_gain_dbi])
            writer.writerow(['Bandwidth (Hz)', params.bandwidth_hz])
            writer.writerow(['Noise Figure (dB)', params.noise_figure_db])
            writer.writerow(['System Loss (dB)', params.system_loss_db])
            writer.writerow(['Required SNR (dB)', params.required_snr_db])
            writer.writerow([])

            # Target parameters
            writer.writerow(['TARGET PARAMETERS'])
            writer.writerow(['Target Name', target.name])
            writer.writerow(['RCS (m²)', target.rcs_m2])
            writer.writerow(['Range (m)', target.range_m])
            writer.writerow([])

            # Results
            writer.writerow(['CALCULATION RESULTS'])
            writer.writerow(['Wavelength (m)', results.wavelength_m])
            writer.writerow(['Path Loss (dB)', results.path_loss_db])
            writer.writerow(['Received Power (dBm)', results.received_power_dbm])
            writer.writerow(['Thermal Noise (dBm)', results.thermal_noise_dbm])
            writer.writerow(['MDS (dBm)', results.mds_dbm])
            writer.writerow(['SNR (dB)', results.snr_db])
            writer.writerow(['Detected', results.detected])
            writer.writerow(['Detection Margin (dB)', results.detection_margin_db])
            writer.writerow(['Max Range (m)', results.max_range_m])
            writer.writerow(['Min Detectable RCS (m²)', results.min_detectable_rcs_m2])

    def _export_tracks(self):
        """Export simulation tracks to CSV."""
        tracks_data = self.simulation_tab.control_panel.get_track_data()

        if not tracks_data:
            QMessageBox.warning(self, "Export Error", "No tracks to export.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Tracks",
            "radar_tracks.csv",
            "CSV Files (*.csv);;All Files (*)"
        )

        if filename:
            try:
                self._write_tracks_csv(filename, tracks_data)
                self.status_bar.showMessage(f"Exported {len(tracks_data)} tracks to {filename}")
            except Exception as e:
                QMessageBox.warning(self, "Export Error", f"Could not export: {e}")

    def _write_tracks_csv(self, filename: str, tracks_data: list):
        """Write track data to CSV file."""
        import csv
        from datetime import datetime

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(['Track ID', 'Target Type', 'Category',
                           'Range (nmi)', 'Azimuth (deg)', 'Altitude (ft)',
                           'Speed (kts)', 'Heading (deg)', 'RCS (m²)', 'RCS (dBsm)'])

            # Data rows
            for track in tracks_data:
                writer.writerow([
                    track['id'],
                    track['name'],
                    track['category'],
                    f"{track['range_nmi']:.2f}",
                    f"{track['azimuth_deg']:.1f}",
                    f"{track['altitude_ft']:.0f}",
                    f"{track['speed_kts']:.0f}",
                    f"{track['heading_deg']:.1f}",
                    f"{track['rcs_m2']:.4f}",
                    f"{track['rcs_dbsm']:.1f}"
                ])

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Radar Parameter Calculator",
            """<h2>Radar Parameter Calculator</h2>
            <p>Version 2.1</p>
            <p>A holistic radar performance analysis tool.</p>
            <p><b>Features:</b></p>
            <ul>
                <li>Radar range equation calculations</li>
                <li>SNR, MDS, and detection analysis</li>
                <li>Multiple radar system profiles</li>
                <li>Waveform/pulse compression analysis</li>
                <li>Interactive plots</li>
                <li>Unit conversion support</li>
                <li>Live PPI track simulator</li>
                <li>Aircraft and vessel target library</li>
                <li><b>NEW:</b> FM Chirp waveform generator</li>
                <li>IQ data export (complex64, int16, numpy)</li>
            </ul>
            <p>Built with PyQt6 and pyqtgraph.</p>
            """
        )
