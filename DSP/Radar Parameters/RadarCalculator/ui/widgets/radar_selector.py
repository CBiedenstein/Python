"""
Radar profile selector widget.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
    QPushButton, QLabel, QGroupBox, QDialog,
    QDialogButtonBox, QLineEdit, QTextEdit, QFormLayout,
    QMessageBox, QFileDialog
)
from PyQt6.QtCore import pyqtSignal
from profiles import ProfileManager, RadarProfile


class RadarSelector(QWidget):
    """
    Widget for selecting and managing radar profiles.
    """

    profile_changed = pyqtSignal(object)  # Emits RadarProfile

    def __init__(self, parent=None):
        super().__init__(parent)
        self._profile_manager = ProfileManager()
        self._current_profile = None
        self._setup_ui()
        self._load_profiles()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Radar System")
        group_layout = QVBoxLayout(group)

        # Profile selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Profile:"))
        self.profile_combo = QComboBox()
        self.profile_combo.setMinimumWidth(150)
        self.profile_combo.currentIndexChanged.connect(self._on_profile_selected)
        selector_layout.addWidget(self.profile_combo, 1)
        group_layout.addLayout(selector_layout)

        # Description label
        self.description_label = QLabel("")
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet("color: #888; font-style: italic;")
        group_layout.addWidget(self.description_label)

        # Buttons
        btn_layout = QHBoxLayout()

        self.save_btn = QPushButton("Save As...")
        self.save_btn.setObjectName("secondaryButton")
        self.save_btn.clicked.connect(self._save_profile_dialog)
        btn_layout.addWidget(self.save_btn)

        self.load_btn = QPushButton("Load...")
        self.load_btn.setObjectName("secondaryButton")
        self.load_btn.clicked.connect(self._load_profile_dialog)
        btn_layout.addWidget(self.load_btn)

        group_layout.addLayout(btn_layout)

        layout.addWidget(group)

    def _load_profiles(self):
        """Load profiles into the combo box."""
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()

        # Add profiles grouped by category
        profiles = self._profile_manager.get_all_profiles()
        categories = self._profile_manager.get_all_categories()

        for category in sorted(categories):
            # Add category header (disabled item)
            self.profile_combo.addItem(f"--- {category} ---")
            idx = self.profile_combo.count() - 1
            self.profile_combo.model().item(idx).setEnabled(False)

            # Add profiles in this category
            for name, profile in sorted(profiles.items()):
                if profile.category == category:
                    self.profile_combo.addItem(name)

        self.profile_combo.blockSignals(False)

        # Select first valid item
        for i in range(self.profile_combo.count()):
            if self.profile_combo.model().item(i).isEnabled():
                self.profile_combo.setCurrentIndex(i)
                break

    def _on_profile_selected(self, index):
        """Handle profile selection change."""
        name = self.profile_combo.currentText()
        if name.startswith("---"):
            return

        profile = self._profile_manager.get_profile(name)
        if profile:
            self._current_profile = profile
            self.description_label.setText(profile.description)
            self.profile_changed.emit(profile)

    def _save_profile_dialog(self):
        """Show dialog to save current settings as a new profile."""
        dialog = SaveProfileDialog(self._current_profile, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_profile = dialog.get_profile()
            self._profile_manager.add_profile(new_profile, save=True)
            self._load_profiles()

            # Select the new profile
            idx = self.profile_combo.findText(new_profile.name)
            if idx >= 0:
                self.profile_combo.setCurrentIndex(idx)

    def _load_profile_dialog(self):
        """Show dialog to load a profile from file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Radar Profile",
            "",
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if filename:
            try:
                profile = RadarProfile.from_yaml_file(filename)
                self._profile_manager.add_profile(profile, save=True)
                self._load_profiles()

                # Select the loaded profile
                idx = self.profile_combo.findText(profile.name)
                if idx >= 0:
                    self.profile_combo.setCurrentIndex(idx)
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Load Error",
                    f"Could not load profile: {e}"
                )

    def get_current_profile(self) -> RadarProfile:
        """Get the currently selected profile."""
        return self._current_profile

    def refresh_profiles(self):
        """Refresh the profile list from disk."""
        self._profile_manager.reload_profiles()
        self._load_profiles()


class SaveProfileDialog(QDialog):
    """Dialog for saving a radar profile."""

    def __init__(self, base_profile: RadarProfile = None, parent=None):
        super().__init__(parent)
        self._base_profile = base_profile
        self.setWindowTitle("Save Radar Profile")
        self.setMinimumWidth(400)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        form = QFormLayout()

        self.name_edit = QLineEdit()
        if self._base_profile:
            self.name_edit.setText(f"{self._base_profile.name} (Copy)")
        form.addRow("Name:", self.name_edit)

        self.category_edit = QLineEdit()
        self.category_edit.setText("Custom")
        form.addRow("Category:", self.category_edit)

        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(80)
        if self._base_profile:
            self.description_edit.setText(self._base_profile.description)
        form.addRow("Description:", self.description_edit)

        layout.addLayout(form)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_profile(self) -> RadarProfile:
        """Get the profile with user-entered metadata."""
        if self._base_profile:
            data = self._base_profile.to_dict()
        else:
            data = {}

        data['name'] = self.name_edit.text()
        data['category'] = self.category_edit.text()
        data['description'] = self.description_edit.toPlainText()

        return RadarProfile.from_dict(data)
