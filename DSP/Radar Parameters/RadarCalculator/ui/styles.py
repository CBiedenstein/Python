"""
Qt stylesheets for the Radar Calculator application.
"""

# Brighter dark theme with more vibrant colors
DARK_THEME = """
QMainWindow {
    background-color: #1a1b26;
}

QWidget {
    background-color: #1a1b26;
    color: #c0caf5;
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-size: 10pt;
}

QGroupBox {
    background-color: #24283b;
    border: 1px solid #414868;
    border-radius: 8px;
    margin-top: 12px;
    padding: 10px;
    padding-top: 20px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    background-color: #24283b;
    color: #7aa2f7;
    font-weight: bold;
}

QLabel {
    color: #c0caf5;
    background-color: transparent;
}

QLineEdit {
    background-color: #414868;
    border: 1px solid #565f89;
    border-radius: 4px;
    padding: 5px 8px;
    color: #c0caf5;
    selection-background-color: #7aa2f7;
}

QLineEdit:focus {
    border: 2px solid #7aa2f7;
}

/* Modern SpinBox styling */
QSpinBox, QDoubleSpinBox {
    background-color: #414868;
    border: 1px solid #565f89;
    border-radius: 4px;
    padding: 4px 6px;
    padding-right: 24px;
    color: #c0caf5;
    selection-background-color: #7aa2f7;
    min-height: 20px;
}

QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #7aa2f7;
}

QSpinBox:hover, QDoubleSpinBox:hover {
    border: 1px solid #7aa2f7;
}

QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 20px;
    border: none;
    border-left: 1px solid #565f89;
    border-top-right-radius: 3px;
    background-color: #4a5178;
    margin: 1px;
    margin-bottom: 0px;
}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
    background-color: #7aa2f7;
}

QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed {
    background-color: #89b4fa;
}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid #c0caf5;
}

QSpinBox::up-arrow:hover, QDoubleSpinBox::up-arrow:hover {
    border-bottom-color: #1a1b26;
}

QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 20px;
    border: none;
    border-left: 1px solid #565f89;
    border-bottom-right-radius: 3px;
    background-color: #4a5178;
    margin: 1px;
    margin-top: 0px;
}

QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #7aa2f7;
}

QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
    background-color: #89b4fa;
}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #c0caf5;
}

QSpinBox::down-arrow:hover, QDoubleSpinBox::down-arrow:hover {
    border-top-color: #1a1b26;
}

/* Modern ComboBox styling */
QComboBox {
    background-color: #414868;
    border: 1px solid #565f89;
    border-radius: 4px;
    padding: 5px 8px;
    padding-right: 28px;
    color: #c0caf5;
    min-width: 80px;
    min-height: 20px;
}

QComboBox:hover {
    border: 1px solid #7aa2f7;
}

QComboBox:focus {
    border: 1px solid #7aa2f7;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 24px;
    border: none;
    border-left: 1px solid #565f89;
    border-top-right-radius: 3px;
    border-bottom-right-radius: 3px;
    background-color: #4a5178;
    margin: 2px;
}

QComboBox::drop-down:hover {
    background-color: #7aa2f7;
}

QComboBox::down-arrow {
    width: 0;
    height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #c0caf5;
}

QComboBox::down-arrow:hover {
    border-top-color: #1a1b26;
}

QComboBox QAbstractItemView {
    background-color: #24283b;
    border: 1px solid #565f89;
    border-radius: 4px;
    padding: 4px;
    selection-background-color: #7aa2f7;
    selection-color: #1a1b26;
    outline: none;
}

QComboBox QAbstractItemView::item {
    padding: 6px 8px;
    border-radius: 2px;
    min-height: 20px;
}

QComboBox QAbstractItemView::item:hover {
    background-color: #414868;
}

QPushButton {
    background-color: #7aa2f7;
    color: #1a1b26;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
    min-height: 16px;
}

QPushButton:hover {
    background-color: #89b4fa;
}

QPushButton:pressed {
    background-color: #7dcfff;
}

QPushButton:disabled {
    background-color: #565f89;
    color: #414868;
}

QPushButton#secondaryButton {
    background-color: #414868;
    color: #c0caf5;
}

QPushButton#secondaryButton:hover {
    background-color: #565f89;
}

QPushButton#dangerButton {
    background-color: #f38ba8;
    color: #1a1b26;
}

QPushButton#dangerButton:hover {
    background-color: #f5a3b7;
}

QPushButton#primaryButton {
    background-color: #a6e3a1;
    color: #1a1b26;
}

QPushButton#primaryButton:hover {
    background-color: #b4e8b0;
}

QCheckBox {
    color: #c0caf5;
    spacing: 5px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 3px;
    border: 2px solid #565f89;
    background-color: #414868;
}

QCheckBox::indicator:checked {
    background-color: #7aa2f7;
    border-color: #7aa2f7;
}

QCheckBox::indicator:hover {
    border-color: #7aa2f7;
}

QTableWidget {
    background-color: #1a1b26;
    gridline-color: #414868;
    color: #c0caf5;
    border: 1px solid #414868;
    border-radius: 4px;
}

QTableWidget::item {
    padding: 4px;
}

QTableWidget::item:selected {
    background-color: #7aa2f7;
    color: #1a1b26;
}

QHeaderView::section {
    background-color: #24283b;
    color: #7aa2f7;
    padding: 6px;
    border: none;
    border-bottom: 1px solid #414868;
    font-weight: bold;
}

QRadioButton {
    color: #c0caf5;
    spacing: 5px;
}

QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border-radius: 8px;
    border: 2px solid #565f89;
    background-color: #414868;
}

QRadioButton::indicator:checked {
    background-color: #7aa2f7;
    border-color: #7aa2f7;
}

QScrollArea {
    border: none;
    background-color: transparent;
}

QScrollBar:vertical {
    background-color: #24283b;
    width: 10px;
    border-radius: 5px;
    margin: 2px;
}

QScrollBar::handle:vertical {
    background-color: #565f89;
    border-radius: 5px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #7aa2f7;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}

QScrollBar:horizontal {
    background-color: #24283b;
    height: 10px;
    border-radius: 5px;
    margin: 2px;
}

QScrollBar::handle:horizontal {
    background-color: #565f89;
    border-radius: 5px;
    min-width: 30px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #7aa2f7;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: none;
}

QStatusBar {
    background-color: #24283b;
    color: #a9b1d6;
    border-top: 1px solid #414868;
}

QMenuBar {
    background-color: #24283b;
    color: #c0caf5;
    border-bottom: 1px solid #414868;
}

QMenuBar::item:selected {
    background-color: #414868;
}

QMenu {
    background-color: #24283b;
    border: 1px solid #414868;
    border-radius: 4px;
    padding: 4px;
}

QMenu::item {
    padding: 6px 24px;
    border-radius: 2px;
}

QMenu::item:selected {
    background-color: #7aa2f7;
    color: #1a1b26;
}

QSplitter::handle {
    background-color: #414868;
}

QSplitter::handle:horizontal {
    width: 4px;
}

QSplitter::handle:vertical {
    height: 4px;
}

QSplitter::handle:hover {
    background-color: #7aa2f7;
}

QTabWidget::pane {
    border: 1px solid #414868;
    border-radius: 4px;
    background-color: #1a1b26;
    top: -1px;
}

QTabBar::tab {
    background-color: #24283b;
    color: #a9b1d6;
    padding: 10px 20px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
    border: 1px solid #414868;
    border-bottom: none;
}

QTabBar::tab:selected {
    background-color: #1a1b26;
    color: #7aa2f7;
    border-bottom: 1px solid #1a1b26;
}

QTabBar::tab:hover:!selected {
    background-color: #414868;
    color: #c0caf5;
}

QSlider::groove:horizontal {
    border: none;
    height: 6px;
    background-color: #414868;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background-color: #7aa2f7;
    border: none;
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background-color: #89b4fa;
}

QToolTip {
    background-color: #24283b;
    color: #c0caf5;
    border: 1px solid #565f89;
    border-radius: 4px;
    padding: 4px 8px;
}
"""


LIGHT_THEME = """
QMainWindow {
    background-color: #eff1f5;
}

QWidget {
    background-color: #eff1f5;
    color: #4c4f69;
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-size: 10pt;
}

QGroupBox {
    background-color: #ffffff;
    border: 1px solid #ccd0da;
    border-radius: 8px;
    margin-top: 12px;
    padding: 10px;
    padding-top: 20px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    background-color: #ffffff;
    color: #1e66f5;
    font-weight: bold;
}

QLabel {
    color: #4c4f69;
    background-color: transparent;
}

QLineEdit {
    background-color: #ffffff;
    border: 1px solid #ccd0da;
    border-radius: 4px;
    padding: 5px 8px;
    color: #4c4f69;
    selection-background-color: #1e66f5;
}

QLineEdit:focus {
    border: 1px solid #1e66f5;
}

/* Modern SpinBox styling */
QSpinBox, QDoubleSpinBox {
    background-color: #ffffff;
    border: 1px solid #ccd0da;
    border-radius: 4px;
    padding: 4px 6px;
    padding-right: 24px;
    color: #4c4f69;
    selection-background-color: #1e66f5;
    min-height: 20px;
}

QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #1e66f5;
}

QSpinBox:hover, QDoubleSpinBox:hover {
    border: 1px solid #1e66f5;
}

QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 20px;
    border: none;
    border-left: 1px solid #ccd0da;
    border-top-right-radius: 3px;
    background-color: #e6e9ef;
    margin: 1px;
    margin-bottom: 0px;
}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
    background-color: #1e66f5;
}

QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed {
    background-color: #2a6df0;
}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid #4c4f69;
}

QSpinBox::up-arrow:hover, QDoubleSpinBox::up-arrow:hover {
    border-bottom-color: #ffffff;
}

QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 20px;
    border: none;
    border-left: 1px solid #ccd0da;
    border-bottom-right-radius: 3px;
    background-color: #e6e9ef;
    margin: 1px;
    margin-top: 0px;
}

QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #1e66f5;
}

QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
    background-color: #2a6df0;
}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #4c4f69;
}

QSpinBox::down-arrow:hover, QDoubleSpinBox::down-arrow:hover {
    border-top-color: #ffffff;
}

/* Modern ComboBox styling */
QComboBox {
    background-color: #ffffff;
    border: 1px solid #ccd0da;
    border-radius: 4px;
    padding: 5px 8px;
    padding-right: 28px;
    color: #4c4f69;
    min-width: 80px;
    min-height: 20px;
}

QComboBox:hover {
    border: 1px solid #1e66f5;
}

QComboBox:focus {
    border: 1px solid #1e66f5;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 24px;
    border: none;
    border-left: 1px solid #ccd0da;
    border-top-right-radius: 3px;
    border-bottom-right-radius: 3px;
    background-color: #e6e9ef;
    margin: 2px;
}

QComboBox::drop-down:hover {
    background-color: #1e66f5;
}

QComboBox::down-arrow {
    width: 0;
    height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #4c4f69;
}

QComboBox::down-arrow:hover {
    border-top-color: #ffffff;
}

QComboBox QAbstractItemView {
    background-color: #ffffff;
    border: 1px solid #ccd0da;
    border-radius: 4px;
    padding: 4px;
    selection-background-color: #1e66f5;
    selection-color: #ffffff;
    outline: none;
}

QComboBox QAbstractItemView::item {
    padding: 6px 8px;
    border-radius: 2px;
    min-height: 20px;
}

QComboBox QAbstractItemView::item:hover {
    background-color: #e6e9ef;
}

QPushButton {
    background-color: #1e66f5;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
    min-height: 16px;
}

QPushButton:hover {
    background-color: #2a6df0;
}

QPushButton:pressed {
    background-color: #1959d5;
}

QPushButton:disabled {
    background-color: #ccd0da;
    color: #9ca0b0;
}

QPushButton#secondaryButton {
    background-color: #e6e9ef;
    color: #4c4f69;
}

QPushButton#secondaryButton:hover {
    background-color: #dce0e8;
}

QPushButton#dangerButton {
    background-color: #d20f39;
    color: #ffffff;
}

QPushButton#dangerButton:hover {
    background-color: #e51f49;
}

QPushButton#primaryButton {
    background-color: #40a02b;
    color: #ffffff;
}

QPushButton#primaryButton:hover {
    background-color: #4ab033;
}

QCheckBox {
    color: #4c4f69;
    spacing: 5px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 3px;
    border: 2px solid #ccd0da;
    background-color: #ffffff;
}

QCheckBox::indicator:checked {
    background-color: #1e66f5;
    border-color: #1e66f5;
}

QCheckBox::indicator:hover {
    border-color: #1e66f5;
}

QTableWidget {
    background-color: #ffffff;
    gridline-color: #ccd0da;
    color: #4c4f69;
    border: 1px solid #ccd0da;
    border-radius: 4px;
}

QTableWidget::item {
    padding: 4px;
}

QTableWidget::item:selected {
    background-color: #1e66f5;
    color: #ffffff;
}

QHeaderView::section {
    background-color: #e6e9ef;
    color: #1e66f5;
    padding: 6px;
    border: none;
    border-bottom: 1px solid #ccd0da;
    font-weight: bold;
}

QRadioButton {
    color: #4c4f69;
    spacing: 5px;
}

QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border-radius: 8px;
    border: 2px solid #ccd0da;
    background-color: #ffffff;
}

QRadioButton::indicator:checked {
    background-color: #1e66f5;
    border-color: #1e66f5;
}

QScrollArea {
    border: none;
    background-color: transparent;
}

QScrollBar:vertical {
    background-color: #e6e9ef;
    width: 10px;
    border-radius: 5px;
    margin: 2px;
}

QScrollBar::handle:vertical {
    background-color: #ccd0da;
    border-radius: 5px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #bcc0cc;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}

QScrollBar:horizontal {
    background-color: #e6e9ef;
    height: 10px;
    border-radius: 5px;
    margin: 2px;
}

QScrollBar::handle:horizontal {
    background-color: #ccd0da;
    border-radius: 5px;
    min-width: 30px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #bcc0cc;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: none;
}

QStatusBar {
    background-color: #e6e9ef;
    color: #5c5f77;
    border-top: 1px solid #ccd0da;
}

QMenuBar {
    background-color: #e6e9ef;
    color: #4c4f69;
    border-bottom: 1px solid #ccd0da;
}

QMenuBar::item:selected {
    background-color: #dce0e8;
}

QMenu {
    background-color: #ffffff;
    border: 1px solid #ccd0da;
    border-radius: 4px;
    padding: 4px;
}

QMenu::item {
    padding: 6px 24px;
    border-radius: 2px;
}

QMenu::item:selected {
    background-color: #1e66f5;
    color: #ffffff;
}

QSplitter::handle {
    background-color: #ccd0da;
}

QSplitter::handle:horizontal {
    width: 4px;
}

QSplitter::handle:vertical {
    height: 4px;
}

QSplitter::handle:hover {
    background-color: #1e66f5;
}

QTabWidget::pane {
    border: 1px solid #ccd0da;
    border-radius: 4px;
    background-color: #eff1f5;
    top: -1px;
}

QTabBar::tab {
    background-color: #e6e9ef;
    color: #5c5f77;
    padding: 10px 20px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
    border: 1px solid #ccd0da;
    border-bottom: none;
}

QTabBar::tab:selected {
    background-color: #eff1f5;
    color: #1e66f5;
    border-bottom: 1px solid #eff1f5;
}

QTabBar::tab:hover:!selected {
    background-color: #dce0e8;
    color: #4c4f69;
}

QSlider::groove:horizontal {
    border: none;
    height: 6px;
    background-color: #ccd0da;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background-color: #1e66f5;
    border: none;
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background-color: #2a6df0;
}

QToolTip {
    background-color: #ffffff;
    color: #4c4f69;
    border: 1px solid #ccd0da;
    border-radius: 4px;
    padding: 4px 8px;
}
"""


# Result card specific styles
DETECTED_STYLE = """
    background-color: #a6e3a1;
    color: #1e1e2e;
    border-radius: 4px;
    padding: 4px 8px;
    font-weight: bold;
"""

NOT_DETECTED_STYLE = """
    background-color: #f38ba8;
    color: #1e1e2e;
    border-radius: 4px;
    padding: 4px 8px;
    font-weight: bold;
"""

DETECTED_STYLE_LIGHT = """
    background-color: #40a02b;
    color: #ffffff;
    border-radius: 4px;
    padding: 4px 8px;
    font-weight: bold;
"""

NOT_DETECTED_STYLE_LIGHT = """
    background-color: #d20f39;
    color: #ffffff;
    border-radius: 4px;
    padding: 4px 8px;
    font-weight: bold;
"""
