# Radar Parameter Calculator - Implementation Plan

## Overview
A holistic radar parameter calculator application that consolidates existing radar calculation scripts into a unified, distributable desktop application with PyQt6.

## User Requirements Summary
- **Distribution**: PyQt6 desktop app (easy to share with others)
- **Radar Profiles**: TPS-43, TPS-70, TPS-75 predefined + custom profile save/load
- **Primary Output**: SNR vs Range curve (most important)
- **Plot Toggling**: Ability to switch between different output visualizations
- **Unit Toggling**: Support switching between unit systems (nmi/km, GHz/MHz, W/kW/MW, etc.)

---

## Project Structure

```
RadarCalculator/
├── __init__.py
├── main.py                      # Application entry point
├── requirements.txt             # Dependencies
│
├── core/                        # Calculation engine (UI-independent)
│   ├── __init__.py
│   ├── constants.py             # Physical constants (c, k, T0)
│   ├── units.py                 # Unit conversion utilities
│   ├── radar_equations.py       # Consolidated radar calculations
│   ├── mie_scattering.py        # RCS coefficient (Mie theory)
│   └── targets.py               # Predefined target RCS library
│
├── profiles/                    # Radar system profiles
│   ├── __init__.py
│   ├── profile_manager.py       # Load/save/manage profiles
│   ├── radar_profile.py         # RadarProfile dataclass
│   └── defaults/                # Built-in YAML profiles
│       ├── tps43.yaml
│       ├── tps70.yaml
│       └── tps75.yaml
│
├── ui/                          # PyQt6 user interface
│   ├── __init__.py
│   ├── main_window.py           # Main application window
│   ├── styles.py                # Qt stylesheets
│   ├── widgets/
│   │   ├── __init__.py
│   │   ├── radar_selector.py    # Radar profile selector/editor
│   │   ├── parameter_panel.py   # Parameter input with unit toggles
│   │   ├── target_panel.py      # Target RCS selection
│   │   └── results_panel.py     # Calculation results display
│   └── plots/
│       ├── __init__.py
│       ├── plot_manager.py      # Plot switching/management
│       ├── snr_range_plot.py    # SNR vs Range
│       ├── rcs_range_plot.py    # Min detectable RCS vs Range
│       ├── max_range_plot.py    # Max range vs RCS
│       ├── link_budget_plot.py  # Waterfall link budget
│       └── detection_envelope.py # Detection envelope (polar)
│
└── export/                      # Export functionality
    ├── __init__.py
    └── csv_exporter.py          # CSV export
```

---

## Phase 1: Core Calculation Engine

### 1.1 constants.py
- Speed of light (c)
- Boltzmann constant (k)
- Standard temperature (T0)
- Pi

### 1.2 units.py
Unit conversion system supporting:
- **Range**: m, km, nmi, mi
- **Frequency**: Hz, kHz, MHz, GHz
- **Power**: W, kW, MW, dBm, dBW
- **RCS**: m², dBsm
- **Gain**: linear, dBi

### 1.3 radar_equations.py
Consolidated from existing scripts:

```python
class RadarCalculator:
    """Core radar equation calculations."""

    def calculate_wavelength(freq_hz) -> float
    def calculate_path_loss(range_m, wavelength_m) -> float
    def calculate_received_power(params: RadarParams, target: Target) -> float
    def calculate_thermal_noise(bandwidth_hz, temp_k=290) -> float
    def calculate_mds(bandwidth_hz, noise_figure_db) -> float
    def calculate_snr(pr_dbm, mds_dbm) -> float
    def calculate_max_range(params: RadarParams, rcs_m2) -> float
    def calculate_min_rcs(params: RadarParams, range_m) -> float

    # Sweep functions for plotting
    def snr_vs_range(params, rcs_m2, range_array) -> np.array
    def min_rcs_vs_range(params, range_array) -> np.array
    def max_range_vs_rcs(params, rcs_array) -> np.array
```

### 1.4 targets.py
Predefined target library:
- Insect, Bird, Small Drone, Large Drone
- Person, Car, Truck
- Small Fighter (F-16), Large Fighter (Su-27), Stealth (F-22/F-35)
- Commercial Aircraft (B737, B777)
- Ship (Small, Large), Corner Reflector

---

## Phase 2: Profile System

### 2.1 radar_profile.py
```python
@dataclass
class RadarProfile:
    name: str
    description: str
    frequency_hz: float
    peak_power_w: float
    tx_gain_dbi: float
    rx_gain_dbi: float
    bandwidth_hz: float
    noise_figure_db: float
    system_loss_db: float
    required_snr_db: float
    # Optional TPS-specific
    beam_config: Optional[dict] = None
```

### 2.2 Default Profiles
- **TPS-43**: 3.0 GHz, 2.4 MW (upgrade), 36/39 dBi gains, 6-beam config
- **TPS-70**: Parameters from technical specs
- **TPS-75**: Parameters from technical specs

### 2.3 profile_manager.py
- Load YAML profiles from defaults/ directory
- Load/save custom profiles to user directory
- Profile validation

---

## Phase 3: PyQt6 User Interface

### 3.1 Main Window Layout
```
+------------------------------------------------------------------+
| [Icon] Radar Parameter Calculator              [Theme] [Settings] |
+------------------------------------------------------------------+
| Radar: [TPS-43 ▼] [Edit] [Save As...]                            |
+------------------------------------------------------------------+
|  LEFT PANEL (300px)    |  CENTER (Plots)        | RIGHT (Results)|
|                        |                        |                |
| ┌─ Parameters ───────┐ | ┌────────────────────┐ | ┌─ Results ──┐ |
| │ Frequency    [GHz] │ | │                    │ | │            │ |
| │ [3.0      ] [▼]    │ | │   SNR vs Range     │ | │ Pr: -85 dBm│ |
| │                    │ | │   (or selected     │ | │ MDS: -105  │ |
| │ Tx Power     [MW]  │ | │    plot)           │ | │ SNR: 20 dB │ |
| │ [2.4      ] [▼]    │ | │                    │ | │ Max: 445km │ |
| │                    │ | │                    │ | │            │ |
| │ Tx Gain     [dBi]  │ | └────────────────────┘ | │ [DETECTED] │ |
| │ [36.0     ]        │ |                        | └────────────┘ |
| │                    │ | Plot: [SNR vs Range ▼] |                |
| │ Rx Gain     [dBi]  │ |                        | ┌─ Budget ───┐ |
| │ [39.0     ]        │ |                        | │ Pt: +93.8  │ |
| │                    │ |                        | │ +Gt: +36   │ |
| │ Bandwidth   [MHz]  │ |                        | │ +Gr: +39   │ |
| │ [1.6      ] [▼]    │ |                        | │ +RCS: +7.8 │ |
| │                    │ |                        | │ -Loss: -5  │ |
| │ Noise Fig   [dB]   │ |                        | │ -Path:-166 │ |
| │ [2.0      ]        │ |                        | │ ────────── │ |
| │                    │ |                        | │ Pr: -85 dBm│ |
| │ Sys Loss    [dB]   │ |                        | └────────────┘ |
| │ [5.0      ]        │ |                        |                |
| │                    │ |                        |                |
| │ Req SNR     [dB]   │ |                        |                |
| │ [13.0     ]        │ |                        |                |
| └────────────────────┘ |                        |                |
|                        |                        |                |
| ┌─ Target ───────────┐ |                        |                |
| │ [B737        ▼]    │ |                        |                |
| │ RCS: [20.0] m²     │ |                        |                |
| │ Range: [240] [nmi▼]│ |                        |                |
| └────────────────────┘ |                        |                |
+------------------------------------------------------------------+
| Range: nmi ○ km  | Power: W ○ kW ○ MW  | Freq: MHz ○ GHz        |
+------------------------------------------------------------------+
```

### 3.2 Plot Selector Options
1. **SNR vs Range** (default, most important)
2. **Min Detectable RCS vs Range**
3. **Max Detection Range vs RCS**
4. **Link Budget Waterfall**
5. **Detection Envelope (Polar)**

### 3.3 Unit Toggle Bar
Bottom toolbar with radio buttons for unit preferences:
- Range: m / km / nmi
- Power: W / kW / MW
- Frequency: MHz / GHz

---

## Phase 4: Implementation Order

### Step 1: Core Engine
1. `core/constants.py`
2. `core/units.py`
3. `core/radar_equations.py`
4. `core/targets.py`
5. Unit tests for all calculations

### Step 2: Profile System
1. `profiles/radar_profile.py`
2. `profiles/defaults/*.yaml` (TPS-43, TPS-70, TPS-75)
3. `profiles/profile_manager.py`

### Step 3: Basic UI
1. `ui/main_window.py` - Window scaffold
2. `ui/widgets/parameter_panel.py` - Input fields
3. `ui/widgets/radar_selector.py` - Profile dropdown
4. `ui/widgets/target_panel.py` - Target selection
5. `ui/widgets/results_panel.py` - Results display

### Step 4: Plots
1. `ui/plots/plot_manager.py` - Plot switching logic
2. `ui/plots/snr_range_plot.py` - Primary plot
3. `ui/plots/rcs_range_plot.py`
4. `ui/plots/max_range_plot.py`
5. `ui/plots/link_budget_plot.py`

### Step 5: Polish
1. Unit toggle implementation
2. Profile save/load dialogs
3. CSV export
4. Styling and themes
5. `main.py` entry point

---

## Dependencies (requirements.txt)

```
PyQt6>=6.4.0
pyqtgraph>=0.13.0
numpy>=1.24.0
scipy>=1.10.0
PyYAML>=6.0
```

---

## Key Design Decisions

1. **Calculation Independence**: Core module has zero UI dependencies - usable standalone or in Jupyter
2. **Real-time Updates**: All results/plots update immediately on parameter change
3. **Unit Flexibility**: Internal calculations use SI base units; UI handles display conversion
4. **Profile Persistence**: YAML format for human-readable, version-controllable configs
5. **Distribution**: Single-folder app, can be packaged with PyInstaller for .exe distribution
