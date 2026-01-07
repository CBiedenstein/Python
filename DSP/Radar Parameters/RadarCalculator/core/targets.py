"""
Predefined target RCS library for common radar targets.
RCS values are typical/representative and vary significantly in practice.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TargetProfile:
    """Represents a radar target with its RCS characteristics."""
    name: str
    rcs_m2: float
    category: str
    description: Optional[str] = None


# Predefined target library with typical RCS values
TARGET_LIBRARY = {
    # Natural/Environmental
    "Insect": TargetProfile("Insect", 1e-5, "Natural", "Small flying insect"),
    "Bird (Small)": TargetProfile("Bird (Small)", 1e-3, "Natural", "Sparrow-sized bird"),
    "Bird (Large)": TargetProfile("Bird (Large)", 1e-2, "Natural", "Goose/eagle-sized bird"),

    # Drones/UAV
    "Drone (Micro)": TargetProfile("Drone (Micro)", 1e-3, "UAV", "Micro UAV, <1kg"),
    "Drone (Small)": TargetProfile("Drone (Small)", 1e-2, "UAV", "Small commercial drone"),
    "Drone (Medium)": TargetProfile("Drone (Medium)", 0.1, "UAV", "Medium tactical UAV"),
    "Drone (Large)": TargetProfile("Drone (Large)", 1.0, "UAV", "Large UAV (Predator-class)"),

    # Ground targets
    "Person": TargetProfile("Person", 0.5, "Ground", "Standing human"),
    "Car": TargetProfile("Car", 5.0, "Ground", "Passenger vehicle"),
    "Truck": TargetProfile("Truck", 20.0, "Ground", "Large truck/semi"),
    "Tank": TargetProfile("Tank", 30.0, "Ground", "Main battle tank"),

    # Military aircraft
    "Missile (Cruise)": TargetProfile("Missile (Cruise)", 0.5, "Missile", "Cruise missile"),
    "Missile (Ballistic)": TargetProfile("Missile (Ballistic)", 1.0, "Missile", "Ballistic missile RV"),
    "Fighter (Small)": TargetProfile("Fighter (Small)", 1.2, "Military Aircraft", "F-16 class"),
    "Fighter (Large)": TargetProfile("Fighter (Large)", 6.0, "Military Aircraft", "F-15/Su-27 class"),
    "Fighter (Stealth)": TargetProfile("Fighter (Stealth)", 1e-4, "Military Aircraft", "F-22/F-35 class"),
    "Bomber": TargetProfile("Bomber", 10.0, "Military Aircraft", "B-1B class"),
    "Bomber (Stealth)": TargetProfile("Bomber (Stealth)", 1e-2, "Military Aircraft", "B-2 class"),
    "Transport (Military)": TargetProfile("Transport (Military)", 50.0, "Military Aircraft", "C-130 class"),

    # Commercial aircraft
    "General Aviation": TargetProfile("General Aviation", 2.0, "Commercial Aircraft", "Cessna class"),
    "Regional Jet": TargetProfile("Regional Jet", 10.0, "Commercial Aircraft", "CRJ/ERJ class"),
    "Narrow Body": TargetProfile("Narrow Body", 20.0, "Commercial Aircraft", "B737/A320 class"),
    "Wide Body": TargetProfile("Wide Body", 40.0, "Commercial Aircraft", "B777/A350 class"),
    "Jumbo": TargetProfile("Jumbo", 100.0, "Commercial Aircraft", "B747/A380 class"),

    # Maritime
    "Small Boat": TargetProfile("Small Boat", 5.0, "Maritime", "Fishing boat/speedboat"),
    "Yacht": TargetProfile("Yacht", 20.0, "Maritime", "Medium yacht"),
    "Cargo Ship": TargetProfile("Cargo Ship", 1000.0, "Maritime", "Container ship"),
    "Warship": TargetProfile("Warship", 10000.0, "Maritime", "Destroyer/cruiser"),
    "Aircraft Carrier": TargetProfile("Aircraft Carrier", 50000.0, "Maritime", "Full-size carrier"),

    # Calibration
    "Corner Reflector (Small)": TargetProfile("Corner Reflector (Small)", 10.0, "Calibration", "0.3m trihedral"),
    "Corner Reflector (Large)": TargetProfile("Corner Reflector (Large)", 1000.0, "Calibration", "1m trihedral"),
    "Calibration Sphere": TargetProfile("Calibration Sphere", 1.0, "Calibration", "Known RCS sphere"),
}


def get_target_rcs(name: str) -> float:
    """Get RCS value for a named target."""
    if name in TARGET_LIBRARY:
        return TARGET_LIBRARY[name].rcs_m2
    raise ValueError(f"Unknown target: {name}")


def get_target_profile(name: str) -> TargetProfile:
    """Get full target profile for a named target."""
    if name in TARGET_LIBRARY:
        return TARGET_LIBRARY[name]
    raise ValueError(f"Unknown target: {name}")


def get_targets_by_category(category: str) -> dict:
    """Get all targets in a specific category."""
    return {k: v for k, v in TARGET_LIBRARY.items() if v.category == category}


def get_all_categories() -> list:
    """Get list of all target categories."""
    return list(set(t.category for t in TARGET_LIBRARY.values()))


def get_rcs_context(rcs_dbsm: float) -> str:
    """Returns a string description of what typically has this RCS."""
    if rcs_dbsm < -30:
        return "Insect"
    if rcs_dbsm < -20:
        return "Bird / Stealth Aircraft"
    if rcs_dbsm < -10:
        return "Small Drone / Missile"
    if rcs_dbsm < 0:
        return "Person / Cruise Missile"
    if rcs_dbsm < 10:
        return "Small Fighter / Car"
    if rcs_dbsm < 20:
        return "Large Fighter / Truck"
    if rcs_dbsm < 30:
        return "Commercial Aircraft"
    if rcs_dbsm < 40:
        return "Corner Reflector / Ship"
    return "Large Ship / Building"
