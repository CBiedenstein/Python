"""
Track simulation engine for radar PPI display.

Generates realistic aircraft and vessel tracks with appropriate RCS,
speeds, and behaviors for radar simulation.
"""

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import time


class TargetCategory(Enum):
    """Target category classifications."""
    COMMERCIAL_AIRCRAFT = "Commercial Aircraft"
    GENERAL_AVIATION = "General Aviation"
    MILITARY_FIGHTER = "Military Fighter"
    MILITARY_STEALTH = "Military Stealth"
    MILITARY_HEAVY = "Military Heavy"
    MILITARY_HELICOPTER = "Military Helicopter"
    CIVILIAN_HELICOPTER = "Civilian Helicopter"
    UAV = "UAV/Drone"
    CARGO_SHIP = "Cargo Ship"
    TANKER = "Tanker"
    CONTAINER_SHIP = "Container Ship"
    CRUISE_SHIP = "Cruise Ship"
    NAVAL_COMBATANT = "Naval Combatant"
    NAVAL_CARRIER = "Naval Carrier"
    PATROL_BOAT = "Patrol Boat"
    FISHING_VESSEL = "Fishing Vessel"
    YACHT = "Yacht"
    SMALL_BOAT = "Small Boat"


@dataclass
class TargetProfile:
    """Profile defining a target type's characteristics."""
    name: str
    category: TargetCategory
    rcs_m2: float  # Radar cross section in square meters
    min_speed_kts: float  # Minimum speed in knots
    max_speed_kts: float  # Maximum/cruise speed in knots
    ceiling_ft: float  # Service ceiling in feet (0 for surface vessels)
    min_altitude_ft: float = 0  # Minimum altitude
    description: str = ""

    @property
    def rcs_dbsm(self) -> float:
        """RCS in dBsm."""
        if self.rcs_m2 <= 0:
            return -60.0
        return 10 * math.log10(self.rcs_m2)


class TargetLibrary:
    """
    Database of aircraft and vessel types with typical S-Band RCS values,
    speeds, and operational parameters.
    """

    def __init__(self):
        self.targets: Dict[str, TargetProfile] = {}
        self._load_aircraft()
        self._load_vessels()

    def _load_aircraft(self):
        """Load aircraft target profiles."""
        aircraft = [
            # Commercial Aircraft
            TargetProfile("Boeing 737", TargetCategory.COMMERCIAL_AIRCRAFT,
                         20.0, 250, 460, 41000, 1000, "Narrow-body airliner"),
            TargetProfile("Boeing 777", TargetCategory.COMMERCIAL_AIRCRAFT,
                         40.0, 250, 490, 43100, 1000, "Wide-body airliner"),
            TargetProfile("Boeing 787", TargetCategory.COMMERCIAL_AIRCRAFT,
                         35.0, 250, 488, 43000, 1000, "Wide-body airliner (composite)"),
            TargetProfile("Airbus A320", TargetCategory.COMMERCIAL_AIRCRAFT,
                         18.0, 250, 450, 39000, 1000, "Narrow-body airliner"),
            TargetProfile("Airbus A380", TargetCategory.COMMERCIAL_AIRCRAFT,
                         100.0, 250, 480, 43000, 1000, "Super jumbo airliner"),
            TargetProfile("Embraer E175", TargetCategory.COMMERCIAL_AIRCRAFT,
                         12.0, 200, 410, 41000, 1000, "Regional jet"),

            # General Aviation
            TargetProfile("Cessna 172", TargetCategory.GENERAL_AVIATION,
                         1.5, 60, 120, 14000, 500, "Single-engine trainer"),
            TargetProfile("Cessna Citation", TargetCategory.GENERAL_AVIATION,
                         5.0, 150, 400, 45000, 1000, "Business jet"),
            TargetProfile("Beechcraft King Air", TargetCategory.GENERAL_AVIATION,
                         4.0, 100, 280, 35000, 1000, "Twin turboprop"),
            TargetProfile("Piper Cherokee", TargetCategory.GENERAL_AVIATION,
                         1.2, 60, 130, 14000, 500, "Single-engine trainer"),
            TargetProfile("Gulfstream G650", TargetCategory.GENERAL_AVIATION,
                         8.0, 200, 516, 51000, 1000, "Large business jet"),

            # Military Fighters
            TargetProfile("F-16 Fighting Falcon", TargetCategory.MILITARY_FIGHTER,
                         1.2, 200, 550, 50000, 100, "Light multirole fighter"),
            TargetProfile("F-15 Eagle", TargetCategory.MILITARY_FIGHTER,
                         5.0, 200, 600, 65000, 100, "Air superiority fighter"),
            TargetProfile("F/A-18 Hornet", TargetCategory.MILITARY_FIGHTER,
                         3.0, 200, 580, 50000, 100, "Carrier-based multirole"),
            TargetProfile("Su-27 Flanker", TargetCategory.MILITARY_FIGHTER,
                         15.0, 200, 580, 62000, 100, "Heavy air superiority"),
            TargetProfile("MiG-29 Fulcrum", TargetCategory.MILITARY_FIGHTER,
                         5.0, 200, 560, 59000, 100, "Light multirole fighter"),
            TargetProfile("Eurofighter Typhoon", TargetCategory.MILITARY_FIGHTER,
                         1.0, 200, 585, 65000, 100, "European multirole"),
            TargetProfile("F-14 Tomcat", TargetCategory.MILITARY_FIGHTER,
                         10.0, 200, 580, 53000, 100, "Fleet defense interceptor"),

            # Military Stealth
            TargetProfile("F-22 Raptor", TargetCategory.MILITARY_STEALTH,
                         0.0001, 250, 650, 65000, 100, "5th gen air superiority"),
            TargetProfile("F-35 Lightning II", TargetCategory.MILITARY_STEALTH,
                         0.005, 250, 630, 50000, 100, "5th gen multirole"),
            TargetProfile("B-2 Spirit", TargetCategory.MILITARY_STEALTH,
                         0.01, 200, 475, 50000, 1000, "Stealth bomber"),
            TargetProfile("F-117 Nighthawk", TargetCategory.MILITARY_STEALTH,
                         0.001, 200, 550, 45000, 100, "Stealth attack aircraft"),

            # Military Heavy
            TargetProfile("B-52 Stratofortress", TargetCategory.MILITARY_HEAVY,
                         100.0, 200, 440, 50000, 1000, "Strategic bomber"),
            TargetProfile("B-1B Lancer", TargetCategory.MILITARY_HEAVY,
                         10.0, 200, 600, 60000, 1000, "Supersonic bomber"),
            TargetProfile("C-130 Hercules", TargetCategory.MILITARY_HEAVY,
                         30.0, 150, 300, 33000, 500, "Tactical transport"),
            TargetProfile("C-17 Globemaster", TargetCategory.MILITARY_HEAVY,
                         50.0, 150, 450, 45000, 500, "Strategic transport"),
            TargetProfile("KC-135 Stratotanker", TargetCategory.MILITARY_HEAVY,
                         40.0, 200, 480, 50000, 1000, "Aerial refueling tanker"),
            TargetProfile("E-3 Sentry AWACS", TargetCategory.MILITARY_HEAVY,
                         60.0, 200, 400, 42000, 1000, "Airborne early warning"),
            TargetProfile("P-8 Poseidon", TargetCategory.MILITARY_HEAVY,
                         25.0, 200, 450, 41000, 500, "Maritime patrol"),

            # Helicopters
            TargetProfile("AH-64 Apache", TargetCategory.MILITARY_HELICOPTER,
                         3.0, 0, 150, 21000, 0, "Attack helicopter"),
            TargetProfile("UH-60 Black Hawk", TargetCategory.MILITARY_HELICOPTER,
                         5.0, 0, 150, 19000, 0, "Utility helicopter"),
            TargetProfile("CH-47 Chinook", TargetCategory.MILITARY_HELICOPTER,
                         15.0, 0, 140, 20000, 0, "Heavy-lift helicopter"),
            TargetProfile("Bell 206", TargetCategory.CIVILIAN_HELICOPTER,
                         2.0, 0, 115, 13500, 0, "Light utility helicopter"),
            TargetProfile("Sikorsky S-76", TargetCategory.CIVILIAN_HELICOPTER,
                         4.0, 0, 155, 15000, 0, "Medium utility helicopter"),

            # UAVs
            TargetProfile("MQ-9 Reaper", TargetCategory.UAV,
                         1.0, 60, 230, 50000, 1000, "Armed reconnaissance UAV"),
            TargetProfile("RQ-4 Global Hawk", TargetCategory.UAV,
                         2.0, 100, 310, 60000, 5000, "High-altitude surveillance"),
            TargetProfile("MQ-1 Predator", TargetCategory.UAV,
                         0.5, 50, 135, 25000, 1000, "Reconnaissance UAV"),
            TargetProfile("DJI Phantom", TargetCategory.UAV,
                         0.01, 0, 35, 500, 0, "Consumer drone"),
        ]

        for target in aircraft:
            self.targets[target.name] = target

    def _load_vessels(self):
        """Load vessel/ship target profiles."""
        vessels = [
            # Cargo Ships
            TargetProfile("Panamax Container Ship", TargetCategory.CONTAINER_SHIP,
                         10000.0, 5, 25, 0, 0, "Large container vessel"),
            TargetProfile("Post-Panamax Container", TargetCategory.CONTAINER_SHIP,
                         15000.0, 5, 24, 0, 0, "Very large container vessel"),
            TargetProfile("Bulk Carrier", TargetCategory.CARGO_SHIP,
                         8000.0, 5, 15, 0, 0, "Bulk cargo vessel"),
            TargetProfile("General Cargo Ship", TargetCategory.CARGO_SHIP,
                         5000.0, 5, 18, 0, 0, "Multi-purpose cargo"),

            # Tankers
            TargetProfile("VLCC Tanker", TargetCategory.TANKER,
                         20000.0, 5, 16, 0, 0, "Very large crude carrier"),
            TargetProfile("Suezmax Tanker", TargetCategory.TANKER,
                         12000.0, 5, 17, 0, 0, "Large crude tanker"),
            TargetProfile("Product Tanker", TargetCategory.TANKER,
                         6000.0, 5, 15, 0, 0, "Refined products tanker"),
            TargetProfile("LNG Carrier", TargetCategory.TANKER,
                         10000.0, 5, 20, 0, 0, "Liquefied natural gas"),

            # Cruise/Passenger
            TargetProfile("Large Cruise Ship", TargetCategory.CRUISE_SHIP,
                         25000.0, 5, 24, 0, 0, "Large passenger vessel"),
            TargetProfile("Medium Cruise Ship", TargetCategory.CRUISE_SHIP,
                         15000.0, 5, 22, 0, 0, "Medium passenger vessel"),
            TargetProfile("Ferry", TargetCategory.CRUISE_SHIP,
                         3000.0, 5, 25, 0, 0, "Passenger/vehicle ferry"),

            # Naval Combatants
            TargetProfile("Arleigh Burke Destroyer", TargetCategory.NAVAL_COMBATANT,
                         2000.0, 5, 35, 0, 0, "Guided missile destroyer"),
            TargetProfile("Ticonderoga Cruiser", TargetCategory.NAVAL_COMBATANT,
                         3000.0, 5, 32, 0, 0, "Guided missile cruiser"),
            TargetProfile("Freedom LCS", TargetCategory.NAVAL_COMBATANT,
                         500.0, 5, 45, 0, 0, "Littoral combat ship"),
            TargetProfile("Type 052D Destroyer", TargetCategory.NAVAL_COMBATANT,
                         1800.0, 5, 32, 0, 0, "Chinese destroyer"),
            TargetProfile("Corvette", TargetCategory.NAVAL_COMBATANT,
                         300.0, 5, 28, 0, 0, "Light naval vessel"),
            TargetProfile("Frigate", TargetCategory.NAVAL_COMBATANT,
                         800.0, 5, 30, 0, 0, "Multi-role frigate"),

            # Aircraft Carriers
            TargetProfile("Nimitz Class Carrier", TargetCategory.NAVAL_CARRIER,
                         50000.0, 5, 35, 0, 0, "Nuclear supercarrier"),
            TargetProfile("Ford Class Carrier", TargetCategory.NAVAL_CARRIER,
                         55000.0, 5, 35, 0, 0, "Advanced supercarrier"),
            TargetProfile("Liaoning Carrier", TargetCategory.NAVAL_CARRIER,
                         40000.0, 5, 32, 0, 0, "STOBAR carrier"),

            # Small Vessels
            TargetProfile("Coast Guard Cutter", TargetCategory.PATROL_BOAT,
                         200.0, 5, 28, 0, 0, "Law enforcement vessel"),
            TargetProfile("Patrol Boat", TargetCategory.PATROL_BOAT,
                         50.0, 5, 35, 0, 0, "Small patrol vessel"),
            TargetProfile("Fishing Trawler", TargetCategory.FISHING_VESSEL,
                         100.0, 3, 12, 0, 0, "Commercial fishing vessel"),
            TargetProfile("Fishing Boat", TargetCategory.FISHING_VESSEL,
                         30.0, 3, 10, 0, 0, "Small fishing vessel"),
            TargetProfile("Large Yacht", TargetCategory.YACHT,
                         50.0, 5, 18, 0, 0, "Motor yacht"),
            TargetProfile("Sailboat", TargetCategory.YACHT,
                         10.0, 0, 8, 0, 0, "Sailing vessel"),
            TargetProfile("Speedboat", TargetCategory.SMALL_BOAT,
                         5.0, 5, 50, 0, 0, "High-speed small craft"),
            TargetProfile("RIB", TargetCategory.SMALL_BOAT,
                         2.0, 5, 40, 0, 0, "Rigid inflatable boat"),
        ]

        for target in vessels:
            self.targets[target.name] = target

    def get_target(self, name: str) -> Optional[TargetProfile]:
        """Get a specific target by name."""
        return self.targets.get(name)

    def get_random_target(self, category: Optional[TargetCategory] = None) -> TargetProfile:
        """Get a random target, optionally filtered by category."""
        if category:
            filtered = [t for t in self.targets.values() if t.category == category]
            if filtered:
                return random.choice(filtered)
        return random.choice(list(self.targets.values()))

    def get_targets_by_category(self, category: TargetCategory) -> List[TargetProfile]:
        """Get all targets in a category."""
        return [t for t in self.targets.values() if t.category == category]

    def get_all_categories(self) -> List[TargetCategory]:
        """Get list of all available categories."""
        return list(set(t.category for t in self.targets.values()))

    def get_all_names(self) -> List[str]:
        """Get list of all target names."""
        return list(self.targets.keys())


@dataclass
class TrackState:
    """Current state of a track."""
    x_m: float  # X position in meters (East positive)
    y_m: float  # Y position in meters (North positive)
    vx_mps: float  # X velocity in m/s
    vy_mps: float  # Y velocity in m/s
    altitude_ft: float
    heading_deg: float

    @property
    def range_m(self) -> float:
        """Distance from radar origin."""
        return math.sqrt(self.x_m**2 + self.y_m**2)

    @property
    def range_nmi(self) -> float:
        """Distance in nautical miles."""
        return self.range_m / 1852.0

    @property
    def azimuth_deg(self) -> float:
        """Azimuth angle from North (clockwise positive)."""
        return math.degrees(math.atan2(self.x_m, self.y_m)) % 360

    @property
    def speed_mps(self) -> float:
        """Speed in m/s."""
        return math.sqrt(self.vx_mps**2 + self.vy_mps**2)

    @property
    def speed_kts(self) -> float:
        """Speed in knots."""
        return self.speed_mps * 1.94384


@dataclass
class Track:
    """
    A simulated radar track representing a moving target.
    """
    track_id: int
    profile: TargetProfile
    state: TrackState
    max_range_m: float = 450000  # 240 nmi default

    # Track history for display
    history: List[Tuple[float, float]] = field(default_factory=list)
    history_limit: int = 50  # Keep last N positions

    # Track behavior
    maneuver_probability: float = 0.02  # Chance to change heading per update
    _time_to_maneuver: float = 0

    def __post_init__(self):
        """Initialize track."""
        # Add initial position to history
        self.history.append((self.state.x_m, self.state.y_m))

    def update(self, dt_seconds: float):
        """Update track position based on velocity."""
        # Check for maneuver
        self._time_to_maneuver -= dt_seconds
        if self._time_to_maneuver <= 0 or random.random() < self.maneuver_probability * dt_seconds:
            self._do_maneuver()
            self._time_to_maneuver = random.uniform(30, 120)  # Next maneuver window

        # Update position
        self.state.x_m += self.state.vx_mps * dt_seconds
        self.state.y_m += self.state.vy_mps * dt_seconds

        # Update heading from velocity
        if self.state.speed_mps > 0:
            self.state.heading_deg = math.degrees(
                math.atan2(self.state.vx_mps, self.state.vy_mps)
            ) % 360

        # Boundary handling - reverse heading if out of range
        if self.state.range_m > self.max_range_m:
            # Turn back toward radar
            angle_to_center = math.atan2(-self.state.x_m, -self.state.y_m)
            speed = self.state.speed_mps
            self.state.vx_mps = speed * math.sin(angle_to_center)
            self.state.vy_mps = speed * math.cos(angle_to_center)

        # Update history
        self.history.append((self.state.x_m, self.state.y_m))
        if len(self.history) > self.history_limit:
            self.history.pop(0)

    def _do_maneuver(self):
        """Execute a random maneuver."""
        # Random heading change (-45 to +45 degrees)
        heading_change = random.uniform(-45, 45)
        new_heading = math.radians(self.state.heading_deg + heading_change)

        # Optional speed change (80% to 120% of current)
        speed_factor = random.uniform(0.8, 1.2)
        new_speed = self.state.speed_mps * speed_factor

        # Clamp speed to profile limits
        min_speed = self.profile.min_speed_kts * 0.514444  # knots to m/s
        max_speed = self.profile.max_speed_kts * 0.514444
        new_speed = max(min_speed, min(max_speed, new_speed))

        # Update velocity
        self.state.vx_mps = new_speed * math.sin(new_heading)
        self.state.vy_mps = new_speed * math.cos(new_heading)


class TrackSimulator:
    """
    Manages multiple simulated tracks for radar display.
    """

    def __init__(self, max_range_nmi: float = 240):
        self.library = TargetLibrary()
        self.tracks: Dict[int, Track] = {}
        self.max_range_m = max_range_nmi * 1852
        self._next_track_id = 1
        self._running = False
        self._last_update = time.time()

    def create_track(self,
                     profile: Optional[TargetProfile] = None,
                     category: Optional[TargetCategory] = None,
                     start_range_nmi: Optional[float] = None,
                     start_azimuth_deg: Optional[float] = None) -> Track:
        """
        Create a new track with given or random parameters.
        """
        # Get profile
        if profile is None:
            profile = self.library.get_random_target(category)

        # Random start position if not specified
        if start_range_nmi is None:
            start_range_nmi = random.uniform(20, 220)
        if start_azimuth_deg is None:
            start_azimuth_deg = random.uniform(0, 360)

        # Convert to Cartesian
        range_m = start_range_nmi * 1852
        azimuth_rad = math.radians(start_azimuth_deg)
        x_m = range_m * math.sin(azimuth_rad)
        y_m = range_m * math.cos(azimuth_rad)

        # Random heading (biased toward radar for aircraft far out)
        if start_range_nmi > 150:
            # Point somewhat toward radar
            base_heading = (start_azimuth_deg + 180) % 360
            heading_deg = base_heading + random.uniform(-60, 60)
        else:
            heading_deg = random.uniform(0, 360)

        heading_rad = math.radians(heading_deg)

        # Random speed within profile limits
        speed_kts = random.uniform(profile.min_speed_kts, profile.max_speed_kts)
        speed_mps = speed_kts * 0.514444  # knots to m/s

        vx = speed_mps * math.sin(heading_rad)
        vy = speed_mps * math.cos(heading_rad)

        # Random altitude within profile limits
        if profile.ceiling_ft > 0:
            altitude_ft = random.uniform(profile.min_altitude_ft, profile.ceiling_ft * 0.8)
        else:
            altitude_ft = 0

        # Create state
        state = TrackState(
            x_m=x_m, y_m=y_m,
            vx_mps=vx, vy_mps=vy,
            altitude_ft=altitude_ft,
            heading_deg=heading_deg
        )

        # Create track
        track = Track(
            track_id=self._next_track_id,
            profile=profile,
            state=state,
            max_range_m=self.max_range_m
        )

        self.tracks[self._next_track_id] = track
        self._next_track_id += 1

        return track

    def remove_track(self, track_id: int):
        """Remove a track."""
        if track_id in self.tracks:
            del self.tracks[track_id]

    def clear_all_tracks(self):
        """Remove all tracks."""
        self.tracks.clear()

    def populate_random_tracks(self,
                               num_aircraft: int = 10,
                               num_vessels: int = 5):
        """
        Populate simulation with random tracks.
        """
        # Aircraft categories to include
        aircraft_categories = [
            TargetCategory.COMMERCIAL_AIRCRAFT,
            TargetCategory.GENERAL_AVIATION,
            TargetCategory.MILITARY_FIGHTER,
            TargetCategory.MILITARY_STEALTH,
            TargetCategory.MILITARY_HEAVY,
            TargetCategory.MILITARY_HELICOPTER,
            TargetCategory.CIVILIAN_HELICOPTER,
            TargetCategory.UAV,
        ]

        # Vessel categories
        vessel_categories = [
            TargetCategory.CARGO_SHIP,
            TargetCategory.TANKER,
            TargetCategory.CONTAINER_SHIP,
            TargetCategory.CRUISE_SHIP,
            TargetCategory.NAVAL_COMBATANT,
            TargetCategory.NAVAL_CARRIER,
            TargetCategory.PATROL_BOAT,
            TargetCategory.FISHING_VESSEL,
            TargetCategory.YACHT,
            TargetCategory.SMALL_BOAT,
        ]

        # Create aircraft
        for _ in range(num_aircraft):
            category = random.choice(aircraft_categories)
            self.create_track(category=category)

        # Create vessels (closer to radar, typically)
        for _ in range(num_vessels):
            category = random.choice(vessel_categories)
            self.create_track(
                category=category,
                start_range_nmi=random.uniform(5, 100)
            )

    def update(self, dt_seconds: Optional[float] = None):
        """
        Update all tracks.

        If dt_seconds is None, uses elapsed time since last update.
        """
        if dt_seconds is None:
            current_time = time.time()
            dt_seconds = current_time - self._last_update
            self._last_update = current_time

        for track in self.tracks.values():
            track.update(dt_seconds)

    def get_all_tracks(self) -> List[Track]:
        """Get list of all active tracks."""
        return list(self.tracks.values())

    def get_track_data(self) -> List[Dict]:
        """
        Get track data in a format suitable for display.
        """
        data = []
        for track in self.tracks.values():
            data.append({
                'id': track.track_id,
                'name': track.profile.name,
                'category': track.profile.category.value,
                'x_m': track.state.x_m,
                'y_m': track.state.y_m,
                'range_nmi': track.state.range_nmi,
                'azimuth_deg': track.state.azimuth_deg,
                'altitude_ft': track.state.altitude_ft,
                'speed_kts': track.state.speed_kts,
                'heading_deg': track.state.heading_deg,
                'rcs_m2': track.profile.rcs_m2,
                'rcs_dbsm': track.profile.rcs_dbsm,
                'history': track.history.copy(),
            })
        return data
