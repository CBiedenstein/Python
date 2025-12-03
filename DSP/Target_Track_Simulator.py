import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import math

# --- 1. RCS and Information Database Class ---
class TargetLibrary:
    """
    Database of aircraft types with typical S-Band RCS (m^2), 
    cruise speeds (knots), and service ceilings (ft).
    """
    def __init__(self):
        self.library = {
            # Commercial
            "B737": {"rcs": 20.0, "speed": 460, "ceiling": 41000, "type": "Commercial"},
            "B777": {"rcs": 40.0, "speed": 490, "ceiling": 43100, "type": "Commercial"},
            "A320": {"rcs": 18.0, "speed": 450, "ceiling": 39000, "type": "Commercial"},
            "Cessna 172": {"rcs": 1.5, "speed": 120, "ceiling": 14000, "type": "GA"},
            
            # Military - Fighters
            "F-16 Fighting Falcon": {"rcs": 1.2, "speed": 550, "ceiling": 50000, "type": "Military"},
            "F-15 Eagle": {"rcs": 5.0, "speed": 600, "ceiling": 65000, "type": "Military"},
            "Su-27 Flanker": {"rcs": 15.0, "speed": 580, "ceiling": 62000, "type": "Military"},
            
            # Military - Stealth / Low Observable
            "F-22 Raptor": {"rcs": 0.0001, "speed": 650, "ceiling": 65000, "type": "Military (Stealth)"},
            "F-35 Lightning II": {"rcs": 0.005, "speed": 630, "ceiling": 50000, "type": "Military (Stealth)"},
            
            # Military - Heavies
            "B-52 Stratofortress": {"rcs": 100.0, "speed": 440, "ceiling": 50000, "type": "Military"},
            "C-130 Hercules": {"rcs": 30.0, "speed": 300, "ceiling": 33000, "type": "Military"},
        }

    def get_random_target(self):
        """Returns a random target name and its properties."""
        name = random.choice(list(self.library.keys()))
        return name, self.library[name]

# --- 2. Track Simulation Class ---
class Track:
    def __init__(self, track_id, radar_lat, radar_lon):
        self.id = track_id
        self.radar_lat = radar_lat
        self.radar_lon = radar_lon
        
        # Initialize Random Kinematics
        self.range_nmi = random.uniform(20, 235) # Start within 240 nmi
        self.azimuth_deg = random.uniform(0, 360)
        self.heading_deg = random.uniform(0, 360)
        
        # Initialize from Library
        db = TargetLibrary()
        self.name, self.props = db.get_random_target()
        
        self.rcs = self.props['rcs']
        self.speed_kts = self.props['speed']
        self.altitude_ft = random.uniform(2000, self.props['ceiling'])
        self.type = self.props['type']
        
        # History for plotting (Lines)
        self.history_lat = []
        self.history_lon = []
        
        # Data storage for export
        self.data_points = []

    def update(self, time_step_idx, dt_seconds):
        """Updates position based on speed and heading."""
        # Distance traveled in nautical miles
        dist_nmi = (self.speed_kts * dt_seconds) / 3600.0
        
        # Convert heading to radians for calculation
        heading_rad = math.radians(self.heading_deg)
        
        # Current position relative to radar (Cartesian approx)
        cur_x_nmi = self.range_nmi * math.sin(math.radians(self.azimuth_deg))
        cur_y_nmi = self.range_nmi * math.cos(math.radians(self.azimuth_deg))
        
        # Update X/Y
        new_x_nmi = cur_x_nmi + (dist_nmi * math.sin(heading_rad))
        new_y_nmi = cur_y_nmi + (dist_nmi * math.cos(heading_rad))
        
        # Convert back to Range/Azimuth
        self.range_nmi = math.sqrt(new_x_nmi**2 + new_y_nmi**2)
        self.azimuth_deg = math.degrees(math.atan2(new_x_nmi, new_y_nmi)) % 360
        
        # Boundary Check (Bounce if out of 240 nmi range)
        if self.range_nmi > 240:
            self.heading_deg = (self.heading_deg + 180) % 360 
            
        # Update Lat/Lon (Approximation for Equator: 1 deg ~ 60 nmi)
        self.current_lat = self.radar_lat + (new_y_nmi / 60.0)
        self.current_lon = self.radar_lon + (new_x_nmi / 60.0)
        
        # Store History for Plotting
        self.history_lat.append(self.current_lat)
        self.history_lon.append(self.current_lon)
        
        # Record Data Point for Export
        point = {
            "Track_ID": self.id,
            "Target_Type": self.name,
            "Category": self.type,
            "Time_Step": time_step_idx,
            "Time_Sec": time_step_idx * dt_seconds,
            "Latitude": self.current_lat,
            "Longitude": self.current_lon,
            "Range_nmi": self.range_nmi,
            "Azimuth_deg": self.azimuth_deg,
            "Altitude_ft": self.altitude_ft,
            "Velocity_kts": self.speed_kts,
            "Heading_deg": self.heading_deg,
            "RCS_m2": self.rcs
        }
        self.data_points.append(point)

# --- 3. Simulation Execution ---
def run_simulation(num_tracks=15, duration_minutes=15):
    radar_lat = 0.0075
    radar_lon = -77.3186
    
    tracks = [Track(i+1, radar_lat, radar_lon) for i in range(num_tracks)]
    
    # Time step (seconds)
    dt = 12 
    steps = int((duration_minutes * 60) / dt)
    
    all_data = []
    
    for t in range(steps):
        for track in tracks:
            track.update(t, dt)
            
    # Aggregate data
    for track in tracks:
        all_data.extend(track.data_points)
            
    return tracks, all_data

# Run
simulated_tracks, export_data = run_simulation(num_tracks=12, duration_minutes=20)
df = pd.DataFrame(export_data)

# --- 4. Plotting (Lines) ---
plt.figure(figsize=(10, 10))
ax = plt.axes()
ax.set_facecolor('black')

# Plot Radar Origin
radar_lat = 0.0075
radar_lon = -77.3186
plt.plot(radar_lon, radar_lat, 'r+', markersize=12, markeredgewidth=2, label='RADAR (TPS-43 Mod)')

# Draw Range Rings
t = np.linspace(0, 2*np.pi, 100)
for r_nmi in [60, 120, 180, 240]:
    r_deg = r_nmi / 60.0
    x_circle = radar_lon + r_deg * np.cos(t)
    y_circle = radar_lat + r_deg * np.sin(t)
    plt.plot(x_circle, y_circle, 'g--', alpha=0.8, linewidth=0.5)

# Plot Tracks as Lines
for track in simulated_tracks:
    # Color coding
    color = 'cyan' if track.props['type'] == 'Commercial' else 'orange'
    if "Stealth" in track.props['type']: color = 'purple'
    if track.props['type'] == 'GA': color = 'white'
    
    # Plot the full history line
    plt.plot(track.history_lon, track.history_lat, color=color, linewidth=1, alpha=0.8)
    
    # Mark the current (end) position
    plt.plot(track.history_lon[-1], track.history_lat[-1], marker='o', color=color, markersize=4)
    
    # Label
    plt.text(track.history_lon[-1], track.history_lat[-1], 
             f" {track.name}", 
             color='white', fontsize=7, alpha=0.8)

plt.title(f"TPS-43 (2GW Upgrade) Simulation - 240 nmi\nLocation: {radar_lat}, {radar_lon}")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(color='green', linestyle=':', linewidth=0.5, alpha=0.3)
plt.xlim(radar_lon - 4.5, radar_lon + 4.5)
plt.ylim(radar_lat - 4.5, radar_lat + 4.5)

plt.show()

# --- 5. Export to CSV ---
csv_filename = 'radar_track_data.csv'
df.to_csv(csv_filename, index=False)
print(f"Exported {len(df)} data points to {csv_filename}")
print(df[['Track_ID', 'Target_Type', 'Time_Sec', 'Range_nmi', 'Azimuth_deg']].head())