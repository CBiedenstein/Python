import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from enum import Enum
import pickle
import os


# --- 1. THE TARGET DATABASE ---
class TargetType(Enum):
    DRONE = 1
    FIGHTER_JET = 2
    MISSILE = 3
    CARGO_PLANE = 4

class TargetDatabase:
    """
    Acts as a lookup table for different physical cross-sections and 
    dynamic constraints.
    """
    def __init__(self):
        self.db = {
            TargetType.DRONE: {
                'name': 'Quadcop Drone',
                'base_rcs': 0.05,       # Very small, hard to see
                'rcs_std': 0.5,         # High fluctuation (rotors)
                'max_speed': 25.0,      # m/s (slow)
                'turn_rate': 4.0,       # rad/s (agile)
                'swerling_model': 1     # Slow fluctuation
            },
            TargetType.FIGHTER_JET: {
                'name': 'F-16 Fighter',
                'base_rcs': 3.0,        # Medium stealth
                'rcs_std': 2.0,         # Aspect dependent
                'max_speed': 350.0,     # m/s (supersonic)
                'turn_rate': 0.5,       # rad/s (limited G-force)
                'swerling_model': 1
            },
            TargetType.MISSILE: {
                'name': 'Cruise Missile',
                'base_rcs': 0.5,        # Stealthy nose-on
                'rcs_std': 0.2,         # Stable structure
                'max_speed': 280.0,     # m/s
                'turn_rate': 1.0,       # Moderate
                'swerling_model': 0     # Constant (often modeled as stable)
            },
            TargetType.CARGO_PLANE: {
                'name': 'C-130 Cargo',
                'base_rcs': 100.0,      # Huge barn door
                'rcs_std': 10.0,        # Massive fluctuations
                'max_speed': 180.0,     # m/s
                'turn_rate': 0.1,       # Very sluggish
                'swerling_model': 1
            }
        }

    def get_profile(self, target_type=None):
        if target_type is None:
            # Pick random if none specified
            target_type = np.random.choice(list(TargetType))
        return self.db[target_type]

# --- 2. THE SIMULATION ENGINE ---
class RadarEnvironment:
    def __init__(self):
        self.target_db = TargetDatabase()
        
    def simulate_rcs(self, profile):
        """
        Simulates RCS fluctuation based on Swerling Models.
        """
        mean_rcs = profile['base_rcs']
        
        # Swerling 0 (Constant / Sphere)
        if profile['swerling_model'] == 0:
            return mean_rcs
            
        # Swerling 1 (Rayleigh Distribution - Slow Fluctuation)
        # P(sigma) = (1/mean) * exp(-sigma/mean)
        # We simulate this using exponential distribution for magnitude
        elif profile['swerling_model'] == 1:
            fluctuation = np.random.exponential(scale=mean_rcs)
            return fluctuation
            
        return mean_rcs

    def generate_run(self, duration_steps=None, specific_type=None):
        """
        Generates a full trajectory run with physics constraints.
        """
        profile = self.target_db.get_profile(specific_type)
        
        # Random Run Length if not specified (e.g., 50 to 200 scans)
        if duration_steps is None:
            duration_steps = np.random.randint(50, 200)
            
        # Initial State
        x = np.random.uniform(-5000, 5000)
        y = np.random.uniform(-5000, 5000)
        speed = np.random.uniform(profile['max_speed'] * 0.5, profile['max_speed'])
        heading = np.random.uniform(0, 2 * np.pi)
        
        data = []
        
        for t in range(duration_steps):
            # 1. Update Kinematics (Physics)
            # Add random maneuver (change in heading) constrained by turn_rate
            turn = np.random.normal(0, profile['turn_rate'] * 0.1) # Small random turns
            heading += turn
            
            vx = speed * np.cos(heading)
            vy = speed * np.sin(heading)
            x += vx
            y += vy
            
            # 2. Radar Measurement Logic
            range_true = np.sqrt(x**2 + y**2)
            az_true = np.arctan2(y, x)
            doppler_true = (x*vx + y*vy) / (range_true + 1e-5)
            
            # 3. RCS & Signal Logic
            rcs_val = self.simulate_rcs(profile)
            
            # Simple Radar Range Equation for SNR
            # SNR drops with 4th power of range
            snr_val = 10 * np.log10((1e12 * rcs_val) / (range_true**4 + 1e-5)) 
            
            # 4. Add Noise (The Sensor Model)
            meas_range = range_true + np.random.normal(0, 15.0) # 15m range error
            meas_az = az_true + np.random.normal(0, 0.005)      # 0.005 rad angle error
            meas_dop = doppler_true + np.random.normal(0, 2.0)  # 2 m/s velocity error
            
            data.append({
                'time_step': t,
                'target_type': profile['name'],
                'range': meas_range,
                'azimuth': meas_az,
                'doppler': meas_dop,
                'rcs': rcs_val,
                'snr': snr_val,
                'x_true': x,  # Label for LSTM
                'y_true': y   # Label for LSTM
            })
            
        return pd.DataFrame(data)

# --- 3. THE PRE-PROCESSING BLOCK (Normalization) ---
class DataPreprocessor:
    def __init__(self):
        # We need separate scalers for Features (Inputs) and Labels (Outputs)
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.label_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_cols = ['range', 'azimuth', 'doppler', 'snr', 'rcs']
        self.label_cols = ['x_true', 'y_true']
        
    def fit(self, df):
        """
        Calibrate the scaler on a training dataset.
        """
        self.feature_scaler.fit(df[self.feature_cols])
        self.label_scaler.fit(df[self.label_cols])
        
    def transform(self, df):
        """
        Apply normalization to data.
        Returns scaled arrays ready for LSTM logic.
        """
        scaled_features = self.feature_scaler.transform(df[self.feature_cols])
        scaled_labels = self.label_scaler.transform(df[self.label_cols])
        return scaled_features, scaled_labels

    def inverse_transform_prediction(self, pred_array):
        """
        Convert LSTM predictions (0-1) back to Meters (World Coordinates).
        """
        return self.label_scaler.inverse_transform(pred_array)

# --- 4. DATA EXPORT ---   

class DataExporter:
    def __init__(self, output_dir='radar_training_data'):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created directory: {self.output_dir}")

    def save_artifacts(self, raw_df, scaled_x, scaled_y, preprocessor):
        """
        Saves all necessary files for training and later inference.
        """
        # 1. Save Raw Data (for audit/debugging)
        csv_path = os.path.join(self.output_dir, 'raw_radar_data.csv')
        raw_df.to_csv(csv_path, index=False)
        print(f"[Saved] Raw CSV:       {csv_path}")

        # 2. Save Scaled Data (The LSTM Food)
        # We use .npz (NumPy Zip) which is fast and compact
        data_path = os.path.join(self.output_dir, 'processed_data.npz')
        np.savez_compressed(
            data_path, 
            features=scaled_x, 
            labels=scaled_y,
            target_types=raw_df['target_type'].values # Helpful for filtering later
        )
        print(f"[Saved] Training Data: {data_path}")

        # 3. Save the Scalers (CRITICAL)
        # Without these, we can't un-scale the predictions later
        scaler_feat_path = os.path.join(self.output_dir, 'feature_scaler.pkl')
        scaler_label_path = os.path.join(self.output_dir, 'label_scaler.pkl')
        
        with open(scaler_feat_path, 'wb') as f:
            pickle.dump(preprocessor.feature_scaler, f)
            
        with open(scaler_label_path, 'wb') as f:
            pickle.dump(preprocessor.label_scaler, f)
            
        print(f"[Saved] Scalers:       {scaler_feat_path} & {scaler_label_path}")
        print("--- Export Complete ---")

# --- 5. MAIN EXECUTION ---

# A. Generate a "Training Set" of multiple runs
print("Generating Simulation Data...")
sim = RadarEnvironment()
all_runs = []

# Generate 5 runs (mix of drones, jets, etc.)
for _ in range(5):
    run_df = sim.generate_run(duration_steps=100)
    all_runs.append(run_df)

# Combine into one massive dataset for scaler fitting
full_dataset = pd.concat(all_runs, ignore_index=True)

# B. Pre-Processing (The "Block" you requested)
print("Running Pre-Processing Block...")
processor = DataPreprocessor()

# 1. Fit the Scaler (Learn Min/Max from data)
processor.fit(full_dataset)

# 2. Transform the data
scaled_X, scaled_Y = processor.transform(full_dataset)

# --- C. Visualization (PPI View) ---
print("\n--- Data Statistics ---")
print(f"Total Measurements: {len(full_dataset)}")
print(f"Target Types Simulated: {full_dataset['target_type'].unique()}")
print(f"Input Shape (Normalized): {scaled_X.shape}")

fig = plt.figure(figsize=(14, 6))

# Plot 1: The PPI Display (Polar Plot)
# We use 'projection=polar' to create the circular grid
ax1 = plt.subplot(1, 2, 1, projection='polar')

# Loop through target types to color-code them
for t_type in full_dataset['target_type'].unique():
    subset = full_dataset[full_dataset['target_type'] == t_type]
    
    # Plotting: Theta (Azimuth), R (Range)
    # Note: 'o' creates dots (detections), '-' creates lines (tracks)
    ax1.plot(subset['azimuth'], subset['range'], 'o-', markersize=3, label=t_type, alpha=0.6)

ax1.set_title("PPI Radar View (Raw Measurements)", va='bottom')
ax1.set_rlabel_position(-22.5)  # Move radial labels out of the way
ax1.grid(True)
ax1.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

# Plot 2: What the Neural Network Sees (Normalized Features)
# This remains a standard scatter plot because the LSTM sees math, not circles
ax2 = plt.subplot(1, 2, 2)
# Visualizing Normalized Range vs Normalized Doppler
sc = ax2.scatter(scaled_X[:, 0], scaled_X[:, 2], c=scaled_X[:, 3], cmap='viridis', alpha=0.5)
plt.colorbar(sc, label='Normalized SNR')

ax2.set_title("Neural Network Input View\n(Normalized 0.0 to 1.0)")
ax2.set_xlabel("Scaled Range")
ax2.set_ylabel("Scaled Doppler")
ax2.grid(True)

plt.tight_layout()
plt.show()

# Show a sample of the raw vs scaled data
print("\nSample Data (Raw vs Scaled):")
print(f"Raw Range (meters): {full_dataset['range'].iloc[0]:.2f} -> Scaled: {scaled_X[0][0]:.4f}")
print(f"Raw Doppler (m/s):  {full_dataset['doppler'].iloc[0]:.2f}   -> Scaled: {scaled_X[0][2]:.4f}")
'''
# Plot 2: What the Neural Network Sees (Normalized)
plt.subplot(1, 2, 2)
# Plotting Normalized Range vs Normalized Doppler as an example
plt.scatter(scaled_X[:, 0], scaled_X[:, 2], alpha=0.3, c='purple', s=5)
plt.title("Neural Network Input View\n(Normalized 0.0 to 1.0)")
plt.xlabel("Scaled Range")
plt.ylabel("Scaled Doppler")
plt.grid(True)

plt.tight_layout()
plt.show()

# Show a sample of the raw vs scaled data
print("\nSample Data (Raw vs Scaled):")
print(f"Raw Range (meters): {full_dataset['range'].iloc[0]:.2f} -> Scaled: {scaled_X[0][0]:.4f}")
print(f"Raw Doppler (m/s):  {full_dataset['doppler'].iloc[0]:.2f}   -> Scaled: {scaled_X[0][2]:.4f}")
'''
# --- EXECUTE EXPORT ---
# Instantiate the exporter
exporter = DataExporter()

# Pass in the variables generated in the previous blocks
exporter.save_artifacts(
    raw_df=full_dataset,      # The Pandas DataFrame
    scaled_x=scaled_X,        # The Normalized Features 
    scaled_y=scaled_Y,        # The Normalized Labels
    preprocessor=processor    # The Class containing the Scalers
)