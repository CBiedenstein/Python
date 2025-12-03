import numpy as np
import matplotlib.pyplot as plt

def generate_trajectory(steps=1000):
    """
    Generates a synthetic 2D flight path (Ground Truth).
    Simulates a target moving in a sine wave pattern (maneuvering).
    """
    t = np.linspace(0, 40, steps)
    
    # Physics: Target moving at constant X velocity, maneuvering in Y
    x_true = t * 10  # Constant velocity in X
    y_true = 50 * np.sin(t / 2)  # Sine wave maneuver in Y
    
    # Combine into a (Steps, 2) matrix representing (x, y)
    ground_truth = np.stack([x_true, y_true], axis=1)
    return ground_truth

def add_radar_noise(trajectory, noise_level=5.0):
    """
    Adds Gaussian noise to simulate radar measurement error.
    """
    noise = np.random.normal(0, noise_level, trajectory.shape)
    measurements = trajectory + noise
    return measurements

def create_lstm_dataset(measurements, ground_truth, look_back=10):
    """
    Slices the data into overlapping windows for the LSTM.
    
    X (Input): Sequence of 'look_back' noisy measurements.
    Y (Label): The SINGLE real (clean) position at the very next step.
    """
    X, Y = [], []
    
    # Loop through the data to create sequences
    for i in range(len(measurements) - look_back):
        # Slice a window of 'look_back' steps
        # This is the "Context" the LSTM sees
        window = measurements[i : i + look_back] 
        
        # The Label is the CLEAN ground truth at the next step (i + look_back)
        # We want the LSTM to predict the True position, not the Noisy one.
        target = ground_truth[i + look_back]
        
        X.append(window)
        Y.append(target)
        
    return np.array(X), np.array(Y)

# --- Main Execution ---

# 1. Generate Data
steps = 500
clean_path = generate_trajectory(steps)
noisy_radar = add_radar_noise(clean_path, noise_level=8.0)

# 2. Format for LSTM
# We look at the past 20 scans to predict the 21st position
LOOK_BACK_WINDOW = 20 

X_train, Y_train = create_lstm_dataset(noisy_radar, clean_path, LOOK_BACK_WINDOW)

# 3. Output Shapes (Crucial for Neural Networks)
print("--- Data Shapes ---")
print(f"Raw Data Length: {steps} points")
print(f"X_train Shape (Samples, Time Steps, Features): {X_train.shape}")
print(f"Y_train Shape (Samples, Features):             {Y_train.shape}")

print("\nExplanation:")
print(f"The LSTM will see {X_train.shape[0]} examples.")
print(f"Each example contains {X_train.shape[1]} history steps (scans).")
print(f"Each step has {X_train.shape[2]} features (X and Y coordinates).")

# 4. Visualization
plt.figure(figsize=(12, 6))

# Plot the whole trajectory
plt.plot(clean_path[:, 0], clean_path[:, 1], 'r-', linewidth=2, label='Ground Truth (Target)')
plt.scatter(noisy_radar[:, 0], noisy_radar[:, 1], c='blue', s=10, alpha=0.5, label='Radar Measurements (Noisy)')

# Highlight one LSTM "sample" window
start_sample = 50
sample_window = noisy_radar[start_sample : start_sample + LOOK_BACK_WINDOW]
target_point = clean_path[start_sample + LOOK_BACK_WINDOW]

plt.plot(sample_window[:, 0], sample_window[:, 1], 'g-', linewidth=1, label='Single LSTM Input Window')
plt.scatter(target_point[0], target_point[1], c='green', s=100, marker='X', label='Target to Predict')

plt.title(f"LSTM Training Data: Learning to Filter Noise\n(Window Size: {LOOK_BACK_WINDOW})")
plt.xlabel("X Position (meters)")
plt.ylabel("Y Position (meters)")
plt.legend()
plt.grid(True)
plt.show()