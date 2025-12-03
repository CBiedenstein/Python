# EXAMPLE LOAD SCRIPT
import numpy as np
import pickle

# 1. Load Data
data = np.load('radar_training_data/processed_data.npz', allow_pickle=True)
X_train = data['features']
Y_train = data['labels']

# 2. Load Scalers
with open('radar_training_data/label_scaler.pkl', 'rb') as f:
    label_scaler = pickle.load(f)

# Now you are ready to slice X_train into windows and feed the LSTM!