import numpy as np
from nptdms import TdmsFile, TdmsWriter, ChannelObject, GroupObject
import pandas as pd
import os

# --- PART 1: SETUP (Create a dummy file for demonstration) ---
def create_dummy_tdms(filename):
    """Creates a sample TDMS file if one doesn't exist."""
    if os.path.exists(filename):
        return

    print(f"Generating sample file: {filename}...")
    points = 100
    root_object = GroupObject("Root", properties={
        "author": "Python Script", 
        "description": "Sample Data"
    })
    
    # Create Group 1 with Sine Wave
    group_1 = GroupObject("Sensor_Data")
    time_data = np.linspace(0, 1, points)
    voltage_data = np.sin(2 * np.pi * 5 * time_data)
    
    channel_1 = ChannelObject("Sensor_Data", "Voltage", voltage_data)
    channel_2 = ChannelObject("Sensor_Data", "Time", time_data)
    
    with TdmsWriter(filename) as tdms_writer:
        tdms_writer.write_segment([root_object, group_1, channel_1, channel_2])

# Define the file to open
filename = "FirstADC_Test.tdms"

# Generate the file for this example (You can skip this if you have your own file)
#create_dummy_tdms(filename)


# --- PART 2: READING THE FILE ---

print(f"\n--- Opening {filename} ---\n")

# 1. Load the TDMS file
tdms_file = TdmsFile.read(filename)

# 2. Inspect File Properties (Root)
print("Root Properties:")
for name, value in tdms_file.properties.items():
    print(f"  {name}: {value}")
print("-" * 30)

# 3. Method A: Iterating through Groups and Channels (Hierarchical)
print("File Structure:")
for group in tdms_file.groups():
    print(f"Found Group: '{group.name}'")
    
    for channel in group.channels():
        print(f"  -> Found Channel: '{channel.name}' (Length: {len(channel)})")
        
        # Access the actual data array
        data = channel[:] 
        
        # Print first 5 data points
        print(f"     First 5 points: {data[:5]}")

print("-" * 30)

# 4. Method B: Accessing a specific channel directly
# If you know the Group and Channel names, you can grab data directly
try:
    # Syntax: tdms_file[Group_Name][Channel_Name]
    specific_data = tdms_file['Sensor_Data']['Voltage'][:]
    print(f"Direct Access 'Voltage' Mean: {np.mean(specific_data):.4f}")
except KeyError:
    print("Could not find specific group/channel for direct access example.")

print("-" * 30)

# 5. Method C: Converting to Pandas DataFrame (The easiest way to analyze)
# This converts the entire file into a neat table
print("Converting to Pandas DataFrame...")
df = tdms_file.as_dataframe()

# Clean up column names (usually appear as 'Group/Channel')
# This optional step removes the Group name to make it cleaner
# df.columns = [col.split('/')[-1] for col in df.columns]

print(df.head())

# Optional: Save to CSV
df.to_csv("converted_data.csv")