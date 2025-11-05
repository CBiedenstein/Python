import matplotlib.pyplot as plt
import numpy as np

# --- Original ADC Data ---
adc_resolution_bits = 14
adc_sampling_rates_gsps = np.array([1, 2, 3, 4, 5, 6])
adc_data_rates_gbps = adc_sampling_rates_gsps * adc_resolution_bits

# --- New DAC Data ---
dac_resolution_bits = 14
# Create an array from 1 to 10 GSPS
dac_sampling_rates_gsps = np.arange(1, 11) # This gives [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
dac_data_rates_gbps = dac_sampling_rates_gsps * dac_resolution_bits

# --- Create the plot ---
plt.figure(figsize=(10, 6)) # Made the figure a bit wider

# Plot the original ADC line
plt.plot(adc_sampling_rates_gsps, adc_data_rates_gbps, marker='o', linestyle='-', color='b', label='ADC 14-bit (1-6 GSPS)')

# Plot the new DAC line
plt.plot(dac_sampling_rates_gsps, dac_data_rates_gbps, marker='s', linestyle='--', color='g', label='DAC 14-bit (1-10 GSPS)')

# Highlight the 5 GSPS point (from original code, still relevant)
specific_rate = 2.5
specific_data_rate = specific_rate * adc_resolution_bits
plt.scatter(specific_rate, specific_data_rate, color='red', s=100, zorder=5)
plt.text(specific_rate, specific_data_rate + 3, f'{specific_data_rate} Gbps', ha='center', color='red', fontweight='bold')

# --- Add labels and title ---
plt.title('ADC/DAC Data Rate (14-bit resolution) vs Sampling Rate')
plt.xlabel('Sampling Rate (GSPS)')
plt.ylabel('Data Rate (Gbps)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()

# --- Set new limits for clarity ---
plt.xlim(0.8, 10.2) # Extend X-axis to 10
new_y_max = dac_data_rates_gbps.max() * 1.1 # Calculate new Y max (140 * 1.1 = 154)
plt.ylim(0, new_y_max)

# Save the plot
plt.show()
plt.savefig('adc_dac_data_rate_comparison.png')
