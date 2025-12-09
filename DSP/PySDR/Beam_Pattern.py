import numpy as np
import matplotlib.pyplot as plt

Nr = 5
d = 0.5 # half wave spacing
N_fft = 512
theta_degrees = 20 # there is no SOI, we aren't processing samples, this is the direction that we want to look at
theta = np.deg2rad(theta_degrees)
w = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # coventional beamformer weights
w_padded = np.concatenate((w, np.zeros(N_fft - Nr))) # zero pad to n_fft elements to get more resolution in the FFT
w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2) # magnitude of fft in dB
w_fft_dB -= np.max(w_fft_dB)

# map the fft bins to angles in radians
theta_bins = np.arcsin(np.linspace(-1, 1, N_fft)) # in radians

# find max so we can add it to plot
theta_max = theta_bins[np.argmax(w_fft_dB)]

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta_bins, w_fft_dB) # make sure to use radian for polar
ax.plot([theta_max], [np.max(w_fft_dB)], 'ro')
ax.text(theta_max -0.1, np.max(w_fft_dB)-4, np.round(np.rad2deg(theta_max)))
ax.set_theta_zero_location('N') # make 0 degrees point up
ax.set_theta_direction(-1) # increase clockwise
ax.set_rlabel_position(55)  # Move grid labels away from other labels
ax.set_thetamin(-90) # only show top half
ax.set_thetamax(90)
ax.set_ylim([-30, 1]) # because there's no noise, only go down 30 dB
plt.show()