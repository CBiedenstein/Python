import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

num_taps = 51 # Helps to use an odd number of taps
cut_off = 3e3 # Cutoff freq in kHz.
sample_rate = 32e3 # kHz

## Creating Low Pass Filter and Plot in Time Domain ##
h = signal.firwin(num_taps, cut_off, fs=sample_rate)
#plt.plot(h, '.-')
#plt.grid(True)
#plt.show()

## Plot the Frequency Response ##
#H = np.abs(np.fft.fft(h, 1024)) # take the 1024-point FFT and magnitude
#H = np.fft.fftshift(H) # make 0 Hz the center
#w = np.linspace(-sample_rate/2, sample_rate/2, len(H))
#plt.plot(w, H, '.-')
#plt.show()

## Complex Version of the above filter ##

## Shift the filter in frequency by multiplying by exp(j*2*pi*f0*t)
f0 = 10e3
Ts = 1.0/sample_rate
t = np.arange(0.0, Ts*len(h), Ts)
exponential = np.exp(2j*np.pi*f0*t)

h_bandpass = h * exponential # do the shift

## Plot impulse response ##
plt.figure('impulse')
plt.plot(np.real(h_bandpass), '.-')
plt.plot(np.imag(h_bandpass), '.-')
plt.legend(['real', 'imag'], loc=1)

## Plot the frequency response ##
H = np.abs(np.fft.fft(h_bandpass, 1024)) # FFT and magnitude
H = np.fft.fftshift(H) # make the center zero
w = np.linspace(-sample_rate/2, sample_rate/2, len(H))
plt.figure('freq')
plt.plot(w, H, '.-')
plt.xlabel('Frequency [Hz]')
plt.show()





