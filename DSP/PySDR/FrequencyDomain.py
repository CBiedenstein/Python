import numpy as np
import matplotlib.pyplot as plt
'''
t = np.arange(100)
s = np.sin(0.15*2*np.pi*t)

#plt.plot(t, s)
#plt.show()

Sf = np.fft.fft(s)
S_mag = np.abs(Sf)
S_phase = np.angle(Sf)
'''

Fs = 1 #Hz
N = 100 # Number of points to sample and the size of the FFT.

t = np.arange(N) # Because the sample rate is 1 Hz
s = np.sin(0.15*2*np.pi*t)
s = s * np.hamming(N)
sf = np.fft.fftshift(np.fft.fft(s))
sf_mag = np.abs(sf)
sf_phase = np.angle(sf)
f = np.arange(Fs/-2, Fs/2, Fs/N)
#plt.figure(0)
#plt.plot(f, sf_mag, '.-')
#plt.grid(True)
#plt.figure(1)
#plt.plot(f, sf_phase, '.-')
#plt.grid(True)
#plt.show() 

#### Making a spectogram in python ####

sample_rate = 1e6 # 1 MHz sampling rate
t1 = np.arange(1024*1000)/sample_rate
f1 = 50e3 # 50 kHz Tone
x = np.sin(2*np.pi*f1*t1) + 0.2*np.random.randn(len(t1))

fft_size = 1024
num_rows = len(x) // fft_size # // is an integer division which rounds down
spectrogram = np.zeros((num_rows, fft_size))
for i in range(num_rows):
    spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i*fft_size:(i+1)*fft_size])))**2)

plt.imshow(spectrogram, aspect='auto', extent = [sample_rate/-2/1e6, sample_rate/2/1e6, len(x)/sample_rate, 0])
plt.xlabel("Frequency [MHz]")
plt.ylabel("Time [s]")
#plt.show()












