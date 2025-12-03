import numpy as np
import matplotlib.pyplot as plt

#H = np.hstack((np.zeros(20), np.arange(10)/10, np.zeros(20))) 
H = np.hstack((np.zeros(200), np.arange(100)/100, np.zeros(200))) ## Results in the impulse response tapering closer to zero but it end with more taps than hamming window
w = np.linspace(-0.5, 0.5, 500)
#plt.plot(w, H, '.-')
#plt.show()

h = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(H)))
window = np.hamming(len(h))
h = h * window ## These two lines apply a hamming window thus smoothing the filter response
plt.figure('Inverse Taps')
plt.plot(np.real(h), '-')
plt.plot(np.imag(h), '-')
plt.legend(['real', 'imag'], loc=2)

H_fft = np.fft.fftshift(np.abs(np.fft.fft(h, 1024)))
plt.figure('Taps Freq Response')
plt.plot(H_fft)
plt.show()

























