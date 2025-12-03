import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

Ns = 1024
x = np.random.randn(Ns)
#plt.plot(x, '.-')
#plt.show()

X = np.fft.fftshift(np.fft.fft(x))
X = X[Ns//2:] # only look at integer frequencies. // is integer divide
#plt.plot(np.real(X), '.-')
#plt.show()

n =np.random.randn(Ns) + 1j * np.random.randn(Ns)/sqrt(2)
power = np.var(x)
print(power)
## Plotting Complex Gaussian Noise ##
#plt.plot(np.real(n), '.-')
#plt.plot(np.imag(n), '.-')
#plt.legend(['real', 'imag'])
#plt.show()

## Plotting Complex Gaussian Noise in the IQ plane ##
plt.plot(np.real(n), np.imag(n), '.')
plt.grid(True, which='both')
plt.axis([-2, 2, -2, 2])
plt.show()