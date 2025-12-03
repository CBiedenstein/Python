## to find the delay from delta D between elements, that is, the time it takes the signal to travel the additional distance to the next adjacent element.
## this is: delta_t = (d*sin(theta))/c. this can also be multiplied by integer multiples for further away elements.

## Simulating a receiving a signal at the array ##

import numpy as np
import matplotlib.pyplot as plt

sample_rate = 1e6
N= 10000 # number of samples to simulate

# create a tone to act the transmiutter signal
t = np.arange(N)/sample_rate # time vector
f_tone = 0.02e6
tx = np.exp(2j * np.pi * f_tone * t)

d = 0.5 # half wavelength spacing
Nr = 3 # number of receive elements
theta_degrees = 20 # direction of arrival (feel free to change this, its arbitrary)
theta = np.deg2rad(theta_degrees)
s = np.exp(2j* np.pi * d * np.arange(Nr) * np.sin(theta)) # steering vector
#print(s) # note that its 3 elements long, its complex and the first element is 1 + 0j.

s = s.reshape(-1, 1) # make s a column vector
#print(s.shape)
tx = tx.reshape(1, -1) # make tx a row vector
#print(tx)

X = s @ tx # simulate the received signal X through a matrix multiply
print(X.shape)

n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
X = X + 0.5*n # X and n are both 3x10000

plt.plot(np.asarray(X[0,:]).squeeze().real[0:200]) # the asarray and squeeze are just annoyances we have to do because we came from a matrix
plt.plot(np.asarray(X[1,:]).squeeze().real[0:200])
plt.plot(np.asarray(X[2,:]).squeeze().real[0:200])
plt.grid(True)
plt.show()

w = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # conventional, aka sum and delay beamformer
X_weighted = w.conj().T @ X # example of apllying the weiughts to the received signal (i.e., perform the beamforming)
print(X_weighted.shape)

