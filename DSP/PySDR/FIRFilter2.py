from scipy.signal import firwin2

sample_rate = 1e6
freqs = [0, 100e3, 110e3, 190e3, 200e3, 300e3, 310e3, 500e3]
gains = [1, 1, 0, 0, 0.5, 0.5, 0, 0]
h2 = firwin2(101, freqs, gains, fs=sample_rate)
print(h2)