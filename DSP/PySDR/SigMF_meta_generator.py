import datetime as dt
import numpy as np
import sigmf
from sigmf import SigMFFile
from math import sqrt

num_symbols = 10000


x_int = np.random.randint(0, 4, num_symbols)
x_degrees = x_int*360/4.0 + 45 # modulation points for QPSK 45, 135, 225, 315
x_radians = np.deg2rad(x_degrees) # sin and cos take radians
x_symbols = np.cos(x_radians) + 1j * np.sin(x_radians) # this produces the complex QPSK symbols.

n = np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols)/sqrt(2) # AWGN with unity power
r = x_symbols + n * sqrt(0.01) # noise power of 0.01
r = r.astype(np.complex64)
print(type(r[0]))
r.tofile('qpsk_in_noise.sigmf-data')

# creating the metadata
meta = SigMFFile(
    data_file = 'qpsk_in_noise.sigmf-data', #extension is optional
    global_info = {
        SigMFFile.DATATYPE_KEY: 'cf32_le',
        SigMFFile.SAMPLE_RATE_KEY: 8000000,
        SigMFFile.AUTHOR_KEY: 'Conner Biedenstein',
        SigMFFile.DESCRIPTION_KEY: 'Simulation of qpsk with noise.',
        SigMFFile.VERSION_KEY: sigmf.__version__,
    }
)

# create a capture key at time index 0
meta.add_capture(0, metadata={
    SigMFFile.FREQUENCY_KEY: 915000000,
    SigMFFile.DATETIME_KEY: dt.datetime.now(dt.timezone.utc).isoformat(),
})

# Check for mistakes and write to disk
meta.validate()
meta.tofile('qpsk_in_noise') # extension is optional










