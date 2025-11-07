import numpy as np
import matplotlib as plt
from numpy.linalg import inv

def exFilter(z, updateNumber):
    dt = 1.0
    j = updateNumber

    # Initialize State
    if updateNumber == 0: # First update

        # compute position values from measurements
        # x = r*sin(b)
        # y = r*cos(b)
        temp_x = z[0][j]*np.sin(z[1][j]*np.pi/180)
        



















