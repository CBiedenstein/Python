import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def getMeasurement (updateNumber) :
    if updateNumber == 1:
        getMeasurement.currentPosition = 0 
        getMeasurement.currentVelocity = 60 # m/s

    dt = 0.1

    w = 8 * np.random.randn(1) # additional random noise  
    v = 8 * np.random.randn(1) # random noise

    z = getMeasurement.currentPosition + getMeasurement.currentVelocity*dt + v
    getMeasurement.currentPosition = z - v
    getMeasurement.currentVelocity = 60 + w
    return( [z, getMeasurement.currentPosition, getMeasurement.currentVelocity] )

def kalmanFilter (z, updateNumber) :
    dt = 0.1 # time between measurements

    #initalize state
    if updateNumber == 1:
        kalmanFilter.x = np.array([[0],
                                   [20]])
        kalmanFilter.p = np.array([[5, 0],
                                   [0, 5]])
        kalmanFilter.A = np.array([[1, dt],
                                   [0,  1]])
        kalmanFilter.H = np.array([[1, 0]])
        kalmanFilter.HT = kalmanFilter.H.T
        kalmanFilter.R = 10
        kalmanFilter.Q = np.array([[1, 0], 
                                   [0, 3]])
        
    # Predict State Forward
    x_p = kalmanFilter.A.dot(kalmanFilter.x)
    # Predict Covariance Forward
    P_p = kalmanFilter.A.dot(kalmanFilter.p).dot(kalmanFilter.A.T) + kalmanFilter.Q 
    # Compute Kalman Gain
    S = kalmanFilter.H.dot(P_p).dot(kalmanFilter.HT) + kalmanFilter.R
    K = P_p.dot(kalmanFilter.HT)*(1/S)

    # Estimate State
    residual = z - kalmanFilter.H.dot(x_p)
    kalmanFilter.x = x_p + K*residual

    # Estimate Covariance
    kalmanFilter.p = P_p - K.dot(kalmanFilter.H).dot(P_p)

    return [kalmanFilter.x[0], kalmanFilter.x[1], kalmanFilter.p]

def testFilter () :
    dt = 0.1
    t = np.linspace(0, 10, num=300)
    numOfMeasurements = len(t)

    measTime = []
    measPos = []
    measDifPos = []
    estDifPos = []
    estPos = []
    estVel = []
    posBound3Sigma = []

    for k in range(1, numOfMeasurements) : 
        z = getMeasurement(k)
        # call Filter and return new State
        f = kalmanFilter(z[0], k)
        # Save that state so that it can be plotted
        measTime.append(k)
        measPos.append(z[0])
        measDifPos.append(z[0]-z[1])
        estDifPos.append(f[0]-z[1])
        estPos.append(f[0])
        estVel.append(f[1])
        posVar = f[2]
        posBound3Sigma.append(3*np.sqrt(posVar[0][0]))

    return [measTime, measPos, estPos, estVel, measDifPos, estDifPos, posBound3Sigma]

t = testFilter()

plot1 = plt.figure(1)
plt.scatter(t[0], t[1])
plt.plot(t[0], t[2])
plt.ylabel('Position')
plt.xlabel('Time')
plt.grid(True)

plot2 = plt.figure(2)
plt.plot(t[0], t[3])
plt.ylabel('Velocity m/s')
plt.xlabel('Update Number')
plt.title('Velocity Estimate on Each Measurement \n', fontweight="bold")
plt.legend(['Estimate'])
plt.grid(True)

plot3 = plt.figure(3)
plt.scatter(t[0], t[4], color='red')
plt.plot(t[0], t[5])
plt.legend(['Estimate', 'Measurement'])
plt.title('Position Errors on Each Measurement Update \n', fontweight='bold')
#plt.plot(t[0], t[6])
plt.ylabel('Position Error (meters)')
plt.xlabel('Update Number')
plt.grid(True)
plt.xlim([0, 300])
plt.show()











