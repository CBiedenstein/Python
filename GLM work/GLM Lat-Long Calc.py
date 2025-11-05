import cmath
import math
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, ITRS, FK5
from angle2dcm import angle2dcm
from astropy.io import fits
from glob import glob

GLM_pos = glob(f"Testing Data/*.fits")[0]
QUART = fits.open(GLM_pos)
print(QUART[0].header)
header = QUART[0].header
QUART.close()

q = [header["QUAT1"], header["QUAT2"], header["QUAT3"], header["QUAT4"]]
q1, q2, q3, q4 = q


print('\n','\n')

fn = r"./Testing Data/GLM_data1200_1400.nc"
ds = nc.Dataset(fn)
var1 = ds.variables['event_x']
var2 = ds.variables['event_y']
var3 = ds.variables['event_intensity']

#print(ds)
#print(ds.variables.keys())


pointx = ds.variables['event_x'][:]
#print("\n",*pointx[:])
pointy = ds.variables['event_y'][:]
intensity_value = ds.variables['event_intensity'][:]

xpixel = np.array(pointx, dtype = 'int64')
ypixel = np.array(pointy, dtype = 'int64')
intensity = np.array(intensity_value)
BGMSB = ds.variables['event_bgmsb'][:] # BGMSB - Background Most Significant Bits
#quaternion = np.array(background, dtype = 'int64')
#print('quaternion:', quaternion[1:20])

frame = ds.variables['event_unique_frame_id']
print(frame)
#print(xpix)
print(max(ypixel))

time = np.array((ds.variables['event_id'], ds.variables['event_day'],ds.variables['event_millisecond'], ds.variables['event_microsecond']))
df = np.asarray(ds.variables['event_pixel'])
ts = np.asarray(ds.variables['event_millisecond'])/1000
#print(ts)

## max values for both x and y pixels fall inside of the shielded area, suggests focusing and exclusion may be needed. 

## Specified Variables :
f_ref = 133.974 #mm section 2.1
c_EFL = -0.00515 #mm/deg C section 2.1
T_LAref = 30 #deg C Section 2.1
k_d = -4.614641E-5 # mm/mm^3 section 2.1


## Needed Variables
T_LA = 0#?? deg C  section 2.1

## Conversion of pixel to CCD
## each pixel is 27.175385 microns, shielding radius is 691.80253

f = f_ref + c_EFL*(T_LA - T_LAref)

x_mcref = xpixel*f_ref/f
y_mcref = ypixel*f_ref/f

## r_iref = r_mcref-k_d*pow(r_mcref, 3)

xpixel = np.array(pointx, dtype = 'int64')
ypixel = np.array(pointy, dtype = 'int64')
intensity = np.array(intensity_value)

## Specified Variables :
f_ref = 133.974 #mm section 2.1
c_EFL = -0.00515 #mm/deg C section 2.1
T_LAref = 30 #deg C Section 2.1
k_d = -4.614641E-5 # mm/mm^3 section 2.1

## Needed Variables
T_LA = 0 # deg C  section 2.1

## Conversion of pixel to CCD
## each pixel is 27.175385 microns, shielding radius is 691.80253

f = f_ref + c_EFL*(T_LA - T_LAref)

x_mcref = xpixel*f_ref/f
y_mcref = ypixel*f_ref/f


r_mcref = np.sqrt(pow(xpixel,2)+pow(ypixel,2))
#print(max(r_mcref[:]))
#print(max(pow(ypixel,1)))
#print(max(pow(r_mcref,3)))

r_iref=r_mcref-(k_d*(pow(r_mcref, 3)))
#print('r_iref:','\n',  r_iref)

theta = np.arctan(ypixel/xpixel)
#print('theta max:',max(theta))

x_irefprime = r_iref * np.cos(theta)
y_irefprime = r_iref * np.sin(theta)

c_x = 0 ## TBD
c_y = 0 ## TBD

x_iref = x_irefprime + c_x
y_iref = y_irefprime + c_y

x_i = x_iref*f/f_ref
y_i = y_iref*f/f_ref

d = pow((pow(f,2) + pow(x_i,2) +pow(y_i,2)), 0.5)

v1 = x_i[1]/d[1]
v2 = y_i[1]/d[1]
v3 = pow((1 - pow(v1,2) - pow(v2,2)), 0.5)

## A(gf->g) transformation matrix

kx_bp = 0 #TBD
ky_bp = 0 #TBD

T_negx = 0 #bipod temperature regions on the spacecraft
T_posx = 0
T_negy = 0
T_posy = 0

phi_bp   = kx_bp*(((T_negx+T_posx)/2-T_negy))
theta_bp = ky_bp*(T_negx - T_posx)

A =  np.array([[np.cos(theta_bp),   np.sin(theta_bp)*np.sin(phi_bp),   -1*np.sin(theta_bp)*np.cos(phi_bp)],
               [       0,                 np.cos(theta_bp),                        np.sin(phi_bp)],
               [np.sin(theta_bp), -1*np.cos(theta_bp)*np.sin(phi_bp),    np.cos(theta_bp)*np.cos(phi_bp) ]])
#print(' A Matrix:','\n', A)
v_gf = np.array([[v1, v2, v3]]).T

v_g=A.dot(v_gf)

alpha_VA = 10.25E-6
orientation = 1 ## value is +1 for upgright spacecraft and -1 for yaw flipped(inverted) spacecraft
alpha_comp = orientation*alpha_VA
A_comp = np.array([[np.cos(alpha_comp), 0, -1*np.sin(alpha_comp)],
                   [      0,            1,           0          ],
                   [np.sin(alpha_comp), 0,    np.cos(alpha_comp)]])
#print('A_comp:', '\n', A_comp)

vc_g = A_comp.dot(v_g)
#print('vc_g:','\n', vc_g)

## Jump to section 3.0 to establish coastline ID, predicted each day -- This could be challenging... might involve more data

#v_m = A(g->m).dot(vc_g) = A(m->g).T.dot(vc_g) # NEED TO ESTABLISH ONCE FILLED OUT TO SECTION 3.0
'''
TIME CONVERSION NOTES
Attitude provided by the ACRS and the position of the SC relative to the J2000 coordinate frame at 20 Hz.
M frame is identcal to the ACRS Frame, the G frame coincides with the M frame.
SC propogates its position at 20 Hz, updated GPS coordinates once every second. Location precision within 100m.
Attitude and position(att/orb) is time stamped, using SC clock. Provided in CDS -- three time fields
A day field, millisecond in day, and microsecond. Time epoch code is 12^h, Jan 1, 2000 UTC.
Day number = td -- number of second in day = ts, like Julian days each td starts at noon/1200
Each att/orb is bounded by time stamps, the native time stamp and the following one.
GLM clock is synchronized with SC clock once a second.
att/orb for spacecraft(SC) is obtained by converting J2000 to ITRS then using linear interpolation.
SC and GLM clock times are converted to CTS (continuous time in seconds) to allow for interpolation at all times.
Described as the number of SI seconds elapsed since 2000 Jan 1, 12^h UTC. CTS time is identified as tc.

CONVERSION OF SC AND GLM CLOCK TIMES TO CTS
Leap second table is kept.... table 1.section 2.3

No new leap seconds in the near future... so nleaps will always be 5 (number of leap seconds added to UTC)
'''

td = np.asarray(ds.variables['event_day']) # Taken from the start of J2000
ts = np.asarray(ds.variables['event_millisecond'])/1000 # Taken from start at 1200
ts_n1 = (np.asarray(ds.variables['event_millisecond'])+50)/1000 # Establishing tc_n+1
nleaps = 5
Sec_day = np.array([86400], dtype = 'int32')

tc = (td*Sec_day) + ts + nleaps # CTS Time
tc_n1 = (td*Sec_day) + ts_n1 + nleaps # CTS Time

## COORDINATE TRANSFORMATION FROM J2000 TO ITRS -- Attempting Astropy -- may need to establish by hand.
#  Need to setup the following matricies: A(ti->it) & A(ci->ti) & A(J->ci)
#  Astropy may be used if right ascension and declination are known or can be mapped. 
#  Establishing by hand... NPB matrix using reference [5] in INROfGLM

#  Below is the building of the frame bias matrix basis and the CIRS transformation step:

#  A(j->ci) = [xci_j, yci_j, zci_j]^T
masRA = 206264806.247 # conversion of milliarcseconds to radians
dalpha_0 = -14.6/masRA # mas (milliarcseconds) the offset in ICRS right Ascension origin with  respect to J2000 Dynamical equinox
xi_0 = -16.6170/masRA # mas ICRS pole offset
etta_0 = -6.8192/masRA # mas ICRS pole offset

B = np.array([[      1    ,   dalpha_0  ,  -1*xi_0  ],
              [-1*dalpha_0,      1      , -1*etta_0 ],
              [    xi_0   ,    etta_0   ,     1     ]])
#print('Frame Bias Matrix:', '\n', B)

## Below the precession matrix is established... may need to refine if outside of tolerable results

T_jc = (tc/86400)/36525 # number of julian centuries since j2000
T_jcn1 = (tc_n1/86400)/36525
epsilon_0 = 84381.406
phi = 0 # TBD 

R1_phi = np.array([[  1  ,        0      ,       0      ],
                   [  0  ,   np.cos(phi) ,   np.sin(phi) ],
                   [  0  ,  -np.sin(phi) ,  np.cos(phi) ]])

R2_phi = np.array([[ np.cos(phi),  0  , -np.sin(phi)],
                   [     0      ,  1  ,      0      ],
                   [ np.sin(phi),  0  , np.cos(phi) ]])

R3_phi = np.array([[ np.cos(phi) , np.sin(phi) ,  0  ],
                   [-np.sin(phi) , np.cos(phi) ,  0  ],
                   [     0       ,      0      ,  1  ]])

psi = ((((-0.0000000951*T_jc[1] + 0.000132851)*T_jc[1] - 0.00114045)*T_jc[1] - 1.0790069)*T_jc[1] + 5038.481507)*T_jc[1]
# psi = 5038.481507*T_jc - 1.0790069*pow(T_jc,2) - 0.00114045*pow(T_jc,3) + 0.000132851*pow(T_jc, 4) - 0.0000000951*pow(T_jc,5)
omega = ((((0.0000003337*T_jc[1] - 0.000000467)*T_jc[1] - 0.00772503)*T_jc[1] + 0.0512623)*T_jc[1] - 0.025754)*T_jc[1] + epsilon_0
chi = ((((-0.0000000560*T_jc[1] + 0.000170663)*T_jc[1] - 0.00121197)*T_jc[1] - 2.3814292)*T_jc[1] + 10.556403)*T_jc[1]

## NEED LOOP FOR T_jc... need to establish an iterative loop to work through each list element and append to array/table.

ps1 = np.sin(epsilon_0)
ps2 = np.sin(-psi)
ps3 = np.sin(-omega)
ps4 = np.sin(chi)
pc1 = np.cos(epsilon_0)
pc2 = np.cos(-psi)
pc3 = np.cos(-omega)
pc4 = np.cos(chi)


P = np.array([[ pc4*pc2 - ps2*ps4*pc3,  pc4*ps2*pc1 + ps4*pc3*pc2*pc1 - ps1*ps4*ps3,  pc4*ps2*ps1 + ps4*pc3*pc2*pc1 + pc1*ps4*ps3],
              [-ps4*pc2 - ps2*pc4*pc3, -ps4*ps2*pc1 + pc4*pc3*pc2*pc1 - ps1*pc4*ps3, -ps4*ps2*ps1 + pc4*pc3*pc2*pc1 + pc1+pc4*ps3],
              [       ps2*ps3     ,           -ps3*pc2*pc1 - ps1*pc3       ,         -ps3*pc2*ps1 + pc3*pc1         ]])
#print('Precession Matrix:','\n', P)

## Below the nutation matrix is established

# Mean heliocentric ecliptic longtitudes of planets Mercury through Neptune

phi1 =  908103.259872 + 538101628.688982*T_jc[1] # Mercury
phi2 =  655127.283060 + 210664136.433548*T_jc[1] # Venus
phi3 =  361679.244588 + 129597742.283429*T_jc[1] # Earth
phi4 = 1279558.798488 +  68905077.493988*T_jc[1] # Mars
phi5 =  123665.467464 +  10925660.377991*T_jc[1] # Jupiter
phi6 =  180278.799480 +   4399609.855732*T_jc[1] # Saturn
phi7 = 1130598.018396 +   1542481.193933*T_jc[1] # Uranus
phi8 = 1095655.195728 +    786550.320744*T_jc[1] # Neptune
phi9 = 5028.8200*T_jc[1] +      1.112022*T_jc[1]**2 # General precession in longitude approx.
phi10 = 485868.249036 + 1717915923.2178*T_jc[1] + 31.8792*T_jc[1]**2 + 0.051635*T_jc[1]**3 - 0.00024470*T_jc[1]**4 # Mean anomaly of the Moon
phi11 = 1287104.79305 + 129596581.0481*T_jc[1] - 0.5532*T_jc[1]**2 + 0.000136*T_jc[1]**3 - 0.00001149*T_jc[1]**4   # Mean anomaly of the Sun
phi12 = 335779.526232 + 1739527262.8478*T_jc[1] - 12.7512*T_jc[1]**2 - 0.001037*T_jc[1]**3 + 0.00000417*T_jc[1]**4 # Mean argument of latitude of the Moon
phi13 = 1072260.70369 + 1602961601.2090*T_jc[1] - 6.3706*T_jc[1]**2 + 0.006593*T_jc[1]**3 - 0.00003169*T_jc[1]**4  # Mean elongation of the Moon from the Sun
phi14 = 450160.398036 - 6962890.5431*T_jc[1] + 7.4722*T_jc[1]**2 + 0.007702*T_jc[1]**3 - 0.00005939*T_jc[1]**4     # Mean longitude of the Moon's mean ascending node

phi_j = np.array([phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8, phi9, phi10, phi11, phi12, phi13, phi14])

M_ij = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,-2, 2],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2,-2, 2],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 2],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 2,-2, 2],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,-2, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 2, 0, 2],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 2, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 2, 2, 2],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 2,-2, 2],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0, 0, 2, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 2],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2,-2, 2],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 2, 0, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 2, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2,-2, 2],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 2, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,-2, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 2, 2, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 2, 2],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0, 2, 0, 0]])

S_i = np.array([-17.2064161, -1.3170906, -0.2276413, 0.2074554, 0.1475877, -0.0516821, 0.0711159, -0.0387298, -0.0301461, 
                0.0215829, 0.0128227, 0.0123457, 0.0156994, 0.0063110, -0.0057976, -0,0.0059641 -0.0051613, 0.0045893, 
                0.0063384, -0.0038571, 0.0032481, -0.0047722, -0.0031046, 0.0028593, 0.0020411, 0.0029243, 0.0025887, 
                -0.0014053, 0.0015164, -0.0015794, 0.0021783, -0.0012873, -0.0012654, -0.0010204, 0.0016707, 0.0007691,
                -0.0011024])
                  
Sdot_i = np.array([-0.0174666, -0.0001675, -0.0000234, 0.0000207, -0.0003633, 0.0001226, 0.0000073, -0.0000367, 0.0000036,
                   -0.0000494, 0.0000137, 0.0000011, 0.0000010, 0.0000063, -0.0000063, -0.0000011, -0.0000042, 0.0000050,
                   0.0000011, -0.0000001, 0, 0, -0.0000001, 0, 0.0000021, 0, 0, -0.0000025, 0.0000010, 0.0000072, 0,
                   -0.0000010, 0.0000011, 0, -0.0000085, 0, 0])
                  
Cprime_i = np.array([0.0033386, -0.0013696, 0.0002796, -0.0000698, 0.0011817, -0.0000524, -0.0000872, 0.0000380, 0.0000816,
                     0.0000111, 0.0000181, 0.0000019, -0.0000168, 0.0000027, -0.0000189, 0.000149, 0.0000129, 0.0000031, 
                     -0.0000150, -0.0000158, 0, -0.0000018, 0.0000131, -0.0000001, 0.0000010, -0.0000074, -0.0000066, 
                     0.0000079, 0.0000011, -0.0000016, 0.0000013, -0.0000037, 0.0000063, 0.0000025, -0.0000010, 0.0000044,
                     -0.0000014]) 
                  
C_i = np.array([9.2052331, 0.5730336, 0.0978459, -0.0897492, 0.0073871, 0.0224386, -0.0006750, 0.0200728, 0.0129025, 
                -.0095929, -0.0068982, -0.0053311, -0.0001235, -0.0033228, 0.0031429, 0.0025543, 0.0026366, -0.0024236,
                -0.0001220, 0.0016452, -0.0013870, 0.0000477, 0.0013238, -0.0012338, -0.0010758, -0.0000609, -0.0000550,
                0.0008551, -0.0008001, 0.0006850, -0.0000167, 0.0006953, 0.0006415, 0.0005222, 0.0000168, 0.0003268, 
                0.0000104])
                  
Cdot_i = np.array([0.0009086, -0.0003015, -0.0000485, 0.0000470, -0.0000184, -0.0000677, 0, 0.0000018, -0.0000063, 0.0000299,
                   -0.0000009, 0.0000032, 0, 0, 0, -0.0000011, 0, -0.0000010, 0, -0.0000011, 0, 0, -0.0000011, 0.0000010, 0, 
                   0, 0, -0.0000002, 0, -0.0000042,0, 0, 0, 0, -0.0000001, 0, 0])
                  
Sprime_i = np.array([0.0015377, -0.0004587, 0.0001374, -0.0000291, -0.0001924, -0.0000174, 0.0000358, 0.0000318, 0.0000367,
                     0.0000132, 0.0000039, -0.0000004, 0.0000082, -0.0000009, -0.0000075, 0.0000066, 0.0000078, 0.0000020, 
                     0.0000029, 0.0000068, 0, -0.0000025, 0.0000059, -0.0000003, -0.0000003, 0.0000013, 0.0000011, -0.0000045,
                     -0.0000001, -0.0000005, 0.0000013, -0.0000014, 0.0000026, 0.0000015, 0.0000010, 0.0000019, 0.0000002])

                  
def deltapsi():
    phi_sum = []

    for i in range(1, 37): 
        phi_sum = sum(M_ij[i,:] * phi_j)
        #print(phi_sum)
            
        return sum((S_i + Sdot_i * T_jc[1]) * np.sin(phi_sum) + Cprime_i * np.cos(phi_sum))

def deltaepsilon():
    phi_sum = []
    
    for i in range(1, 37):
        phi_sum = sum(M_ij[i,:] * phi_j)
        #print(phi_sum)
        
        return sum((C_i + Cdot_i * T_jc[1]) * np.cos(phi_sum) + Sprime_i * np.sin(phi_sum))
    
def arc_rad(n):
    R = n * np.pi/(180*3600)
    return R
    
deltaepsilon(), deltapsi()

epsilon_0 = 84381.406
epsilon = epsilon_0 - 46.836769*T_jc[1] - 0.0001831*T_jc[1]**2 + 0.00200340*T_jc[1]**3 - 0.000000576*T_jc[1]**4 - 0.0000000434*T_jc[1]**5
epsilon_prime = epsilon + deltaepsilon()

#print('epsilon prime:', epsilon_prime)
#print('epsilon:', epsilon)

ns1 = np.sin(epsilon)
ns2 = np.sin(-deltapsi())
ns3 = np.sin(epsilon - deltaepsilon())
nc1 = np.cos(epsilon)
nc2 = np.cos(-deltapsi())
nc3 = np.cos(-epsilon - deltaepsilon())

N = np.array([[     nc2    ,          ns2*nc1        ,          ns2*ns1        ],
              [  -ns2*nc3  ,  nc3*nc2*nc1 - ns1*ns3  ,  nc3*nc2*ns1 + nc1*ns3  ],
              [   ns2*nc3  , -ns3*nc2*nc1 - ns1*nc3  , -ns3*nc2*ns1 + nc3*nc1  ]])
#print('Nutation Matrix:','\n', N)

#result = N*P*B # matrix positions are n-1 for python indexing
#print('result', '\n', result)
#result1 = N*P*B[2,0] # matrix positions are n-1 for python indexing
#print('result', '\n', result1)
#result2 = N*P*B[2,1] # matrix positions are n-1 for python indexing
#print('result', '\n', result2)

## A(j->ci) construction

zci_j1 = (N*P*B)[2,0] # X
zci_j2 = (N*P*B)[2,1] # Y
zci_j3 = (N*P*B)[2,2] # Z

zci_j = np.array([zci_j1, zci_j2, zci_j3])
#print('zci_j:', zci_j)

xci_j1 = zci_j3/np.sqrt((zci_j1**2 + zci_j3**2))
xci_j2 = 0
xci_j3 = -zci_j1/np.sqrt(zci_j1**2 + zci_j3**2)

xci_j = np.array([xci_j1, xci_j2, xci_j3])
#print('xci_j:', xci_j)

yci_j = zci_j * xci_j
#print('yci_j:', yci_j)

A_jci = np.vstack((xci_j, yci_j, zci_j))
#print('A(j->ci):', '\n', A_jci)

## A(ci->ti) construction

tc_ci = tc - 0.001059 # delay of the time stamp event relative to the center of integration... 1.059 milliseconds

import astropy
from astropy.time import Time

## interpolated ut1-utc for a 5 week period, 35 days, is 0.0036 @ 02/17/2025 16:42 PM

deltaUT1 = 0.0036
DeltaUT1C = deltaUT1 - nleaps
D_u = (tc_ci[1] + DeltaUT1C)/86400
#print('D_u:', '\n', D_u)

theta_E = (2*np.pi*(0.7790572732640+1.00273781191135448*D_u)) ## May need to convert to radians using 1 arcsecond = 4.8481368E-6 radians
#print('Theta_E:', '\n', theta_E)

A_citi = np.array([[ np.cos(theta_E), np.sin(theta_E),  0  ],
                   [-np.sin(theta_E), np.cos(theta_E),  0  ],
                   [        0       ,        0       ,  1  ]])
#print('A(ci->ti):', '\n', A_citi)

MJD = td[1] + np.array([51544], dtype = 'int32')#51544.5 to convert to MJD
MJD_table = MJD - 41684 # to grab correct values for 2025.... I think. This comment was added after days after this was done.
#print('MJD:', MJD)
#print('MJD_table:', MJD_table)

from astropy.utils import iers
tiers = Time('2010:001')
#iers_a = iers.IERS_A.open(iers.IERS_A_URL)
#iers_a.ut1_utc(tiers)
#iers.earth_orientation_table.set(iers_a)
#with iers.earth_orientation_table.set(iers_a):
#    print(tiers.ut1.iso)
dat = iers.earth_orientation_table.get()
type(dat)
dat['MJD', 'UT1_UTC', 'PM_x', 'PM_y']

IERS_Array = np.array([dat['MJD'], 
                       dat['UT1_UTC'],
                       dat['PM_x'],
                       dat['PM_y']])
#print(IERS_Array[0:,:])
print('PM_x @', MJD, 'MJD:', IERS_Array[2,MJD_table]) #columns, rows ... PM_x is index 2 & PM_y is index 3
print('PM_y @', MJD, 'MJD:', IERS_Array[3,MJD_table]) #columns, rows ... PM_x is index 2 & PM_y is index 3
print(IERS_Array[2, MJD_table])
## A(ti->it) Matrix ... May need to refine MJD Determination ... MUST BE CONVERTED TO RADIANS

PM_x = np.squeeze(IERS_Array[2, MJD_table])
PM_y = np.squeeze(IERS_Array[3, MJD_table])
print(PM_x, PM_y)

AR_IERS2 = np.asarray(arc_rad(PM_x), dtype='float32')
AR_IERS3 = np.asarray(arc_rad(PM_y), dtype='float32')
print(AR_IERS2, AR_IERS3)

A_tiit = np.array([[    1    ,       0    ,  AR_IERS2],
                   [    0    ,       1    , -AR_IERS3],
                   [-AR_IERS2,    AR_IERS3,     1    ]])

#np.array([[1,0,AR_IERS2],[0,1,-1.4698873e-06],[-8.3464556e-07,1.4698873e-06,1]]) troubleshooting line for A_tiit
#print('A(ti->it)', '\n', A_tiit)

## Establising the A(j->m) Matrix -- NEED THE QUATERNION VALUES PROVIDED BY THE SPACECRAFT

A_jm = np.array([[q1**2 - q2**2 - q3**2 + q4**4 ,       2*(q1*q2 + q3*q4)        ,        2*(q1*q3 - q2*q4)      ],
                 [      2*(q1*q2 - q3*q4)       , -q1**2 + q2**2 - q3**2 + q4**4 ,        2*(q2*q3 + q1*q4)      ],
                 [      2*(q1*q3 + q2*q4)       ,       2*(q2*q3 - q1*q4)        , -q1**2 - q2**2 + q3**2 + q4**4]])
print('A_jm:', '\n', A_jm)


# Transformation from j2000 to ITRS A(j->it)

A_jit = A_tiit*A_citi*A_jci
#print('A_jit:', '\n', A_jit)

# Transformation of M relative to the ITRS coordinate frame

A_itm = A_jm*A_jit.T
#print('A_itm:', '\n', A_itm)

### Working to establish A(m_n -> m_n+1)

psi_mn = ((((-0.0000000951*T_jcn1[1] + 0.000132851)*T_jcn1[1] - 0.00114045)*T_jcn1[1] - 1.0790069)*T_jcn1[1] + 5038.481507)*T_jcn1[1]
omega_mn = ((((0.0000003337*T_jcn1[1] - 0.000000467)*T_jcn1[1] - 0.00772503)*T_jcn1[1] + 0.0512623)*T_jcn1[1] - 0.025754)*T_jcn1[1] + epsilon_0
chi_mn = ((((-0.0000000560*T_jcn1[1] + 0.000170663)*T_jcn1[1] - 0.00121197)*T_jcn1[1] - 2.3814292)*T_jcn1[1] + 10.556403)*T_jcn1[1]

ps1mn = np.sin(epsilon_0) # The 'mn' indicates it is part of the m_n+1 block.
ps2mn = np.sin(-psi_mn)
ps3mn = np.sin(-omega_mn)
ps4mn = np.sin(chi_mn)
pc1mn = np.cos(epsilon_0)
pc2mn = np.cos(-psi_mn)
pc3mn = np.cos(-omega_mn)
pc4mn = np.cos(chi_mn)


P_mn = np.array([[ pc4mn*pc2mn - ps2mn*ps4mn*pc3mn,     pc4mn*ps2mn*pc1mn + ps4mn*pc3mn*pc2mn*pc1mn - ps1mn*ps4mn*ps3mn,  pc4mn*ps2mn*ps1mn + ps4mn*pc3mn*pc2mn*pc1mn + pc1mn*ps4mn*ps3mn],
                 [-ps4mn*pc2mn - ps2mn*pc4mn*pc3mn,    -ps4mn*ps2mn*pc1mn + pc4mn*pc3mn*pc2mn*pc1mn - ps1mn*pc4mn*ps3mn, -ps4mn*ps2mn*ps1mn + pc4mn*pc3mn*pc2mn*pc1mn + pc1mn+pc4mn*ps3mn],
                 [           ps2mn*ps3mn          ,                    -ps3mn*pc2mn*pc1mn - ps1mn*pc3mn                ,                -ps3mn*pc2mn*ps1mn + pc3mn*pc1mn                 ]])
#print('Precession Matrix for m_n+1:','\n', P_mn)

phi1mn =  908103.259872 + 538101628.688982*T_jcn1[1] # Mercury
phi2mn =  655127.283060 + 210664136.433548*T_jcn1[1] # Venus
phi3mn =  361679.244588 + 129597742.283429*T_jcn1[1] # Earth
phi4mn = 1279558.798488 +  68905077.493988*T_jcn1[1] # Mars
phi5mn =  123665.467464 +  10925660.377991*T_jcn1[1] # Jupiter
phi6mn =  180278.799480 +   4399609.855732*T_jcn1[1] # Saturn
phi7mn = 1130598.018396 +   1542481.193933*T_jcn1[1] # Uranus
phi8mn = 1095655.195728 +    786550.320744*T_jcn1[1] # Neptune
phi9mn = 5028.8200*T_jcn1[1] +      1.112022*T_jcn1[1]**2 # General precession in longitude approx.
phi10mn = 485868.249036 + 1717915923.2178*T_jcn1[1] + 31.8792*T_jcn1[1]**2 + 0.051635*T_jcn1[1]**3 - 0.00024470*T_jcn1[1]**4 # Mean anomaly of the Moon
phi11mn = 1287104.79305 + 129596581.0481*T_jcn1[1] - 0.5532*T_jcn1[1]**2 + 0.000136*T_jcn1[1]**3 - 0.00001149*T_jcn1[1]**4   # Mean anomaly of the Sun
phi12mn = 335779.526232 + 1739527262.8478*T_jcn1[1] - 12.7512*T_jcn1[1]**2 - 0.001037*T_jcn1[1]**3 + 0.00000417*T_jcn1[1]**4 # Mean argument of latitude of the Moon
phi13mn = 1072260.70369 + 1602961601.2090*T_jcn1[1] - 6.3706*T_jcn1[1]**2 + 0.006593*T_jcn1[1]**3 - 0.00003169*T_jcn1[1]**4  # Mean elongation of the Moon from the Sun
phi14mn = 450160.398036 - 6962890.5431*T_jcn1[1] + 7.4722*T_jcn1[1]**2 + 0.007702*T_jcn1[1]**3 - 0.00005939*T_jcn1[1]**4     # Mean longitude of the Moon's mean ascending node

phi_jmn = np.array([phi1mn, phi2mn, phi3mn, phi4mn, phi5mn, phi6mn, phi7mn, phi8mn, phi9mn, phi10mn, phi11mn, phi12mn, phi13mn, phi14mn])

def deltapsi_mn():
    phi_sum = []

    for i in range(1, 37): 
        phi_sum = sum(M_ij[i,:] * phi_jmn)
        #print(phi_sum)
            
        return sum((S_i + Sdot_i * T_jcn1[1]) * np.sin(phi_sum) + Cprime_i * np.cos(phi_sum))

def deltaepsilon_mn():
    phi_sum = []
    
    for i in range(1, 37):
        phi_sum = sum(M_ij[i,:] * phi_jmn)
        #print(phi_sum)
        
        return sum((C_i + Cdot_i * T_jcn1[1]) * np.cos(phi_sum) + Sprime_i * np.sin(phi_sum))
    
    
epsilon_mn = epsilon_0 - 46.836769*T_jcn1[1] - 0.0001831*T_jcn1[1]**2 + 0.00200340*T_jcn1[1]**3 - 0.000000576*T_jcn1[1]**4 - 0.0000000434*T_jcn1[1]**5
epsilon_prime_mn = epsilon_mn + deltaepsilon_mn()

#print('epsilon prime_mn:', epsilon_prime_mn)
#print('epsilon_mn:', epsilon_mn)

ns1mn = np.sin(epsilon_mn)
ns2mn = np.sin(-deltapsi_mn())
ns3mn = np.sin(epsilon_mn - deltaepsilon_mn())
nc1mn = np.cos(epsilon_mn)
nc2mn = np.cos(-deltapsi_mn())
nc3mn = np.cos(-epsilon_mn - deltaepsilon_mn())

N_mn = np.array([[      nc2mn     ,            ns2mn*nc1mn            ,             ns2mn*ns1mn           ],
                 [  -ns2mn*nc3mn  ,  nc3mn*nc2mn*nc1mn - ns1mn*ns3mn  ,  nc3mn*nc2mn*ns1mn + nc1mn*ns3mn  ],
                 [   ns2mn*nc3mn  , -ns3mn*nc2mn*nc1mn - ns1mn*nc3mn  , -ns3mn*nc2mn*ns1mn + nc3mn*nc1mn  ]])
#print('Nutation Matrix for m_n+1:','\n', N_mn)

## A(j->ci)mn matrix

zci_j1_mn = (N_mn*P_mn*B)[2,0] # X
zci_j2_mn = (N_mn*P_mn*B)[2,1] # Y
zci_j3_mn = (N_mn*P_mn*B)[2,2] # Z

zci_j_mn = np.array([zci_j1_mn, zci_j2_mn, zci_j3_mn])
#print('zci_j:', zci_j_mn)

xci_j1_mn = zci_j3_mn/np.sqrt((zci_j1_mn**2 + zci_j3_mn**2))
xci_j2_mn = 0
xci_j3_mn = -zci_j1_mn/np.sqrt(zci_j1_mn**2 + zci_j3_mn**2)

xci_j_mn = np.array([xci_j1_mn, xci_j2_mn, xci_j3_mn])
#print('xci_j_mn:', xci_j_mn)

yci_j_mn = zci_j_mn * xci_j_mn
#print('yci_j_mn:', yci_j_mn)

A_jci_mn = np.vstack((xci_j_mn, yci_j_mn, zci_j_mn))
#print('A(j->ci)mn:', '\n', A_jci_mn)

## A(ci->ti)mn matrix

tc_ci_n1 = tc_n1 - 0.001059 # delay of the time stamp event relative to the center of integration... 1.059 milliseconds + 50 ms for event shift

## interpolated ut1-utc for a 5 week period, 35 days, is 0.0036 @ 02/17/2025 16:42 PM

deltaUT1 = 0.0036
DeltaUT1C = deltaUT1 - nleaps
D_u_mn = (tc_ci_n1[1] + DeltaUT1C)/86400
#print('D_u_mn:', '\n', D_u_mn)

theta_Emn = (2*np.pi*(0.7790572732640+1.00273781191135448*D_u_mn)) ## May need to convert to radians using 1 arcsecond = 4.8481368E-6 radians
#print('Theta_E_mn:', '\n', theta_Emn)

A_citi_mn = np.array([[ np.cos(theta_Emn), np.sin(theta_Emn),  0  ],
                      [-np.sin(theta_Emn), np.cos(theta_Emn),  0  ],
                      [        0         ,        0         ,  1  ]])
#print('A(ci->ti)mn:', '\n', A_citi_mn)

A_jit_mn = A_tiit*A_citi_mn*A_jci_mn
#print('A_jit:', '\n', A_jit)

# Transformation of M relative to the ITRS coordinate frame

A_itm_mn = A_jm*A_jit_mn.T
#print('A_itm_mn:', '\n', A_itm)

A_deltam = A_itm_mn*A_itm.T
#print('A(mn->mn+1)','\n', A_deltam)

frac = (tc_ci[1] - tc[1])/(tc_n1[1] - tc[1])
#print('frac:', frac)

Theta_euler = np.arcsin(A[2,0]) * frac
Phi_euler = np.arcsin(-A[2,1]/np.cos(Theta_euler)) * frac
Psi_euler = np.arcsin(-A[1,0]/np.cos(Theta_euler)) * frac
#print(Theta_euler, Phi_euler, Psi_euler)

A_mnm = angle2dcm(Theta_euler, Psi_euler, Psi_euler,  input_units='rad', rotation_sequence='321')
#print('A(mn->m)','\n', A_mnm)

A_itm1 = A_mnm*A_itm_mn
#print('A(it->m)', '\n', A_itm1)

'''v_it = A_itm1.T * v_m''' 

#### HIGHLY IMPORTANT 
## using the spacecraft Att/orb packet, compute the position at tc and tc_n+1,. converting it to the ITRS coordinate from 
# then interpolating from tc and tc_n+1 using frac. 
'''We need to obtain the SC att/orb packet. It is not included in the netCDF4 file, also need to acquire the quaternion data'''

R_jn = 1 #this is what is determined from the cartesian plane when the ITRS coordinates of the Spacecraft are determined at Tc
R_jn1 = 1 #this is what is determined from the cartesian plane when the ITRS coordinates of the Spacecraft are determined at Tc_n+1

R_itn = A_jit*R_jn
R_it_n1 = A_jit *R_jn1

### Establishing the Coastline Identification Model -- GSHHS Database 
# land masses are described as polygons, where lat-long pairs are taken at the verices defined in the database

R_e = 6378140*1000 # 6,378,140 km equatorial radius of the earth. 
ff_e = 1/298.267 # flattening factor 

## EXPERIMENTAL -- attempting to you cartopy for gshhs integration.

#import cartopy.crs as ccrs
#import cartopy.feature as cfeature

#fig = plt.figure(figsize=(2,2))
#ax = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
#ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
#ax.add_feature(cfeature.GSHHSFeature(scale='auto', levels=[1], facecolor='none', edgecolor='black'))

#lon = np.linspace(-80, 80, 25)
#lat = np.linspace(30, 70, 25)
#lon2d, lat2d = np.meshgrid(lon, lat)

#data = np.cos(np.deg2rad(lat2d) * 4) + np.sin(np.deg2rad(lon2d * 4))

#ax = plt.axes(projection=ccrs.PlateCarree())
#ax.set_global()
#ax.coastlines()

#ax.contourf(lon, lat, data)
#plt.show()
'''
import pygmt
import math

Map = pygmt.Figure()
lat_map = 28.5744 # latitude for KSC  --  Will be updated to display resulting location from GSHHS Database.
long_map = -80.6520 # Longitude for KSC
Map.coast(
    region=[long_map-0.5, long_map+0.5, lat_map-0.5, lat_map+0.5],
    shorelines=True,
    land= 'lightgreen',
    water= 'lightblue',
    lakes = 'lightblue',
    rivers = 'lightblue',
    frame = ['a', '+tKennedy Space Center']
)
Map.show() #-- for repeated testing this is turned off.
'''
## using the lat and long from KSC to generate process. The end goal is to map and plot the point of the IR spike.

lat_map = 28.5744 # latitude for KSC  --  Will be updated to display resulting location from GSHHS Database.
long_map = -80.6520 # Longitude for KSC

lat_prime = np.arctan((1-ff_e)**2)*np.tan(np.deg2rad(lat_map)) # may need to do np.deg2rad for arctan... not sure how the math shakes out.
print('lat_prime:', lat_prime)
P_e = R_e*(1-ff_e)/(np.sqrt(1-ff_e*(2-ff_e)*math.cos(np.deg2rad(lat_map)**2)))
print('P_e:', P_e)
delta_theta = np.rad2deg(0.125*6.300387487/86400) # where omega_e is 6.300387487 rad/day
Pe_it = np.array([P_e*np.cos(np.deg2rad(lat_prime))*np.cos(np.deg2rad(long_map-delta_theta)),
                  P_e*np.cos(np.deg2rad(lat_prime))*np.sin(np.deg2rad(long_map-delta_theta)), 
                  P_e*np.sin(np.deg2rad(lat_prime))])
print('Pe_it:', Pe_it)
