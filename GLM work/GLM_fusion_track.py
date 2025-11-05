from astropy.io import fits
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, EarthLocation
from astropy.time import Time
import astropy

import cv2

from datetime import datetime

from glob import glob

import matplotlib.pyplot as plt

import numpy as np


GLM_SIZE = (1370, 1300) # Height, Wdith
#GLM_SIZE = (1216, 1216) # Because we crop the frame

def convert_millis(ms):
    seconds = (ms / 1000) % 60
    minutes = (ms // (1000 * 60)) % 60
    hours = (ms // (1000 * 60 *60)) 
    return hours, minutes, seconds


def simple_plots():

    G16 = np.loadtxt("./BOLIDE_10032023_GE_locations_test.txt", delimiter=",")
    G18 = np.loadtxt("./BOLIDE_10032023_GW_locations_test.txt", delimiter=",")



    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    print(G18.shape)


    ax.scatter3D(G16[:, 1], G16[:, 2], G16[:, 0], label="G16")
    ax.scatter3D(G18[:, 1], G18[:, 2], G18[:, 0], label="G18")

    ax.set_xlabel("X Postion (pixel)")
    ax.set_ylabel("Y Position (pixel)")
    ax.set_zlabel("Time (ms)")

    fig.suptitle("Bolide 10/03/2023 13:21:06.441 UTC")
    fig.legend()

    plt.show()

    fig, ax = plt.subplots(2,2)

    ax[0,0].plot(G16[:,0], G16[:,1])
    ax[0,0].set_title("G16 X Position")
    ax[0,1].plot(G16[:,0], G16[:,2])
    ax[0,1].set_title("G16 Y Position")

    ax[1,0].plot(G18[:,0], G18[:,1])
    ax[1,0].set_title("G18 X Position")
    ax[1,1].plot(G18[:,0], G18[:,2])
    ax[1,1].set_title("G18 Y Position")

    plt.show()



def GLM_Triangulate(YYYY=2023, DOY=276, EAST="G16_2021_323", WEST="G17_2021_323"):

    # Configure the correct Day
    date = datetime.strptime(f"{YYYY}-{DOY}", "%Y-%j").strftime("%Y-%m-%d")

    YYYY, MM, DD = date.split("-")
    YYYY = int(YYYY)
    MM = int(MM)
    DD = int(DD)
    print(YYYY, MM, DD)
    # Get the trinagulation 

    # First set is create a camera matrix P for each satellite


    ## Specified Variables :
    f_ref = 133.974 #mm section 2.1
    c_EFL = -0.00515 #mm/deg C section 2.1
    T_LAref = 30 #deg C Section 2.1
    k_d = -4.614641E-5 # mm/mm^3 section 2.1

    # Following values are from the INRO of GLM document
    focal_mm = 133.974 / 1000 # This is actually a function based on LA temp, but we just need a rough estimate for now
    sensor_width_mm = 17664 #mm
    sensor_width_pixels = GLM_SIZE[1]
    sensor_height_mm = 17800 #mm
    sensor_height_pixels = GLM_SIZE[0]

    avg_px_per_mm = ((GLM_SIZE[0]+GLM_SIZE[1]) / 2) / ((sensor_height_mm + sensor_width_mm) / 2)

    cx = 649.5
    cy = 685.5

    #cx = 608
    #cy = 608

    fx = (focal_mm / (sensor_width_mm / sensor_width_pixels ) ) 
    fy = (focal_mm / (sensor_height_mm / sensor_height_pixels) ) 
    #fx  = focal_mm  * avg_px_per_mm
    #fy = focal_mm * avg_px_per_mm

    # The projection matrix is the same for each camera
    GOES_K = [
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ]


    # Each of 16, 17,18,19 have different rotations because they are offset from the world (0,0,0) (which for GLM is the center of the earth)
    # Rotation is based on the quantarions (that will be expresed in pixels)


    g16_bg_file = glob(f"./GLM_DATA/BACKGROUNDS/{EAST}/*.fits")[0]
    G16_BG = fits.open(g16_bg_file)
    print(G16_BG[0].header)
    header = G16_BG[0].header
    G16_BG.close()

    T_LA_GE = header["CCDTEMP"]
    GE_fref = f_ref + c_EFL * (T_LA_GE - T_LAref)


    R = np.asarray([header["POS1"], header["POS2"], header["POS3"]])  # Keep the transform in ECI J2000 so the output is in those units
    #R *= 1000 # Convert to mm
    #R *= avg_px_per_mm # Convert from mm to pixels 

    q = [header["QUAT1"], header["QUAT2"], header["QUAT3"], header["QUAT4"]]
    q1, q2, q3, q4 = q

    # Get G16 LLA
    HH = int(header["TIME"][9:11])
    mm = int(header["TIME"][11:13])
    ss = int(float(header["TIME"][13:]))
    print(HH, mm, ss)
    obstime = Time(datetime(YYYY, MM, DD, HH, mm, ss), scale="utc")
    gcrs = GCRS(CartesianRepresentation(R[0] * astropy.units.m, R[1] * astropy.units.m , R[2] * astropy.units.m), obstime=obstime)
    itrs = gcrs.transform_to(ITRS(obstime=obstime))
    earth_loc = EarthLocation(itrs.x, itrs.y, itrs.z)
    lon, lat, alt = earth_loc.geodetic
    lat, lon, alt = lat.to(astropy.units.deg).value, lon.to(astropy.units.deg).value, alt.to(astropy.units.km).value
   # assert True == False


    G16_Rt = [
        #[2*(q0**2+q1**2)-1, 2*(q1*q2 - q0*q3),   2*(q1*q3 + q0*q2),   R[0]],
        #[2*(q1*q2 + q0*q3), 2*(q0**2 + q2**2)-1, 2*(q2*q3 - q0*q1),   R[1]],
        #[2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1),   2*(q0**2 + q3**2)-1, R[2]]

        [q1**2 - q2**2 - q3**2 + q4**2, 2*(q1*q2 + q3*q4), 2*(q1*q3 - q2*q4),  R[0]], # 0
        [2*(q1*q2 - q3*q4), -q1**2 + q2**2 - q3**2 + q4**2, 2*(q2*q3 + q1*q4), R[2]], # 2
        [2*(q1*q3 + q2*q4), 2*(q2*q3 - q1*q4), -q1**2 - q2**2 + q3**2 + q4**2, R[1]], # 1
    ]

    G16_P = np.matmul(GOES_K, G16_Rt)

    g18_bg_file = glob(f"./GLM_DATA/BACKGROUNDS/{WEST}/*.fits")[0]
    G18_BG = fits.open(g18_bg_file)
    print(G18_BG[0].header)
    header = G18_BG[0].header
    G18_BG.close()

    T_LA_GW = header["CCDTEMP"]
    GW_fref = f_ref + c_EFL * (T_LA_GW - T_LAref)
    
    R = np.asarray([header["POS1"], header["POS2"], header["POS3"]]) 
    #R *= 1000 # Convert to mm
    #R *= avg_px_per_mm # Convert from mm to pixels 

    q = [header["QUAT1"], header["QUAT2"], header["QUAT3"], header["QUAT4"]]
    q1, q2, q3, q4 = q

    # Get G18 LLA
    HH = int(header["TIME"][9:11])
    mm = int(header["TIME"][11:13])
    ss = int(float(header["TIME"][13:]))
    #print(HH, mm, ss)
    obstime = Time(datetime(YYYY, MM, DD, HH, mm, ss), scale="utc")
    gcrs = GCRS(CartesianRepresentation(R[0] * astropy.units.m, R[1] * astropy.units.m , R[2] * astropy.units.m), obstime=obstime)
    itrs = gcrs.transform_to(ITRS(obstime=obstime))
    earth_loc = EarthLocation(itrs.x, itrs.y, itrs.z)
    lon, lat, alt = earth_loc.geodetic
    lat, lon, alt = lat.to(astropy.units.deg).value, lon.to(astropy.units.deg).value, alt.to(astropy.units.km).value


    G18_Rt = [
        [q1**2 - q2**2 - q3**2 + q4**2, 2*(q1*q2 + q3*q4), 2*(q1*q3 - q2*q4),  R[0]],
        [2*(q1*q2 - q3*q4), -q1**2 + q2**2 - q3**2 + q4**2, 2*(q2*q3 + q1*q4), R[2]],
        [2*(q1*q3 + q2*q4), 2*(q2*q3 - q1*q4), -q1**2 - q2**2 + q3**2 + q4**2, R[1]],
    ]

    G18_P = np.matmul(GOES_K, G18_Rt)
    


    #G16_Loc = np.loadtxt("./Bolide_2021_323/BOLIDE_2021_323_GE_locations.txt", delimiter=",")
    #G18_Loc = np.loadtxt("./Bolide_2021_323/BOLIDE_2021_323_GW_locations.txt", delimiter=",")
    
    G16_Loc = np.loadtxt(f"./BOLIDE_{EAST[4:]}_GE_test_locations.txt", delimiter=",")
    G18_Loc = np.loadtxt(f"./BOLIDE_{WEST[4:]}_GW_test_locations.txt", delimiter=",")

    # Triangulated coordinates
    tri = []
    LLA = []

    # Iterate over the unique times so we can correlate pixels at the correct time
    times = np.unique(G16_Loc[:, 0])
    #print(times)
    shared_times = []
    for t in times:
        g16_idx = np.where(G16_Loc[:, 0] == t)[0]
        g18_idx = np.where(G18_Loc[:, 0] == t)[0]
        
        # Make sure we have data at the same time
        if g16_idx.size == 0 or g18_idx.size == 0:
            continue

        shared_times.append(t)

        g16_xy = G16_Loc[g16_idx, 1:]
        # Take the average positions in each dimension so that both images have the same number of points to triangulate
        g16_xy = [
            [np.mean(g16_xy[:,1]) + 42],
            [np.mean(g16_xy[:,0]) + 77],
            [1]
        ]
        g18_xy = G18_Loc[g18_idx, 1:]
        g18_xy = [
            [np.mean(g18_xy[:,1]) + 42],
            [np.mean(g18_xy[:,0]) + 77],


            [1]
        ]

        
        g16_xy = np.asarray(g16_xy)
        g18_xy = np.asarray(g18_xy)

        if False:
            # convert from pixel frame ref to glm frame ref
            g16_xy = g16_xy*f_ref/GE_fref

            #r_mcref = np.sqrt(pow(g16_xy[0],2)+pow(g16_xy[1],2))
            #r_iref=r_mcref-(k_d*(pow(r_mcref, 3)))
            #theta = np.arctan(g16_xy[1]/g16_xy[0])
            #x_irefprime = r_iref * np.cos(theta)
            #y_irefprime = r_iref * np.sin(theta)

            #x_iref = x_irefprime + cx
            #y_iref = y_irefprime + cy

            #g16_xy[0] = x_iref*GE_fref/f_ref
            #g16_xy[1] = y_iref*GE_fref/f_ref

            g18_xy = g18_xy*f_ref/GW_fref

            #r_mcref = np.sqrt(pow(g18_xy[0],2)+pow(g18_xy[1],2))
            #r_iref=r_mcref-(k_d*(pow(r_mcref, 3)))
            #theta = np.arctan(g18_xy[1]/g18_xy[0])
            #x_irefprime = r_iref * np.cos(theta)
            #y_irefprime = r_iref * np.sin(theta)

            #x_iref = x_irefprime + cx
            #y_iref = y_irefprime + cy

            #g18_xy[0] = x_iref*GW_fref/f_ref
            #g18_xy[1] = y_iref*GW_fref/f_ref



        #print(g16_xy)
        #print(g18_xy)

        POINTS = cv2.triangulatePoints(G16_P, G18_P, g16_xy[:2], g18_xy[:2])
        #print(POINTS)
        POINTS *= (1/POINTS[3]) # Multiply is the faster op
        tri.append([t, POINTS[0][0], POINTS[1][0], POINTS[2][0]]) # time, X, Y, Z
        
        # Get the time to convert from ECI to LLA
        HH, mm, SS = convert_millis(t)
        #print(HH, mm, SS)
        obstime = Time(datetime(YYYY, MM, DD, int(HH), int(mm), int(SS)), format="datetime", scale="utc")
        gcrs = GCRS(CartesianRepresentation(POINTS[0][0] * astropy.units.m, POINTS[1][0] * astropy.units.m , POINTS[2][0] * astropy.units.m), obstime=obstime)
        itrs = gcrs.transform_to(ITRS(obstime=obstime))
        earth_loc = EarthLocation(itrs.x, itrs.y, itrs.z)
        lon, lat, alt = earth_loc.geodetic
        LLA.append([t, lat.to(astropy.units.deg).value, lon.to(astropy.units.deg).value, alt.to(astropy.units.km).value])




    LLA = np.asarray(LLA)
    tri = np.asarray(tri)
    # Now we can work on plotting the 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    #print(tri)
    ax.scatter3D(tri[:, 1], tri[:, 2], tri[:, 3])
    ax.plot(tri[:, 1], tri[:, 2], tri[:, 3])

    ax.invert_zaxis()
    ax.ticklabel_format(useOffset=False)




    HH, mm, ss = convert_millis(shared_times[0])
    #print(HH, mm, ss)
    fig.suptitle(f"Bolide {date} {int(HH):02d}:{int(MM):02d}:{ss:06.3f} UTC 3D Fusion Track")
    ax.set_xlabel("X position (m from center of Earth [ECI])")
    ax.set_ylabel("Y position (m from center of Earth [ECI])")
    ax.set_zlabel("Z position (m from center of Earth [ECI])")
    plt.show()


    #print(np.unique(LLA, axis=1))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter3D(LLA[:, 1], LLA[:, 2], LLA[:,3])
    ax.plot(LLA[:, 1], LLA[:, 2], LLA[:,3])


    fig.suptitle(f"Bolide {date} {int(HH):02d}:{int(MM):02d}:{ss:06.3f} UTC 3D Fusion Track")
    ax.set_xlabel("Lat position (degrees)")
    ax.set_ylabel("Lon position (degrees)")
    ax.set_zlabel("Alt position (km from sea level)")
    
    ax.ticklabel_format(useOffset=False)
    
    plt.savefig(f"BOLIDE_{YYYY}_{DOY}_fusiontrack.png")
    plt.show()












if __name__ == '__main__':

    #GLM_Triangulate(YYYY=2023, DOY=276, EAST="G16_2023_276", WEST="G18_2023_276")

    GLM_Triangulate(YYYY=2019, DOY=213, EAST="G16_2019_213", WEST="G17_2019_213")

    #GLM_Triangulate(YYYY=2021, DOY=323, EAST="G16_2021_323", WEST="G17_2021_323")