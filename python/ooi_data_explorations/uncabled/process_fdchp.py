import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import integrate, interpolate
from scipy.signal import detrend, filtfilt


PADLENGTH = 12 # (3*(np.max([len(bhi), len(ahi)]) - 1) == 12)


def process_fdchp(raw_data, latitude, anemometer_relative_position, tc1=20, tcwave=30, despike=True, despikecompass=False, flux_filepath=None):
    # JBE 06/29 JW 11Aug2014
    # 1.5 JW 3Sep2014
    # 2.0 JBE 06/12/2021
    # 3.0 JBE 05/11/2022
    # 3.5 JBE 01/02/2023
    # 4.0 JKP 2024-07-26 (pythonification)

    """
    tc1 is the cutoff period that determines where the pitch and roll Euler angles are 
    determined by the tilt of the accelerometers at low frequencies versus the 
    integrated angular rates at high frequencies.  This is known as complementary filtering

    tcwave is the same cutoff period as tc1, but for the wave measurements, which requires a large cutoff period
    """
    
    # Note: raw_data is expected to be in NWU coordinates, which is the default returned by the fdchp_utils.particles_to_pandas() function
    dt = 0.1 # Sampling period for FDCHP; 10 Hz
    fs = 1/dt # Sampling frequency for Windmaster

    # Define constants for filters
    tc2=tc1
    Rvec=np.zeros(3)             # Distance vector between the sonic anemometer and motion sensors
    Rvec[2]=0.85   # z offset
    FREEZING_POINT_K = 273.15

    G = gravity_from_latitude(latitude)

    version_number=4.0
    status_val = 1 #uint32

    #JBE Redefine files for 10 Hz and tc1=12 or 15 or 20.  Use digits(16) and
    #vpa(ahiwaves)
    if tc1==12:        #JBE Redefine filters for 10 Hz and tc1=12 with ludo=1
        ahi=[1.000000000000000,  -3.869797539975555,   5.617802044587569,  -3.625896801659086,   0.877898078061702]
        bhi=[0.9, 36962154017745,  -3.747848616070978,   5.621772924106467,  -3.747848616070978,   0.936962154017745]
    elif tc1==15:           #JBE Redefine filters for 10 Hz and tc1=15 with ludo=1
        ahi=[1.000000000000000,  -3.895833876325376,   5.692892648957240,  -3.698121672490409,   0.901065298297354]
        bhi=[0.949244593504399,  -3.796978374017595,   5.695467561026392,  -3.796978374017595,   0.949244593504399]
    else:
        ahi=[1.000000000000000,  -3.921872982100935,   5.768656400578301,  -3.771625137827138,   0.924842488052324]
        bhi=[0.961687313034919,  -3.846749252139674,   5.770123878209512,  -3.846749252139674,   0.961687313034919]

    #JBE Redefine files for 10 Hz and fcwaves=1/40;
    if tcwave==40:
        ahiwaves = [1.000000000000000,  -3.960935321365416,  5.883567180614652, -3.884319737084527,  0.961687926819144]
        bhiwaves = [0.980656885367734,  -3.922627541470935,  5.883941312206403, -3.922627541470935,  0.980656885367734]
    else:
        #JBE Redefine files for 10 Hz and fcwaves=1/30;
        ahiwaves = [1.0000000000000000, -3.947914166208924,  5.845094680183927, -3.846426389902994,  0.9492460297427443]
        bhiwaves = [0.9742925791274119, -3.897170316509647,  5.845755474764471, -3.897170316509647,  0.9742925791274119]

    num_datapoints=len(raw_data)
    mean_time = raw_data['time'].mean()

    #Some indices to be used later
    edge = int(np.fix( 20 * fs))
    tot=int(len(raw_data)-edge*2)
    start_index = int(edge)
    end_index = int(start_index+tot-1)
    
    if tot <= 0:
        # The datset is too short
        print("Unexpectedly short dataset in process_fdchp with shape: {}".format(raw_data.shape))
        return None
    
    #UNITS Velocities seem to be stored as cm/s, and must be converted to m/s
    #TODO: is this right for the case where this already represents a temperature?
    # This is the sonic temperature computed from speed of sound
    sos=raw_data['fdchp_speed_of_sound_sonic'].to_numpy()*0.01 
    if np.nanmedian(sos) < 50:
        Tv = sos
    else:
        # We convert SoS to temperature T1=C1^2/403 
        # print("adjusting speed of sound")
        Tv = (sos**2)/403.0 - FREEZING_POINT_K

    if despike:
        Tv = despikesimple(Tv)

    # sonic velocities
     # These are the 3-axis sonic wind velocities
     # Apparently stored as cm/s on disk; converted here to m/s.
    sonics=raw_data[['fdchp_wind_x', 'fdchp_wind_y', 'fdchp_wind_z']].to_numpy()*0.01

    if despike:                                # Remove obvious spikes in the data
        sonics = despikesimple(sonics)


    Tavg = np.nanmean(Tv)
    Tstd = np.nanstd(Tv)
    Tvmax = np.nanmax(Tv)
    Tvmin = np.nanmin(Tv)

    #***********************************************
    # Deal with the compass 
    # JBE Fill in bad points first units and signs 
    # UNITS-Roll, pitch and yaw are in radians
    #***********************************************

    compass=raw_data['fdchp_heading']  # heading
    roll=raw_data['fdchp_roll']
    pitch=raw_data['fdchp_pitch'] 


    gx=np.cos(compass)
    gy=np.sin(compass)
    compcos = np.mean(gx)
    compsin = np.mean(gy)
    compavg = np.arctan2(compsin,compcos)
    if compavg < 0:
        compavg=compavg + 2.0*np.pi

    if despikecompass:
        gx = despikesimple(gx)
        gy = despikesimple(gy)
        gsmooth=np.arctan2(gy,gx)
        gyro=gsmooth
    else:
        gyro=compass

    #***********************************************
    # Then the angular rates
    # UNITS - Rates are in radian/sec
    #***********************************************
    # This will hold the angular rates
    ang_rates = raw_data[['fdchp_x_ang_rate','fdchp_y_ang_rate','fdchp_z_ang_rate']].to_numpy()
    if despike:
        ang_rates = despikesimple(ang_rates)

    #***********************************************
    # Then the accelerations
    #***********************************************
    # This will hold the accelerations
    platform_accelerations = raw_data[['fdchp_x_accel_g', 'fdchp_y_accel_g', 'fdchp_z_accel_g']].to_numpy()
    if despike:
        platform_accelerations = despikesimple(platform_accelerations)

    #***************************************************************
    # We no longer force the mean accelerometers to equal gravity
    # UNITS#8 convert platform_accelerations accels to m/s^2
    #***************************************************************
    platform_accelerations=platform_accelerations*G

    #*********************************************
    #  Waves first
    #*********************************************
    # rpy = euler angles; dr = ang_rates rotated
    #TODO: change rpy and dr shapes to be 12000x3 instead of 3x12000
    rpy,dr = get_euler_angles(ahiwaves,bhiwaves,fs,platform_accelerations,ang_rates,gyro,G)  # euler angles are right-handed
    #[acc, uvwplatwave, xyzplat] = get_platform_vel_pos(bhiwaves,ahiwaves,fs,platform_accelerations,rpy,G);
    uvwplatwave,xyzplat = heave_calc(dr,rpy,platform_accelerations,fs,bhiwaves,ahiwaves,anemometer_relative_position,G)

    #*********************************************
    #  Then sonics
    #*********************************************
    euler,dr = get_euler_angles(ahi,bhi,fs,platform_accelerations,ang_rates,gyro,G)    # euler angles are right-handed
    acc, uvwplat, nope = get_platform_vel_pos(bhi,ahi,fs,platform_accelerations,euler,G)
    uvw,uvwr,uvwrot = sonic(sonics,dr,euler,uvwplat,Rvec)

    UVW = uvw[start_index:end_index, :]
    Ts = Tv[start_index:end_index]

    heave = detrend(xyzplat[start_index:end_index, 2]) #z-axis to find heave
    waveheight = 4*np.std(heave)

    U = np.mean(UVW[:, 0])
    V = np.mean(UVW[:, 1])
    W = np.mean(UVW[:, 2])

    # UNITS - wdir is in radians
    wdir = np.arctan2(V,-U)
    if wdir < 0:
        wdir = wdir + 2*np.pi

    u, alpha, beta = alignwind(UVW)

    # Relative to Earth
    Uearth=np.mean(u[0:tot, 0])

    for col in np.arange(u.shape[-1]):    # TODO: check axis
        u[:, col] = detrend(u[:, col])
    
    # Ts=detrend(Ts)
    
    Ts = detrend(Ts)

    fluxes = np.zeros(3)
    uwavg = np.mean(u[0:tot, 2]*u[0:tot, 0]) # Along-Wind component of momentum flux
    vwavg = np.mean(u[0:tot, 2]*u[0:tot, 1]) # Cross-Wind component of momentum flux
    wTavg = np.mean(u[0:tot, 2]*Ts) # Heat flux (vertical velocity times temp)
    #JBE Fixed a bug here by replacing mean with std

    Ucorstd = np.std(u[0:tot, 0])
    Vcorstd = np.mean(u[0:tot, 1]*u[0:tot, 1])
    Wcorstd = np.mean(u[0:tot, 2]*u[0:tot, 2])

    fluxes[0]=uwavg
    fluxes[1]=vwavg
    fluxes[2]=wTavg

    Uavg=np.mean(sonics[:, 0])
    Vavg=np.mean(sonics[:, 1])
    Wavg=np.mean(sonics[:, 2])
    rdir=np.arctan2(Vavg, Uavg) #TODO: not used?

    Umax=np.max(sonics[:, 0])
    Vmax=np.max(sonics[:, 1])
    Wmax=np.max(sonics[:, 2])
    Umin=np.min(sonics[:, 0])
    Vmin=np.min(sonics[:, 1])
    Wmin=np.min(sonics[:, 2])

    #JBE  This is not the standard deviation. 
    #Ustd = mean(abs(sonics(1,1:L)-Uavg));
    Ustd = np.std(sonics[:, 0])
    Vstd = np.std(sonics[:, 1])
    Wstd = np.std(sonics[:, 2])
    
    compstd=np.std(np.unwrap(compass))
    compmin=compass.min()
    compmax=compass.max()
    
    gx=np.cos(compass)
    gy=np.sin(compass)
    compcos = np.mean(gx)
    compsin = np.mean(gy)
    compavg = np.arctan2(compsin, compcos)
    
    mean_ang_rate=np.mean(ang_rates, axis=0)
    std_ang_rate=np.std(ang_rates, axis=0)

    Phiavg = mean_ang_rate[0]
    Thetaavg = mean_ang_rate[1]
    Psiavg = mean_ang_rate[2]
    Phistd = std_ang_rate[0]
    Thetastd = std_ang_rate[1]
    Psistd = std_ang_rate[2]
    Phimax = np.max(ang_rates[:, 0])
    Thetamax = np.max(ang_rates[:, 1])
    Psimax = np.max(ang_rates[:, 2])
    Phimin = np.min(ang_rates[:, 0])
    Thetamin = np.min(ang_rates[:, 1])
    Psimin = np.min(ang_rates[:, 2])
    
    gcomp = np.mean(platform_accelerations, axis=0)/G

    Axavg = gcomp[0]
    Ayavg = gcomp[1]
    Azavg = gcomp[2]
    Axmax=np.max(platform_accelerations[:, 0])/G
    Aymax=np.max(platform_accelerations[:, 1])/G
    Azmax=np.max(platform_accelerations[:, 2])/G
    Axmin=np.min(platform_accelerations[:, 0])/G
    Aymin=np.min(platform_accelerations[:, 1])/G
    Azmin=np.min(platform_accelerations[:, 2])/G

    gstd = np.std(platform_accelerations, axis=0)/G
    Axstd = gstd[0]/G
    Aystd = gstd[1]/G
    Azstd = gstd[2]/G
    
    # UNITS - Cal mean roll and pitch in radians
    Pitch = np.mean(pitch)
    Roll = np.mean(roll)
    Pitchstd = np.std(pitch)
    Rollstd = np.std(roll)
    Pitchmax = np.max(pitch)
    Rollmax = np.max(roll)
    Pitchmin = np.min(pitch)
    Rollmin = np.min(roll)

    version_number = 0
    status_val = 0

    flux_out = [
        mean_time,
        version_number,
        status_val,
        Uavg,
        Vavg,
        Wavg,
        Tavg,
        Ustd,
        Vstd,
        Wstd,
        Tstd,
        Umax,
        Vmax,
        Wmax,
        Tvmax,
        Umin,
        Vmin,
        Wmin,
        Tvmin,
        Axavg, #20
        Ayavg,
        Azavg,
        Axstd,
        Aystd,
        Azstd,
        Axmax,
        Aymax,
        Azmax,
        Axmin,
        Aymin,
        Azmin,
        Phiavg, #32
        Thetaavg,
        Psiavg,
        Phistd,
        Thetastd,
        Psistd,
        Phimax,
        Thetamax,
        Psimax,
        Phimin,
        Thetamin,
        Psimin,
        compavg,
        Pitch,
        Roll,
        compstd,
        Pitchstd,
        Rollstd,
        compmax,
        Pitchmax,
        Rollmax,
        compmin,
        Pitchmin,
        Rollmin,
        U,
        V,
        W,
        Ucorstd,
        Vcorstd,
        Wcorstd,
        Uearth,
        uwavg,
        vwavg,
        wTavg,
        waveheight
    ]

    file_path = flux_filepath if flux_filepath else "fluxes"
    write_metrics(flux_out, file_path=file_path)
    # calculate_and_write_metrics(mean_time, sonics, ang_rates, platform_accelerations, compass, roll, pitch, waveheight, fluxes)
    
    return (fluxes, Uearth, waveheight)


def process_fdchp_xarray(dataset, latitude, anemometer_relative_position, tc1=20, tcwave=30, despike=True, despikecompass=False, flux_filepath=None):
    # JBE 06/29 JW 11Aug2014
    # 1.5 JW 3Sep2014
    # 2.0 JBE 06/12/2021
    # 3.0 JBE 05/11/2022
    # 3.5 JBE 01/02/2023
    # 4.0 JKP 2024-07-26 (pythonification)
    # 5.0 JKP 2025-04-03 (xarray)

    """
    tc1 is the cutoff period that determines where the pitch and roll Euler angles are 
    determined by the tilt of the accelerometers at low frequencies versus the 
    integrated angular rates at high frequencies.  This is known as complementary filtering

    tcwave is the same cutoff period as tc1, but for the wave measurements, which requires a large cutoff period
    """
    
    # Note: dataset is expected to be in NWU coordinates, which is the default returned by the fdchp_utils.particles_to_pandas() function
    dt = 0.1 # Sampling period for FDCHP; 10 Hz
    fs = 1/dt # Sampling frequency for Windmaster

    # Define constants for filters
    Rvec=np.zeros(3)    # Distance vector between the sonic anemometer and motion sensors
    Rvec[2]=0.85        # z offset
    FREEZING_POINT_K = 273.15

    G = gravity_from_latitude(latitude)

    version_number=4.0
    status_val = 1 #uint32

    #JBE Redefine files for 10 Hz and tc1=12 or 15 or 20.  Use digits(16) and
    #vpa(ahiwaves)
    if tc1==12:        #JBE Redefine filters for 10 Hz and tc1=12 with ludo=1
        ahi=[1.000000000000000,  -3.869797539975555,   5.617802044587569,  -3.625896801659086,   0.877898078061702]
        bhi=[0.9, 36962154017745,  -3.747848616070978,   5.621772924106467,  -3.747848616070978,   0.936962154017745]
    elif tc1==15:           #JBE Redefine filters for 10 Hz and tc1=15 with ludo=1
        ahi=[1.000000000000000,  -3.895833876325376,   5.692892648957240,  -3.698121672490409,   0.901065298297354]
        bhi=[0.949244593504399,  -3.796978374017595,   5.695467561026392,  -3.796978374017595,   0.949244593504399]
    else:
        ahi=[1.000000000000000,  -3.921872982100935,   5.768656400578301,  -3.771625137827138,   0.924842488052324]
        bhi=[0.961687313034919,  -3.846749252139674,   5.770123878209512,  -3.846749252139674,   0.961687313034919]

    #JBE Redefine files for 10 Hz and fcwaves=1/40;
    if tcwave==40:
        ahiwaves = [1.000000000000000,  -3.960935321365416,  5.883567180614652, -3.884319737084527,  0.961687926819144]
        bhiwaves = [0.980656885367734,  -3.922627541470935,  5.883941312206403, -3.922627541470935,  0.980656885367734]
    else:
        #JBE Redefine files for 10 Hz and fcwaves=1/30;
        ahiwaves = [1.0000000000000000, -3.947914166208924,  5.845094680183927, -3.846426389902994,  0.9492460297427443]
        bhiwaves = [0.9742925791274119, -3.897170316509647,  5.845755474764471, -3.897170316509647,  0.9742925791274119]

    num_datapoints=len(dataset)
    mean_time = dataset['time'].mean()

    #Some indices to be used later
    edge = int(np.fix( 20 * fs))
    tot=int(len(dataset['time'])-edge*2)
    start_index = int(edge)
    end_index = int(start_index+tot-1)
    
    if tot <= 0:
        # The datset is too short
        print("Unexpectedly short dataset in process_fdchp with length: {}".format(dataset['time'].shape))
        return None
    
    #UNITS Velocities seem to be stored as cm/s, and must be converted to m/s
    #TODO: is this right for the case where this already represents a temperature?
    # This is the sonic temperature computed from speed of sound
    sos=dataset['fdchp_speed_of_sound_sonic'].to_numpy().T*0.01 
    if np.nanmedian(sos) < 50:
        Tv = sos
    else:
        # We convert SoS to temperature T1=C1^2/403 
        # print("adjusting speed of sound")
        Tv = (sos**2)/403.0 - FREEZING_POINT_K

    if despike:
        Tv = despikesimple(Tv)

    # sonic velocities
     # These are the 3-axis sonic wind velocities
     # Apparently stored as cm/s on disk; converted here to m/s.
    sonics=dataset[['fdchp_wind_x', 'fdchp_wind_y', 'fdchp_wind_z']].to_array().to_numpy().T*0.01

    if despike:                                # Remove obvious spikes in the data
        sonics = despikesimple(sonics)


    Tavg = np.nanmean(Tv)
    Tstd = np.nanstd(Tv)
    Tvmax = np.nanmax(Tv)
    Tvmin = np.nanmin(Tv)

    #***********************************************
    # Deal with the compass 
    # JBE Fill in bad points first units and signs 
    # UNITS-Roll, pitch and yaw are in radians
    #***********************************************

    compass=dataset['fdchp_heading'].T  # heading
    roll=dataset['fdchp_roll'].T
    pitch=dataset['fdchp_pitch'].T 


    gx=np.cos(compass)
    gy=np.sin(compass)
    compcos = np.mean(gx)
    compsin = np.mean(gy)
    compavg = np.arctan2(compsin,compcos)
    if compavg < 0:
        compavg=compavg + 2.0*np.pi

    if despikecompass:
        gx = despikesimple(gx)
        gy = despikesimple(gy)
        gsmooth=np.arctan2(gy,gx)
        gyro=gsmooth
    else:
        gyro=compass

    #***********************************************
    # Then the angular rates
    # UNITS - Rates are in radian/sec
    #***********************************************
    # This will hold the angular rates
    ang_rates = dataset[['fdchp_x_ang_rate','fdchp_y_ang_rate','fdchp_z_ang_rate']].to_array().to_numpy().T
    if despike:
        ang_rates = despikesimple(ang_rates)

    #***********************************************
    # Then the accelerations
    #***********************************************
    # This will hold the accelerations
    platform_accelerations = dataset[['fdchp_x_accel_g', 'fdchp_y_accel_g', 'fdchp_z_accel_g']].to_array().to_numpy().T
    if despike:
        platform_accelerations = despikesimple(platform_accelerations)

    #***************************************************************
    # We no longer force the mean accelerometers to equal gravity
    # UNITS#8 convert platform_accelerations accels to m/s^2
    #***************************************************************
    platform_accelerations=platform_accelerations*G

    #*********************************************
    #  Waves first
    #*********************************************
    # rpy = euler angles; dr = ang_rates rotated
    #TODO: change rpy and dr shapes to be 12000x3 instead of 3x12000
    rpy,dr = get_euler_angles(ahiwaves,bhiwaves,fs,platform_accelerations,ang_rates,gyro,G)  # euler angles are right-handed
    #[acc, uvwplatwave, xyzplat] = get_platform_vel_pos(bhiwaves,ahiwaves,fs,platform_accelerations,rpy,G);
    uvwplatwave,xyzplat = heave_calc(dr,rpy,platform_accelerations,fs,bhiwaves,ahiwaves,anemometer_relative_position,G)

    #*********************************************
    #  Then sonics
    #*********************************************
    euler,dr = get_euler_angles(ahi,bhi,fs,platform_accelerations,ang_rates,gyro,G)    # euler angles are right-handed
    acc, uvwplat, nope = get_platform_vel_pos(bhi,ahi,fs,platform_accelerations,euler,G)
    uvw,uvwr,uvwrot = sonic(sonics,dr,euler,uvwplat,Rvec)

    UVW = uvw[start_index:end_index, :]
    Ts = Tv[start_index:end_index]

    heave = detrend(xyzplat[start_index:end_index, 2]) #z-axis to find heave
    waveheight = 4*np.std(heave)

    U = np.mean(UVW[:, 0])
    V = np.mean(UVW[:, 1])
    W = np.mean(UVW[:, 2])

    # UNITS - wdir is in radians
    wdir = np.arctan2(V,-U)
    if wdir < 0:
        wdir = wdir + 2*np.pi

    u, alpha, beta = alignwind(UVW)

    # Relative to Earth
    Uearth=np.mean(u[0:tot, 0])

    for col in np.arange(u.shape[-1]):    # TODO: check axis
        u[:, col] = detrend(u[:, col])
    
    # Ts=detrend(Ts)
    
    Ts = detrend(Ts)

    fluxes = np.zeros(3)
    uwavg = np.mean(u[0:tot, 2]*u[0:tot, 0]) # Along-Wind component of momentum flux
    vwavg = np.mean(u[0:tot, 2]*u[0:tot, 1]) # Cross-Wind component of momentum flux
    wTavg = np.mean(u[0:tot, 2]*Ts) # Heat flux (vertical velocity times temp)

    fluxes[0]=uwavg
    fluxes[1]=vwavg
    fluxes[2]=wTavg

    return (fluxes, Uearth, waveheight)


def convert_data_to_nwu(data):
    """
    Rotate direction, angular rate, and accelerations to North-West-Up coordinate system.
    """
    data['fdchp_heading'] = -data['fdchp_heading']   # z heading(yaw) counter clockwise
    data['fdchp_pitch'] = -data['fdchp_pitch']   # y pitch east to west
    data['fdchp_y_ang_rate'] = -data['fdchp_y_ang_rate'] # angular rate around y-axis
    data['fdchp_z_ang_rate'] = -data['fdchp_z_ang_rate'] # angular rate around z-axis 
    data['fdchp_y_accel_g'] = -data['fdchp_y_accel_g'] # linear acceleration along y-axis
    data['fdchp_z_accel_g'] = -data['fdchp_z_accel_g'] # linear acceleration along z-axis (positve up)
    
def exception_handler(exception):
    print("Exception! {}".format(exception))


def read_file(file_path):
    """Read a file and return a list of Particles.
    Parameters
    ----------
    file_path : str
        Path to the raw FDCHP data file.
    Returns
    -------
    List[FdchpADataParticle] or None
        List of FdchpADataParticles parsed from the raw FDCHP data file.
    Notes
    -----
    - The function uses `FdchpAParser` to parse the input file.
    """
    from mi.dataset.parser.fdchp_a import FdchpAParser
    
    data = []
    with open(file_path, 'rb') as input:
        parser = FdchpAParser(input, exception_handler)
        parser.parse_file()
        parser._file_parsed = True
        num_particles = len(parser._record_buffer)
        data = parser.get_records(num_particles)
    return data


def read_file_to_pandas(file_path, convert_to_nwu=True):
    """Read a file and convert its contents to a pandas DataFrame.
    Parameters
    ----------
    file_path : str
        Path to the raw FDCHP data file.
    convert_to_nwu : bool, optional
        If True, converts FDCHP readings from the North East Down (NED) coordinate system
        to the North West Up (NWU) coordinate system. Default is True.
    Returns
    -------
    pandas.DataFrame or None
        DataFrame containing the parsed FDCHP data. Returns None if no data is found.
    Notes
    -----
    - The function uses `FdchpAParser` to parse the input file.
    - The resulting DataFrame includes a 'time' column constructed from individual date and time components.
    """
    from mi.dataset.parser.fdchp_a import FdchpAParser

    data = []
    with open(file_path, 'rb') as input:
        parser = FdchpAParser(input, exception_handler)
        # parser.parse_file()
        particle = parser.get_records()
        
        while particle:
            data.append({ value['value_id']: value['value'] for value in particle[0].generate_dict()['values']})
            particle = parser.get_records()
    if data:
        df = pd.DataFrame(data)
    else:
        df = None
    if df is not None:
        df['time'] = df.apply(lambda row: datetime.datetime( int(row.year), int(row.month), int(row.day), int(row.hour), int(row.minute), int(row.second), int(row.millisecond)*1000), axis=1)
        
        if convert_to_nwu:
            # Convert IMU from North East Down coordinate system to North West Up coordinate system to match Sonic.
            # Note that we can leave the x-axis variable as measured.
            convert_data_to_nwu(df)
            
    return df


def particles_to_pandas(particles, convert_to_nwu=True):
    """Convert a list of FdchpADataParticles to a pandas DataFrame.
    Parameters
    ----------
    particles : List[FdchpADataParticle]
        List of FdchpADataParticles to be converted.
    convert_to_nwu : bool, optional 
        If True, converts FDCHP readings from the North East Down (NED) coordinate system
        to the North West Up (NWU) coordinate system. Default is True.
    Returns
    -------
    pandas.DataFrame or None
        DataFrame containing the parsed FDCHP data. Returns None if no data is found.
    Notes
    -----
    - The resulting DataFrame includes a 'time' column constructed from individual date and time components
    """
    df = None
    for particle in particles:
        values = particle.generate_dict()['values']
        data_dict = { value['value_id']: [value['value']] for value in values}
        entry = pd.DataFrame.from_dict(data_dict)
        df = pd.concat([df,entry])
        
    if df is not None:
        df['time'] = df.apply(lambda row: datetime.datetime( int(row.year), int(row.month), int(row.day), int(row.hour), int(row.minute), int(row.second), int(row.millisecond)*1000), axis=1)
        
        if convert_to_nwu:
            # Convert IMU from North East Down coordinate system to North West Up coordinate system to match Sonic.
            # Note that we can leave the x-axis variable as measured.
            convert_data_to_nwu(df)
        
    return df
    

def despikesimple(Y, exclusion_stdevs = 4, iterations = 3):
    """
    Removes outliers from a numpy array by excluding points outside a 
    specified number of standard deviations from the median, then 
    interpolates over the excluded points. The process is repeated for a 
    given number of iterations.

    Parameters
    ----------
    Y : numpy.ndarray
        Input data array. Can be one-dimensional or two-dimensional.
    exclusion_stdevs : float, optional
        Number of standard deviations from the median beyond which data 
        points are considered outliers and excluded. Default is 4.
    iterations : int, optional
        Number of passes to perform the despiking and interpolation process. 
        Default is 3.

    Returns
    -------
    numpy.ndarray
        Array with outliers replaced by interpolated values.

    Notes
    -----
    - Interpolation is performed over the acceptable (non-outlier) points.
    - For multi-dimensional arrays, despiking is performed independently for each column.
    """

    data_shape = Y.shape
    rows = data_shape[0]

    if len(data_shape) == 1:
        # one-dimensional
        cols = 1
    else:
        cols = data_shape[1]
    t = np.arange(rows)

    def handle_interp(rows, data):
        median = np.nanmedian(data)
        stdev = np.nanstd(data)
        good_data = (data < median + exclusion_stdevs*stdev) & (data > median - exclusion_stdevs*stdev)
        acceptable_points = data[good_data]
        k = np.sum(good_data)

        if k == len(rows):
            return data# don't bother iterating when there aren't any points to exclude
        if k > 0:
            interp = interpolate.interp1d(rows[good_data], acceptable_points, fill_value='extrapolate')
            data[:] = interp(rows)

        return data

    for iter in range(iterations):
        if cols == 1:
            X = Y[:]
        else:
            for col in range(cols):
                X = Y[:, col]
                Y[:, col] = handle_interp(t, X)

    return Y
                

def gravity_from_latitude(latitude):
    # Get gravitational acceleration g based on latitude
    if latitude < 0 or latitude > 90:
        latitude = 45.0

    gamma = 9.7803267715
    c1 = 0.0052790414
    c2 = 0.0000232718
    c3 = 0.0000001262
    c4 = 0.0000000007

    phi = latitude*np.pi/180.0
    x = np.sin(phi)
    g = gamma*(1+c1*x**2+c2*x**4+c3*x**6+c4*x**8)

    return g


def rotate_buoy_to_world(input_angles, euler_angles, iflag = False):
    # Rotate from buoy frame to world frame

    sinp  = np.sin(euler_angles[:, 0])
    cosp  = np.cos(euler_angles[:, 0])
    sint  = np.sin(euler_angles[:, 1])
    cost  = np.cos(euler_angles[:, 1])
    sinps = np.sin(euler_angles[:, 2])
    cosps = np.cos(euler_angles[:, 2])

    up = input_angles[:, 0]
    vp = input_angles[:, 1]
    wp = input_angles[:, 2]

    if iflag: #  from xyz to x'y'z'
        u = up*cost*cosps                   + vp*cost*sinps                   - wp*sint
        v = up*(sinp*sint*cosps-cosp*sinps) + vp*(sinp*sint*sinps+cosp*cosps) + wp*(cost*sinp)
        w = up*(cosp*sint*cosps+sinp*sinps) + vp*(cosp*sint*sinps-sinp*cosps) + wp*(cost*cosp)

    else: # from x'y'z' to xyz
        u = up*cost*cosps + vp*(sinp*sint*cosps-cosp*sinps) + wp*(cosp*sint*cosps+sinp*sinps)
        v = up*cost*sinps + vp*(sinp*sint*sinps+cosp*cosps) + wp*(cosp*sint*sinps-sinp*cosps)
        w = up*(-sint)    + vp*(cost*sinp)                  + wp*(cost*cosp)

    return np.array([u, v, w]).T


def sonic(sonics, omegam, euler, uvwplat, R):
    """
    Function from EDDYCORR toolbox
    CORRECT SONIC ANEMOMETER COMPONENTS FOR PLATFORM MOTION AND ORIENTATION.
    INPUTS:
       sonics     - row of integers corre to sonic numbers which are to be 
          corrected
       omegam     - (3XN) measured angular rate 'vector' in platform frame
       euler      - (3XN) array of euler angles (phi, theta, psi)
       uvwplat    - (3XN) array of platform velocities (output from accels_.m)
       R          - Distance vector between the sonic anemometer and motion sensors

    OUTPUTS:
       uvw        - (MXN) array of corrected sonic anemometer components, in the 
                        fixed earth reference frame  (North-West-up)
    CALCULATE WINDS IN EARTH BASED FRAME. THE ANGULAR VELOCITY IS CALCULATED AS
    THE CROSS PRODUCT BETWEEN THE ANGULAR RATE VECTOR AND POSITION VECTOR.THE 
    MEASURED AND ANGULAR VELOCITIES ARE IN THE PLATFORM FRAME AND MUST BE
    ROTATED INTO THE EARTH FRAME. THE PLATFORM VELOCITY IS ALREADY IN THE EARTH
    FRAME (FROM ACCELS.M), SO HERE THEY CAN JUST BE ADDED.
     UVW =  MEASURED VELOCITY + ANGULAR RATE INDUCED VELOCITIES + 
      INTEGRATED ACCELEROMETERS  
  """

    Rvec = np.array(R * np.ones(omegam.shape))
    uvwrot = np.cross(omegam,Rvec)

    uvw  = rotate_buoy_to_world(sonics + uvwrot, euler) + uvwplat
    uvwr = rotate_buoy_to_world(sonics + uvwrot, euler)
    return (uvw, uvwr, uvwrot)



def get_angle_update_matrix(angular_rates, angles):
    """
    Function from EDDYCORR toolbox
    This function computes the angular update matrix
    as described in Edson et al. (1998) and Thwaites
    (1995) page 50.
    """
    p  = angles[:, 0]
    t  = angles[:, 1]
    ps = angles[:, 2]

    up = angular_rates[:, 0]
    vp = angular_rates[:, 1]
    wp = angular_rates[:, 2]

    u = up  + vp*np.sin(p)*np.tan(t) + wp*np.cos(p)*np.tan(t)
    v =  0  + vp*np.cos(p)           - wp*np.sin(p)
    w =  0  + vp*np.sin(p)/np.cos(t) + wp*np.cos(p)/np.cos(t)

    return np.array([u, v, w]).T


def get_euler_angles(ahi, bhi, sampling_frequency, accelerations, angular_rates, gyro, gravity, iterations = 5):
    """
     Function from EDDYCORR toolbox

     Sept 2018  Made compass right-handed by changing:
      psi_slow= -gyro - filtfilt(bhi,ahi,-gyro) to
      psi_slow=  gyro - filtfilt(bhi,ahi, gyro)

     Sept 2000     Replaced integrations with cumtrapz function

     May 16 1997 - modified to remove the first estimate of the euler
        angles in the nonlinear euler angle update matrix, F^-1 matrix
        is approximated by the identity matrix. still uses trapezoidal
        intetgration

     INPUT:
        ahi,bhi - 5 element array of filter coefficients for filtfilt
        sampling_frequency    - sampling frequency
        accelerations  - (3xN) array of recalibrated linear accelerations: acc_x,acc_y,acc_z
        angular_rates - (3xN) array of recalibrated angular rates: rate_x, rate_y, rate_z
        gyro  - (1xN) array of gyro signal
        iterations   - number of interations

     OUTPUT:
        euler    - (3XN) array of the euler angles (phi, theta, psi) in radians.
        dr       - ???

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     THE ANGLES ARE  ESTIMATED FROM

     angle = slow_angle (from accelerometers) + fast_angle (integrated rate sensors)
    """
    # Unwrap compass
    gyro = np.unwrap(gyro)

    ang_rates = np.copy(angular_rates)
    # REMOVE MEAN FROM RATE SENSORS
    for col in np.arange(ang_rates.shape[-1]):    # TODO: check axis
        ang_rates[:, col] = detrend(ang_rates[:,col])

    # LOW FREQUENCY ANGLES FROM ACCELEROMETERS AND GYRO
    # SLOW ROLL FROM GRAVITY EFFECTS ON HORIZONTAL ACCELERATIONS. LOW PASS
    # FILTER SINCE HIGH FREQUENCY HORIZONTAL ACCELERATIONS MAY BE 'REAL'

    # PITCH
    normalized_x_acceleration = -accelerations[:,0]/gravity
    theta = np.minimum(normalized_x_acceleration, 1) # Cap accelerations by g; small angles
    theta = np.maximum(theta,-1)                    
    sensible_x_acc =np.nonzero(np.abs(normalized_x_acceleration) < 1)  # Get indices where x-acceleration is less than g
    theta[sensible_x_acc] = np.arcsin(normalized_x_acceleration[sensible_x_acc])
    # Scipy default filtering differs from MATLAB's:
    # https://dsp.stackexchange.com/questions/11466/differences-between-python-and-matlab-filtfilt-function
    theta_slow = theta - filtfilt(bhi, ahi, theta, padlen = PADLENGTH) # 3*(np.max([len(bhi), len(ahi)]) - 1)) # padlen set to match MATLAB default

    # ROLL
    normalized_y_acceleration = accelerations[:, 1]/gravity
    phi = np.minimum(normalized_y_acceleration, 1) # Cap accelerations by g; small angles
    phi = np.maximum(phi,-1)
    acc_y_grav_cos_theta_slow = normalized_y_acceleration/np.cos(theta_slow)
    sensible_y_acc = np.nonzero(np.abs(acc_y_grav_cos_theta_slow) < 1)
    phi[sensible_y_acc] = np.arcsin(acc_y_grav_cos_theta_slow[sensible_y_acc]) # Get indices where y-acceleration is less than g
    phi_slow = phi - filtfilt(bhi, ahi, phi, padlen = PADLENGTH) # 3*(np.max([len(bhi), len(ahi)]) - 1)) # padlen set to match MATLAB default

    # YAW
    # HERE, WE ESTIMATE THE SLOW HEADING. THE 'FAST HEADING' IS NOT NEEDED
    # FOR THE EULER ANGLE UPDATE MATRIX.
    # Using the microstrain compass for heading
    # THE COMPASS IS PASSED RIGHT HANDED SO IT CAN BE TREATED LIKE PITCH AND ROLL
    psi_slow = gyro - filtfilt(bhi, ahi, gyro, padlen = PADLENGTH) #3*(max([len(bhi), len(ahi)]) - 1)) # padlen set to match MATLAB default

    # USE SLOW ANGLES AS FIRST GUESS
    euler  = np.array([phi_slow, theta_slow, psi_slow]).T
    rates = get_angle_update_matrix(ang_rates, euler)
    # INTEGRATE AND FILTER ANGLE RATES, AND ADD TO SLOW ANGLES
    for i in np.arange(iterations):
        phi_int = integrate.cumulative_trapezoid(rates[:, 0], dx = 1/sampling_frequency, initial=0)
        phi = phi_slow + filtfilt(bhi, ahi, phi_int, padlen = PADLENGTH) # 3*(np.max([len(bhi), len(ahi)]) - 1)) # padlen set to match MATLAB default 
        theta_int = integrate.cumulative_trapezoid(rates[:, 1], dx = 1/sampling_frequency, initial=0)
        theta = theta_slow + filtfilt(bhi, ahi, theta_int, padlen = PADLENGTH) # 3*(np.max([len(bhi), len(ahi)]) - 1)) # padlen set to match MATLAB default
        psi_int = integrate.cumulative_trapezoid(rates[:, 2], dx = 1/sampling_frequency, initial=0)
        psi = psi_slow + filtfilt(bhi, ahi, psi_int, padlen = PADLENGTH) #3*(np.max([len(bhi), len(ahi)]) - 1)) # padlen set to match MATLAB default 

        euler  = np.array([phi, theta, psi]).T
        rates = get_angle_update_matrix(ang_rates, euler)
        for col in np.arange(rates.shape[-1]):
            rates[:, col] = detrend(rates[:, col])
        #TODO: could I replace the detrend above with this simple mean subtraction?                       
        # rates = rates - np.mean(rates, axis=0)

    return (euler, rates)


def heave_calc(omegam, euler, accm, sampling_frequency, bhi, ahi, R, gravity):
    """
    Function from EDDYCORR toolbox
    CORRECT SONIC ANEMOMETER COMPONENTS FOR PLATFORM MOTION AND ORIENTATION.
    INPUTS:
       omegam     - (3XN) measured angular rate 'vector' in platform frame
       euler      - (3XN) array of euler angles (phi, theta, psi)
       accm       - (3XN) array of platform accelerations
       R          - vector distance from motionPak to wave sensor

    OUTPUTS:
      platform_velocity - platform velocity at sensor location: [u, v, w]
      platform_disp - platform displacement at sensor location: [x, y, z]
    """

    Rvec = np.array(R * np.ones(accm.shape))
    uvwrot = np.cross(omegam, Rvec)
    uvwrot  = rotate_buoy_to_world(uvwrot, euler)

    acc = rotate_buoy_to_world(accm, euler) # first rotate
    acc[:, 2] = acc[:, 2] - gravity      # remove gravity from z-component of acceleration

    platform_velocity = np.zeros(acc.shape)
    platform_disp = np.zeros(acc.shape)
    
    for i in range(3):
        platform_velocity[:, i] = integrate.cumulative_trapezoid(acc[:, i], dx = 1/sampling_frequency, initial=0) + uvwrot[:, i]
        platform_velocity[:, i] = filtfilt(bhi, ahi, platform_velocity[:, i], padlen = PADLENGTH) # 3*(np.max([len(bhi), len(ahi)]) - 1)) # padlen set to match MATLAB default 
        # INTEGRATE AGAIN TO GET DISPLACEMENTS
        platform_disp[:, i] = integrate.cumulative_trapezoid(platform_velocity[:, i], dx = 1/sampling_frequency, initial=0)
        platform_disp[:, i] = filtfilt(bhi, ahi, platform_disp[:, i], padlen = PADLENGTH) #3*(np.max([len(bhi), len(ahi)]) - 1)) # padlen set to match MATLAB default 

    return (platform_velocity, platform_disp)


def get_platform_vel_pos(bhi, ahi, sampling_frequency, accm, euler, gravity):
    """
    Function from EDDYCORR toolbox
    2008          Allows filter to have different cutoff from angular filter
    Sept 2000     Replaced integrations with cumtrapz function
    Mar 3 1998	Redesigned the high pass filter (see below).
    Revised: June 10, 1997 - high pass filter with higher cutoff 
               frequency than previous version
    Calculates platform velocity and positions from accelerometer data.
    Integrates linear accelerations to get platform velocity
    and displacement. After each integration, signals are 
    high pass filtered to remove low frequency effects.
    INPUT
       bhigh,ahigh - high pass filter coefficients
       sampling_frequency      - sampling frequency
       accm    - calibrated linear accelerations (output from recal.m)
       euler   - (3xN) Euler angles phi, theta, psi 
    OUTPUT:
       acc     - (3XN) linear accelerations in FLIP/Earth reference 
       lin_velocities - (3XN) linear velocities at the point of motion measurement: [u, v, w]
       platform_disp - (3XN) platform displacements from mean position: [x, y, z]
    """
    acc = rotate_buoy_to_world(accm, euler) # first rotate
    acc[:, 2] = acc[:, 2] - gravity # remove gravity

    # INTEGRATE ACCELERATIONS TO GET PLATFORM VELOCITIES
    lin_velocities = np.zeros(acc.shape)
    platform_disp = np.zeros(acc.shape)
    for i in range(3):
        lin_velocities[:, i] = integrate.cumulative_trapezoid(acc[:, i], dx = 1/sampling_frequency, initial=0)
        lin_velocities[:, i] = filtfilt(bhi, ahi, lin_velocities[:, i], padlen = PADLENGTH)
        # INTEGRATE AGAIN TO GET DISPLACEMENTS
        platform_disp[:, i] = integrate.cumulative_trapezoid(lin_velocities[:, i], dx = 1/sampling_frequency, initial=0)
        platform_disp[:, i] = filtfilt(bhi, ahi, platform_disp[:, i], padlen = PADLENGTH)

    return (acc, lin_velocities, platform_disp)


def alignwind(U):
    # u,v,w are in platform coordinates;
    # rotate motion corrected velocities into mean wind
    # fluxes are relative to mean wind (for whole 20-minute segment)

    Ub = np.mean(U[:, 0])
    Vb = np.mean(U[:, 1])
    Wb = np.mean(U[:, 2])
    Sb = np.sqrt(Ub**2 + Vb**2)
    beta  = np.arctan2(Wb,Sb)
    alpha = np.arctan2(Vb,Ub)
    Ur =  U[:, 0]*np.cos(alpha)*np.cos(beta) + U[:, 1]*np.sin(alpha)*np.cos(beta) + U[:,2]*np.sin(beta)
    Vr = -U[:, 0]*np.sin(alpha) + U[:, 1]*np.cos(alpha)
    Wr = -U[:, 0]*np.cos(alpha)*np.sin(beta) - U[:, 1]*np.sin(alpha)*np.sin(beta) + U[:, 2]*np.cos(beta)

    u=np.zeros(U.shape)

    u[:, 0] = Ur
    u[:, 1] = Vr
    u[:, 2] = Wr

    beta  = beta*180/np.pi
    alpha = alpha*180/np.pi
    return [u, alpha, beta]


def calculate_and_write_metrics(mean_time, sonics, ang_rates, platform_accelerations, compass, roll, pitch, fluxes, waveheight, file_path='fluxes'):
    
    Uavg=np.mean(sonics[:, 0])
    Vavg=np.mean(sonics[:, 1])
    Wavg=np.mean(sonics[:, 2])
    rdir=np.arctan2(Vavg,Uavg) #TODO: not used?

    Umax=np.max(sonics[0,:])
    Vmax=np.max(sonics[1,:])
    Wmax=np.max(sonics[2,:])
    Umin=np.min(sonics[0,:])
    Vmin=np.min(sonics[1,:])
    Wmin=np.min(sonics[2,:])

    #JBE  This is not the standard deviation. 
    #Ustd = mean(abs(sonics(1,1:L)-Uavg));
    Ustd = np.std(sonics[0,:])
    Vstd = np.std(sonics[1,:])
    Wstd = np.std(sonics[2,:])
    
    compstd=np.std(np.unwrap(compass))
    compmin=compass.min()
    compmax=compass.max()
    
    gx=np.cos(compass)
    gy=np.sin(compass)
    compcos = np.mean(gx)
    compsin = np.mean(gy)
    compavg = np.arctan2(compsin,compcos)
    
    mean_ang_rate=np.mean(ang_rates, axis=0) #TODO: check axis
    std_ang_rate=np.std(ang_rates, axis=0) #TODO: check axis

    Phiavg = mean_ang_rate[0]
    Thetaavg = mean_ang_rate[1]
    Psiavg = mean_ang_rate[2]
    Phistd = std_ang_rate[0]
    Thetastd = std_ang_rate[1]
    Psistd = std_ang_rate[2]
    Phimax = np.max(ang_rates[0,:])
    Thetamax = np.max(ang_rates[1,:])
    Psimax = np.max(ang_rates[2,:])
    Phimin = np.min(ang_rates[0,:])
    Thetamin = np.min(ang_rates[1,:])
    Psimin = np.min(ang_rates[2,:])
    
    gcomp = np.mean(platform_accelerations, axis=0)

    Axavg = gcomp[0]
    Ayavg = gcomp[1]
    Azavg = gcomp[2]
    Axmax=np.max(platform_accelerations[:,0])
    Aymax=np.max(platform_accelerations[:, 1])
    Azmax=np.max(platform_accelerations[:, 2])
    Axmin=np.min(platform_accelerations[:, 0])
    Aymin=np.min(platform_accelerations[:, 1])
    Azmin=np.min(platform_accelerations[:, 2])

    gstd = np.std(platform_accelerations, axis=0)
    Axstd = gstd[0]
    Aystd = gstd[1]
    Azstd = gstd[2]
    
    # UNITS - Cal mean roll and pitch in radians
    Pitch = np.mean(pitch)
    Roll = np.mean(roll)
    Pitchstd = np.std(pitch)
    Rollstd = np.std(roll)
    Pitchmax = np.max(pitch)
    Rollmax = np.max(roll)
    Pitchmin = np.min(pitch)
    Rollmin = np.min(roll)
                 
    with open(file_path, 'w') as outfile:
        outfile.write(str(fluxes))


def write_metrics(metrics_list, file_path='fluxes', rounding=2):
    with open(file_path, 'w') as outfile:
        for metric in metrics_list:
            met = np.round(metric, rounding) if np.isscalar(metric) else metric
            outfile.write(str(met) + ", ")

def plot_x_velocity_vs_wind_speed(speeds, x_velocities, x_min = 0, x_max = 20, y_min = -1.0, y_max = 0.0):
    fig, ax = plt.subplots()
    ax.scatter(speeds, x_velocities, marker='o', alpha=0.5)
    ax.set_xlabel('U (m/s)')
    ax.set_xlim(x_min, x_max)
    ax.set_ylabel('<uw> (m/s)')
    ax.set_ylim(y_min, y_max)
    plt.show()
    
    
    
def plot_wave_height_vs_wind_speed(speeds, wave_heights,  x_min = 0, x_max = 15, y_min = 0.0, y_max = 4.0):
    fig, ax = plt.subplots()
    ax.scatter(speeds, wave_heights, marker='o', alpha=0.5)
    ax.set_xlabel('U (m/s)')
    ax.set_xlim(x_min, x_max)
    ax.set_ylabel('$\\sigma_H$ (m)')
    ax.set_ylim(y_min, y_max)
    plt.show()