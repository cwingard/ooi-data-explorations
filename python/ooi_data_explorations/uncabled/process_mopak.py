import os
import numpy as np
import xarray as xr

from scipy.signal import buttord, butter, filtfilt, detrend, welch
from scipy.fft import fft
from scipy.signal.windows import hann
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

from ooi_data_explorations.common import ENCODINGS, get_vocabulary, get_deployment_dates, m2m_request, \
    m2m_collect, update_dataset, inputs
from pyseas.data.generic_functions import magnetic_declination, magnetic_correction

MOPAK = {
    'sample': {
        'long_name': 'Sample Number',
        'units': 'counts',
        'comment': ('The burst sample number of the data. Used to distinguish bursts in the data, so processing can '
                    'be done on a burst-by-burst basis.')
    },
    'acceleration_x': {
        'long_name': 'X-axis Acceleration',
        'units': 'gravity',
        'comment': ('This is a vector quantifying the direction and magnitude of the acceleration that the '
                    'mooring is exposed to. This quantity is fully temperature compensated and scaled into '
                    'physical units of g (1 g = 9.80665 m/sec^2). It is expressed in terms of the MOPAK''s '
                    'local coordinate system (in this case the X-axis).')
    },
    'acceleration_y': {
        'long_name': 'Y-axis Acceleration',
        'units': 'gravity',
        'comment': ('This is a vector quantifying the direction and magnitude of the acceleration that the '
                    'mooring is exposed to. This quantity is fully temperature compensated and scaled into '
                    'physical units of g (1 g = 9.80665 m/sec^2). It is expressed in terms of the MOPAK''s '
                    'local coordinate system (in this case the Y-axis).')
    },
    'acceleration_z': {
        'long_name': 'Z-axis Acceleration',
        'units': 'gravity',
        'comment': ('This is a vector quantifying the direction and magnitude of the acceleration that the '
                    'mooring is exposed to. This quantity is fully temperature compensated and scaled into '
                    'physical units of g (1 g = 9.80665 m/sec^2). It is expressed in terms of the MOPAK''s '
                    'local coordinate system (in this case the Z-axis).')
    },
    'angular_rate_x': {
        'long_name': 'X-axis Angular Rate',
        'units': 'radians/s',
        'comment': ('This is a vector quantifying the rate of rotation of the mooring. This quantity is '
                    'is fully temperature compensated and scaled into units of radians/second. It is '
                    'expressed in terms of the MOPAK''s local coordinate system (in this case the X-axis) '
                    'in units of radians/second.')
    },
    'angular_rate_y': {
        'long_name': 'Y-axis Angular Rate',
        'units': 'radians/s',
        'comment': ('This is a vector quantifying the rate of rotation of the mooring. This quantity is '
                    'is fully temperature compensated and scaled into units of radians/second. It is '
                    'expressed in terms of the MOPAK''s local coordinate system (in this case the Y-axis) '
                    'in units of radians/second.')
    },
    'angular_rate_z': {
        'long_name': 'Z-axis Angular Rate',
        'units': 'radians/s',
        'comment': ('This is a vector quantifying the rate of rotation of the mooring. This quantity is '
                    'is fully temperature compensated and scaled into units of radians/second. It is '
                    'expressed in terms of the MOPAK''s local coordinate system (in this case the Z-axis) '
                    'in units of radians/second.')
    },
    'magnetometer_x': {
        'long_name': 'X-axis Magnetometer',
        'units': 'gauss',
        'comment': ('This is a vector which gives the instantaneous magnetometer direction and magnitude. It '
                    'is fully temperature compensated and is expressed in terms of the MOPAK''s local '
                    'coordinate system (in this case the X-axis) in units of Gauss.')
    },
    'magnetometer_y': {
        'long_name': 'Y-axis Magnetometer',
        'units': 'gauss',
        'comment': ('This is a vector which gives the instantaneous magnetometer direction and magnitude. It '
                    'is fully temperature compensated and is expressed in terms of the MOPAK''s local '
                    'coordinate system (in this case the Y-axis) in units of Gauss.')
    },
    'magnetometer_z': {
        'long_name': 'Z-axis Magnetometer',
        'units': 'gauss',
        'comment': ('This is a vector which gives the instantaneous magnetometer direction and magnitude. It '
                    'is fully temperature compensated and is expressed in terms of the MOPAK''s local '
                    'coordinate system (in this case the Z-axis) in units of Gauss.')
    }
}

WAVES = {
    'deployment': {
        'long_name': 'Deployment Number',
        'comment': 'The deployment number of the instrument.'
    },
    'time': {
        'long_name': 'Time',
        'standard_name': 'time',
        'comment': ('The time given here is the start time of the sample collection. Sample collection continues '
                    'for 20 minutes at 10 Hz.')
    },
    'number_zero_crossings': {
        'long_name': 'Number of Wave Zero-Crossings',
        'type': 'zero-crossing',
        'comment': ('Zero-crossing is defined as when the buoy vertical displacement crosses a mean sea surface '
                    'level. The total number of zero-crossings is twice the total number of waves observed during '
                    'a measurement period.')
    },
    'significant_wave_height': {
        'long_name': 'Significant Wave Height',
        'standard_name': 'sea_surface_wave_significant_height',
        'units': 'm',
        'type': 'zero-crossing',
        'comment': ('Wave height is defined as the vertical distance from a wave trough to the following wave crest. '
                    'The significant wave height is the mean trough to crest distance measured during the observation '
                    'period of the highest one-third of waves. Calculated from the zero down-crossing method.')
    },
    'significant_wave_period': {
        'long_name': 'Significant Wave Period',
        'standard_name': 'sea_surface_wave_significant_period',
        'units': 's',
        'type': 'zero-crossing',
        'comment': ('Significant wave period corresponds to the mean wave period of the highest one-third of measured '
                    'waves during the observation period. Wave period is defined as the interval of time between '
                    'repeated features on the waveform such as crests, troughs, or upward/downward passes through '
                    'the mean sea surface level.')
    },
    'wave_height_10': {
        'long_name': 'Height of Highest Tenth of Waves',
        'standard_name': 'sea_surface_wave_mean_height_of_highest_tenth',
        'units': 'm',
        'type': 'zero-crossing',
        'comment': ('Wave height is defined as the vertical distance from a wave trough to the following wave crest. '
                    'The height of the highest tenth is defined as the mean of the highest 10 per cent of trough to '
                    'crest distances measured during the observation period. Calculated from the zero down-crossing '
                    'method.')
    },
    'wave_period_10': {
        'long_name': 'Period of Highest Tenth of Waves',
        'standard_name': 'sea_surface_wave_mean_period_of_highest_tenth',
        'units': 's',
        'type': 'zero-crossing',
        'comment': ('Wave mean period is the mean period measured over the observation duration. The period of the '
                    'highest tenth of waves is the mean period of the highest 10 per cent of waves measured during '
                    'the observation period. Calculated from the zero down-crossing method.')
    },
    'mean_wave_height': {
        'long_name': 'Mean wave height',
        'standard_name': 'sea_surface_wave_mean_height',
        'units': 'm',
        'type': 'zero-crossing',
        'comment': ('Wave height is defined as the vertical distance from a wave trough to the following wave crest. '
                    'The mean wave height is the mean trough to crest distance measured during the observation period. '
                    'This is calculated from the average zero down-crossing wave height')
    },
    'mean_wave_period': {
        'long_name': 'Mean Wave Period',
        'standard_name': 'sea_surface_wave_mean_period',
        'units': 's',
        'type': 'zero-crossing',
        'comment': ('Wave period is the interval of time between repeated features on the waveform such as crests, '
                    'troughs or upward passes through the mean level. Wave mean period is the mean period measured '
                    'over the observation duration. Calculated as the average zero down-crossing wave period. ')
    },
    'peak_wave_period': {
        'long_name': 'Peak Wave Period',
        'standard_name': 'sea_surface_wave_period_at_variance_spectral_density_maximum',
        'units': 's',
        'type': 'directional',
        'comment': ('Wave period is the interval of time between repeated features on the waveform such as crests, '
                    'troughs or upward passes through the mean level. The peak wave period, is the period of the most '
                    'energetic waves in the total wave spectrum at a specific location.')
    },
    'peak_wave_direction': {
        'long_name': 'Peak Wave Direction',
        'standard_name': 'sea_surface_wave_from_direction_at_variance_spectral_density_maximum',
        'units': 'degrees',
        'type': 'directional',
        'comment': ('Peak wave direction is the direction from which the most energetic waves are coming. The '
                    'spectral peak is the most energetic wave in the total wave spectrum. The direction is a '
                    'bearing in the usual geographical sense, measured positive clockwise from due north (compass '
                    'data has been corrected for magnetic declination). This parameter is derived via the PUV-method.')
    },
    'peak_wave_spread': {
        'long name': 'Peak Wave Spread',
        'standard_name': 'sea_surface_wave_directional_spread_at_variance_spectral_density_maximum',
        'units': 'degrees',
        'type': 'directional',
        'comment': ('Peak wave spread is the directional spread of the most energetic waves in the total wave '
                    'spectrum. Directional spread is the (one-sided) directional width within a given sub-domain '
                    'of the wave directional spectrum. This parameter is derived via the PUV-method.')
    },
    'peak_wave_period_puv': {
        'long_name': 'Peak Wave Period',
        'standard_name': 'sea_surface_wave_period_at_variance_spectral_density_maximum',
        'units': 's',
        'type': 'directional',
        'comment': ('Wave period is the interval of time between repeated features on the waveform such as crests, '
                    'troughs or upward passes through the mean level. The peak wave period, is the period of the most '
                    'energetic waves in the total wave spectrum at a specific location. This parameter is derived '
                    'via the PUV-method and by parabolic fitting of the log-averaged frequency bands.')
    },
    'wave_height_hm0': {
        'long_name': 'Significant Wave Height from Spectral Moment 0',
        'standard_name': 'sea_surface_wave_significant_height_from_variance_spectral_density',
        'units': 'm',
        "type": 'directional',
        'comment': ('Wave height is defined as the vertical distance from a wave trough to the following '
                    'wave crest. The significant wave height (hm0) is the mean wave height of the highest '
                    'one-third of waves as estimated from the zeroth-spectral moment m0, where '
                    'hm0 = 4*sqrt(m0), and m0 is the integral of the S(f)*df with f = F1 to F2 in Hz. This '
                    'parameter is derived via the PUV-method.')
    }
}


def filter_coefficients(fs, fc, ludo=True):
    """
    High-pass filter which retains real accelerations but removes drift
    
    The filter was chosen so that the spectra of the doubly integrated
    acceleration matched the frequency domain integrated power spectrum
    in the pass band. The comparison is most sensitive to the transition
    width.
    
    The selected transition region to lie in the overlap region between 
    the integrated angle rate and the accelerometer based angle estimates.
    The filter is shift to higher frequencies by 1/4 to 1/2 decade.
    
    NOTE: The scipy buttord function optimizes for the stopband, whereas
    MatLab Buttord optimizes for the passband, resulting in a -3 dB shift
    
    Parameters
    ----------
    fs: float, int
        Sampling frequency
    fc: float, int
    ludo: bool, Default = True
        
    Returns
    -------
    b_high, a_high: array_like, array_like
        The numerator (b) and denominator (a) polynomials of the IIR filter
    """
    n_freq = fs / 2
    wp = fc / n_freq
    if ludo:
        ws = 0.8 * wp
        n, wn = buttord(wp, ws, 3, 7)
    else:
        ws = 0.7 * wp
        n, wn = buttord(wp, ws, 10, 25)

    # noinspection PyTupleAssignmentBalance
    b_high, a_high = butter(n, wn, btype='highpass')
    
    return b_high, a_high


def identify_samples(ds, threshold):
    """
    Defines sample intervals for burst/pulse sampling instruments.
    
    Parameters
    ----------
    ds: xarray.Dataset, xarray.DataArray
        An xarray dataset or dataarray with time as the primary dimension
    threshold: int
        The threshold in seconds that separates burst sample intervals
        
    Returns
    -------
    sample: array_like
        A numpy array the length of the input time dimension with the 
        sample interval number
    """
    # First, get the difference of the time in seconds
    dt = ds["time"].diff(dim="time").dt.seconds
    
    # Next, find where the gaps in the time series occur
    ends = np.where(dt > threshold)[0]
    
    # increase index of ends by 1 since diff drops the first entry, and python is left index inclusive,
    # right index exclusive
    ends = ends + 1
    
    # Find the sampling groups
    sample = np.zeros(ds.time.shape, dtype=int)
    for i, end in enumerate(ends):
        if i == 0:
            start = 0
        elif end == ends[-1]:
            start = ends[i-1]
            end = len(sample)+1
        else:
            start = ends[i-1]
        sample[start:end] = i
        
    return sample


def zero_crossing(displace, fs):
    """
    Calculate the wave statistics using a zero-crossing algorithm.
    
    This method utilizes a zero down-crossing wave algorithm to
    compute the bulk wave statistics. The code, as written, actually
    looks for the up-crossing waves; inverting the heave values
    identifies the down-crossing waves. Additionally, waves with 
    either a crest or trough that falls below a detection limit
    are joined to either the following or preceding wave.
    
    Parameters
    ----------
    displace: array_like
        An array of vertical displacement (heave)
    fs: float
        Sampling frequency
        
    Returns
    -------
    n: int
        The number of zero-crossings detected
    H_sig: float
        The significant wave height, defined as the average
        wave height of the 1/3 highest waves
    T_sig: float
        The significant wave period
    H_10: float
        The wave height of the 10% highest waves
    T_10: float
        The mean period of the 10% highest waves
    H_avg: float
        The mean wave height
    T_avg: float
        The mean wave period

    References
    ----------
    Neumeier, Urs. 2003. Waves [Software: MatLab]
     """
    # Code is written looking at up-crossing - to use down-crossing we invert the heave
    z = -1 * displace
    z = detrend(z)

    # Find and remove values near zero
    z0 = z[z != 0]

    # Create an index
    back0 = np.arange(0, len(z), 1)
    back0 = back0[z != 0]

    # Find the zero crossings
    f = np.where(z0[0:-1]*z0[1:] < 0)[0]
    crossing = back0[f]

    # Reject the first crossing if it is upward-crossing
    if z[0] > 0:
        crossing = crossing[1:]

    # Take every other crossing to get the zero-downward crossings
    crossing = crossing[np.arange(0, len(crossing), 2)]

    # #### CALCULATE WAVE PARAMETERS #### #
    # Initialize arrays to save the results in
    wave = np.zeros((len(crossing)-1, 4))

    # Get the max (crest) and min (trough) values between each crossing
    for n in np.arange(0, len(crossing)-1, 1):
        wave[n, 1] = np.max(z[crossing[n]:crossing[n+1]])
        wave[n, 2] = -1 * np.min(z[crossing[n]:crossing[n+1]])

    # Check the size of the wave and if no wave found do nothing
    if len(wave[:, 1]) >= 1:

        # Calculate elapsed time between each measurement
        wave[:, 3] = np.diff(crossing) / fs

        # Calculate the threshold wave size
        threshold = 0.01 * np.max(wave[:, 1] + wave[:, 2])
        if threshold < 0:
            raise ValueError(f"Wave threshold must not be negative")
        
        # Now remove wave which are too small by joining them to adjacent waves
        for idx, (crest, trough) in enumerate(wave[:, 1:3]):
            if crest < threshold:
                if idx != 0:
                    # Join the values to the preceding wave
                    wave[idx-1, 1] = np.max(wave[idx-1:idx+1, 1])
                    wave[idx-1, 2] = np.max(wave[idx-1:idx+1, 2])
                    wave[idx-1, 3] = np.sum(wave[idx-1:idx+1, 3])
                # Replace the values with NaNs
                wave[idx, :] = np.nan
            elif trough < threshold:
                if idx+1 != len(wave):
                    # Join the values to the next wave
                    wave[idx, 1] = np.max(wave[idx:idx+2, 1])
                    wave[idx, 2] = np.max(wave[idx:idx+2, 2])
                    wave[idx, 3] = np.sum(wave[idx:idx+2, 3])
                    # Replace the values with NaNs
                    wave[idx+1, :] = np.nan
                else:
                    wave[idx, :] = np.nan
    
    # Drop the NaNs
    wave = wave[np.all(~np.isnan(wave), axis=1)]

    # Now calculate the wave height from the crest -> trough distance
    wave[:, 0] = np.sum(wave[:, 1:3], axis=1)

    # Sort based on wave height
    wave_sorted = np.sort(wave, axis=0)
    wave_sorted = np.flipud(wave_sorted)

    # #### CALCULATE WAVE STATISTICS #### #
    # Get number of waves measured
    n = len(wave_sorted)

    # Calculate significant wave height and period
    n_sig = int(np.round(n / 3))
    h_sig = np.mean(wave_sorted[0:n_sig, 0])
    T_sig = np.mean(wave_sorted[0:n_sig, 3])

    # Calculate the 10-highest waves
    n_10 = int(np.round(n / 10))
    h_10 = np.mean(wave_sorted[0:n_10, 0])
    T_10 = np.mean(wave_sorted[0:n_10, 3])

    # Calculate the mean height and period
    h_avg = np.mean(wave[:, 0])
    T_avg = np.mean(wave[:, 3])

    return n, h_sig, T_sig, h_10, T_10, h_avg, T_avg


def non_directional_statistics(displace, fs, npt):
    """
    Calculate the wave statistics from the wave time series.

    This method utilizes a mixed approach to calculating wave statistics. The
    significant wave height is calculated as the 4 * std(heave), the
    significant wave period is from the frequency at the spectral max, and the
    average wave period uses a zero-crossing approach.

    Parameters
    ----------
    displace: array_like
        An array of vertical displacement (heave)
    fs: float
        Sampling frequency
    npt: int
        Number of points in the fft
        
    Returns
    -------
    Hsig: float
        Significant wave height
    Havg: float
        Mean sea level height. This value should be near-zero.
    Tsig: float
        Significant wave period calculated from the peak spectrum.
    Tavg: float
        Average wave period
    
    References
    ----------
    Edson, Jim. 2023. Motion Calculations Toolbox. [Software: MatLab]
    """
    # Detrend the heave and calculate the significant and average wave height
    displace = detrend(displace)
    Hsig = 4 * np.std(displace)
    Havg = np.mean(displace)  # Not actually average wave height
    
    bw = np.ones(5) / 5
    aw = 1
    
    # Calculate the significant wave period from the spectral density
    if Hsig > 0.2:
        [fr, wxx] = welch(displace, fs, window=hann(npt), nfft=npt, noverlap=0, detrend=False)
        wxx[0] = wxx[1]
        wxxf = filtfilt(bw, aw, wxx)
        i = np.where(fr < 0.01)[0]
        wxxf[i] = 1e-7
        i = np.where(wxxf == np.max(wxxf))[0]
        fr1 = np.mean(fr[i])
        Tsig = 1 / fr1
        Tavg = wave_period(displace, fs)
    else:
        Tsig = np.nan
        Tavg = np.nan
        
    return Hsig, Havg, Tsig, Tavg


def wave_period(displace, fs):
    """
    Compute average wave period using the zero-crossing method
    
    Parameters
    ----------
    displace: array_like
        An array of vertical displacement (heave)
    fs: float
        The sampling frequency

    Returns
    -------
    tm: float
        The calculated average wave period

    References
    ----------
    Edson, Jim. 2023. Motion Calculations Toolbox. [Software: MatLab]
    """
    displace = detrend(displace)
    n = len(displace)
    T = (n - 1) / fs
    sheave = np.sign(displace)
    sw1 = sheave[0:len(displace) - 1]
    sw2 = sheave[1:len(displace)]
    i = np.where(sw1 != sw2)[0]
    fm = len(i) / 2 / T
    tm = 1 / fm
    return tm


def updater(rates, angles):
    """
    Computes the angular update matrix described in Edson et al. (1998) and
    Thwaites (1995)
    
    Parameters
    ----------
    rates: array_like
        A (3 x n) matrix of the angular rates
    angles: array_like
        A (3 x n) matrix of the euler angles phi, theta, psi, where:
            phi = angles[0,:] - rotation of x'y'z' about x-axis (roll)
            theta = angles[1,:] - rotation of x'y'z' about y-axis (pitch)
            psi = angles[2,:] - rotations of x'y'z' about z-axis (yaw)
            
    Returns
    -------
    OUT: array_like
        A (3 x n) matrix of the updated angular rates

    References
    ----------
    Beardsley, Bob. 1999. AIR SEA Toolbox. Ver. 2.0. [Software: MatLab]
    Edson, Jim. 2023. Motion Calculations Toolbox. [Software: MatLab]
    """
    p = angles[0, :]
    t = angles[1, :]
    # ps = angles[2, :]
    
    up = rates[0, :]
    vp = rates[1, :]
    wp = rates[2, :]
    
    u = up + vp * np.sin(p) * np.tan(t) + wp * np.cos(p) * np.tan(t)
    v = 0 + vp * np.cos(p) - wp * np.sin(p)
    w = 0 + vp * np.sin(p) / np.cos(t) + wp * np.cos(p) / np.cos(t)
    
    return np.vstack([u, v, w])


def euler_angles(ahi, bhi, fs, accm, ratem, gyro, gravity, iters=5):
    """
    Derive the euler angles from the accelerometers and rate sensors.
    
    The equation for the derivation of the euler angles are:
        angle = slow_angle (from accelerometers) + fast_angle (integrated rate sensors)
        
    Parameters
    ----------
    ahi: array_like
        The filter coefficients a
    bhi: array_like
        The filter coefficients b
    fs: float
        The sample frequency
    accm: array_like
        A (3 x n) array of recalibrated linear accelerations in (x, y, z)
    ratem: array_like
        A (3 x n) array of recalibrated angular rates in (x, y, z)
    gyro: array_like
        A (1 x n) array of the gyro signal
    gravity: float
        The gravitational constant
    iters: int, Default = 5
        Number of iterations to filter
    
    Returns
    -------
    euler: array_like
        A (3 x n) array of the euler angles (phi, theta, psi) in radians
    dr: array_like
        A (3 x n) array of detrended angular rates in (x, y, z)

    References
    ----------
    Beardsley, Bob. 1999. AIR SEA Toolbox. Ver. 2.0. [Software: MatLab]
    Edson, Jim. 2023. Motion Calculations Toolbox. [Software: MatLab]
    """
    # Unwrap the compass data
    gyro = np.unwrap(gyro)
    
    # Remove the mean from the rate sensors
    ratem = detrend(ratem)
    
    # ==================================================================
    # Low frequency angles from accelerometers and gyro
    # Slow roll from gravity effects on horizontal accelerations
    # Low pass filter since high frequency horizontal accelerations may be real
    
    # PITCH
    # Use small angles
    theta_th = np.minimum(-1 * accm[0, :] / gravity, 1)
    theta = theta_th
    
    # Remove freefall values
    ind = np.where(np.abs(accm[0, :]) < gravity)
    theta[ind] = np.arcsin(-1 * accm[0, ind] / gravity)
    
    # Calculate the slow angles
    theta_slow = theta - filtfilt(bhi, ahi, theta)
    
    # ROLL
    # Use small angles
    phi_th = accm[1] / gravity
    phi = phi_th
    
    # Find well-behaved angles
    ind = np.where(np.abs(accm[1] / gravity / np.cos(theta_slow)) < 1)[0]
    phi[ind] = np.arcsin(accm[1, ind] / gravity / np.cos(theta_slow[ind]))
    
    # Filter the roll angles
    phi_slow = phi - filtfilt(bhi, ahi, phi)
    
    # YAW
    psi_slow = gyro[0] - filtfilt(bhi, ahi, gyro[0])
    
    # ==================================================================
    # EULER ANGLES
    # Calculate the euler angles, starting with the slow angles as first guess
    euler = np.vstack([phi_slow, theta_slow, psi_slow])
    rates = updater(ratem, euler)
    
    # Recalculate the euler angles by adding the integrated rates to the
    # slow angles, updating the euler angles, and repeating for N iterations
    cnt = 0
    while cnt < iters:
        phi = phi_slow + filtfilt(bhi, ahi, 1 / fs * cumulative_trapezoid(rates[0, :], initial=0))
        theta = theta_slow + filtfilt(bhi, ahi, 1 / fs * cumulative_trapezoid(rates[1, :], initial=0))
        psi = psi_slow + filtfilt(bhi, ahi, 1 / fs * cumulative_trapezoid(rates[2, :], initial=0))
        euler = np.vstack([phi, theta, psi])
        rates = updater(ratem, euler)
        rates[0] = detrend(rates[0], type='constant')
        rates[1] = detrend(rates[1], type='constant')
        rates[2] = detrend(rates[2], type='constant')
        cnt += 1

    return euler, ratem


def rotate(vectors, angles, iflag=0):
    """
    Rotate a vector from one cartesian basis to another based on Euler angles.
    
    This function rotates a vector from one cartesian basis to another based on
    the associated Euler angles, defined as rotations around the reference axes
    (x,y,z). The axis in the rotated frame are (x',y',z').
    
    Parameters
    ----------
    vectors: array_like
        A (3 x n) matrix of the input vector components
    angles: array_like
        A (3 x n) matrix of the euler angles phi, theta, psi, where:
            phi = angles[0,:] - rotation of x'y'z' about x-axis (roll)
            theta = angles[1,:] - rotation of x'y'z' about y-axis (pitch)
            psi = angles[2,:] - rotations of x'y'z' about z-axis (yaw)
    iflag: int, Default = 0
        For rotation of measurements from body coordinates to earth coordinates
        the flag is set to FALSE. This is a 321 rotation, where the first
        rotation is around the 3-axis (z-axis, angle psi), the second rotation
        is then about the intermediate 2 axis (y-axis, angle theta), and the
        third rotation is about the intermediate 1 axis (x-axis, angle phi)
        
        An integer value indicates which direction the rotation is in:
            0: "vectors" vector transformed from x'y'z' -> xyz
            1: "vectors" vector transformed from xyz -> x'y'z'
            
    Returns
    -------
    OUT: array_like
        A (3 x n) matrix of the rotated input vector components

    References
    ----------
    Beardsley, Bob. 1999. AIR SEA Toolbox. Ver. 2.0. [Software: MatLab]
    """
    # Grab the Euler angles
    phi = angles[0, :]
    theta = angles[1, :]
    psi = angles[2, :]
    
    # Get the input vector components
    up = vectors[0, :]
    vp = vectors[1, :]
    wp = vectors[2, :]
    
    # Perform rotation
    # If True: xyz -> x'y'z'
    if iflag == 1:
        u = up * np.cos(theta) * np.cos(psi) + vp * np.cos(theta)*np.sin(psi) - wp * np.sin(theta)
        v = (up * (np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi)) + vp *
             (np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi)) + wp *
             (np.cos(theta) * np.sin(phi)))
        w = (up * (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)) + vp *
             (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)) + wp *
             (np.cos(theta) * np.cos(phi)))
    # If False: x'y'z' -> xyz
    else:
        u = up * np.cos(theta) * np.cos(psi) + vp * (np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) *
                                                     np.sin(psi)) + wp * (np.cos(phi) * np.sin(theta) * np.cos(psi) +
                                                                          np.sin(phi) * np.sin(psi))
        v = up * np.cos(theta) * np.sin(psi) + vp * (np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) *
                                                     np.cos(psi)) + wp * (np.cos(phi) * np.sin(theta) * np.sin(psi) -
                                                                          np.sin(phi) * np.cos(psi))
        w = up * (-np.sin(theta)) + vp * (np.cos(theta) * np.sin(phi)) + wp * (np.cos(theta) * np.cos(phi))
        
    # Return the rotated vector
    OUT = np.vstack((u, v, w))
    return OUT


# noinspection PyUnusedLocal
def heave(omegam, euler, accm, fs, bhi, ahi, R, g):
    """
    Correct components for platform motion and orientation
    
    Parameters
    ----------
    omegam: array_like
        A (3 x n) measured angular rate vector in the platform frame
    euler: array_like
        A (3 x n) array of euler angles (phi, theta, psi)
    accm: array_like
        A (3 x n) array of platform accelerations
    fs: float
        The sampling frequency
    bhi: array_like
        The filter coefficients b
    ahi: array_like
        The filter coefficients a
    R: array_like
        Vector distance from motion pack to wave sensor
    g: float
        Standard gravity constant

    Returns
    -------
    uvw_plat: array_like
        The platform velocity at sensor location
    xyz_plat: array_like
        The platform displacement at sensor location

    References
    ----------
    Beardsley, Bob. 1999. AIR SEA MatLab Toolbox. Ver. 2.0. [Software: MatLab]
    """
    n, m = omegam.shape
    Rvec = np.vstack((R[0] * np.ones(m), R[1] * np.ones(m), R[2] * np.ones(m)))
    # noinspection PyUnreachableCode
    uvw_rot = np.cross(omegam, Rvec, axis=0)
    uvw_rot = rotate(uvw_rot, euler, 0)
    
    # Rotate the accelerometer and remove gravity
    acc = rotate(accm, euler, 0)
    acc[2, :] = acc[2, :] - g
    
    motion = np.ones(acc.shape)
    uvw_plat = np.ones(acc.shape)
    for i in range(0, 3):
        acc[i, :] = filtfilt(bhi, ahi, acc[i, :])
        motion[i, :] = cumulative_trapezoid(acc[i, :], initial=0) / fs + uvw_rot[i, :]
        uvw_plat[i, :] = filtfilt(bhi, ahi, motion[i, :])
    
    # Integrate again to get the displacements
    xyz_plat = np.ones(uvw_plat.shape)
    for i in range(0, 3):
        xyz_plat[i, :] = cumulative_trapezoid(uvw_plat[i, :], initial=0) / fs
        xyz_plat[i, :] = filtfilt(bhi, ahi, xyz_plat[i, :])
        
    return uvw_plat, xyz_plat


def uvw_xyz(gyro, platform, a_rates, fs, f_cutoff=1/30, com_offset=None, g=9.80665):
    """
    Calculate the displacements (xyz) and velocities (uvw) from accelerometer data
    
    Parameters
    ----------
    gyro: array_like
        An array of the processed and cleaned compass directions
        measured by the MOPAK
    platform: array_like
        A (3 x n) array of the processed and cleaned mopak accelerations in the
        (x, y, z) directions
    a_rates: array_like
        A (3 x n) array of the processed and cleaned angular rates measured by
        the MOPAK
    fs: float
        The sampling frequency
    f_cutoff: float, Default=1/30
        The cutoff period for waves
    com_offset: list[x, y, z], Default=[0, 0, 0.5]
        A list of the offsets from the center of mass of the MOPAK (m)
    g: float, Default = 9.80665
        Standard gravity constant
        
    Returns
    -------
    uvw: array_like
        A (3 x n) array of platform velocities in the (u, v, w) directions
        based at sensor location
    xyz: array_like
        A (3 x n) array of platform displacements in the (x, y, z) directions
        at sensor location

    References
    ----------
    Beardsley, Bob. 1999. AIR SEA MatLab Toolbox. Ver. 2.0. [Software: MatLab]
    Edson, Jim. 2023. Motion Calculations Toolbox. [Software: MatLab]
    """
    if not com_offset:
        # set the default center of mass offset if not provided
        com_offset = [0, 0, 0.5]

    # 30-second cutoff period for waves
    bhiwaves, ahiwaves = filter_coefficients(fs, f_cutoff)
    
    #  despike the data
    platform, bad_platform = despike(platform)
    ang_rate, bad_ang_rate = despike(a_rates)

    # Calculate the compass offset angles
    gx = np.cos(gyro)
    gy = np.sin(gyro)

    # Remove spikes from the compass offset angles
    gx, _ = despike(gx)
    gy, _ = despike(gy)

    # Smooth the compass angles
    g_smooth = np.arctan2(gy, gx)
    gyro = g_smooth

    # Calculate the Euler angles, velocities, and displacements
    euler, dr = euler_angles(ahiwaves, bhiwaves, fs, platform, ang_rate, gyro, g)
    uvw, xyz = heave(dr, euler, platform, fs, bhiwaves, ahiwaves, com_offset, g)
    
    return uvw, xyz


def despike(data, n_std=4, iters=3):
    """
    Despike MOPAK data
    
    Parameters
    ----------
    data: array_like
        An array containing the x, y, z accelerations as rows
    n_std: int (default = 4)
        Number of standard deviations from the median which defines
        a spike
    iters: int (default = 3)
        The number of iterations (passes) over the data to make in
        order to identify and remove spikes
        
    Returns
    -------
    data: array_like
        A two-dimensional array of the data despiked. Identified bad
        data points have been filled via linearly interpolation.
    bad: array_like
        The bad data points

    References
    ----------
    Edson, Jim. 2023. Motion Calculations Toolbox. [Software: MatLab]
    """
    # Coerce entries to be at least 2d
    data = np.atleast_2d(data)
    
    # initialize the variables needed to process the data
    rows, n = data.shape
    t = np.arange(n)
    bad = []
    n_bad = 0

    # Iterate over each row of the column separately
    for row in np.arange(rows):
        # Run multiple iterations to remove all possible spikes (default = 3)
        cnt = 0
        while cnt < iters:
            # Calculate the median and standard deviations
            median = np.nanmedian(data[row, :])
            std = np.nanstd(data[row, :])
            
            # Find where the data is out-of-range
            good = np.where((data[row, :] < median + n_std * std) & (data[row, :] > median - n_std * std) &
                            (~np.isnan(data[row, :])))[0]

            # determine the number of bad points in the data set for the first iteration
            if cnt == 0:
                n_bad = n - len(good)

            # Now use the good data points to linearly interpolate to fill in the bad data points
            if len(good) > 0:
                ft = interp1d(t[good], data[row, good], kind="nearest", fill_value="extrapolate")
                data[row, :] = ft(t)

            # increment the counter for the next iteration
            cnt += 1

        # Save the total number of bad points
        bad.append(n_bad)
    
    return data, bad


def arctan3(y, x):
    """
    Four quadrant inverse tangents of the real elements of (y, x).
    0 <= np.atan2(y, x) <= 2*pi. The number of elements in (y, x)
    must be equal.

    Parameters
    ----------
    y: float, array_like
    x: float, array_like

    Returns
    -------
    theta: float, array_like
        The four quadrant inverse tangents

    References
    ----------
    Beardsley, Bob. 1999. AIR SEA Toolbox. Ver. 2.0. [Software: MatLab]
    """
    theta = np.arctan2(y, x)
    theta = np.atleast_1d(theta)
    index = np.where(theta < 0)[0]
    if len(index) > 0:
        theta[index] = theta[index] + 2 * np.pi
    
    return theta


def log_avg(f, s, n):
    """
    Logarithmically average the input spectrum s over frequencies f into n
    uniform bands.
    
    Parameters
    ----------
    f: array_like
        Frequencies f associated with the input spectrum
    s: array_like
        Input spectrum defined over frequencies f
    n: int
        Number of uniformly-spaced log10 frequencies bands over which
        the spectrum is averaged
        
    Returns
    -------
    F: array_like
        The array of averaged bandwidth-centered frequencies
    S: array_like
        The averaged spectrum
    dF: array_like
        The bandwidths of each frequency-band
    Ns: array_like
        The start indices of each frequency band
    Ne: array_like
        The end indices of each frequency band

    References
    ----------
    Gordon, Lee. 2001. NortekUSA LLC. [Software: MatLab]
    """
    lf = np.log(f)
    
    # Log frequency increment
    dlf = 1.000000001 * (lf[len(f) - 1] - lf[0]) / n
    
    # Get the array of transitions plus the final frequency
    NDX = 1 + np.floor((lf - lf[0]) / dlf)
    AA = np.where(np.diff(NDX) > 0)[0]
    AA = np.concatenate([AA, np.ones(1) * (len(f) - 1)]).astype(int)

    # Calculate the averaged spectrum and frequencies
    Cs = np.cumsum(s)
    Cf = np.cumsum(f)
    F = np.concatenate([np.array([Cf[AA[0]]]), np.diff(Cf[AA])]) / np.concatenate([np.array([AA[0] + 1]), np.diff(AA)])
    S = np.concatenate([np.array([Cs[AA[0]]]), np.diff(Cs[AA])]) / np.concatenate([np.array([AA[0] + 1]), np.diff(AA)])

    # Calculate the frequency bandwidths
    dF = np.concatenate([np.array([AA[0] + 1]), np.diff(AA)]) * (f[9] - f[8])

    # Get the start and end indices of each band
    Ns = np.concatenate([np.array([0]), AA[0:-1] + 1])
    Ne = AA
    
    return F, S, dF, Ns, Ne


def wave_spectra(u, v, p, dt, nF, params=None):
    """
    Calculate the wave direction and spreading using the Nortek PUV-method.
    
    This method derives the power spectra for the surface elevation based on
    both the pressure (heave) and velocity components, the directional wave
    statistics, and the associated frequency bands and degrees of freedom. 
    This is done by computing both the power spectra and cross-spectra, then
    band averaging in log-frequency space.
    
    Parameters
    ----------
    u: array_like
        East components of velocity (m/s)
    v: array_like
        North component of velocity (m/s)
    p: array_like
        Pressure (m)
    dt: float
        Sample interval (sec) (typically 0.5 or 1 sec)
    nF: array_like
        Nominal number of output frequencies
    params: list[lf_cutoff, max_fac, min_spec, n_dir]
        lf_cutoff: float, Default=0.03
            Low frequency cutoff where F < lf are not outputted
        max_fac: float, Default=200
            Largest factor scoping pressure to surface elevation
            Spectra and directions at F > max_fac are NaNs
        min_spec: float, Default= 0.03
            Minimum spectral level for which direction is computed
            Directions for spectra < minspec are returned NaNs
        n_dir: float, Default=0
            Direction of the "north" component (degrees)
    
    Returns
    -------
    u_spectra:
        Surface elevation spectra (m**2/Hz) based on velocity data
    p_spectra:
        Surface elevation spectra (m**2/Hz) based on pressure data
    Tdir:
        Mean wave direction (degree)
    Ts:
        Wave spreading (deg) in an array size [nf, nt] where
        nf is the number of output frequency bands 
        nt is the number of input time series
    F:
        Center frequency of each band
    dF:
        Bandwidth of each band
    dof:
        Degrees of freedom for each band

    References
    ----------
    Gordon, Lee. 2001. NortekUSA LLC. [Software: MatLab]
    """
    if not params:
        params = [0.03, 200, 0.03, 0]

    # Parse out the wave parameters
    lf_cutoff = params[0]
    # max_fac = params[1]
    min_spec = params[2]
    # n_dir = params[3]
    
    # Number of points in time series
    n = len(p)
    
    # If the length of the time series is odd, make it even
    if n % 2 == 1:
        u = u[:-1]
        v = v[:-1]
        p = p[:-1]
        n = len(p)
    Dt = n*dt
    
    # Frequency array up to the Nyquist frequency
    f = np.arange(1, (n + 1) / 2) / Dt
    
    # Compute the spectrum from velocity and pressure
    u_fft = fft(u)
    v_fft = fft(v)
    p_fft = fft(p)
    
    # Compute power spectrum and ignore zero frequency
    u_power = abs(u_fft[1:int((n / 2) + 1)]**2)
    v_power = abs(v_fft[1:int((n / 2) + 1)]**2)
    p_power = abs(p_fft[1:int((n / 2) + 1)]**2)
    
    # Scale power spectrum
    u_power = (u_power * 2) / n**2 / f[0]
    v_power = (v_power * 2) / n**2 / f[0]
    p_power = (p_power * 2) / n**2 / f[0]

    # Scale the cross-spectra and limit frequencies
    pu_power = np.real((p_fft * np.conj(u_fft)) * 2 / n**2 / f[0])
    pu_power = pu_power[1:int((n / 2) + 1)]
    pv_power = np.real((p_fft * np.conj(v_fft)) * 2 / n**2 / f[0])
    pv_power = pv_power[1:int((n / 2) + 1)]
    uv_power = np.real((u_fft * np.conj(v_fft)) * 2 / n**2 / f[0])
    uv_power = uv_power[1:int((n / 2) + 1)]
    
    # Average the power spectrum into log bands
    F, Cuu, _, _, _ = log_avg(f, u_power, nF)
    F, Cvv, _, _, _ = log_avg(f, v_power, nF)
    F, Cpp, _, _, _ = log_avg(f, p_power, nF)
    F, Cpu, _, _, _ = log_avg(f, pu_power, nF)
    F, Cpv, dF, Ns, Ne = log_avg(f, pv_power, nF)
    F, Cuv, _, _, _ = log_avg(f, uv_power, nF)
    dof = 2 * (Ne - Ns + 1)
    
    # Find low-frequency cutoff
    aa = np.where(F > lf_cutoff)[0]
    # lF = len(aa)

    # Filter the frequencies
    F = F[aa]
    dF = dF[aa]
    dof = dof[aa]
    
    # Filter the cross spectrum
    Cuu = Cuu[aa]
    Cvv = Cvv[aa]
    Cpp = Cpp[aa]
    Cpu = Cpu[aa]
    Cpv = Cpv[aa]
    Cuv = Cuv[aa]
    
    # Calculate the spectra
    u_spectra = (Cuu + Cvv)
    p_spectra = Cpp
    
    # Calculate the wave direction and spreading
    Tdir = 57.296 * arctan3(Cpu, Cpv)
    R2 = ((Cuu - Cvv)**2 + 4 * Cuv**2)**0.5 / (Cuu + Cvv)
    Ts = 57.296 * ((1 - R2) / 2)**0.5
    
    # Filter the direction and spreads for values below the minimum spectral levels
    ad = np.where(p_spectra < min_spec)[0]
    Tdir[ad] = np.nan
    Ts[ad] = np.nan
    
    return u_spectra, p_spectra, Tdir, Ts, F, dF, dof


def directional_statistics(p_spectra, Tdir, Ts, F, dF):
    """
    Calculate the wave statistics from the wave spectra using the Nortek PUV-method.
    
    Parameters
    -------
    p_spectra: array_like
        The surface elevation spectra (m**2/hz) based on pressure data
    Tdir: array_like
        The wave direction (deg)
    Ts: array_like
        The wave spreading (deg)
    F: array_like
        The center frequency of each band
    dF: array_like
        The bandwidth of each frequency band
        
    Returns
    -------
    Hm0: float
        The significant wave height
    Fs: float
        The peak frequency
    Tdir: float
        Wave direction at the peak frequency
    Ts:
        Wave spreading at the peak frequency

    References
    ----------
    Gordon, Lee. 2001. NortekUSA LLC. [Software: MatLab]
    """
    # Calculate the significant wave height
    Hm0 = np.max(np.cumsum(p_spectra * dF))**0.5 * 4

    # Find the peak spectra from the pressure
    B = np.max(p_spectra)
    nP = np.where(p_spectra == B)[0]
    
    A = p_spectra[nP-1][0]
    C = p_spectra[nP+1][0]
    
    # Calculate the peak frequency by interpolating using a parabolic fit
    Fs = (np.log(F[nP + 1]) - np.log(F[nP - 1])) * (-1 * (C - A) / (2 * (A - 2 * B + C))) / 2
    Fs = np.exp(np.log(F[nP]) + Fs)[0]
    
    # Identify the peak direction and spread
    Tdir = Tdir[nP][0]
    Ts = Ts[nP][0]
    
    return Hm0, Fs, Tdir, Ts


def magnetometer(data):
    """
    Process and clean the magnetometer data to get compass directions
    
    This function grabs the xyz magnetometer data, gets the headings,
    corrects for the orientation of z-positive downwards, adjusts for
    the magnetic to true north misalignment, and calculates the compass
    headings (in radians)
    
    Parameters
    ----------
    data: xarray.DataSet
        An xarray dataset object containing the MOPAK data
        
    Returns
    -------
    compass: array_like 
        An array of the calculated compass directions (radians)

    References
    ----------
    Edson, Jim. 2023. Motion Calculations Toolbox. [Software: MatLab]
    """
    # ----------------------------------------------------
    # First, grab the xyz magnetometer data
    magx = data.mopak_magx.values
    magy = data.mopak_magy.values
    magz = data.mopak_magz.values
    
    # Put the data into a matrix
    magnet = np.vstack([magx, magy, magz])
    
    # ----------------------------------------------------
    # Get headings and account for z is positive downwards
    # Account for magnetic to true north misalignment
    dev = -10 * np.pi / 180
    magnet[0] = magnet[0]
    magnet[1] = -1 * magnet[1]
    magnet[2] = -1 * magnet[2]
    
    # Calculate the compass directions in radians
    compass = np.arctan2(magnet[1], magnet[0]) + dev

    # Correct for values that fall outside real ranges
    mask = compass > np.pi
    compass[mask] = compass[mask] - np.pi
    mask = compass < -1 * np.pi
    compass[mask] = compass[mask] + np.pi

    # Reorient to account for z-direction
    compass = -1 * compass
    
    return compass


def accelerations(data):
    """
    Process and clean the xyz accelerations
    
    This function get the MOPAK accelerations in the x, y, z
    directions (in units of g-force), corrects for the
    orientation of z-positive downwards, and derives the local
    free-fall values.
    
    Parameters
    ----------
    data: xarray.DataSet
        An xarray dataset object containing the MOPAK data downloaded
        from OOINet
    
    Returns
    -------
    platform: array_like
        A (3 x n) array of the mopak accelerations
        in the x, y, z directions in units of m/s^2
    gravity: float
        The local free-fall estimate

    References
    ----------
    Edson, Jim. 2023. Motion Calculations Toolbox. [Software: MatLab]
    """
    # standard gravity constant
    g = 9.80665

    # Mopak accelerations (in g-force units)
    ax = data.mopak_accelx.values
    ay = data.mopak_accely.values
    az = data.mopak_accelz.values

    # Put the accelerations into a (3 x n) array
    platform = np.vstack([ax, ay, az])
    
    # Reorient y & z positions (z is positive down)
    platform[0] = platform[0]
    platform[1] = -1 * platform[1]
    platform[2] = -1 * platform[2]
    
    # Calculate the local gravity values
    gravxyz = np.mean(platform.transpose(), axis=0)
    gravity = np.sqrt(np.sum(gravxyz**2))
    
    # Adjust the accelerations to be in m/s^2
    platform = platform * g
    
    return platform, gravity

    
def angular_rates(data):
    """
    Process and clean the angular rates
    
    This function gets the MOPAK angular rates in the x, y, z
    directions and corrects for the orientation of z-positive 
    downwards.
    
    Parameters
    ----------
    data: xarray.DataSet
        An xarray dataset object containing the MOPAK data
    
    Returns
    -------
    a_rates: array_like
        A (3 x n) array of the mopak angular rates

    References
    ----------
    Edson, Jim. 2023. Motion Calculations Toolbox. [Software: MatLab]
    """
    # Get the xyz angular rates
    dx = data.mopak_ang_ratex.values
    dy = data.mopak_ang_ratey.values
    dz = data.mopak_ang_ratez.values

    # Put into a matrix
    angular_rate = np.vstack([dx, dy, dz])
    
    # Reorient the y and z positions (right-hand-rule)
    angular_rate[0] = angular_rate[0]
    angular_rate[1] = -1*angular_rate[1]
    angular_rate[2] = -1*angular_rate[2]
    
    return angular_rate


def build_dataset(ds, number_zero_crossings, significant_wave_height, significant_wave_period, wave_height_10,
                  wave_period_10, peak_wave_period, mean_wave_height, mean_wave_period, peak_wave_direction_puv,
                  peak_wave_spread_puv, peak_wave_period_puv, significant_wave_height_puv, sample_start_time,
                  deployment):
    """
    Takes in the calculated wave statistics are builds an xarray dataset
    
    Parameters
    ----------
    ds: xarray.Dataset
        An xarray dataset object containing the MOPAK data downloaded
        from OOINet
    number_zero_crossings: array_like
        The number of zero-crossings (downwards) identified during the
        observation period
    significant_wave_height: array_like
        The wave height of the highest 1/3 of waves measured during the
        observation period (units: m)
    significant_wave_period: array_like
        The mean period of the highest 1/3 of waves measured during the
        observation period (units: s)
    wave_height_10: array_like
        The wave height of the highest tenth of waves measured during the
        observation period (units: m)
    wave_period_10: array_like
        The wave period of the highest tenth of waves measured during the
        observation period (units: s)
    peak_wave_period: array_like
        The period of the wave calculated from the frequency associated with
        the peak in the wave spectra (units: s)
    mean_wave_height: array_like
        The mean wave height (units: m)
    mean_wave_period: array_like
        The mean wave period (units: s)
    peak_wave_direction_puv: array_like
        The peak wave direction calculated using the Nortek PUV-method (units: degrees)
    peak_wave_spread_puv: array_like
        The wave spread of the peak wave calculated using the Nortek PUV-method
        (units: degrees)
    peak_wave_period_puv: array_like
        The peak wave period calculated using the Nortek PUV-method and a 
        parabolic fit across the peak frequency band (units: s)
    significant_wave_height_puv: array_like
        The significant wave height calculated using the Nortek PUV-method (hm0)
        (units: m)
    sample_start_time: array_like
        Either an array of datetime strings or datetime objects that correspond
        to the start of each sampling period
    deployment: int, float, str
        The deployment number of the dataset being processed
        
    Returns
    -------
    ds: xarray.Dataset
        An xarray dataset which contains the wave statistics data, indexed via the
        sample_start_time, with associated metadata    
    """
    # Check that the data is all 1-d arrays
    number_zero_crossings = np.atleast_1d(number_zero_crossings)
    significant_wave_height = np.atleast_1d(significant_wave_height)
    significant_wave_period = np.atleast_1d(significant_wave_period)
    wave_height_10 = np.atleast_1d(wave_height_10)
    wave_period_10 = np.atleast_1d(wave_period_10)
    peak_wave_period = np.atleast_1d(peak_wave_period)
    mean_wave_height = np.atleast_1d(mean_wave_height)
    mean_wave_period = np.atleast_1d(mean_wave_period)
    peak_wave_direction_puv = np.atleast_1d(peak_wave_direction_puv)
    peak_wave_spread_puv = np.atleast_1d(peak_wave_spread_puv)
    peak_wave_period_puv = np.atleast_1d(peak_wave_period_puv)
    significant_wave_height_puv = np.atleast_1d(significant_wave_height_puv)
    sample_start_time = np.atleast_1d(sample_start_time)
    
    # Check that the sample_start_times are datetime objects
    sample_start_time = sample_start_time.astype("datetime64[ns]")
    
    # Build an array of the deployment number
    deployment = np.ones(sample_start_time.shape) * int(deployment)
    deployment = deployment.astype("int")
    
    # Create a dictionary object of the data variables
    data_vars = dict(
        number_zero_crossings=(["time"], number_zero_crossings),
        significant_wave_height=(["time"], significant_wave_height),
        significant_wave_period=(["time"], significant_wave_period),
        wave_height_10=(["time"], wave_height_10),
        wave_period_10=(["time"], wave_period_10),
        peak_wave_period=(["time"], peak_wave_period),
        mean_wave_height=(["time"], mean_wave_height),
        mean_wave_period=(["time"], mean_wave_period),
        peak_wave_direction=(["time"], peak_wave_direction_puv),
        peak_wave_spread=(["time"], peak_wave_spread_puv),
        peak_wave_period_puv=(["time"], peak_wave_period_puv),
        wave_height_hm0=(["time"], significant_wave_height_puv),
        deployment=(["time"], deployment))

    coords = dict(time=sample_start_time)
       
    # Build a new dataset, reusing the original dataset attributes so we keep that metadata
    waves = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=ds.attrs
    )

    # replace the dataset description and title
    waves.attrs['Description'] = 'Bulk and directional wave statistics derived from the 3-axis motion sensor data'
    waves.attrs['title'] = 'Bulk and Directional Wave Statistics'

    # Add the variable attributes
    for var in waves.variables:
        waves[var].attrs = WAVES[var]
    
    return waves


def calculate_wave_statistics(ds, fs, com_offset=None, f_cutoff=1/30, lf_cutoff=0.03, max_fac=200,
                              min_spec=0.03, n_dir=0):
    """
    Calculate the directional and non-directional wave statistics and return a
    new dataset.
    
    This function takes in a dataset from the 3-axis motion pack (MOPAK) and
    processes it to derive the directional and non-directional wave statistics,
    which are returned as a new dataset. First, the accelerometer, angular
    rate, and magnetic declination data from the MOPAK are reprocessed to
    derive the displacements (x,y,z) and velocities (u,v,w). Next, the bulk
    wave statistics are calculated using a zero down-crossing algorithm. The
    directional statistics are derived from the wave power and cross-spectra.
    
    Parameters
    ----------
    ds: xarray.DataSet
        A dataset containing the unprocessed MOPAK data
    fs: float
        The sampling frequency of the MOPAK data (Hz)
    f_cutoff: float, Default=1/30
        The frequency cutoff period for waves (Hz)
    com_offset: list[x, y, z], Default=[0, 0, 0.5]
        A list of the offsets from the center of mass of the MOPAK (m)
    lf_cutoff: float, Default=0.03
        Low frequency cutoff where F < lf are not output (Hz)
    max_fac: float, Default=200
        Largest factor scoping pressure to surface elevation
        Spectra and directions at F > max_fac are NaNs
    min_spec: float, Default= 0.03
        Minimum spectral level for which direction is computed
        Directions for spectra < minspec are returned NaNs
    n_dir: float, Default=0
        Direction of the "north" component (degrees)

    Returns
    -------
    wave_stats: xarray.DataSet
        A dataset containing the computed bulk and directional wave statistics
        from the associated 3-axis motion sensor data. The returned dataset
        variables are:
            * number_zero_crossings
                The number of zero-crossings (downwards) identified during the
                observation period
            * significant_wave_height
                The wave height of the highest 1/3 of waves measured during the
                observation period (units: m)
            * significant_wave_period
                The mean period of the highest 1/3 of waves measured during the
                observation period (units: s)
            * wave_height_10
                The wave height of the highest tenth of waves measured during
                the observation period (units: m)
            * wave_period_10
                The wave period of the highest tenth of waves measured during
                the observation period (units: s)
            * peak_wave_period
                The period of the wave calculated from the frequency associated
                with the peak in the wave spectra (units: s)
            * mean_wave_height
                The mean wave height (units: m)
            * mean_wave_period
                The mean wave period (units: s)
            * peak_wave_direction_puv
                The peak wave direction calculated using the Nortek PUV-method
                (units: degrees)
            * peak_wave_spread_puv
                The wave spread of the peak wave calculated using the Nortek
                PUV-method (units: degrees)
            * peak_wave_period_puv
                The peak wave period calculated using the Nortek PUV-method and a 
                parabolic fit across the peak frequency band (units: s)
            * significant_wave_height_puv
                The significant wave height calculated using the Nortek PUV-method (hm0)
                (units: m)
            * sample_start_time
                The timestamp hat correspond to the start of each sampling period
            * deployment: int, float, str
                The deployment number of the dataset being processed

    References
    ----------
    1998. Edson, J.B., A.A. Hinton, K.E. Prada, J.E. Hare, & C.W. Fairall, “Direct covariance flux estimates 
        from mobile platforms at sea,”  J. Atmos. Oceanic Tech., 15, 547-562
    2001. McGillis, W.R., J.B. Edson, J.E. Hare, & C.W. Fairall, “Direct covariance air-sea CO2 fluxes,” 
        J. Geophys. Res., 106, 16729-16745.
    2003. Fairall, C.W., E.F. Bradley, J.E. Hare, A.A. Grachev, & J.B. Edson, “Bulk parameterization of 
        air–sea fluxes: Updates and verification for the COARE algorithm,” J. Climate, 16, 571–591.
    2004. Edson, J.B., C.J. Zappa, J.A. Ware, W.R. McGillis, & J.E. Hare, “Scalar flux profile relationships
        over the open ocean,” J. Geophys. Res., 109, C08S09, doi:10.1029/2003JC001960.
    2008. Miller, S., C. Friehe, T. Hristov, & J. Edson, “Platform motion effects on measurements of 
        turbulence and air-sea exchange over the open ocean,” J. Atmos. Oceanic Tech., 25, 1683-1694.
    2012. Flügge, M., J.B. Edson, & J. Reuder, “Sensor Movement Correction for Direct Turbulence Measurements 
        in the Marine Atmospheric Boundary Layer,” Energy Procedia, 24, 159-165.
    2013. Edson, J.B., V. Jampana, R.A. Weller, S. Bigorre, A.J. Plueddemann, C.W. Fairall, S.D. Miller, 
        L. Mahrt, D. Vickers, and H. Hersbach, “On the exchange of momentum over the open ocean,” 
        J. Phys. Oceanogr., 43, 1589–1610.
    """
      
    # --------------------------------------------------------------------
    # First, identify each wave sample
    sample = identify_samples(ds, 2400)
    
    # Add the sample as a new variable to the dataset
    ds["sample"] = ("time", sample)
    
    # --------------------------------------------------------------------
    # Grab the constants and parameters
    if com_offset is None:  # Center of mass offset from the MOPAK (m)
        com_offset = [0, 0, 0.5]

    g = 9.80665  # standard gravity
    params = [lf_cutoff, max_fac, min_spec, n_dir]
       
    # --------------------------------------------------------------------
    # Process and return the magnetic directions, platforms accelerations,
    # and angular rates from the MOPAK data
    Compass = magnetometer(ds)
    Platform, gravity = accelerations(ds)
    Deg_rate = angular_rates(ds)

    # --------------------------------------------------------------------
    # Calculate the wave statistics by iterating through each wave sample
    # (Note: can probably speed this portion up using Dask)
    number_zero_crossings = []
    significant_wave_height = []
    significant_wave_period = []
    wave_height_10 = []
    wave_period_10 = []
    peak_wave_period = []
    mean_wave_height = []
    mean_wave_period = []
    peak_wave_direction_puv = []
    peak_wave_spread_puv = []
    peak_wave_period_puv = []
    significant_wave_height_puv = []
    sample_start_time = []

    # Remove edge effects
    edge = np.fix(1 * 30 * fs)

    for sample in np.unique(ds.sample):
        # Find the data associated with each sample
        index, = np.where(ds.sample == sample)

        # Get the start time
        tstart = ds.time[index].min().values

        # Check the length of the index
        if len(index) < 10000:
            pass
        else:
            # ----------------------------------------------------------------
            # Get the angular rates, accelerations, and compass data 
            # associated with the given sample data
            platform = Platform[:, index]
            deg_rate = Deg_rate[:, index]
            gyro = Compass[index]

            # ----------------------------------------------------------------
            # Calculate the wave statistics
            # Determine if to use dask or go serial
            n, Hsig, t_sig, h_10, t_10, Tsig, h_avg, t_avg, Tdir, Ts, Fs, Hm0 = wave_statistics(platform, deg_rate,
                                                                                                gyro, fs, f_cutoff,
                                                                                                com_offset, g, params,
                                                                                                edge)

            # correct the peak wave direction for the magnetic declination
            declination = magnetic_declination(ds.attrs["lat"], ds.attrs["lon"], tstart.astype(float) / 1e9, 0)
            Tdir = Tdir + declination
            if Tdir > 360:
                Tdir = Tdir - 360
            elif Tdir < 0:
                Tdir = Tdir + 360

            # Save the results
            number_zero_crossings.append(n)          # Calculated from zero-crossings: Method B
            significant_wave_height.append(Hsig)     # Calculated from zero-crossings: Method B
            significant_wave_period.append(t_sig)    # Calculated from zero-crossings: Method B
            wave_height_10.append(h_10)              # Calculated from zero-crossings: Method B
            wave_period_10.append(t_10)              # Calculated from zero-crossings: Method B
            peak_wave_period.append(Tsig)            # Calculated from zero-crossings: Method A
            mean_wave_height.append(h_avg)           # Calculated from zero-crossings: Method B
            mean_wave_period.append(t_avg)           # Calculated from zero-crossings: Method B
            peak_wave_direction_puv.append(Tdir)     # Calculated from the PUV method
            peak_wave_spread_puv.append(Ts)          # Calculated from the PUV method
            peak_wave_period_puv.append(1/Fs)        # Calculated from the PUV method
            significant_wave_height_puv.append(Hm0)  # Calculated from the PUV method (Hm0)
            sample_start_time.append(tstart)

    # --------------------------------------------------------------------
    # Get the deployment number
    deployment = np.unique(ds.deployment)

    # --------------------------------------------------------------------
    # Build the wave statistics dataset
    wave_stats = build_dataset(ds, number_zero_crossings, significant_wave_height, significant_wave_period,
                               wave_height_10, wave_period_10, peak_wave_period, mean_wave_height, mean_wave_period,
                               peak_wave_direction_puv, peak_wave_spread_puv, peak_wave_period_puv,
                               significant_wave_height_puv, sample_start_time, deployment)
    
    return wave_stats


def wave_statistics(platform, deg_rate, gyro, fs, f_cutoff, com_offset, g, params, edge):
    """Wrapper function to calculate the directional and non-directional wave statistics."""
    # Get the velocities and displacements
    uvw, xyz = uvw_xyz(gyro, platform, deg_rate, fs, f_cutoff, com_offset, g)

    # Remove the data at the beginning and end of the timeseries to 
    # reduce edge effects due to filtering
    incra = edge
    incrb = xyz.shape[-1] - edge
    incr = np.arange(incra, incrb+1, 1)
    incr = [int(x) for x in incr]

    # Calculate non-directional statistics from the zero-crossings
    z = xyz[2, :][incr]
    npt = np.min([2**13, len(incr)])
    
    # Method A (Peak Wave Period) and check on displacement (Havg here should be zero)
    Hsig, Havg, Tsig, Tavg = non_directional_statistics(z, fs, npt)
    
    # Method B - zero crossings algorithm
    n, h_sig, t_sig, h_10, t_10, h_avg, t_avg = zero_crossing(z, fs)

    # Calculate the wave spectra and the directional statistics
    vu = uvw[1, :][incr]
    vv = -uvw[0, :][incr]
    vp = z

    # Wave spectra
    u_spectra, p_spectra, wave_direction, wave_spread, F, dF, dof = wave_spectra(vu, vv, vp, 1/fs, 100, params)
    
    # Directional statistics
    Hm0, Fs, Tdir, Ts = directional_statistics(p_spectra, wave_direction, wave_spread, F, dF)
    
    # Return all the calculated wave statistics
    return n, Hsig, t_sig, h_10, t_10, Tsig, h_avg, t_avg, Tdir, Ts, Fs, Hm0


def mopak_datalogger(ds):
    """
    Takes MOPAK data recorded by the data loggers used in the CGSN/EA moorings
    and cleans up the data set to make it more user-friendly.  Primary task is
    renaming parameters and dropping some that are of limited use. Additionally,
    use the 3-axis motion pack (MOPAK) data to calculate the wave statistics.

    :param ds: initial MOPAK data set downloaded from OOI via the M2M system
    :return ds: cleaned up data set with renamed parameters
    :return wave_stats: calculated wave statistics from the MOPAK data
    """
    # drop some of the variables:
    #   internal_timestamp == redundant with time, can remove
    #   mopak_timer == internal_timestamp, can remove as well
    drop_vars = ['internal_timestamp', 'mopak_timer']
    for var in ds.variables:
        if var in drop_vars:
            ds = ds.drop_vars(var)

    # use the MOPAK data as-is to calculate the wave statistics
    wave_stats = calculate_wave_statistics(ds, fs=10)

    # lots of renaming here to get a better defined data set with cleaner attributes
    rename = {
        'mopak_accelx': 'accelerometer_x',
        'mopak_accely': 'accelerometer_y',
        'mopak_accelz': 'accelerometer_z',
        'mopak_ang_ratex': 'angular_rate_x',
        'mopak_ang_ratey': 'angular_rate_y',
        'mopak_ang_ratez': 'angular_rate_z',
        'mopak_magx': 'magnetometer_x',
        'mopak_magy': 'magnetometer_y',
        'mopak_magz': 'magnetometer_z',
    }
    ds = ds.rename(rename)

    # reset some attributes
    for key, value in MOPAK.items():
        for atk, atv in value.items():
            if key in ds.variables:
                ds[key].attrs[atk] = atv

    # add the original variable name as an attribute, if renamed
    for key, value in rename.items():
        ds[value].attrs['ooinet_variable_name'] = key
        
    return ds, wave_stats


def main(argv=None):
    args = inputs(argv)
    site = args.site
    node = args.node
    sensor = args.sensor
    method = args.method
    stream = args.stream
    deploy = args.deploy
    start = args.start
    stop = args.stop

    # check if we are specifying a deployment or a specific date and time range
    if not deploy or (start and stop):
        return SyntaxError('You must specify either a deployment number or beginning and end dates of interest.')

    # if we are specifying a deployment number, then get the start and end dates for that deployment
    if deploy:
        start, stop = get_deployment_dates(site, node, sensor, deploy)
        tag = '.*deployment{:04d}.*MOPAK.*\\.nc$'.format(deploy)  # download MOPAK files for the specified deployment
    else:
        # otherwise, set the tag to download all MOPAK files for the specified date range regardless of deployment
        tag = '.*MOPAK.*\\.nc$'

    # make the M2M request for the MOPAK data
    r = m2m_request(site, node, sensor, method, stream, start, stop)
    if not r:
        exit_text = ('Request failed for %s-%s-%s, %s, %s, from %s to %s.' % (site, node, sensor, method,
                                                                              stream, start, stop))
        raise SystemExit(exit_text)

    # Valid M2M request, start downloading the data
    mopak = m2m_collect(r, tag)

    # check to see if we downloaded any data
    if not mopak:
        exit_text = ('Data unavailable for %s-%s-%s, %s, %s, from %s to %s.' % (site, node, sensor, method,
                                                                                stream, start, stop))
        raise SystemExit(exit_text)

    mopak, waves = mopak_datalogger(mopak)
    vocab = get_vocabulary(site, node, sensor)[0]
    mopak = update_dataset(mopak, vocab['maxdepth'])
    waves = update_dataset(waves, vocab['maxdepth'])

    # set the output file name and directory, creating it if it doesn't exist
    out_file = os.path.abspath(args.outfile)
    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    # save the MOPAK data to disk
    mopak.to_netcdf(out_file, mode='w', format='NETCDF4', engine='h5netcdf', encoding=ENCODINGS)

    # save the wave statistics to disk
    out_file = os.path.abspath(args.outfile.replace('.nc', '_waves.nc'))
    waves.to_netcdf(out_file, mode='w', format='NETCDF4', engine='h5netcdf', encoding=ENCODINGS)


if __name__ == '__main__':
    main()
