#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Christopher Wingard
@brief common functions for processing profiler data
"""
import numpy as np
import xarray as xr


def create_profile_id(ds):
    """
    Use a combination of the deployment number and profile sequence within a
    deployment to create a unique profile identifier for each profile in the
    data set.

    :param ds: dataset containing the profile data
    :return: dataset with a profile variable added
    """
    # create an initial profile number for each record in the data set
    ds['profile'] = ds['deployment'].astype('int32') * 0

    # group the data set by deployment
    deployments = ds.groupby('deployment')

    # loop over each deployment
    for deploy in deployments:
        # split the deployment into individual profiles
        profiles = split_profiles(deploy[1])
        profile_id = []
        # add the sequential number for each profile, creating a data array object
        for i, p in enumerate(profiles):
            profile_id.append(p['profile'] + i + 1)

        # combine the profile_id's back into a single data array
        profile_id = xr.concat(profile_id, 'time')

        # update the profile number for each profile in the deployment
        ds['profile'].loc[{'time': profile_id['time']}] = profile_id

    # update the attributes for the profile variable
    ds['profile'].attrs = dict({
        'long_name': 'Profile Number',
        'cf_role': 'profile_id',
        'units': 'count',
        'comment': ('Unique identifier for each profile in a deployment, created by separating the data set into '
                    'individual profiles and then adding a sequential number to each profile. The profile number '
                    'resets to 1 at the start of each deployment. ')
    })
    return ds


def split_profiles(ds):
    """
    Split the data set into individual profiles, where each profile is a
    collection of data from a single deployment and profile sequence. The
    resulting data sets are returned in a list.

    :param ds: data set containing the profile data
    :return: a list of data sets, one for each profile
    """
    # split the data into profiles, assuming at least 120 seconds between profiles
    dt = ds.where(ds['time'].diff('time') > np.timedelta64(120, 's'), drop=True).get_index('time')

    # process each profile, adding the results to a list of profiles
    profiles = []
    jback = np.timedelta64(30, 's')  # 30 second jump back to avoid collecting data from the following profile
    for i, d in enumerate(dt):
        # pull out the profile
        if i == 0:
            profile = ds.sel(time=slice(ds['time'].values[0], d - jback))
        else:
            profile = ds.sel(time=slice(dt[i - 1], d - jback))

        # add the profile to the list
        profiles.append(profile)

    # grab the last profile and append it to the list
    profile = ds.sel(time=slice(d, ds['time'].values[-1]))
    profiles.append(profile)
    return profiles


def bin_profiles(profile, site_depth, bin_size=0.25):
    """
    Bin the data in the profile into a set of bins of a given size (default is
    25 cm). The bin depth is set to the center of the bin, using the median
    value of the data in each bin.

    :param profile: data set containing the profile data
    :param site_depth: maximum depth of the site, used to set binning range
    :param bin_size: size of the bin, in meters
    :return: a data set containing the binned data
    """
    # test the length of the profile, short ones with less than 15 records will be skipped
    if len(profile.time) < 15:
        return None

    # test the time range of the profile, short ones (less than 15 seconds) will be skipped
    if (profile.time[-1] - profile.time[0]).astype(float) / 10 ** 9 < 15:
        return None

    # use a set of median boxcar filters to help despike the data
    smth = profile.rolling(time=5, min_periods=3, center=True).median(keep_attrs=True).dropna("time", subset=['deployment'])
    smth = smth.rolling(time=5, min_periods=3, center=True).median(keep_attrs=True).dropna("time", subset=['deployment'])

    # bin the data using the center of the bin as the depth value
    bins = smth.groupby_bins('depth', np.arange(bin_size / 2, site_depth + bin_size / 2, bin_size))
    binning = []
    for grp in bins:
        avg = grp[1].mean('time', keepdims=True, keep_attrs=True)
        avg = avg.assign_coords({'time': np.atleast_1d(grp[1].time.mean().values)})  # add time back
        avg['depth'] = avg['depth'] * 0 + grp[0].mid  # set depth to bin midpoint
        binning += avg,  # append to the list

    binned = xr.concat(binning, 'time')
    return binned


def updown(db, db_level):
    """
    Creates an indexing array marking when a moving profiler has changed
    directions. Also creates a flag array, equal in length to the original
    data, indicating the direction of travel. Note, the direction is relative
    to the pressure record's syntax.

    :param db: depth/pressure record
    :param db_level: threshold indicating a "true" change in direction.
    :return pks: indexing array marking when the direction of travel (up or
        down) changes.
    :return dzdt: flag array indicating the direction the profiler is moving,
        equal in length to db (note: dzdt is relative to pressure record
        syntax).
    """
    # determine the up/down cycle of the profiler and mark the local min/max extrema
    # via an indexing array.
    j = np.sign(np.diff(np.concatenate(([-np.inf], db, [-np.inf]))))
    j = np.where(np.diff(j + (j == 0)) == -2)[0]
    k = np.sign(np.diff(np.concatenate(([np.inf], db, [np.inf]))))
    k = np.where(np.diff(k + (k == 0)) == 2)[0]
    pks = np.sort(np.concatenate((k, j)))

    # mark and delete index locations where there isn't a "true" direction change
    cnt = 0
    step = 1
    m = np.ones(len(pks), dtype=bool)
    while cnt < len(pks):
        if cnt + step == len(pks):
            break

        # compare the magnitude of the difference between the extrema
        dz1 = db[pks[cnt]] - db[pks[cnt + step]]  # magnitude of difference
        sg1 = np.sign(dz1)  # sign of difference

        if abs(dz1) < db_level:  # change is not large enough
            m[cnt + step] = False  # mark extrema for deletion
            step += 1  # bump the stepper
        else:  # change is large enough, but does it qualify as a direction change?
            if cnt + step + 1 == len(pks):
                break

            j = cnt + step
            dz2 = db[pks[j]] - db[pks[j + 1]]  # magnitude of difference
            sg2 = np.sign(dz2)  # sign of difference

            if abs(dz2) < db_level:  # change is not large enough -- or is it ???
                if sg1 == sg2:  # nahh, we're still going the same way
                    m[j] = False  # mark extrema for deletion
                    step = step + 1  # bump the stepper
                else:  # maybe -- this slows things down quite a bit
                    # find extrema > db_level to the left of the current point
                    extrema = np.abs(db[pks[:j]] - db[pks[j]])
                    lft = extrema[extrema > db_level]
                    if len(lft) == 0:
                        a = 0
                    else:
                        a = np.where(extrema == lft[-1])[0][0]

                    # find extrema > db_level to the right of the current point
                    extrema = np.abs(db[pks[j]+1:] - db[pks[j]])
                    rght = extrema[extrema > db_level]
                    if len(rght) == 0:
                        b = len(pks)
                    else:
                        b = j + np.where(extrema == rght[0])[0][0]

                    # set the min and max of the window
                    mn = np.min(db[pks[a:b]])
                    mx = np.max(db[pks[a:b]])

                    # see if our point is a min or max of this window
                    if mx == db[pks[j]] or mn == db[pks[j]]:
                        # we have a valid turn -- reset the counters
                        cnt = j
                        step = 1
                    else:  # it's not a valid peak
                        m[j] = False  # mark extrema for deletion
                        step = step + 1  # bump the stepper
            else:  # we have a valid turn -- reset the counters
                cnt = j
                step = 1

    pks = pks[m]  # remove non-valid extrema

    # mark the tail
    if pks[-1] != len(db):
        pks = np.concatenate((pks, [len(db) - 1]))

    # refine the pks array to catch minor, mid-profile changes that slipped through
    flg = np.ones(len(pks)).astype(bool)
    for i in range(1, len(pks) - 1):
        flg[i] = (db[pks[i]] > db[pks[i - 1]]) & (db[pks[i]] > db[pks[i + 1]]) or \
                 (db[pks[i]] < db[pks[i - 1]]) & (db[pks[i]] < db[pks[i + 1]])

    pks = pks[flg]  # remove mid-profile changes

    # create the direction array
    dz = np.sign(np.diff(db[pks])).astype(int)
    dzdt = np.zeros(len(db)).astype(int)
    for i in range(len(pks) - 1):
        dzdt[pks[i]:pks[i + 1]+1] = dz[i]

    return pks, dzdt


def pair_profiles(ds, break_t = 60):
    """
    Function to pair profiles with their partner
    profile.
    
    :param ds: the profile dataset with the profile variable added
    :param break_t: the time (min) separation below which a profile is paired
    :return ds: the profile dataset with the paired profile variable added
    """

    parameters = [p for p in ds.variables]
    if 'profile' not in parameters:
        raise NameError('Dataset must have the variable "profile"')

    # First, identify the start times of each individual profile
    t = []
    for p in np.unique(ds['profile']):
        idx, = np.where(ds['profile'] == p)
        t.append(ds['time'][idx[0]].values)

    # Next, calculate the time differences
    dt = np.diff(t).astype('timedelta64[m]')
    
    # Find where the breaks exceed 1 hour
    breaks, = np.where(dt > np.timedelta64(60))

    # Now, we need to use the locations of the breaks to group profiles
    profiles = np.unique(ds['profile'])
    p1 = profiles[breaks]
    p2 = profiles[breaks] + 1

    grouping = []
    for p in profiles:
        if p==1:
            if p in p1:
                g = np.nan
            else:
                g = p + 1
        elif p==profiles[-1]:
            if p in p2:
                g = np.nan
            else:
                g = p - 1
        else:
            if (p in p1) and (p not in p2):
                g = p - 1
            elif (p in p2) and (p not in p1):
                g = p + 1
            else:
                g = np.nan
        grouping.append(g)

    # Now, add in the paired profile variable to the dataset
    pairs = np.zeros(ds['profile'].shape)
    for p in profiles:
        idx, = np.where(ds['profile'] == p)
        pairs[idx] = grouping[p-1]
        
    ds['profile_pair'] = (['time'], pairs)
    ds['profile_pair'].attrs = {
        'long_name': 'Paired Profile Number',
        'units': 'count',
        'comment': ('The unique identifier for the profile that is paired to the '
                    '"profile". The standard operation of deployed profilers is '
                    'to collect a pair of profiles. However, mechanical issues or '
                    'power limitations may cause the profiler to operate in "singe" '
                    'profile mode.')
    }

    return ds
