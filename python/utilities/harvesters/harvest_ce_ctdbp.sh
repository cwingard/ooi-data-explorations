#!/usr/bin/env bash
#
# harvest_ce_ctdbp.sh
#
# Harvest the ctdbp data from all of the OOI Coastal Endurance moorings. Data
# sets include telemetered and recovered host and instrument data. Data is
# downloaded from OOI Net and reworked to create a cleaner and more consistent
# set of files named and organized by the mooring, mooring sub-location, data
# delivery method and deployment.
#
# C. Wingard, 2019-07-22 -- Initial code

# set the base directory python command for all subsequent processing
. $(dirname $CONDA_EXE)/../etc/profile.d/conda.sh
conda activate ooi
PYTHON="python -m ooi_data_explorations.uncabled.process_ctdbp"

### CE01ISSM ###
BASE_FLAGS="-s CE01ISSM -n SBD17 -sn 06-CTDBPC000"
BASE_FILE="${HOME}/ooidata/m2m/ce01issm/buoy/ctdbp/ce01issm.buoy.ctdbp"
for i in $(seq -f "%02g" 1 14); do
    $PYTHON $BASE_FLAGS -mt telemetered -st ctdbp_cdef_dcl_instrument -dp $i -o "$BASE_FILE.deploy$i.telemetered.ctdbp_cdef_dcl_instrument.nc"
    $PYTHON $BASE_FLAGS -mt recovered_host -st ctdbp_cdef_dcl_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_host.ctdbp_cdef_dcl_instrument_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_inst -st ctdbp_cdef_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.ctdbp_cdef_instrument_recovered.nc"
done

BASE_FLAGS="-s CE01ISSM -n RID16 -sn 03-CTDBPC000"
BASE_FILE="${HOME}/ooidata/m2m/ce01issm/nsif/ctdbp/ce01issm.nsif.ctdbp"
for i in $(seq -f "%02g" 1 14); do
    $PYTHON $BASE_FLAGS -mt telemetered -st ctdbp_cdef_dcl_instrument -dp $i -o "$BASE_FILE.deploy$i.telemetered.ctdbp_cdef_dcl_instrument.nc"
    $PYTHON $BASE_FLAGS -mt recovered_host -st ctdbp_cdef_dcl_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_host.ctdbp_cdef_dcl_instrument_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_inst -st ctdbp_cdef_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.ctdbp_cdef_instrument_recovered.nc"
done

BASE_FLAGS="-s CE01ISSM -n MFD37 -sn 03-CTDBPC000"
BASE_FILE="${HOME}/ooidata/m2m/ce01issm/seafloor/ctdbp/ce01issm.seafloor.ctdbp"
for i in $(seq -f "%02g" 1 14); do
    $PYTHON $BASE_FLAGS -mt telemetered -st ctdbp_cdef_dcl_instrument -dp $i -o "$BASE_FILE.deploy$i.telemetered.ctdbp_cdef_dcl_instrument.nc"
    $PYTHON $BASE_FLAGS -mt recovered_host -st ctdbp_cdef_dcl_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_host.ctdbp_cdef_dcl_instrument_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_inst -st ctdbp_cdef_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.ctdbp_cdef_instrument_recovered.nc"
done

### CE02SHSM ###
BASE_FLAGS="-s CE02SHSM -n RID27 -sn 03-CTDBPC000"
BASE_FILE="${HOME}/ooidata/m2m/ce02shsm/nsif/ctdbp/ce02shsm.nsif.ctdbp"
for i in $(seq -f "%02g" 1 12); do
    if [ $i -le 7 ]; then
      # average the bursts for deployments prior to deployment 7
      $PYTHON $BASE_FLAGS -mt telemetered -st ctdbp_cdef_dcl_instrument -ba -dp $i -o "$BASE_FILE.deploy$i.telemetered.ctdbp_cdef_dcl_instrument.nc"
      $PYTHON $BASE_FLAGS -mt recovered_host -st ctdbp_cdef_dcl_instrument_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.recovered_host.ctdbp_cdef_dcl_instrument_recovered.nc"
      $PYTHON $BASE_FLAGS -mt recovered_inst -st ctdbp_cdef_instrument_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.ctdbp_cdef_instrument_recovered.nc"
    else
      # no longer using burst mode, just collect and rework
      $PYTHON $BASE_FLAGS -mt telemetered -st ctdbp_cdef_dcl_instrument -dp $i -o "$BASE_FILE.deploy$i.telemetered.ctdbp_cdef_dcl_instrument.nc"
      $PYTHON $BASE_FLAGS -mt recovered_host -st ctdbp_cdef_dcl_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_host.ctdbp_cdef_dcl_instrument_recovered.nc"
      $PYTHON $BASE_FLAGS -mt recovered_inst -st ctdbp_cdef_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.ctdbp_cdef_instrument_recovered.nc"
    fi
done

### CE04OSSM ###
BASE_FLAGS="-s CE04OSSM -n RID27 -sn 03-CTDBPC000"
BASE_FILE="${HOME}/ooidata/m2m/ce04ossm/nsif/ctdbp/ce04ossm.nsif.ctdbp"
for i in $(seq -f "%02g" 1 11); do
    if [ $i -le 6 ]; then
      # average the bursts for deployments prior to deployment 6
      $PYTHON $BASE_FLAGS -mt telemetered -st ctdbp_cdef_dcl_instrument -ba -dp $i -o "$BASE_FILE.deploy$i.telemetered.ctdbp_cdef_dcl_instrument.nc"
      $PYTHON $BASE_FLAGS -mt recovered_host -st ctdbp_cdef_dcl_instrument_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.recovered_host.ctdbp_cdef_dcl_instrument_recovered.nc"
      $PYTHON $BASE_FLAGS -mt recovered_inst -st ctdbp_cdef_instrument_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.ctdbp_cdef_instrument_recovered.nc"
    else
      # no longer using burst mode, just collect and rework
      $PYTHON $BASE_FLAGS -mt telemetered -st ctdbp_cdef_dcl_instrument -dp $i -o "$BASE_FILE.deploy$i.telemetered.ctdbp_cdef_dcl_instrument.nc"
      $PYTHON $BASE_FLAGS -mt recovered_host -st ctdbp_cdef_dcl_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_host.ctdbp_cdef_dcl_instrument_recovered.nc"
      $PYTHON $BASE_FLAGS -mt recovered_inst -st ctdbp_cdef_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.ctdbp_cdef_instrument_recovered.nc"
    fi
done

### CE06ISSM ###
BASE_FLAGS="-s CE06ISSM -n SBD17 -sn 06-CTDBPC000"
BASE_FILE="${HOME}/ooidata/m2m/ce06issm/buoy/ctdbp/ce06issm.buoy.ctdbp"
for i in $(seq -f "%02g" 1 13); do
    $PYTHON $BASE_FLAGS -mt telemetered -st ctdbp_cdef_dcl_instrument -dp $i -o "$BASE_FILE.deploy$i.telemetered.ctdbp_cdef_dcl_instrument.nc"
    $PYTHON $BASE_FLAGS -mt recovered_host -st ctdbp_cdef_dcl_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_host.ctdbp_cdef_dcl_instrument_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_inst -st ctdbp_cdef_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.ctdbp_cdef_instrument_recovered.nc"
done

BASE_FLAGS="-s CE06ISSM -n RID16 -sn 03-CTDBPC000"
BASE_FILE="${HOME}/ooidata/m2m/ce06issm/nsif/ctdbp/ce06issm.nsif.ctdbp"
for i in $(seq -f "%02g" 1 13); do
    $PYTHON $BASE_FLAGS -mt telemetered -st ctdbp_cdef_dcl_instrument -dp $i -o "$BASE_FILE.deploy$i.telemetered.ctdbp_cdef_dcl_instrument.nc"
    $PYTHON $BASE_FLAGS -mt recovered_host -st ctdbp_cdef_dcl_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_host.ctdbp_cdef_dcl_instrument_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_inst -st ctdbp_cdef_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.ctdbp_cdef_instrument_recovered.nc"
done

BASE_FLAGS="-s CE06ISSM -n MFD37 -sn 03-CTDBPC000"
BASE_FILE="${HOME}/ooidata/m2m/ce06issm/seafloor/ctdbp/ce06issm.seafloor.ctdbp"
for i in $(seq -f "%02g" 1 13); do
    $PYTHON $BASE_FLAGS -mt telemetered -st ctdbp_cdef_dcl_instrument -dp $i -o "$BASE_FILE.deploy$i.telemetered.ctdbp_cdef_dcl_instrument.nc"
    $PYTHON $BASE_FLAGS -mt recovered_host -st ctdbp_cdef_dcl_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_host.ctdbp_cdef_dcl_instrument_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_inst -st ctdbp_cdef_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.ctdbp_cdef_instrument_recovered.nc"
done

### CE07SHSM ###
BASE_FLAGS="-s CE07SHSM -n RID27 -sn 03-CTDBPC000"
BASE_FILE="${HOME}/ooidata/m2m/ce07shsm/nsif/ctdbp/ce07shsm.nsif.ctdbp"
for i in $(seq -f "%02g" 1 12); do
    if [ $i -le 7 ]; then
      # average the bursts for deployments prior to deployment 7
      $PYTHON $BASE_FLAGS -mt telemetered -st ctdbp_cdef_dcl_instrument -ba -dp $i -o "$BASE_FILE.deploy$i.telemetered.ctdbp_cdef_dcl_instrument.nc"
      $PYTHON $BASE_FLAGS -mt recovered_host -st ctdbp_cdef_dcl_instrument_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.recovered_host.ctdbp_cdef_dcl_instrument_recovered.nc"
      $PYTHON $BASE_FLAGS -mt recovered_inst -st ctdbp_cdef_instrument_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.ctdbp_cdef_instrument_recovered.nc"
    else
      # no longer using burst mode, just collect and rework
      $PYTHON $BASE_FLAGS -mt telemetered -st ctdbp_cdef_dcl_instrument -dp $i -o "$BASE_FILE.deploy$i.telemetered.ctdbp_cdef_dcl_instrument.nc"
      $PYTHON $BASE_FLAGS -mt recovered_host -st ctdbp_cdef_dcl_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_host.ctdbp_cdef_dcl_instrument_recovered.nc"
      $PYTHON $BASE_FLAGS -mt recovered_inst -st ctdbp_cdef_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.ctdbp_cdef_instrument_recovered.nc"
    fi
done

BASE_FLAGS="-s CE07SHSM -n MFD37 -sn 03-CTDBPC000"
BASE_FILE="${HOME}/ooidata/m2m/ce07shsm/seafloor/ctdbp/ce07shsm.seafloor.ctdbp"
for i in $(seq -f "%02g" 1 12); do
    $PYTHON $BASE_FLAGS -mt telemetered -st ctdbp_cdef_dcl_instrument -dp $i -o "$BASE_FILE.deploy$i.telemetered.ctdbp_cdef_dcl_instrument.nc"
    $PYTHON $BASE_FLAGS -mt recovered_host -st ctdbp_cdef_dcl_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_host.ctdbp_cdef_dcl_instrument_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_inst -st ctdbp_cdef_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.ctdbp_cdef_instrument_recovered.nc"
done

### CE09OSSM ###
BASE_FLAGS="-s CE09OSSM -n RID27 -sn 03-CTDBPC000"
BASE_FILE="${HOME}/ooidata/m2m/ce09ossm/nsif/ctdbp/ce09ossm.nsif.ctdbp"
for i in $(seq -f "%02g" 1 12); do
    if [ $i -le 7 ]; then
      # average the bursts for deployments prior to deployment 7
      $PYTHON $BASE_FLAGS -mt telemetered -st ctdbp_cdef_dcl_instrument -ba -dp $i -o "$BASE_FILE.deploy$i.telemetered.ctdbp_cdef_dcl_instrument.nc"
      $PYTHON $BASE_FLAGS -mt recovered_host -st ctdbp_cdef_dcl_instrument_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.recovered_host.ctdbp_cdef_dcl_instrument_recovered.nc"
      $PYTHON $BASE_FLAGS -mt recovered_inst -st ctdbp_cdef_instrument_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.ctdbp_cdef_instrument_recovered.nc"
    else
      # no longer using burst mode, just collect and rework
      $PYTHON $BASE_FLAGS -mt telemetered -st ctdbp_cdef_dcl_instrument -dp $i -o "$BASE_FILE.deploy$i.telemetered.ctdbp_cdef_dcl_instrument.nc"
      $PYTHON $BASE_FLAGS -mt recovered_host -st ctdbp_cdef_dcl_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_host.ctdbp_cdef_dcl_instrument_recovered.nc"
      $PYTHON $BASE_FLAGS -mt recovered_inst -st ctdbp_cdef_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.ctdbp_cdef_instrument_recovered.nc"
    fi
done

BASE_FLAGS="-s CE09OSSM -n MFD37 -sn 03-CTDBPE000"
BASE_FILE="${HOME}/ooidata/m2m/ce09ossm/seafloor/ctdbp/ce09ossm.seafloor.ctdbp"
for i in $(seq -f "%02g" 1 12); do
    $PYTHON $BASE_FLAGS -mt telemetered -st ctdbp_cdef_dcl_instrument -dp $i -o "$BASE_FILE.deploy$i.telemetered.ctdbp_cdef_dcl_instrument.nc"
    $PYTHON $BASE_FLAGS -mt recovered_host -st ctdbp_cdef_dcl_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_host.ctdbp_cdef_dcl_instrument_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_inst -st ctdbp_cdef_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.ctdbp_cdef_instrument_recovered.nc"
done
