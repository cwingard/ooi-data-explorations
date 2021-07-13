#!/usr/bin/env bash
#
# harvest_ce_pco2a.sh
#
# Harvest the pco2a data from all of the OOI Coastal Endurance moorings. Data
# sets include telemetered and recovered host. Data is downloaded from OOI Net
# and reworked to create a cleaner and more consistent set of files named and
# organized by the mooring, mooring sub-location, data delivery method and
# deployment.
#
# C. Wingard, 2019-12-16 -- Initial code

# set the base directory python command for all subsequent processing
. $(dirname $CONDA_EXE)/../etc/profile.d/conda.sh
conda activate ooi
PYTHON="python -m ooi_data_explorations.uncabled.process_pco2a"

### CE02SHSM ###
BASE_FLAGS="-s CE02SHSM -n SBD12 -sn 04-PCO2AA000"
BASE_FILE="${HOME}/ooidata/m2m/ce02shsm/buoy/pco2a/ce02shsm.buoy.pco2a"
for i in $(seq -f "%02g" 1 12); do
  $PYTHON $BASE_FLAGS -mt telemetered -st pco2a_a_dcl_instrument_air -ba -dp $i -o "$BASE_FILE.deploy$i.telemetered.pco2a_a_dcl_instrument.nc"
  $PYTHON $BASE_FLAGS -mt telemetered -st pco2a_a_dcl_instrument_water -ba -dp $i -o "$BASE_FILE.deploy$i.telemetered.pco2a_a_dcl_instrument.nc"
  $PYTHON $BASE_FLAGS -mt recovered_host -st pco2a_a_dcl_instrument_air_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.recovered_host.pco2a_a_dcl_instrument_recovered.nc"
  $PYTHON $BASE_FLAGS -mt recovered_host -st pco2a_a_dcl_instrument_water_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.recovered_host.pco2a_a_dcl_instrument_recovered.nc"
done

### CE04OSSM ###
BASE_FLAGS="-s CE04OSSM -n SBD12 -sn 04-PCO2AA000"
BASE_FILE="${HOME}/ooidata/m2m/ce04ossm/buoy/pco2a/ce04ossm.buoy.pco2a"
for i in $(seq -f "%02g" 1 11); do
  $PYTHON $BASE_FLAGS -mt telemetered -st pco2a_a_dcl_instrument_air -ba -dp $i -o "$BASE_FILE.deploy$i.telemetered.pco2a_a_dcl_instrument.nc"
  $PYTHON $BASE_FLAGS -mt telemetered -st pco2a_a_dcl_instrument_water -ba -dp $i -o "$BASE_FILE.deploy$i.telemetered.pco2a_a_dcl_instrument.nc"
  $PYTHON $BASE_FLAGS -mt recovered_host -st pco2a_a_dcl_instrument_air_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.recovered_host.pco2a_a_dcl_instrument_recovered.nc"
  $PYTHON $BASE_FLAGS -mt recovered_host -st pco2a_a_dcl_instrument_water_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.recovered_host.pco2a_a_dcl_instrument_recovered.nc"
done

### CE07SHSM ###
BASE_FLAGS="-s CE07SHSM -n SBD12 -sn 04-PCO2AA000"
BASE_FILE="${HOME}/ooidata/m2m/ce07shsm/buoy/pco2a/ce07shsm.buoy.pco2a"
for i in $(seq -f "%02g" 1 12); do
  $PYTHON $BASE_FLAGS -mt telemetered -st pco2a_a_dcl_instrument_air -ba -dp $i -o "$BASE_FILE.deploy$i.telemetered.pco2a_a_dcl_instrument.nc"
  $PYTHON $BASE_FLAGS -mt telemetered -st pco2a_a_dcl_instrument_water -ba -dp $i -o "$BASE_FILE.deploy$i.telemetered.pco2a_a_dcl_instrument.nc"
  $PYTHON $BASE_FLAGS -mt recovered_host -st pco2a_a_dcl_instrument_air_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.recovered_host.pco2a_a_dcl_instrument_recovered.nc"
  $PYTHON $BASE_FLAGS -mt recovered_host -st pco2a_a_dcl_instrument_water_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.recovered_host.pco2a_a_dcl_instrument_recovered.nc"
done

### CE09OSSM ###
BASE_FLAGS="-s CE09OSSM -n SBD12 -sn 04-PCO2AA000"
BASE_FILE="${HOME}/ooidata/m2m/ce09ossm/buoy/pco2a/ce09ossm.buoy.pco2a"
for i in $(seq -f "%02g" 1 12); do
  $PYTHON $BASE_FLAGS -mt telemetered -st pco2a_a_dcl_instrument_air -ba -dp $i -o "$BASE_FILE.deploy$i.telemetered.pco2a_a_dcl_instrument.nc"
  $PYTHON $BASE_FLAGS -mt telemetered -st pco2a_a_dcl_instrument_water -ba -dp $i -o "$BASE_FILE.deploy$i.telemetered.pco2a_a_dcl_instrument.nc"
  $PYTHON $BASE_FLAGS -mt recovered_host -st pco2a_a_dcl_instrument_air_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.recovered_host.pco2a_a_dcl_instrument_recovered.nc"
  $PYTHON $BASE_FLAGS -mt recovered_host -st pco2a_a_dcl_instrument_water_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.recovered_host.pco2a_a_dcl_instrument_recovered.nc"
done
