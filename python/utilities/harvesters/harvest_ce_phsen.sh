#!/usr/bin/env bash
#
# harvest_ce_phsen.sh
#
# Harvest the phsen data from all of the OOI Coastal Endurance moorings. Data
# sets include telemetered, recovered host and instrument data. Data is
# downloaded from OOI Net and reworked to create a cleaner and more consistent
# set of files named and organized by the mooring, mooring sub-location, data
# delivery method and deployment.
#
# C. Wingard, 2019-07-22 -- Initial code

# set the base directory python command for all subsequent processing
. $(dirname $CONDA_EXE)/../etc/profile.d/conda.sh
conda activate ooi
PYTHON="python -m ooi_data_explorations.uncabled.process_phsen"

### CE01ISSM ###
BASE_FLAGS="-s CE01ISSM -n RID16 -sn 06-PHSEND000"
BASE_FILE="${HOME}/ooidata/m2m/ce01issm/nsif/phsen/ce01issm.nsif.phsen"
for i in $(seq -f "%02g" 1 14); do
    $PYTHON $BASE_FLAGS -mt telemetered -st phsen_abcdef_dcl_instrument -dp $i -o "$BASE_FILE.deploy$i.telemetered.phsen_abcdef_dcl_instrument.nc"
    $PYTHON $BASE_FLAGS -mt recovered_host -st phsen_abcdef_dcl_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_host.phsen_abcdef_dcl_instrument_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_inst -st phsen_abcdef_instrument -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.phsen_abcdef_instrument.nc"
done

BASE_FLAGS="-s CE01ISSM -n MFD35 -sn 06-PHSEND000"
BASE_FILE="${HOME}/ooidata/m2m/ce01issm/seafloor/phsen/ce01issm.seafloor.phsen"
for i in $(seq -f "%02g" 1 14); do
    $PYTHON $BASE_FLAGS -mt telemetered -st phsen_abcdef_dcl_instrument -dp $i -o "$BASE_FILE.deploy$i.telemetered.phsen_abcdef_dcl_instrument.nc"
    $PYTHON $BASE_FLAGS -mt recovered_host -st phsen_abcdef_dcl_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_host.phsen_abcdef_dcl_instrument_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_inst -st phsen_abcdef_instrument -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.phsen_abcdef_instrument.nc"
done

### CE02SHSM ###
BASE_FLAGS="-s CE02SHSM -n RID26 -sn 06-PHSEND000"
BASE_FILE="${HOME}/ooidata/m2m/ce02shsm/nsif/phsen/ce02shsm.nsif.phsen"
for i in $(seq -f "%02g" 1 12); do
    $PYTHON $BASE_FLAGS -mt telemetered -st phsen_abcdef_dcl_instrument -dp $i -o "$BASE_FILE.deploy$i.telemetered.phsen_abcdef_dcl_instrument.nc"
    $PYTHON $BASE_FLAGS -mt recovered_host -st phsen_abcdef_dcl_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_host.phsen_abcdef_dcl_instrument_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_inst -st phsen_abcdef_instrument -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.phsen_abcdef_instrument.nc"
done

### CE04OSSM ###
BASE_FLAGS="-s CE04OSSM -n RID26 -sn 06-PHSEND000"
BASE_FILE="${HOME}/ooidata/m2m/ce04ossm/nsif/phsen/ce04ossm.nsif.phsen"
for i in $(seq -f "%02g" 1 11); do
    $PYTHON $BASE_FLAGS -mt telemetered -st phsen_abcdef_dcl_instrument -dp $i -o "$BASE_FILE.deploy$i.telemetered.phsen_abcdef_dcl_instrument.nc"
    $PYTHON $BASE_FLAGS -mt recovered_host -st phsen_abcdef_dcl_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_host.phsen_abcdef_dcl_instrument_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_inst -st phsen_abcdef_instrument -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.phsen_abcdef_instrument.nc"
done

### CE06ISSM ###
BASE_FLAGS="-s CE06ISSM -n RID16 -sn 06-PHSEND000"
BASE_FILE="${HOME}/ooidata/m2m/ce06issm/nsif/phsen/ce06issm.nsif.phsen"
for i in $(seq -f "%02g" 1 13); do
    $PYTHON $BASE_FLAGS -mt telemetered -st phsen_abcdef_dcl_instrument -dp $i -o "$BASE_FILE.deploy$i.telemetered.phsen_abcdef_dcl_instrument.nc"
    $PYTHON $BASE_FLAGS -mt recovered_host -st phsen_abcdef_dcl_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_host.phsen_abcdef_dcl_instrument_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_inst -st phsen_abcdef_instrument -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.phsen_abcdef_instrument.nc"
done

BASE_FLAGS="-s CE06ISSM -n MFD35 -sn 06-PHSEND000"
BASE_FILE="${HOME}/ooidata/m2m/ce06issm/seafloor/phsen/ce06issm.seafloor.phsen"
for i in $(seq -f "%02g" 1 13); do
    $PYTHON $BASE_FLAGS -mt telemetered -st phsen_abcdef_dcl_instrument -dp $i -o "$BASE_FILE.deploy$i.telemetered.phsen_abcdef_dcl_instrument.nc"
    $PYTHON $BASE_FLAGS -mt recovered_host -st phsen_abcdef_dcl_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_host.phsen_abcdef_dcl_instrument_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_inst -st phsen_abcdef_instrument -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.phsen_abcdef_instrument.nc"
done

### CE07SHSM ###
BASE_FLAGS="-s CE07SHSM -n RID26 -sn 06-PHSEND000"
BASE_FILE="${HOME}/ooidata/m2m/ce07shsm/nsif/phsen/ce07shsm.nsif.phsen"
for i in $(seq -f "%02g" 1 12); do
    $PYTHON $BASE_FLAGS -mt telemetered -st phsen_abcdef_dcl_instrument -dp $i -o "$BASE_FILE.deploy$i.telemetered.phsen_abcdef_dcl_instrument.nc"
    $PYTHON $BASE_FLAGS -mt recovered_host -st phsen_abcdef_dcl_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_host.phsen_abcdef_dcl_instrument_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_inst -st phsen_abcdef_instrument -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.phsen_abcdef_instrument.nc"
done

BASE_FLAGS="-s CE07SHSM -n MFD35 -sn 06-PHSEND000"
BASE_FILE="${HOME}/ooidata/m2m/ce07shsm/seafloor/phsen/ce07shsm.seafloor.phsen"
for i in $(seq -f "%02g" 1 12); do
    $PYTHON $BASE_FLAGS -mt telemetered -st phsen_abcdef_dcl_instrument -dp $i -o "$BASE_FILE.deploy$i.telemetered.phsen_abcdef_dcl_instrument.nc"
    $PYTHON $BASE_FLAGS -mt recovered_host -st phsen_abcdef_dcl_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_host.phsen_abcdef_dcl_instrument_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_inst -st phsen_abcdef_instrument -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.phsen_abcdef_instrument.nc"
done

### CE09OSSM ###
BASE_FLAGS="-s CE09OSSM -n RID26 -sn 06-PHSEND000"
BASE_FILE="${HOME}/ooidata/m2m/ce09ossm/nsif/phsen/ce09ossm.nsif.phsen"
for i in $(seq -f "%02g" 1 12); do
    $PYTHON $BASE_FLAGS -mt telemetered -st phsen_abcdef_dcl_instrument -dp $i -o "$BASE_FILE.deploy$i.telemetered.phsen_abcdef_dcl_instrument.nc"
    $PYTHON $BASE_FLAGS -mt recovered_host -st phsen_abcdef_dcl_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_host.phsen_abcdef_dcl_instrument_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_inst -st phsen_abcdef_instrument -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.phsen_abcdef_instrument.nc"
done

BASE_FLAGS="-s CE09OSSM -n MFD35 -sn 06-PHSEND000"
BASE_FILE="${HOME}/ooidata/m2m/ce09ossm/seafloor/phsen/ce09ossm.seafloor.phsen"
for i in $(seq -f "%02g" 1 12); do
    $PYTHON $BASE_FLAGS -mt telemetered -st phsen_abcdef_dcl_instrument -dp $i -o "$BASE_FILE.deploy$i.telemetered.phsen_abcdef_dcl_instrument.nc"
    $PYTHON $BASE_FLAGS -mt recovered_host -st phsen_abcdef_dcl_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_host.phsen_abcdef_dcl_instrument_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_inst -st phsen_abcdef_instrument -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.phsen_abcdef_instrument.nc"
done
