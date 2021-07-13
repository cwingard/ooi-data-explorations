#!/usr/bin/env bash
#
# harvest_ce_pco2w.sh
#
# Harvest the pco2w data from all of the OOI Coastal Endurance moorings. Data
# sets include telemetered, recovered host and instrument data. Data is
# downloaded from OOI Net and reworked to create a cleaner and more consistent
# set of files named and organized by the mooring, mooring sub-location, data
# delivery method and deployment.
#
# C. Wingard, 2021-01-28 -- Initial code

# set the base directory python command for all subsequent processing
. $(dirname $CONDA_EXE)/../etc/profile.d/conda.sh
conda activate ooi
PYTHON="python -m ooi_data_explorations.uncabled.process_pco2w"

### CE01ISSM ###
BASE_FLAGS="-s CE01ISSM -n RID16 -sn 05-PCO2WB000"
BASE_FILE="${HOME}/ooidata/m2m/ce01issm/nsif/pco2w/ce01issm.nsif.pco2w"
for i in $(seq -f "%02g" 1 14); do
    $PYTHON $BASE_FLAGS -mt telemetered -st pco2w_abc_dcl_instrument -dp $i -o "$BASE_FILE.deploy$i.telemetered.pco2w_abc_dcl_instrument.nc"
    $PYTHON $BASE_FLAGS -mt recovered_host -st pco2w_abc_dcl_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_host.pco2w_abc_dcl_instrument_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_inst -st pco2w_abc_instrument -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.pco2w_abc_instrument.nc"
done

BASE_FLAGS="-s CE01ISSM -n MFD35 -sn 05-PCO2WB000"
BASE_FILE="${HOME}/ooidata/m2m/ce01issm/seafloor/pco2w/ce01issm.seafloor.pco2w"
for i in $(seq -f "%02g" 1 14); do
    $PYTHON $BASE_FLAGS -mt telemetered -st pco2w_abc_dcl_instrument -dp $i -o "$BASE_FILE.deploy$i.telemetered.pco2w_abc_dcl_instrument.nc"
    $PYTHON $BASE_FLAGS -mt recovered_host -st pco2w_abc_dcl_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_host.pco2w_abc_dcl_instrument_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_inst -st pco2w_abc_instrument -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.pco2w_abc_instrument.nc"
done

### CE06ISSM ###
BASE_FLAGS="-s CE06ISSM -n RID16 -sn 05-PCO2WB000"
BASE_FILE="${HOME}/ooidata/m2m/ce06issm/nsif/pco2w/ce06issm.nsif.pco2w"
for i in $(seq -f "%02g" 1 13); do
    $PYTHON $BASE_FLAGS -mt telemetered -st pco2w_abc_dcl_instrument -dp $i -o "$BASE_FILE.deploy$i.telemetered.pco2w_abc_dcl_instrument.nc"
    $PYTHON $BASE_FLAGS -mt recovered_host -st pco2w_abc_dcl_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_host.pco2w_abc_dcl_instrument_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_inst -st pco2w_abc_instrument -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.pco2w_abc_instrument.nc"
done

BASE_FLAGS="-s CE06ISSM -n MFD35 -sn 05-PCO2WB000"
BASE_FILE="${HOME}/ooidata/m2m/ce06issm/seafloor/pco2w/ce06issm.seafloor.pco2w"
for i in $(seq -f "%02g" 1 13); do
    $PYTHON $BASE_FLAGS -mt telemetered -st pco2w_abc_dcl_instrument -dp $i -o "$BASE_FILE.deploy$i.telemetered.pco2w_abc_dcl_instrument.nc"
    $PYTHON $BASE_FLAGS -mt recovered_host -st pco2w_abc_dcl_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_host.pco2w_abc_dcl_instrument_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_inst -st pco2w_abc_instrument -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.pco2w_abc_instrument.nc"
done

### CE07SHSM ###
BASE_FLAGS="-s CE07SHSM -n MFD35 -sn 05-PCO2WB000"
BASE_FILE="${HOME}/ooidata/m2m/ce07shsm/seafloor/pco2w/ce07shsm.seafloor.pco2w"
for i in $(seq -f "%02g" 1 12); do
    $PYTHON $BASE_FLAGS -mt telemetered -st pco2w_abc_dcl_instrument -dp $i -o "$BASE_FILE.deploy$i.telemetered.pco2w_abc_dcl_instrument.nc"
    $PYTHON $BASE_FLAGS -mt recovered_host -st pco2w_abc_dcl_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_host.pco2w_abc_dcl_instrument_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_inst -st pco2w_abc_instrument -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.pco2w_abc_instrument.nc"
done

### CE09OSSM ###
BASE_FLAGS="-s CE09OSSM -n MFD35 -sn 05-PCO2WB000"
BASE_FILE="${HOME}/ooidata/m2m/ce09ossm/seafloor/pco2w/ce09ossm.seafloor.pco2w"
for i in $(seq -f "%02g" 1 12); do
    $PYTHON $BASE_FLAGS -mt telemetered -st pco2w_abc_dcl_instrument -dp $i -o "$BASE_FILE.deploy$i.telemetered.pco2w_abc_dcl_instrument.nc"
    $PYTHON $BASE_FLAGS -mt recovered_host -st pco2w_abc_dcl_instrument_recovered -dp $i -o "$BASE_FILE.deploy$i.recovered_host.pco2w_abc_dcl_instrument_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_inst -st pco2w_abc_instrument -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.pco2w_abc_instrument.nc"
done
