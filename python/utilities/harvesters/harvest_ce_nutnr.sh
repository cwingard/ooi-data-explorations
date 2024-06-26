#!/usr/bin/env bash
#
# harvest_ce_nutnr.sh
#
# Harvest the NUTNR data from all of the OOI Coastal Endurance moorings. Data
# sets (SUNA only) include telemetered and recovered host. Data is downloaded 
# from OOI Net and reworked to create a cleaner and more consistent set of 
# files named and organized by the mooring, mooring sub-location, data delivery
# method and deployment.
#
# C. Wingard, 2021-02-15 -- Initial code

# set the base directory python command for all subsequent processing
. $(dirname $CONDA_EXE)/../etc/profile.d/conda.sh
conda activate ooi
PYTHON="python -m ooi_data_explorations.uncabled.process_nutnr"

### CE01ISSM ###
BASE_FLAGS="-s CE01ISSM -n RID16 -sn 07-NUTNRB000"
BASE_FILE="${HOME}/ooidata/m2m/ce01issm/nsif/nutnr/ce01issm.nsif.nutnr"
for i in $(seq -f "%02g" 8 17); do
    $PYTHON $BASE_FLAGS -mt telemetered -st suna_dcl_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.telemetered.suna_dcl_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_host -st suna_dcl_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.recovered_host.suna_dcl_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_inst -st suna_instrument_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.suna_instrument_recovered.nc"
done

### CE02SHSM ###
BASE_FLAGS="-s CE02SHSM -n RID26 -sn 07-NUTNRB000"
BASE_FILE="${HOME}/ooidata/m2m/ce02shsm/nsif/nutnr/ce02shsm.nsif.nutnr"
for i in $(seq -f "%02g" 7 15); do
    $PYTHON $BASE_FLAGS -mt telemetered -st suna_dcl_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.telemetered.suna_dcl_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_host -st suna_dcl_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.recovered_host.suna_dcl_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_inst -st suna_instrument_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.suna_instrument_recovered.nc"
done

### CE04OSSM ###
BASE_FLAGS="-s CE04OSSM -n RID26 -sn 07-NUTNRB000"
BASE_FILE="${HOME}/ooidata/m2m/ce04ossm/nsif/nutnr/ce04ossm.nsif.nutnr"
for i in $(seq -f "%02g" 6 14); do
    $PYTHON $BASE_FLAGS -mt telemetered -st suna_dcl_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.telemetered.suna_dcl_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_host -st suna_dcl_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.recovered_host.suna_dcl_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_inst -st suna_instrument_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.suna_instrument_recovered.nc"
done

### CE06ISSM ###
BASE_FLAGS="-s CE06ISSM -n RID16 -sn 07-NUTNRB000"
BASE_FILE="${HOME}/ooidata/m2m/ce06issm/nsif/nutnr/ce06issm.nsif.nutnr"
for i in $(seq -f "%02g" 8 16); do
    $PYTHON $BASE_FLAGS -mt telemetered -st suna_dcl_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.telemetered.suna_dcl_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_host -st suna_dcl_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.recovered_host.suna_dcl_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_inst -st suna_instrument_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.suna_instrument_recovered.nc"
done

### CE07SHSM ###
BASE_FLAGS="-s CE07SHSM -n RID26 -sn 07-NUTNRB000"
BASE_FILE="${HOME}/ooidata/m2m/ce07shsm/nsif/nutnr/ce07shsm.nsif.nutnr"
for i in $(seq -f "%02g" 7 15); do
    $PYTHON $BASE_FLAGS -mt telemetered -st suna_dcl_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.telemetered.suna_dcl_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_host -st suna_dcl_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.recovered_host.suna_dcl_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_inst -st suna_instrument_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.suna_instrument_recovered.nc"
done

### CE09OSSM ###
BASE_FLAGS="-s CE09OSSM -n RID26 -sn 07-NUTNRB000"
BASE_FILE="${HOME}/ooidata/m2m/ce09ossm/nsif/nutnr/ce09ossm.nsif.nutnr"
for i in $(seq -f "%02g" 7 15); do
    $PYTHON $BASE_FLAGS -mt telemetered -st suna_dcl_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.telemetered.suna_dcl_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_host -st suna_dcl_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.recovered_host.suna_dcl_recovered.nc"
    $PYTHON $BASE_FLAGS -mt recovered_inst -st suna_instrument_recovered -ba -dp $i -o "$BASE_FILE.deploy$i.recovered_inst.suna_instrument_recovered.nc"
done
