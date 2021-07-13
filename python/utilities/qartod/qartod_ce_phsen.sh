#!/usr/bin/env bash
#
# qartod_ce_phsen.sh
#
# Collect the phsen data from all of the OOI Coastal Endurance moorings to
# calculate QARTOD test ranges and generate the different lookup values and
# tables.
#
# C. Wingard, 2021-06-17 -- Initial code

# set the base directory python command for all subsequent processing
. $(dirname $CONDA_EXE)/../etc/profile.d/conda.sh
conda activate ooi
PYTHON="python -m ooi_data_explorations.qartod.endurance.qartod_ce_phsen"

### CE01ISSM ###
$PYTHON -s CE01ISSM -n RID16 -sn 06-PHSEND000
$PYTHON -s CE01ISSM -n MFD35 -sn 06-PHSEND000

### CE02SHSM ###
$PYTHON -s CE02SHSM -n RID26 -sn 06-PHSEND000

### CE04OSSM ###
$PYTHON -s CE04OSSM -n RID26 -sn 06-PHSEND000

### CE06ISSM ###
$PYTHON -s CE06ISSM -n RID16 -sn 06-PHSEND000
$PYTHON -s CE06ISSM -n MFD35 -sn 06-PHSEND000

### CE07SHSM ###
$PYTHON -s CE07SHSM -n RID26 -sn 06-PHSEND000
$PYTHON -s CE07SHSM -n MFD35 -sn 06-PHSEND000

### CE09OSSM ###
$PYTHON -s CE09OSSM -n RID26 -sn 06-PHSEND000
$PYTHON -s CE09OSSM -n MFD35 -sn 06-PHSEND000
