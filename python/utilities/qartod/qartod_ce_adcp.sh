#!/usr/bin/env bash
#
# qartod_ce_adcp.sh
#
# Collect the ADCP data from all of the OOI Coastal Endurance moorings to
# calculate QARTOD test ranges and generate the different lookup values and
# tables.
#
# C. Wingard, 2022-01-11 -- Initial code

# set the base directory python command for all subsequent processing
. $(dirname $CONDA_EXE)/../etc/profile.d/conda.sh
conda activate ooi
PYTHON="python -m ooi_data_explorations.qartod.endurance.qartod_ce_adcp"

### CE01ISSM ###
$PYTHON -s CE01ISSM -n MFD35 -sn 04-ADCPTM000 -co 2025-01-01T00:00:00

### CE02SHSM ###
$PYTHON -s CE02SHSM -n RID26 -sn 01-ADCPTA000 -co 2025-01-01T00:00:00

### CE04OSSM ###
$PYTHON -s CE04OSSM -n RID26 -sn 01-ADCPTC000 -co 2025-01-01T00:00:00

### CE06ISSM ###
$PYTHON -s CE06ISSM -n MFD35 -sn 04-ADCPTM000 -co 2025-01-01T00:00:00

### CE07SHSM ###
$PYTHON -s CE07SHSM -n RID26 -sn 01-ADCPTA000 -co 2025-01-01T00:00:00
$PYTHON -s CE07SHSM -n MFD35 -sn 01-ADCPTC000 -co 2025-01-01T00:00:00

### CE09OSSM ###
$PYTHON -s CE09OSSM -n RID26 -sn 01-ADCPTC000 -co 2025-01-01T00:00:00
$PYTHON -s CE09OSSM -n MFD35 -sn 04-ADCPSJ000 -co 2025-01-01T00:00:00
