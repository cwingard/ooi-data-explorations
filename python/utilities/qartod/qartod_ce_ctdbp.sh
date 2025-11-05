#!/usr/bin/env bash
#
# qartod_ce_ctdbp.sh
#
# Collect the CTDBP data from all of the OOI Coastal Endurance moorings to
# calculate QARTOD test ranges and generate the different lookup values and
# tables.
#
# C. Wingard, 2022-01-11 -- Initial code

# set the base directory python command for all subsequent processing
. $(dirname $CONDA_EXE)/../etc/profile.d/conda.sh
conda activate ooi
PYTHON="python -m ooi_data_explorations.qartod.endurance.qartod_ce_ctdbp"

### CE01ISSM ###
$PYTHON -s CE01ISSM -n SBD17 -sn 06-CTDBPC000 -co 2024-01-01T00:00:00
$PYTHON -s CE01ISSM -n RID16 -sn 03-CTDBPC000 -co 2024-01-01T00:00:00
$PYTHON -s CE01ISSM -n MFD37 -sn 03-CTDBPC000 -co 2024-01-01T00:00:00

### CE02SHSM ###
$PYTHON -s CE02SHSM -n RID27 -sn 03-CTDBPC000 -co 2024-01-01T00:00:00

### CE04OSSM ###
$PYTHON -s CE04OSSM -n RID27 -sn 03-CTDBPC000 -co 2024-01-01T00:00:00

### CE06ISSM ###
$PYTHON -s CE06ISSM -n SBD17 -sn 06-CTDBPC000 -co 2024-01-01T00:00:00
$PYTHON -s CE06ISSM -n RID16 -sn 03-CTDBPC000 -co 2024-01-01T00:00:00
$PYTHON -s CE06ISSM -n MFD37 -sn 03-CTDBPC000 -co 2024-01-01T00:00:00

### CE07SHSM ###
$PYTHON -s CE07SHSM -n RID27 -sn 03-CTDBPC000 -co 2024-01-01T00:00:00
$PYTHON -s CE07SHSM -n MFD37 -sn 03-CTDBPC000 -co 2024-01-01T00:00:00

### CE09OSSM ###
$PYTHON -s CE09OSSM -n RID27 -sn 03-CTDBPC000 -co 2024-01-01T00:00:00
$PYTHON -s CE09OSSM -n MFD37 -sn 03-CTDBPE000 -co 2024-01-01T00:00:00
