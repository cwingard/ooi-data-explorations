#!/usr/bin/env bash
#
# qartod_ce_mopak.sh
#
# Collect the MOPAK data from the four OOI Coastal Endurance moorings to
# calculate QARTOD test ranges and generate the different lookup values and
# tables.
#
# C. Wingard, 2025-02-20 -- Initial code

# set the base directory python command for all subsequent processing
. $(dirname $CONDA_EXE)/../etc/profile.d/conda.sh
conda activate ooi
PYTHON="python -m ooi_data_explorations.qartod.endurance.qartod_ce_mopak"

### CE02SHSM ###
$PYTHON -s CE02SHSM -n SBD11 -sn 01-MOPAK0000 -co 2025-01-01T00:00:00

### CE04OSSM ###
$PYTHON -s CE04OSSM -n SBD11 -sn 01-MOPAK0000 -co 2025-01-01T00:00:00

### CE07SHSM ###
$PYTHON -s CE07SHSM -n SBD11 -sn 01-MOPAK0000 -co 2025-01-01T00:00:00

### CE09OSSM ###
$PYTHON -s CE09OSSM -n SBD11 -sn 01-MOPAK0000 -co 2025-01-01T00:00:00
