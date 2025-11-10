#!/usr/bin/env bash
#
# qartod_ce_dosta.sh
#
# Collect the dosta data from all of the OOI Coastal Endurance moorings and the
# CSPPs to calculate QARTOD test ranges and generate the different lookup
# values and tables.
#
# C. Wingard, 2022-06-17 -- Initial code

# set the base directory python command for all subsequent processing
. $(dirname $CONDA_EXE)/../etc/profile.d/conda.sh
conda activate ooi
PYTHON="python -m ooi_data_explorations.qartod.endurance.qartod_ce_dosta"

### CE01ISSM ###
#$PYTHON -s CE01ISSM -n RID16 -sn 03-DOSTAD000 -co 2025-12-31T00:00:00
#$PYTHON -s CE01ISSM -n MFD37 -sn 03-DOSTAD000 -co 2025-12-31T00:00:00

### CE01ISSP ###
$PYTHON -s CE01ISSP -n SP001 -sn 02-DOSTAJ000 -co 2025-12-31T00:00:00

### CE02SHSM ###
#$PYTHON -s CE02SHSM -n RID27 -sn 04-DOSTAD000 -co 2025-12-31T00:00:00

### CE02SHSP ###
$PYTHON -s CE02SHSP -n SP001 -sn 01-DOSTAJ000 -co 2025-12-31T00:00:00

### CE04OSSM ###
#$PYTHON -s CE04OSSM -n RID27 -sn 04-DOSTAD000 -co 2025-12-31T00:00:00

### CE06ISSM ###
#$PYTHON -s CE06ISSM -n RID16 -sn 03-DOSTAD000 -co 2025-12-31T00:00:00
#$PYTHON -s CE06ISSM -n MFD37 -sn 03-DOSTAD000 -co 2025-12-31T00:00:00

### CE06ISSP ###
$PYTHON -s CE06ISSP -n SP001 -sn 02-DOSTAJ000 -co 2025-12-31T00:00:00

### CE07SHSM ###
#$PYTHON -s CE07SHSM -n RID27 -sn 04-DOSTAD000 -co 2025-12-31T00:00:00
#$PYTHON -s CE07SHSM -n MFD37 -sn 03-DOSTAD000 -co 2025-12-31T00:00:00

### CE07SHSP ###
$PYTHON -s CE07SHSP -n SP001 -sn 01-DOSTAJ000 -co 2025-12-31T00:00:00

### CE09OSSM ###
#$PYTHON -s CE09OSSM -n RID27 -sn 04-DOSTAD000 -co 2025-12-31T00:00:00
#$PYTHON -s CE09OSSM -n MFD37 -sn 03-DOSTAD000 -co 2025-12-31T00:00:00
