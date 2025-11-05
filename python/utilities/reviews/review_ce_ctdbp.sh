#!/usr/bin/env bash
# review_ce_ctdbp.sh
#
# Collect the CTDBP data from all of the OOI Coastal Endurance moorings to
# create reviews of the data for further analysis and reporting.
#
# C. Wingard, 2024-07-11 -- Initial code

# set the base directory python command for all subsequent processing
# shellcheck disable=SC2046
. $(dirname "$CONDA_EXE")/../etc/profile.d/conda.sh
conda activate ooi
PYTHON="python -m ooi_data_explorations.qartod.endurance.reports.reports_ce_ctdbp"

### CE01ISSM ###
for i in $(seq -f "%02g" 1 19); do
    $PYTHON -s CE01ISSM -n SBD17 -sn 06-CTDBPC000 -dp "$i"
    $PYTHON -s CE01ISSM -n RID16 -sn 03-CTDBPC000 -dp "$i"
    $PYTHON -s CE01ISSM -n MFD37 -sn 03-CTDBPC000 -dp "$i"
done

### CE02SHSM ###
for i in $(seq -f "%02g" 1 17); do
    $PYTHON -s CE02SHSM -n RID27 -sn 03-CTDBPC000 -dp "$i"
done

### CE04OSSM ###
for i in $(seq -f "%02g" 1 16); do
   $PYTHON -s CE04OSSM -n RID27 -sn 03-CTDBPC000 -dp "$i"
done

### CE06ISSM ###
for i in $(seq -f "%02g" 1 18); do
    $PYTHON -s CE06ISSM -n SBD17 -sn 06-CTDBPC000 -dp "$i"
    $PYTHON -s CE06ISSM -n RID16 -sn 03-CTDBPC000 -dp "$i"
    $PYTHON -s CE06ISSM -n MFD37 -sn 03-CTDBPC000 -dp "$i"
done

### CE07SHSM ###
for i in $(seq -f "%02g" 1 17); do
    $PYTHON -s CE07SHSM -n RID27 -sn 03-CTDBPC000 -dp "$i"
    $PYTHON -s CE07SHSM -n MFD37 -sn 03-CTDBPC000 -dp "$i"
done

### CE09OSSM ###
for i in $(seq -f "%02g" 1 17); do
    $PYTHON -s CE09OSSM -n RID27 -sn 03-CTDBPC000 -dp "$i"
    $PYTHON -s CE09OSSM -n MFD37 -sn 03-CTDBPE000 -dp "$i"
done
