#!/usr/bin/env bash
#
# combine_datasets.sh
#
# Combines the data from the multiple sources (includes telemetered, recovered
# host and recovered instrument data) into a single dataset with common
# parameters and timestep.
#
# C. Wingard, 2020-02-15 -- Initial code

# set the base python command, using the ooi environments for all subsequent
# processing
. $(dirname $CONDA_EXE)/../etc/profile.d/conda.sh
conda activate ooi
COMBINE="python -m ooi_data_explorations.combine_data"

# Parse the command line inputs
if [ $# -ne 5 ]; then
    echo "$0: required inputs are the base directory with the downloaded data,"
    echo "    the deployment number, the resampling time in minutes, the data"
    echo "    file to save the combined data in, and the number of data set"
    echo "    types to process (either 2 for telemetered and recovered_host,"
    echo "    or 3 for telemetered, recovered_host and recovered_inst). The"
    echo "    path for the base directory can be absolute or relative, and the"
    echo "    name for the combined data file can include absolute or relative"
    echo "    path information."
    echo ""
    echo "    example: $0 ~/ooidata/m2m/ce01issm/nsif/ctdbp 2 60 combined.nc 3"
    exit 1
fi
BASE_DIR=$1
DEPLOY=$2
RESAMPLE=$3
OUT_FILE=$4
NUM_DATA=$5

case $NUM_DATA in
    2 )
        echo "Combining telemetered and recovered_host data sets."
        $COMBINE -t -rh -d $BASE_DIR -dp $DEPLOY -rt $RESAMPLE  -o $OUT_FILE
        ;;
    3 )
        echo "Combining telemetered, recovered_host and recovered_inst data sets."
        $COMBINE -t -rh -ri -d $BASE_DIR -dp $DEPLOY -rt $RESAMPLE  -o $OUT_FILE
        ;;
    * )
        echo "Incorrect value for the number of data set types (either 2 or 3)."
        exit 1
        ;;
esac
