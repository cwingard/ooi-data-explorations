# OOI Data Explorations with Python
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/ooi-data-explorations/badges/version.svg)](https://anaconda.org/conda-forge/ooi-data-explorations) [![Anaconda-Server Badge](https://anaconda.org/conda-forge/ooi-data-explorations/badges/latest_release_date.svg)](https://anaconda.org/conda-forge/ooi-data-explorations) [![Anaconda-Server Badge](https://anaconda.org/conda-forge/ooi-data-explorations/badges/platforms.svg)](https://anaconda.org/conda-forge/ooi-data-explorations) [![Anaconda-Server Badge](https://anaconda.org/conda-forge/ooi-data-explorations/badges/license.svg)](https://anaconda.org/conda-forge/ooi-data-explorations) [![Anaconda-Server Badge](https://anaconda.org/conda-forge/ooi-data-explorations/badges/downloads.svg)](https://anaconda.org/conda-forge/ooi-data-explorations) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17380223.svg)](https://doi.org/10.5281/zenodo.17380223)

## Overview
The python code provided here was developed primarily as a toolset for the OOI Data Team to facilitate accessing data 
from OOINet for the QC reviews, gap analyses, metadata checks, etc. that OOI performs to quality check its datasets. 
This code is provided to the larger community in the hopes that it will be of use. The code uses several methods to 
access data:

* [OOI M2M API](https://oceanobservatories.org/ooi-m2m-interface/)
* [OOI THREDDS Data Server](https://thredds.dataexplorer.oceanobservatories.org/thredds/catalog/ooigoldcopy/public/catalog.html)
* [JupyterHub-mounted NetCDF store](`/home/jovyan/ooi/kdata`) from an OOI JupyterHub session

Datasets are loaded into the user workspace as an [xarray](http://xarray.pydata.org/en/stable/) dataset, or saved to disk as a NetCDF file in different 
examples. There are instructions below of how to set up and use the package, with several example notebooks and scripts 
available in the examples directory.

If you have any comments, questions or issues, please don't hesitate to 
[open an issue]((https://github.com/oceanobservatories/ooi-data-explorations/issues)).

## Table of Contents
* [Installation](#installation)
    * [Download and Install OOI Data Explorations](#obtaining-the-code-and-configuring-the-environment)
    * [Setup Access Credentials](#access-credentials)
    * [Installing Bash, Git and Python](#configuring-system-for-python)
* [Usage](#usage)
    * [M2M Terminology](#m2m-terminology)
    * [Requesting As-Is (Mostly) Data](#requesting-as-is-mostly-data)
    * [Simplifying the As-Is Requests](#simplifying-the-as-is-requests)
        * [YAML Structure](#yaml-structure)
        * [Simplified Request](#simplified-request)
    * [Additional Utilities](#additional-utilities)
    * [Requesting Processed Data](#requesting-processed-data)
    * [QARTOD Workflows](#qartod-workflows)

## Installation
### Installing via conda or pip
The `ooi-data-explorations` package is available on [conda-forge](https://anaconda.org/conda-forge/ooi-data-explorations) 
and can be installed into an existing conda environment with the following command:
```shell
conda install -c conda-forge ooi-data-explorations
```
Alternatively, the package can be installed via `pip` from [PyPI](https://pypi.org/project/ooi-data-explorations/) with 
the following command:
```shell
pip install ooi-data-explorations
```

### Obtaining the Code and Configuring the Environment
If you do not have python installed, read about [Installing Bash, Git and Python](#configuring-system-for-python) below 
before following these instructions to use this code repository.

This section describes downloading a copy of the python code, setting up a virtual environment, and installing this 
module for use in that environment. All commands below are to be run from a terminal.

Clone the `ooi-data-explorations` code to your local machine:
```shell
# download the ooi-data-explorations code
git clone https://github.com/oceanobservatories/ooi-data-explorations.git
```
What follows are two ways to set up a code environment to run `ooi-data-explorations` examples and use the python code
base using either `conda` or `pip` as the package manager.

#### Create conda environment
If you prefer to use the `conda` package manager, follow this section to set up the `ooi` environment which has the 
dependencies needed to run the `ooi-data-explorer` python code and example notebooks.
```shell
# configure the OOI python environment
cd ooi-data-explorations/python
conda env create -f environment.yml
conda activate ooi
conda develop .
```

#### Create a pip environment
If you prefer to use the `pip` package manager, follow this section to set up the `ooi` environment which has the 
dependencies needed to run the `ooi-data-explorer` python code and example notebooks.
```shell
cd ooi-data-explorations/python
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

#### Ensure the python environment is available in JupyterHub
If using this code in a JupyterHub environment, an additional step will be needed to ensure the environment is 
available for running in a JupyterHub kernel. If using a pip environment, a couple of additional dependencies are 
required. Install them with pip:
```shell
pip install ipykernel ipympl
```
For either the conda or pip environments, the environment must be added to a list of available kernels using the 
following command:
```shell
python -m ipykernel install --user --name=ooi
```
Now the `ooi` kernel should be listed as available when running a Jupyter Notebook.

### Access Credentials
Access credentials are required to download data from OOINet via the M2M interface. Directions on how to obtain 
these, in addition to details about the M2M system, are [available on the OOI website](https://oceanobservatories.org/ooi-m2m-interface/).

* If you haven't already done so, either create a user account on the [OOI Data Portal](https://ooinet.oceanobservatories.org), or use the CILogon button 
with an academic or Google account (login button is towards the upper right corner of the web page) to log in to the 
portal.
* Navigate to the drop-down menu screen in the top-right corner of the menu bar 
* Click on the "User Profile" element of the drop-down.
* Copy and save the following data from the user profile: API Username and API Token.

The python code uses the [netrc](https://docs.python.org/3.6/library/netrc.html) utility to obtain your access credentials. Users need to create a `.netrc` file 
in their home directory to store these access credentials. Using either a text editor or the bash terminal, create the 
`.netrc` file (replacing the `<API Username>` and `<API Token>` in the example below with the corresponding values from 
your login credentials for the [OOI Data Portal](https://ooinet.oceanobservatories.org)):
```shell
cd ~
touch .netrc
chmod 600 .netrc
cat <<EOT >> .netrc
machine ooinet.oceanobservatories.org
    login <API Username>
    password <API Token>
EOT
```

### Configuring System for Python
If you already have python installed or are using the OOI JupyterHub, you can skip this section, as the required tools 
are already available.

In order to use the python code in this repository, you will need to set up the proper tools. There  are several 
examples on how to this, so I'll avoid reinventing the wheel here. One of the best tutorials I've found has been 
developed by the folks at [Earth Lab](https://www.earthdatascience.org/). The [tutorial](https://www.earthdatascience.org/workshops/setup-earth-analytics-python/setup-git-bash-conda/) they have 
prepared will guide you through the process of setting up a system to use Python for Earth Science analysis from start
to finish, regardless of your computer's operating system. Experienced users can easily skip and skim through to the 
sections relevant to them.

You do not need to install Bash or Git for the code to work. You can [directly download the code](https://github.com/oceanobservatories/ooi-data-explorations/archive/master.zip) instead of
using Git, use a text editor to [set up your access credentials](#access-credentials), and/or use the Anaconda 
Prompt or a terminal of your choice instead of following the examples given below. I am trying to be OS independent, 
thus the examples below assume you are using some form of bash (Git Bash if you followed the tutorial from above). 
Adjust as you need and see fit.  

Note, for Windows users only and assuming you are using Git Bash, if you already have Anaconda/Miniconda installed on
your machine, you do not need to uninstall/reinstall as described in the [tutorial](https://www.earthdatascience.org/workshops/setup-earth-analytics-python/setup-git-bash-conda/).
You can leave everything as-is. However, you do need to link Git Bash to Anaconda (or Miniconda); this happens
automagically if you follow the sequence in the tutorial by installing Git Bash before Anaconda. If you already have 
Anaconda installed, however, from the bash terminal add the following code to the `.bash_profile` file in your home 
directory (assuming you installed Anaconda in your home directory, which is the default):
```shell script
cd ~
echo ". ${HOME}/Anaconda3/etc/profile.d/conda.sh" >> ~/.bash_profile
source .bash_profile
```

## Usage
The code is available in the [ooi_data_explorations](ooi_data_explorations) directory with examples (both scripts
and notebooks) in the [examples](examples) directory. The python code has been developed and used with Python 3.10 
on Windows and Linux machines. The functions are configured in a granular fashion to allow users to access the
data in a few different ways. Additionally, multiple steps are applied behind the scenes to address common issues
that arise when working with OOI data. The following sections describe the different ways to access the data, as well
as some of the additional utilities available to work with OOI data and metadata.

### M2M Terminology
Before using these functions, it is important to understand how requests to the [OOI M2M API](https://oceanobservatories.org/ooi-m2m-interface/) 
are structured. A request is built around the reference designator (comprised of the site, node, and sensor names), 
the data delivery method, and data stream (think of a stream as a dataset). Beginning and ending dates for the time 
period of interest are optional inputs. If omitted, all the data for a particular instrument of interest will be 
downloaded.

* Site -- 8 character uppercase string denoting the array and location within the array of the system. These are 
[defined](https://oceanobservatories.org/research-arrays/) on the OOI website.
* Node -- 5 character uppercase string (of which the first 2 characters are really the key) denoting the assembly the
instrument is connected to/mounted on. These can be thought of as physical locations within/under the higher level site 
designator.
* Sensor -- 12 character uppercase string that indicates, among other things, the instrument class and series. The 
instrument class and series are [defined](https://oceanobservatories.org/instruments/) on the OOI website.
* Delivery Method -- Method of data delivery (lowercase).
    * `streamed` -- Real-time data delivery method for all cabled assets. Data is "streamed" to shore over the fiber
    optic network as it outputs from an instrument. 
    * `telemetered` -- Near real-time data delivery method for most uncabled assets. Data is recorded remotely
    by a data logger system and delivered in batches over a satellite or cellular network link on a recurring schedule 
    (e.g. every 2 hours).
    * `recovered_host` -- Usually the same data set as telemetered for uncabled assets. Key difference is this data is 
    downloaded from the data logger system after the asset is recovered. In most cases, this is 1:1 with the 
    telemetered data unless there was an issue with telemetry during the deployment or the data was decimated
    (temporal and/or # of parameters) by the data logger system  prior to transmission.
    * `recovered_inst` -- Data recorded on and downloaded directly from an individual instrument after the instrument is
    recovered. Not all instruments internally record data, so this method will not be available for all instruments.
    * `recovered_wfp` -- Data recorded on and downloaded from the McLane Moored Profiler system used at several sites 
    in OOI. Telemetered data is decimated, this data set represents the full-resolution data.
    * `recovered_cspp` -- Data recorded on and downloaded from the Coastal Surface Piercing Profiler system used in
    the Endurance array. Telemetered data is decimated, this data set represents the full-resolution data.
* Stream -- A collection of parameters output by an instrument or read from a file, and parsed into a named data set.
Stream names are all lowercase. Streams are mostly associated with the data delivery methods and there may be more than
one stream per method.

### Requesting As-Is (Mostly) Data
The core functions used to request and download data are `m2m_request` and `m2m_collect`, located in the 
[`common.py`](ooi_data_explorations/common.py) module. From those two functions, you can pretty much create your own 
library of functions to download and process whatever data you want from the system and either save it locally or
continue to work on it within your python environment. It is important to note, these functions require inputs that
map directly to those required by the [OOI M2M API](https://oceanobservatories.org/ooi-m2m-interface/).

Also, these two functions are probably the slowest way to get data from OOI. The M2M request process can take
some time to complete, depending on the size of the request and the current load on the M2M system. If you need data
quickly, consider using the other two functions described further below: `load_gc_thredds` and `load_kdata`.

The data requested and downloaded by the `m2m_request` and `m2m_collect` functions (true for `load_gc_thredds` and 
`load_kdata` as well) is somewhat modified from the original: I switch the dimensions from `obs` to `time`, drop 
certain timing variables that were originally never meant to be exposed to the user, and clean up some basic metadata 
attributes. Beyond that, the data is provided as-is. No effort is made to select a subset of the variables, conduct QC, 
clean up the variable names, or in any other way alter the data obtained from the [OOI Data Portal](https://ooinet.oceanobservatories.org). For example:
```python
import os
from  ooi_data_explorations.common import m2m_request, m2m_collect

# Set up the needed information to request data from the pH sensor on the Oregon
# Shelf Surface Mooring near-surface (7 m depth) instrument frame (NSIF).
site = 'CE02SHSM'           # OOI Net site designator
node = 'RID26'              # OOI Net node designator
sensor = '06-PHSEND000'     # OOI Net sensor designator
method = 'telemetered'      # OOI Net data delivery method
stream = 'phsen_abcdef_dcl_instrument'  # OOI Net stream name
start = '2019-04-01T00:00:00.000Z'  # data for spring 2019 ...
stop = '2019-09-30T23:59:59.999Z'   # ... through the beginning of fall

# Request the data (this may take some time).
r = m2m_request(site, node, sensor, method, stream, start, stop)

# Use a regex tag to download only the pH sensor data from the THREDDS catalog
# created by our request.
tag = '.*PHSEN.*\\.nc$'
data = m2m_collect(r, tag)

# Save the data to the users home directory under a folder called ooidata for
# further processing
out_path = os.path.join(os.path.expanduser('~'), 'ooidata')
out_path = os.path.abspath(out_path)
if not os.path.exists(out_path):
    os.makedirs(out_path)

# set up the output file
out_file = ('%s.%s.%s.%s.%s.nc' % (site, node, sensor, method, stream))
nc_out = os.path.join(out_path, out_file)

# save the data to disk
data.to_netcdf(nc_out, mode='w', format='NETCDF4', engine='h5netcdf')
```
The example above will request data from the pH sensor (PHSEN) on the Oregon Shelf Surface Mooring (CE02SHSM) 
near-surface (7 m depth) instrument frame (NSIF) via `m2m_request`. The requested data is gathered by the system in a 
THREDDS catalog specific to the user and the request. One or more NetCDF files with the requested data will be in the 
catalog. The second function, `m2m_collect`, will load the content of those files into an xarray dataset. A key input to
`m2m_collect` is the regex tag used to select the files to load. In most cases, a user could use `r'.*\.nc$'` as the 
tag, downloading all available NetCDF files. In some cases, however, the tag used needs to be more selective. If an
instrument requires data from a co-located sensor, those NetCDF files will be present as well. Part of the process in
collecting the requested data is concatenating the downloaded data into a single xarray dataset. That will fail if the 
individual data files contain different variables. In the above example, the pH sensor requires salinity data from a 
co-located CTD. Both pH sensor and CTD NetCDF files will be present in the THREDDS catalog. The tag `r'.*PHSEN.*\.nc$'` 
is used to select only the pH sensor data.

Conversely, OOI auto-generates the same datasets for all instruments and makes those available (excluding the so-called 
engineering data) on the [OOI Gold Copy THREDDS Data Server](https://thredds.dataexplorer.oceanobservatories.org/thredds/catalog/ooigoldcopy/public/catalog.html). 
This is the source of the data used in [OOI Data Explorer](https://dataexplorer.oceanobservatories.org/). You can 
bypass the M2M request process (which can take some time to complete) and directly access the data from the THREDDS 
server using a different function: `load_gc_thredds`. You will still need to know the site, node, sensor, method, and 
stream name to use this function. The structure of the request is nearly identical to that used in the M2M request 
process. The key difference is how the regex tag is used to select the files of interest. For example:
```python
from  ooi_data_explorations.common import load_gc_thredds

# Set up the needed information to request data from the pH sensor on the Oregon
# Shelf Surface Mooring near-surface (7 m depth) instrument frame (NSIF).
site = 'CE02SHSM'           # OOI Net site designator
node = 'RID26'              # OOI Net node designator
sensor = '06-PHSEND000'     # OOI Net sensor designator
method = 'telemetered'      # OOI Net data delivery method
stream = 'phsen_abcdef_dcl_instrument'  # OOI Net stream name

# Download the data from deployment 10 only from the THREDDS Server ...
tag = r'deployment0010.*PHSEN.*\.nc$'  # data from deployment 10 only
data_d10 = load_gc_thredds(site, node, sensor, method, stream, tag)

# ... or download the data collected in 2016
tag = r'.*PHSEN.*2016.*\.nc$'  # data from 2016 only
data_2016 = load_gc_thredds(site, node, sensor, method, stream, tag)

# ... or download all data available for that instrument
tag = r'.*PHSEN.*\.nc$'  # all data for the pH sensor
data_all = load_gc_thredds(site, node, sensor, method, stream, tag)
```

If you are working in an OOI JupyterHub environment, you can also access the data directly from the mounted NetCDF store
(`/home/jovyan/ooi/kdata`). The function `load_kdata` is used to access the data in a similar manner to the `load_gc_thredds`
function. Two key differences for this data source: all the data, engineering included, is available, and the regex 
tag is changed to a file glob in order to select the files of interest. For example:
```python   
from ooi_data_explorations.common import load_kdata

# Set up the needed information to request data from the pH sensor on the Oregon
# Shelf Surface Mooring near-surface (7 m depth) instrument frame (NSIF).
site = 'CE02SHSM'           # OOI Net site designator
node = 'RID26'              # OOI Net node designator
sensor = '06-PHSEND000'     # OOI Net sensor designator
method = 'telemetered'      # OOI Net data delivery method
stream = 'phsen_abcdef_dcl_instrument'  # OOI Net stream name

# Download all the data from the kdata store
tag = r'*PHSEN*.nc'  # all data for the pH sensor
data_all = load_kdata(site, node, sensor, method, stream, tag)
```

All of these functions take advantage of parallel processing to speed up the download and loading of data. 

### Simplifying the As-Is Requests
Users really only need to use `m2m_request` and `m2m_collect` for the data requests. However, the user needs to
explicitly know all the details (e.g. correct regex tag) and terms from above. Outside OOI (and even inside 
OOI), the terminology used for sites, nodes, sensors, methods, and streams can be intimidating and confusing. In an 
attempt to clean up some of that terminology, limit the need for the user to learn all the OOI lingo, and to align 
with some of the functionality in the Matlab and R utilities, I've organized a subset of all the sources of OOI data 
into a [YAML structure](ooi_data_explorations/m2m_urls.yml) that users can query using a simpler set of terms as part of 
a data request. The YAML structure removes from consideration all so-called engineering sensors and instruments that 
cannot be accessed through the API (e.g. cameras or bioacoustic sonar), as well as most of the non-science streams 
(engineering or metadata streams). The idea is to cover the most common needs, rather than all possible cases. Users can
still access any data set desired, but they need to use the [method from above](#requesting-as-is-mostly-data) to 
explicitly call any stream not represented in the YAML structure.

#### YAML Structure
The YAML structure I've created uses the OOI site codes as-is. It simplifies the node designation by taking the 100+
nodes and groups them according to an assembly type indicating where co-located sensors can be found. There are 6 
assembly types with differing numbers of subassemblies (see table below). Either the assembly or subassembly name can 
be used to request the data.

| Assembly | Subassembly | Description |
| --- | --- | --- |
| buoy | n/a | Surface buoys with meteorological, wave, and/or sea surface (~1 m depth) instrumentation |
| midwater | nsif, riser, sphere, 200m_platform | Platforms located at various depths below the sea surface and above the seafloor |
| seafloor |mfn, bep, low-pwr-jbox, medium-pwr-jbox| Platforms resting directly on the seafloor |
| profiler | cspp, coastal-wfp, global-wfp, shallow-profiler, deep-profiler | Profiling systems with integrated instrumentation |
| glider | coastal-glider, global-glider, profiling-glider | Autonomous, buoyancy driven underwater vehicles with integrated instrumentation |
| auv | n/a | Autonomous underwater vehicles, currently only deployed in the Pioneer Array |

The shorter OOI instrument class name is used instead of the full sensor designator. The data delivery methods are as
defined above and are used to determine the stream(s) available. There is usually just one. In the few cases where 
there is more than one, the code defaults to selecting the first. I've curated the list to make sure this is the stream 
of interest to 99.9% of users. The key utility here is users do not have to know the stream name. You can still get at 
the other streams, if needed, but you have to explicitly know what they are and call them as shown in example above.

The last things to consider are the date ranges to bound the request and an aggregation value. Date ranges are fairly 
self-explanatory. You need to select a starting and ending date for the data of interest, otherwise you will get all 
the data for that instrument. That could potentially be a large request, so be careful. The dates entered need to be
recognizable as such. I'm using the [dateutil parser function](https://dateutil.readthedocs.io/en/stable/parser.html) to 
convert the dates you enter into the proper form for the M2M API. Alternatively, you can use the deployment number and
the M2M API will determine the dates. 

The aggregation value addresses a few cases where more than one instance of an instrument class is associated with
an assembly. For example the Global surface moorings have a midwater chain of 10 CTDs connected to an inductive modem 
line. If you do not specify the aggregation flag, you will only get the first of those 10 CTDs. If, however, you set the
aggregation flag to `0`, you will get all of them in a data set with a variable added called `sensor_count`, so you can
distinguish between them. Conversely, you can request a specific instrument by using its sequential number.

#### Simplified Request
At the end of the day, users need to know the site, assembly, instrument class and data delivery method. Somewhat
simpler than the full site, node, sensor, method, and stream name, and hopefully more meaningful. Additionally, any
date/time string format, so long as it can be recognized by the 
[dateutil parser function](https://dateutil.readthedocs.io/en/stable/parser.html), will work for setting starting and
ending dates for the data requests as opposed to explicitly setting the date to a format of `YYYY-MM-DDThh:mm:ss.fffZ`.
Or, you can just use the deployment number and let the system figure out the dates. Take a look at the examples below: 
```python
from ooi_data_explorations.data_request import data_request

# Set up the needed information to request data from the pH sensor on the Oregon
# Shelf Surface Mooring near-surface (7 m depth) instrument frame (NSIF).
site = 'ce02shsm'           # OOI site designator
assembly = 'midwater'       # Assembly grouping name
instrument = 'phsen'        # OOI instrument class 
method = 'telemetered'      # data delivery method

# the first four inputs are required in the order given above, the following
# inputs are semi optional, You need to specify at least a start date and/or
# stop date, or use the deployment number
start = '2019-04-01'       # data for April 2019 ...
stop = '2019-05-30'        # ... through May 2019
deploy = 9                 # The Spring 2019 deployment number

# request and download the data using specific dates
data_01 = data_request(site, assembly, instrument, method, start=start, stop=stop)

# request and download the data using the deployment number
data_02 = data_request(site, assembly, instrument, method, deploy=deploy)

# Set up the needed information to request data from the CTDMO sensor on the 
# Global Irminger Surface Mooring inductive modem line.
site = 'gi01sumo'           # OOI site designator
assembly = 'riser'          # Subassembly grouping name
instrument = 'ctdmo'        # OOI instrument class 
method = 'recovered_inst'   # data delivery method

start = '2019-04-01'        # data for April 2019 ...
stop = '2019-05-30'         # ... through May 2019

# request and download the data using specific dates, this only returns the 
# first instance of the CTDMOs
data_03 = data_request(site, assembly, instrument, method, start=start, stop=stop)

# request and download the data for all 10 CTDMOs
data_04 = data_request(site, assembly, instrument, method, start=start, stop=stop, aggregate=0)

# request and download the data for the CTDMO 5 out of 10.
data_05 = data_request(site, assembly, instrument, method, start=start, stop=stop, aggregate=5)
```

### Additional Utilities
In addition to `m2m_request` and `m2m_collect`, a collection of additional utilities are available to access instrument
and site deployment information. This information is collected in the OOI [Asset Management](https://github.com/oceanobservatories/asset-management)
database. It includes the dates, times and locations of deployments, instrument serial numbers, calibration coefficients,
and all the other pieces of information that combine to form the OOI metadata. These utilities and their use are 
demonstrated in a [Jupyter notebook](examples/notebooks/m2m_explorations.ipynb) available in the [examples](examples/notebooks) 
directory.

### Requesting Processed Data
For most individuals, the above code should satisfy your needs. For some of the data QC tasks I work through, the data
needs organizational reworking, renaming of variables or different processing to fit within my workflow. The 
process_*.py modules in the [cabled](./ooi_data_explorations/cabled) and [uncabled](./ooi_data_explorations/uncabled) 
directories represent an attempt on my part to rework the data sets into more useful forms before conducting any 
further work. Primarily, these re-works are for my own use, but they are available for others to use. The primary steps
are:

* Deleting certain variables that are of no use to my needs (helps to reduce file sizes)
* Renaming some parameters to more consistent names (across and within datasets). The original OOI names are preserved 
as variable level attributes termed `ooinet_variable_name`.
* Resetting the QC parameters to use the `flag_mask` and `flag_meaning` attributes from the 
[CF Metadata conventions](http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html#flags).
* Resetting incorrectly set units and other attributes for some variables.
* Reworking certain parameters by splitting or reshaping the data into more useful forms. 
* Update global attributes and otherwise cleaning up the data set.

Additionally, some instrument data is collected in burst mode (e.g. every 15 minutes for 3 minutes at 1 Hz). This
can make the data sets fairly large. By applying a median average to each of the bursts, the size of the data set can
be reduced to a more workable form, and the point-to-point variability in each burst can be smoothed out. Burst 
averaging is optional. Most of the processing functions are set to run from the command line. Examples of how these are
run can be found in the [examples directory](examples). Bash scripts to automate downloading date using these processing
scripts are in the [utilities/harvesters](utilities/harvesters) directory.

### QARTOD Workflows
OOI is beginning the process of replacing the current automated QC algorithms with [QARTOD](https://ioos.noaa.gov/project/qartod/)
[tests](https://github.com/ioos/ioos_qc) developed by [IOOS](https://ioos.noaa.gov/). The workflows and functions used 
to generate the test limits for Endurance Array assets are available under the [qartod](ooi_data_explorations/qartod) 
directory. These workflows rely on the processing functions described above. As more QARTOD tests are developed, the
processing functions and the QARTOD workflows will be extended. The goal is to create a record for the community 
detailing how the test limits were created and to facilitate regenerating those limits as more data is collected over
time.
