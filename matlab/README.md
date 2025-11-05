# OOI Data Explorations with MATLAB

## Installation

Download the main [ooi-data-explorations repository](https://github.com/oceanobservatories/ooi-data-explorations/archive/refs/heads/master.zip) as a ZIP, 
extract the ZIP file, and navigate to the matlab sub directory.
Rename this director to something unique to prevent possible confusion with other toolboxes (e.g. ooi_m2m).

Alternatively, clone this repository with `git` and navigate to the `matlab` subdirectory to see the available MATLAB functions and examples:
```shell
git clone https://github.com/oceanobservatories/ooi-data-explorations.git
cd ooi-data-explorations/matlab
```

### Windows
1. Navigate to C:\Program Files\MATLAB\RYYYYY\toolbox. 
	- YYYYY is your MATLAB version. These functions and example scripts have been tested using Matlab 2019b and above.  
2. Move the ooi_m2m folder to the above location.
3. Open MATLAB.
4. Via MATLAB, select "Set Path".
5. Select "Add with Subfolders".
6. Select the ooi_m2m folder.
7. Select "Save".
8. Select "Close".

### Testing
To test that the toolbox was successfully added to path, you can open one of the functions using the MATLAB console. Example: 

 ```
 open M2M_Call
 ```

The return should be the code that issues requests to OOINet. If an error appears, the toolbox was not successfully added to path.


### Examples
Example scripts can be found in the [examples folder](https://github.com/oceanobservatories/ooi-data-explorations/tree/master/matlab/examples).

### Usage
If you have not done so already, you will need to create an [OOINet](https://ooinet.oceanobservatories.org/) account to make requests for data
(see section on Access Credentails below). If you are not familiar with OOI nomenclature, we recommend starting with the list of 
[OOI sites](https://oceanobservatories.org/site-list/).

#### Access Credentials for the OOI M2M API
In order to access the OOI M2M API, you need to setup your OOINet credentials as
a `weboptions` object that Matlab can then use as part of a `webread` request
to the API.  To set this up, in Matlab (replacing `<API Username>` and 
`<API Token>` with your OOINet credentials):

``` matlab
>> username = '<API Username>';
>> password = '<API Token>';
>> options = weboptions('CertificateFilename', '', 'HeaderFields', {'Authorization', ...
    ['Basic ' matlab.net.base64encode([username ':' password])]}, ...
    'Timeout', 180);
>> save('ooinet.credentials.mat', 'options');
```

Put the `ooinet.credentials.mat` file in your Matlab path (best option is your
Matlab home directory as it is on the Matlab path by default) so it is available
for this application. The code uses these credentials to make requests to the M2M
API for data and metadata.

#### M2M_URLs
After obtaining your OOI API username and token, you will then need to identify the site, node, instrument (sensor), and method of data delivery that you are interested in. 
Plugging this information into the M2M_URLs function will generate a request string that can be input into the M2M_Call function. The list of variables associated with the request is also returned.

```
% Example: Returns a requestable string and list of variables that are attached to the oxygen optode located on the seafloor at CE01ISSM (Oregon Inshore Surface Mooring).
site = 'CE01ISSM'
node = 'MFN'
instrument = 'DOSTA'
method = 'RecoveredHost'

[ce01issm_mfn,ce01issm_mfn_vars] = M2M_URLs(site,node,instrument,'RecoveredHost')
```

#### M2M_Call

This function takes the first return from M2M_URLs and issues the request to OOINet. The return is a list of NetCDF extensions that can be used to download data or bring it into the Workspace using the M2M_Data function.
An OOI API Username and Token, as well as the request options, are required to issue the request. You will also need to identify the start and end datetimes of your data request.
Start and stop times must follow the ISO 8601 format ('YYYY-mm-ddTHH:MM:SS.sssZ')

```
% Example: Issues the request for the seafloor oxygen optode data at CE01ISSM 
% between May 01, 2019 and August 31, 2019.
% Either create a weboptions object for your credentials, or load one already
% created and saved (as above).
% load('ooinet.credentials.mat', 'options')
user = 'OOI-API-USER-HERE'
token = 'OOI-API-TOKEN-HERE'
options = weboptions('CertificateFilename','','HeaderFields',{'Authorization',...
    ['Basic ' matlab.net.base64encode([user ':' token])]}, 'Timeout', 120);

start_time = '2019-05-01T00:00:00.000Z'; 
end_time = '2019-08-31T23:59:59.999Z';

ce01issm_nc = M2M_Call(ce01issm_mfn,start_time,end_time,options);  
```

#### M2M_Data
This function takes the return from M2M_Call and will bring the data into the MATLAB Workspace. If you positionally specify a third parameter as true (default) data is brought into the Workspace. 
If the third parameters is set to false, data are downloaded. 
The return for this function is the data as a series of arrays, time as an array, and the filenames of where the data was downloaded from.

```
% Example: Returns data from the request for oxygen optode data from the CE01ISSM seafloor package.
[ce01_vars,ce01_time,ce01_files] = M2M_Data(ce01issm_mfn_vars,ce01issm_nc);
```
