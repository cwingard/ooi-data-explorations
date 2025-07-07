function data = load_gc_thredds(site, node, sensor, method, stream, tag, parallel)
% LOAD_GC_THREDDS Download data from the OOI Gold Copy THREDDS catalog
%
% Download data from the OOI Gold Copy THREDDS catalog, using reference
% designator parameters to select the catalog of interest and a regex tag to
% select the NetCDF file(s) of interest.
% 
% INPUTS (required):
%
%   site -- Site designator, extracted from the first part of the
%       reference designator
%   node -- Node designator, extracted from the second part of the
%       reference designator
%   sensor -- Sensor designator, extracted from the third and fourth part
%       of the reference designator
%   method -- Delivery method for the data (either telemetered,
%       recovered_host, recovered_inst, recovered_cspp or recovered_wfp)
%   stream -- Stream name that contains the data of interest
%   tag --  Regex pattern to select the NetCDF files to download. This really
%       should NOT be too broad (e.g., ".*\.nc$") or you will be downloading
%       all of the data files in a catalog! Instead, refine the regex (e.g.,
%       add the deployment number and the instrument class name to limit the
%       the data download: "deployment0003.*OPTAA.*\.nc$") 
%
% INPUTS (optional)
%
%   parallel -- true/false boolean to enable use of the parallel processing
%       toolbox (if the user has it installed) to download the data files.
%       Can be helpful if downloading large numbers of large files (e.g.
%       all the ADCP data for a site). For smaller datasets, or a small
%       number of files, there isn't much benefit to using this option. The
%       default is false.
%
% OUTPUTS:
%
%   data -- All the data for the particular dataset of interest, combined into
%       a single timetable
%
% C. Wingard, 2023-07-10

% define the argument data types and set the default value for the optional
% parallel input (false if not set)
arguments
    site char
    node char
    sensor char
    method char
    stream char
    tag char
    parallel logical = false;
end

% load the common file processing utilities
ph = process_files;

% set up the Gold Copy THREDDS catalog URL, the OpenDAP server URL, and the
% dataset ID
base_url = "https://thredds.dataexplorer.oceanobservatories.org/thredds/catalog/ooigoldcopy/public/";
dap_url = "http://thredds.dataexplorer.oceanobservatories.org/thredds/dodsC/ooigoldcopy/public/";
dataset_id = join([upper(string(site)), upper(string(node)), upper(string(sensor)), lower(string(method)), lower(string(stream))], "-");
options = weboptions("Timeout", 300);

% determine the number of available files to download based on the dataset ID
% and the regex tag
url = join([base_url dataset_id "/catalog.html"], "");
files = ph.list_files(url, tag);
nfiles = numel(files);      % determine number of files to download
if nfiles == 0
    data = [];  % no data to download
    return
end %if
data = cell(nfiles, 1);     % pre-allocate a cell array for downloaded results

% check to see if the user has the parallel processing toolbox and there are
% enough files to justify its use (buildup and teardown of the parallel
% processing utilities costs time, so we only want to use it when it benefits
% us)
fprintf('Downloading %d files from the Gold Copy THREDDS Catalog ...\n', nfiles)
tic
if parallel == true && exist("parpool", "file") == 2 && numel(files) > 5
    % start up a local parallel pool to handle downloading the multiple files.
    % ideally this would be a multithreaded pool, but only subset of functions
    % are available in the multithreaded environment and the NetCDF utilities
    % are not included
    delete(gcp('nocreate'));
    mypool = parpool('local');
    startS = ticBytes(mypool);  % track pool stats
    % use a parallel processing loop to download the files, adding them to the
    % pre-allocated array
    parfor i = 1:nfiles
        [~, file, ext] = fileparts(files(i));
        nc_url = join([dap_url, dataset_id, '/', file, ext, '#fillmismatch'], '');
        source = websave([tempname, '.nc'], strrep(nc_url, 'dodsC', 'fileServer'), options);
        data{i} = ph.process_file(source); %#ok<PFBNS>
        delete(source);
    end %parfor
    tocBytes(mypool, startS)
    delete(mypool)
    clear mypool startS
else
    % use a traditional for loop to download the files, adding them to the 
    % pre-allocated cell array
    percentage = 0;
    backspaces = '';
    percent_step = 100 / nfiles;
    for i = 1:nfiles
        [~, file, ext] = fileparts(files(i));
        nc_url = join([dap_url, dataset_id, '/', file, ext, '#fillmismatch'], '');
        source = websave([tempname, '.nc'], strrep(nc_url, 'dodsC', 'fileServer'), options);
        data{i} = ph.process_file(source);
        delete(source);

        % Print percentage progress
        percentage = percentage + percent_step;
        perc_str = sprintf('... percent loaded: %3.1f', percentage);
        fprintf([backspaces, perc_str]);
        backspaces = repmat(sprintf('\b'), 1, length(perc_str));
    end %for
    fprintf("\n")
    clear percentage backspaces percent_step perc_str
end %if
toc
clear base_url dap_url dataset_id url files nfiles options i file ext nc_url source

% finally, concatenate the timetables together returning the results
if length(data) > 1
    fprintf("Merge and finalize the data set, returning final product as a timetable.\n")
    data = ph.merge_frames(data);
else
    fprintf("Returning final product as a timetable.\n")
    data = data{1};
end %if
clear ph
end %function
