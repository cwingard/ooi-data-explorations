function file_list = get_fdchp_raw_files_for_deployment(site, deployment)
% Get a list of fdchp raw data files from the OOI kdata directory for a given
% site and deployment.
%
% Get list of matching files from the local raw data directory available on the OOI JupyterHub
% using reference site and deployment to select the files of interest.
% 
% INPUTS:
%
%   site -- Site designator, extracted from the first part of the
%       reference designator
%   deployment -- integer deployment number
%
% OUTPUTS:
%
%   file_list -- All the data for the particular dataset of interest, combined into
%       a single list
%
% J. Peters, 2024-11-31, based on load_kdata.m by C. Wingard, 2023-07-10

base_dir = "/home/jovyan/ooi/uncabled/";
if ~isfolder(base_dir)
    error("This function can only be used on the OOI JupyterHub")
end %if

% determine the number of available files to download based on the site and deployment
deployment_dir = join(["R" num2str(deployment,'%05.f')], "");
data_dir = join([base_dir upper(string(site)) deployment_dir "instruments" "dcl12" "FDCHP*" "*" "*.dat"],"/");
fprintf('Looking for data files in  %s\n', data_dir)
file_list = dir(data_dir);
nfiles = numel(file_list);      % determine number of files to download

fprintf('Found %d files from the local raw data directory ...\n', nfiles)

end %function
