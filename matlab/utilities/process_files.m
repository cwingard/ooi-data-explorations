function fh = process_files()
% PROCESS_FILES Function handles to common files used to load NetCDF data
%
% cwingard 2023-07-09
fh.list_files = @list_files;
fh.process_file = @process_file;
fh.merge_frames = @merge_frames;
end %function

function nc_files = list_files(url, tag)
% LIST_FILES create a list of the NetCDF data files in a THREDDS catalog
%
% Inputs are the URL for the THREDDS catalog of interest and a regex tag
% used to subselect only the NetCDF files of interest
%
% cwingard 2023-07-09

% use webread to scrape the catalog
options = weboptions('ContentType', 'text');
catalog = webread(url, options);

% pull a list of netCDF files out of the catalog
nc_all = regexp(catalog, '<a href="([^>]+.nc)">', 'tokens');

% cell elements are themselves cells, convert to cell array of character vectors
nc_all = [nc_all{:}]';

% drop 'catalog.html?dataset=' from URL
nc_all = cellfun(@(x) x(22:end), nc_all, 'UniformOutput', 0);
[~, files, ~] = fileparts(nc_all);

% use the regex tag to select only files of interest
files = cellfun(@(x) regexp([x, '.nc'], tag, 'match'), files, 'UniformOutput', 0);
nc_all(cellfun('isempty', files)) = [];  % remove empty matches

% convert the list to a string array
nc_files = string(nc_all(:));

% check to see if we found any files to download, if not throw an error
if size(nc_files, 1) == 0
    warning("Unable to find any files to download using tag '%s' from '%s'", tag, url)
end %if
end %function

function data = process_file(filename)
% PROCESS_FILE Load NetCDF files (local or from OpenDAP) into a timetable
%
% cwingard 2023-07-09

% Get information about the NetCDF data file
file_info = ncinfo(filename);

% load the data into a timetable
data = nc_reader(filename);

% add global attributes to the timetable
m = reshape(strcmp({file_info.Attributes.Name}, 'publisher_name'), size(file_info.Attributes));
if sum(m) == 1
    if strcmp(file_info.Attributes(m).Value, 'Ocean Observatories Initiative')
        % assign the NetCDF title attribute to the timetable description
        m = reshape(strcmp({file_info.Attributes.Name}, 'title'), size(file_info.Attributes));
        data.Properties.Description = file_info.Attributes(m).Value;
        
        % create custom properties for the timetable, pulling different global 
        % attributes out of the NetCDF file and assigning them
        prop_names = {'subsite', 'node', 'sensor', 'collection_method', ...
            'stream', 'date_created', 'date_modified', 'requestUUID', ...
            'Manufacturer', 'ModelNumber', 'Description', 'SerialNumber', ...
            'AssetUniqueID', 'AssetManagementRecordLastModified', ...
            'publisher_name', 'acknowledgement'};
        prop_types = {'table', 'table', 'table', 'table', 'table', 'table', ...
            'table', 'table', 'table', 'table', 'table', 'table', 'table', ...
            'table', 'table', 'table'};
        data = addprop(data, prop_names, prop_types);    
        for i = 1:size(prop_types, 2)
            m = reshape(strcmp({file_info.Attributes.Name}, prop_names{i}), size(file_info.Attributes));
            if sum(m) == 1
                data.Properties.CustomProperties.(prop_names{i}) = file_info.Attributes(m).Value;
            end %if
        end %for
        clear prop_names prop_types i
    end %if
end %if
clear file_info m
end %function

function data = merge_frames(frames)
% MERGE_frames Combine downloaded data into a single timetable
%
% Created by Christopher Wingard, 2023-07-10

% determine the number of frames to process
nframes = numel(frames);
if nframes > 1
    % merge the frames into a single timetable
    try
        % simplest method, works most of the time, is to just vertically
        % concatenate the timetables together
        data = cat(1, frames{:});
    catch
        % usually one or more of the timetables will be missing a variable(s)
        % and the vertical concatentation will fail.
        for i = 2:nframes
            % find the missing variable(s) and add using a fill value
            nvars1 = numel(frames{i-1}.Properties.VariableNames);
            nvars2 = numel(frames{i}.Properties.VariableNames);
            if nvars1 > nvars2
                m = ~ismember(frames{i-1}.Properties.VariableNames, frames{i}.Properties.VariableNames);
                varnames = frames{i-1}.Properties.VariableNames(m);
                t = frames{i};
                for j = 1:numel(varnames)
                    t = addvars(t, repmat(convertTo(t.Time, "datenum"), 1, 1) .* -inf, 'NewVariableNames', varnames{j});
                end %for
                frames{i} = t;
            elseif nvars1 < nvars2
                m = ~ismember(frames{i}.Properties.VariableNames, frames{i-1}.Properties.VariableNames);
                varnames = frames{i}.Properties.VariableNames(m);
                t = frames{i-1};
                for j = 1:numel(varnames)
                    t = addvars(t, repmat(convertTo(t.Time, "datenum"), 1, 1) .* -inf, 'NewVariableNames', varnames{j});
                end %for
                frames{i-1} = t;
            else
                % frame mismatch corrected
                continue
            end %if
        end %for
        % try merging the frames one more time
        try
            data = cat(1, frames{:});
        catch ME
            % something is fundamentally wrong here....
            rethrow(ME)
        end %try
    end %try
end %if
clear nframes nvars1 nvars2 m t

% finally, sort the data first by deployment and then time
data = unique(data, 'rows');  % remove any duplicate entries
data = sortrows(data, {'deployment', 'Time'});
end %function
