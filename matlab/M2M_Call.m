function [nclist] = M2M_Call(uframe_dataset_name,start_date,end_date,options)
%.. 2019-10-XX: CMRisien. original code.
%.. 2019-11-04: RADesiderio.
%..             (a) put in a for loop to iterate over multiple streams
%..             (b) restored the first try-catch block that CMR removed.
%..             (c) restored the incompleteness check that CMR removed.
%..             (d) removed data_url from the input argument list;
%..             (e) variable m2m_url transferred in from function M2M_URLs;
%..             (f) transferred thredds_url out to be a variable in M2M_Data;
%..             (g) changed output argument from nc_urls to nclist so that the
%..                 option of using different sources in the websave calls
%..                 in M2M_Data could be implemented.
%.. 2019-12-16: RADesiderio.
%..             (a) reverted from a previous version which had added a urlread
%..                 branch to work with matlab version R2019b. this branch no
%..                 longer exists in the code; compatibility with R2019b is
%..                 achieved by changing the 'options' input argument in the
%..                 calling program. See Notes.
%..             (b) added parsing for catalog and async URLs from m2m_response
%
%.. NOTES:
%..
%.. Use the following sentence in the calling program to ensure compatibility
%.. with all (recent) versions of Matlab (tested up to R2019b). A different 
%.. value can be used for the Timeout setting. Note that if the webread call
%.. fails using the following weboptions call and the user is running on 
%.. linux, an additional CertificateFilename setting to '' (empty character
%.. string) may be required.
%
% options = weboptions('HeaderFields', {'Authorization', ...
%    ['Basic ' matlab.net.base64encode([api_key ':' api_token])]}, 'Timeout', 120);
%
m2m_url = "https://ooinet.oceanobservatories.org/api/m2m/12576/sensor/inv/";
% Check type of uframe_dataset_name; when it comes in as a char array, must be converted to string
if isa(uframe_dataset_name, "char")
    dataset_name = string(uframe_dataset_name);
else
    dataset_name = uframe_dataset_name;
end

data_url = strcat(m2m_url, dataset_name);
data_options = "?beginDT=" + start_date + "&endDT=" + end_date + "&format=application/netcdf&email=None";

%.. Make M2M Call(s)
nclist = cell((1:length(data_url)), 1);
response_status(1:length(data_url)) = {"No Uframe data found."};
for jj = 1:length(data_url)
    %.. request the data and start monitoring for a completed request.
    %.. .. cannot distinguish between the following two errors, because they
    %.. .. both give identical error messages, including "NOT FOUND":
    %.. .. (1) no data in requested time range
    %.. .. (2) incorrect spelling of dataset name.
    try
        m2m_response = webread(data_url{jj} + data_options, options);
        response_status{jj} = 'M2M REQUEST MADE, not yet completed:';
        disp(response_status{jj});
        %.. the following two URLs are not always presented in the same order, so:
        catalogURL = string(m2m_response.allURLs(contains(m2m_response.allURLs, 'catalog')));
        asyncURL = string(m2m_response.allURLs(contains(m2m_response.allURLs, 'async')));
    catch ME
        %.. comment out ME diagnostics, display abbreviated diagnostic instead
        %disp(ME.identifier);
        %disp(ME.message);
        
        %.. for NUTNR requests, several different datastreams are required to
        %.. span the entire time that these instruments were deployed. These
        %.. datastreams do not exist for all of these time periods, therefore
        %.. throw a warning and not a fatal error if no Uframe data are found.
        %.. (For example, the SUNA replaced the ISUS on surface moorings, and
        %.. while the data collected are identical, the data formats are not,
        %.. requiring different streams that were applicable at different
        %.. times).
        if strcmp(ME.identifier, 'MATLAB:webservices:HTTP401StatusCodeError')
            response_status{jj}  = ['ERROR: ' ...
            'UFrame credentials error. Request unauthorized: ' ...
            ME.identifier];
            error(['uframe_m2m_status: ' response_status{jj}]);
        end
        response_status{jj} = ['WARNING: ' ...
            'No UFrame data found for datastream ' dataset_name{jj}];
        disp(['uframe_m2m_status: ' response_status{jj}]);
        disp(' ');
        continue
    end
    
    for ii = 1:1800
        try
            check_complete = webread(join([asyncURL, "status.txt"], "/"));
            if strip(check_complete) == "complete"
                response_status{jj} = 'request complete';
                fprintf("done\n");
                break
            else
                pause(1);
                response_status{jj} = 'request incomplete';
                if mod(ii,10)==0, fprintf('%u ', ii); end
            end
        catch
            pause(1);
            response_status{jj} = 'request incomplete';
            if mod(ii,10)==0, fprintf('%u ', ii); end
        end
    end
    disp(['uframe_m2m_status: ' response_status{jj}]);
    if ~strcmp(response_status, 'request complete')
        disp(' ');
        error(['***WARNING: ' dataset_name{jj} ' NEEDS TO BE RE-RUN***']);
    end
    
    % now put together the list of available NetCDF files from the THREDDS server
    catalog = webread(catalogURL, 'ContentType', 'text');
    %nc_all = regexp(catalog, '<a href=''([^>]+.nc)''>', 'tokens'); % cell elements are themselves cells
    nc_all = regexp(catalog, '<a href="(.*?\.nc)">', 'tokens', 'dotexceptnewline');

    %.. 2019 fdchp case: m2m request is successfully completed, but
    %..                  there are no netcdf files in the catalog.
    if isempty(nc_all)
        response_status{jj} = 'WARNING: m2m request completed: no uframe netcdf files';
        disp(['uframe_m2m_status: ' response_status{jj}]);
        continue
    else
        response_status{jj} = 'successful';
        disp(['uframe_m2m_status: ' response_status{jj}]);
    end
    
    disp('THREDDS Catalog URL:')
    disp(catalogURL);
    disp(' ');
    
    nc_all = [nc_all{:}]';  % nc_all is now a cell array of character vectors
    nc_all = cellfun(@(x) x(22:end), nc_all, 'UniformOutput', 0); % drop 'catalog.html?dataset=' from URL
    
    %.. prune the list of mapped files to our instrument of interest by eliminating
    %.. files associated with other instruments used to calculate data products
    strings_to_match = sprintf('.*/deployment.*%s.*\\.nc$', strrep(dataset_name{jj}, '/', '-'));
    nc_all = cellfun(@(x) regexp(x, strings_to_match, 'match'), nc_all, 'UniformOutput', 0);
    nc_all(cellfun('isempty', nc_all)) = [];  % cell elements are themselves cells
    nclist{jj} = string(nc_all(:));  % string array
end
%.. stop execution if no uframe data found
if ~any(contains(lower(response_status{:}), 'successful'))
    error('***NO UFRAME DATA FOUND IN ANY DATASTREAM(S)***');
end
%.. consolidate nclist into one string array
nclist = vertcat(nclist{:});
nclist(cellfun('isempty', nclist)) = [];
if isempty(nclist)
    error('***NO NETCDF FILES FOUND IN ANY DATASTREAM(S)***');
end

end  % function
