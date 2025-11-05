function [timedata,rawdata] = fdchpread_raw(filename)
fid = fopen(filename,'r','ieee-be');
fseek(fid,0,'eof');
bytecnt=ftell(fid);
fseek(fid,0,'bof');
if ((rem(bytecnt,55)==0) & (bytecnt ~= 0))
    num_rec=bytecnt/55;
    timedata = [];
else
    disp('Bad Length');
    timedata = [];
    dcfsdata = [];
    fclose(fid);
    return;
end

fseek(fid,0,'bof');
yeardat=fread(fid,inf,'uint16',53,'ieee-be');  %Year
fseek(fid,2,'bof');
tdat=fread(fid,inf,'5*uint8',50,'ieee-be');    %Month, Day, Hr, Min, Sec
fseek(fid,7,'bof');
sdat=fread(fid,inf,'uint16',53,'ieee-be');
fseek(fid,9,'bof');
sondat=fread(fid,inf,'3*int16',49,'ieee-be');
fseek(fid,15,'bof');
sontempdat=fread(fid,inf,'uint16',53,'ieee-be');

fseek(fid,17,'bof');
imudat=fread(fid,inf,'9*float',19,'ieee-be');
fseek(fid,53,'bof');
statdat=fread(fid,inf,'uint16',53,'ieee-be');

tdatlen=size(tdat,1);
sondatlen=size(sondat,1);
imudatlen=size(imudat,1);
stlen=size(sontempdat);

fclose(fid);

rawdata=zeros(21,num_rec);
rawdata(1,:) = yeardat;
rawdata(2,:) = tdat(1:5:tdatlen); 
rawdata(3,:) = tdat(2:5:tdatlen);
rawdata(4,:) = tdat(3:5:tdatlen);
rawdata(5,:) = tdat(4:5:tdatlen);
rawdata(6,:) = tdat(5:5:tdatlen);
rawdata(7,:) = sdat/(1000*86400)';
rawdata(8,:) = sondat(1:3:sondatlen)'*0.01;  %U - 
rawdata(9,:) = sondat(2:3:sondatlen)'*0.01;  %V
rawdata(10,:) = sondat(3:3:sondatlen)'*0.01; %W
rawdata(11,:) = sontempdat'*0.01;            %SoS
rawdata(12,:) = imudat(1:9:imudatlen)';  %Dx 
rawdata(13,:) = imudat(2:9:imudatlen)';  %Dy
rawdata(14,:) = imudat(3:9:imudatlen)';  %Dz
rawdata(15,:) = imudat(4:9:imudatlen)';  %Ax
rawdata(16,:) = imudat(5:9:imudatlen)';  %Ay
rawdata(17,:) = imudat(6:9:imudatlen)';  %Az
rawdata(18,:) = imudat(7:9:imudatlen)';  %roll
rawdata(19,:) = imudat(8:9:imudatlen)';  %pitch
rawdata(20,:) = imudat(9:9:imudatlen)';  %heading
rawdata(21,:) = statdat';




