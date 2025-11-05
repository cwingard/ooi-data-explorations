function [fluxes,Uearth,waveheight] = ProcessFDCHP_new(rawdata,L,bugfix,lat,Rwaves);

% HACK-10Hz Ts1 = 20/100;    % Sampling period for FDCHP
Ts1 = 10/100;                % Sampling period for FDCHP
fs = 1/Ts1;                  % Sampling frequency for Windmaster
dt=1/fs;                     % Time step for 10 Hz data

%JBE Change tc1=tc2=15;
tc1=20;                      % Define constants for filters
tc2=tc1;                     % tc1 is the cutoff period that determines where the pitch and roll Euler angles are 
fc1=1/tc1;                   %    determined by the tilt of the accelerometers at low frequencies versus the 
fc2=1/tc2;                   %    integrated angular rates at high frequencies.  This is known as complementary
tcwave=30;                   %    filtering
fcwaves=1/tcwave;            % Same as above but for the wave measurements, which requires a large cutoff period
rad2deg=180.0/pi;            %    for swell
Rvec=zeros(1,3);             % Distance vector between the sonic anemometer and motion sensors
Rvec(3)=0.85;
roffset=0;                   %These are not used
poffset=0;

%******************************************************
% bugfix, lat and Rwaves are now passed as variables 
% so the same code can be use for the 3 flux moorings
%******************************************************
%bugfix=0;
%lat=40.1334;
%lat = 59.93370;
%Rwaves=[-0.75 0 -5];
Lat=double(lat);
G=grv(Lat);
gfix=1;
despike=1;

% JBE 06/29 JW 11Aug2014
% 1.5 JW 3Sep2014
% 2.0 JBE 06/12/2021
% 3.0 JBE 05/11/2022
% 3.5 JBE 01/02/2023

version_number=3.5;
status_val = uint32(1);

%JBE Redefine files for 10 Hz and tc1=12 or 15 or 20.  Use digits(16) and
%vpa(ahiwaves)
if tc1==12;           %JBE Redefine filters for 10 Hz and tc1=12 with ludo=1
    ahi=[1.000000000000000  -3.869797539975555   5.617802044587569  -3.625896801659086   0.877898078061702];
    bhi=[0.936962154017745  -3.747848616070978   5.621772924106467  -3.747848616070978   0.936962154017745];
elseif tc1==15;           %JBE Redefine filters for 10 Hz and tc1=15 with ludo=1
    ahi=[1.000000000000000  -3.895833876325376   5.692892648957240  -3.698121672490409   0.901065298297354];
    bhi=[0.949244593504399  -3.796978374017595   5.695467561026392  -3.796978374017595   0.949244593504399];
else
    ahi=[1.000000000000000  -3.921872982100935   5.768656400578301  -3.771625137827138   0.924842488052324];
    bhi=[0.961687313034919  -3.846749252139674   5.770123878209512  -3.846749252139674   0.961687313034919];
end

%JBE Redefine files for 10 Hz and fcwaves=1/40;
if tcwave==40;
    ahiwaves = [1.000000000000000  -3.960935321365416  5.883567180614652 -3.884319737084527  0.961687926819144];
    bhiwaves = [0.980656885367734  -3.922627541470935  5.883941312206403 -3.922627541470935  0.980656885367734];
else
    %JBE Redefine files for 10 Hz and fcwaves=1/30;
    ahiwaves = [1.0000000000000000 -3.947914166208924  5.845094680183927 -3.846426389902994  0.9492460297427443];
    bhiwaves = [0.9742925791274119 -3.897170316509647  5.845755474764471 -3.897170316509647  0.9742925791274119];
end
 
% HACK-10Hz L=12000;
% JBE change
% L=12000;
dcfsdata=zeros(15,L);

%    dcfsdata(1,i) = i;
%    dcfsdata(2,i) = sonic5hz(i,1);
%    dcfsdata(3,i) = sonic5hz(i,2);
%    dcfsdata(4,i) = sonic5hz(i,3);
%    dcfsdata(5,i) = sonic5hz(i,4);
%    dcfsdata(6,i) = heading(i);
%    dcfsdata(7,i) = roll(i);
%    dcfsdata(8,i) = pitch(i);
%    dcfsdata(9,i) = rateX(i);
%    dcfsdata(10,i) = rateY(i);
%    dcfsdata(11,i) = rateZ(i);
%    dcfsdata(12,i) = mpaktemp(i);
%    dcfsdata(13,i) = accX(i);
%    dcfsdata(14,i) = accY(i);
%    dcfsdata(15,i) = accZ(i);

%dcfsdata(1,1:L) = 1:L';
mtime = datenum(rawdata(1,1:L),rawdata(2,1:L),rawdata(3,1:L),rawdata(4,1:L),rawdata(5,1:L),rawdata(6,1:L)) + rawdata(7,1:L); %/(1000*86400);
Mtime=mean(mtime);
%disp(datestr(Mtime));
%dcfsdata(1,i) = datenum(yeardat,tdat(i,2),timedata(i,3),timedata(i,4),timedata(i,5),timedata(i,6))+ timedata(i,7)/(1000*86400);
%dcfsdata(1,:) = datenum(yeardat,tdat(1:5:tdatlen),tdat(2:5:tdatlen),tdat(3:5:tdatlen),tdat(4:5:tdatlen),tdat(5:5:tdatlen))+sdat/(1000*86400)';
%UNITS#3 Velocities are m/s

dcfsdata(2,1:L) = rawdata(8,1:L);  %wind x -- fdchp_wind_x
dcfsdata(3,1:L) = rawdata(9,1:L);  %wind y -- fdchp_wind_y
dcfsdata(4,1:L) = rawdata(10,1:L); %wind z -- fdchp_wind_z
dcfsdata(5,1:L) = rawdata(11,1:L); %Speed of sound -- fdchp_speed_of_sound_sonic
%UNITS#4 - Rates are in radians/s, accels are in G=9.80665 m/s^2, pitch,roll and yaw are in radians. 
dcfsdata(6,1:L) = rawdata(20,1:L); %*180.0/pi; %heading  -- fdchp_heading
dcfsdata(7,1:L) = rawdata(18,1:L); %*180.0/pi; %roll  -- fdchp_roll
dcfsdata(8,1:L) = rawdata(19,1:L); %*180.0/pi; %pitch  -- fdchp_pitch
dcfsdata(9,1:L) = rawdata(12,1:L); %*180.0/pi; %rate x  -- fdchp_x_ang_rate
dcfsdata(10,1:L) = rawdata(13,1:L); %*180.0/pi; %rate y  -- fdchp_y_ang_rate
dcfsdata(11,1:L) = rawdata(14,1:L); %*180.0/pi; %rate z  -- fdchp_z_ang_rate
%JBE The temperature is never used, so we may want to remove it
dcfsdata(12,1:L) = rawdata(21,1:L); % not used  -- ????
dcfsdata(13,1:L) = rawdata(15,1:L); % accel x -- fdchp_x_accel_g
dcfsdata(14,1:L) = rawdata(16,1:L); % accel y -- fdchp_y_accel_g
dcfsdata(15,1:L) = rawdata(17,1:L); % accel z -- fdchp_z_accel_g
    
%tsec = typecast(rawdata(1,1:L),'uint32');
%tmsec = cast(rawdata(2,1:L),'uint16');

% Convert Sonic Speed of Sound to temperature
% Not required with R3

sos=dcfsdata(5,1:L);
if median(sos, "omitnan")<50
    dcfsdata(5,1:L)=sos;
else
    dcfsdata(5,1:L)=sos.^2./403 - 273.15;
end

% Convert IMU from North East Down coordinate system to North West Up coordinate system to match Sonic.
% Note that we can leave the x-axis variable as measured.

dcfsdata(6,1:L) = -1.0*dcfsdata(6,1:L);   % z heading(yaw) counter clockwise
dcfsdata(8,1:L) = -1.0*dcfsdata(8,1:L);   % y pitch east to west
dcfsdata(10,1:L) = -1.0*dcfsdata(10,1:L); % angular rate around y-axis
dcfsdata(11,1:L) = -1.0*dcfsdata(11,1:L); % angular rate around z-axis 
dcfsdata(14,1:L) = -1.0*dcfsdata(14,1:L); % linear acceleration along y-axis
dcfsdata(15,1:L) = -1.0*dcfsdata(15,1:L); % linear acceleration along z-axis (positve up)

%pre-define arrays
sonics=zeros(3,L);        % These are the 3-axis sonic velocities
Tv=zeros(1,L);            % This is the sonic temperature computed from speed of sound
compass=zeros(1,L);        
roll=zeros(1,L);           
pitch=zeros(1,L);
platform=zeros(3,L);      % This will hold the accelerations
deg_rate=zeros(3,L);      % This will hold the angular rates

% sonic velocities
sonics(1:3,1:L)=dcfsdata(2:4,1:L);
if despike                                % Remove obvious spikes in the data
    [sonics] = despikesimple(sonics,L);
end

% fixes problem due to Gill transducers if bugfix = 1
if bugfix
    Ww=sonics(3,:);
    iup=find(Ww>=0);
    sonics(3,iup)=Ww(iup)*1.166;
    idn=find(Ww<0);
    sonics(3,idn)=Ww(idn)*1.289;
end

Uavg=mean(sonics(1,1:L));
Vavg=mean(sonics(2,1:L));
Wavg=mean(sonics(3,1:L));
rdir=atan2(Vavg,Uavg); %*180/pi;

Umax=max(sonics(1,1:L));
Vmax=max(sonics(2,1:L));
Wmax=max(sonics(3,1:L));
Umin=min(sonics(1,1:L));
Vmin=min(sonics(2,1:L));
Wmin=min(sonics(3,1:L));

%JBE  This is not the standard deviation. 
%Ustd = mean(abs(sonics(1,1:L)-Uavg));
Ustd = std(sonics(1,1:L));
Vstd = std(sonics(2,1:L));
Wstd = std(sonics(3,1:L));

% Above we convert SoS to temperature T1=C1^2/403 
% Sonic temperature
Tv=dcfsdata(5,1:L);
if despike
    [Tv] = despikesimple(Tv,L);
end

Tavg=mean(Tv);
Tstd = std(dcfsdata(5,1:L));

Tvmax=max(Tv);
Tvmin=min(Tv);

tmot=dcfsdata(12,1:L);

%***********************************************
% Deal with the compass 
% JBE Fill in bad points first units and signs 
% UNITS#5-Roll, pitch and yaw are in radians
%***********************************************

compass=dcfsdata(6,1:L);  %*pi/180;
roll=dcfsdata(7,1:L); %*pi/180;
pitch=dcfsdata(8,1:L); %*pi/180;
compstd=std(unwrap(compass)); %*180/pi;
compmin=min(compass); %*180/pi;
compmax=max(compass); %*180/pi;

gx=cos(compass);
gy=sin(compass);
compcos = mean(gx);
compsin = mean(gy);
compavg = atan2(compsin,compcos);
if (compavg < 0)
    compavg=compavg + 2.0*pi;
end

despikecompass=0;
if despikecompass
    [gx] = despikesimple(gx,L);
    [gy] = despikesimple(gy,L);
    gsmooth=atan2(gy,gx);
    gyro=gsmooth;
else
    gyro=compass;
end
gchk=unwrap(gyro); %*180/pi;
stdhdg=std(gchk);
goodcompass=1;

%***********************************************
% Then the angular rates
% UNITS#9 Rates are in radian/sec
%***********************************************
deg_rate = dcfsdata(9:11,1:L);
if despike
    [deg_rate] = despikesimple(deg_rate,L);
end

dcomp=mean(deg_rate');
dstd=std(deg_rate');

Phiavg = dcomp(1);
Thetaavg = dcomp(2);
Psiavg = dcomp(3);
Phistd = dstd(1);
Thetastd = dstd(2);
Psistd = dstd(3);
Phimax = max(deg_rate(1,1:L));
Thetamax = max(deg_rate(2,1:L));
Psimax = max(deg_rate(3,1:L));
Phimin = min(deg_rate(1,1:L));
Thetamin = min(deg_rate(2,1:L));
Psimin = min(deg_rate(3,1:L));

%***********************************************
% Then the accelerations
%***********************************************
platform = dcfsdata(13:15,1:L);
if despike
    [platform] = despikesimple(platform,L);
end

gcomp=zeros(1,3);
gcomp(1)=mean(platform(1,1:L));
gcomp(2)=mean(platform(2,1:L));
gcomp(3)=mean(platform(3,1:L));
Axavg=gcomp(1);
Ayavg=gcomp(2);
Azavg=gcomp(3);
Axmax=max(platform(1,1:L));
Aymax=max(platform(2,1:L));
Azmax=max(platform(3,1:L));
Axmin=min(platform(1,1:L));
Aymin=min(platform(2,1:L));
Azmin=min(platform(3,1:L));

gstd=std(platform');
Axstd=gstd(1);
Aystd=gstd(2);
Azstd=gstd(3);

%***************************************************************
% We no longer force the mean accelerometers to equal gravity
% UNITS#8 convert platform accels to m/s^2
%***************************************************************
gravity=sqrt(sum(gcomp.*gcomp));
platform=platform*gfix*G; 
gcomp(1)=mean(platform(1,1:L));
gcomp(2)=mean(platform(2,1:L));
gcomp(3)=mean(platform(3,1:L));

%*********************************************
%  Waves first
%*********************************************
its=5;

[rpy,dr] = anglesclimodeyaw(ahiwaves,bhiwaves,fs,platform,deg_rate,gyro,its,goodcompass,G,L);    % euler angles are right-handed
%[acc, uvwplatwave, xyzplat] = accelsclimode(bhiwaves,ahiwaves,fs,platform,rpy,G,L);
[uvwplatwave,xyzplat] = heavecalc(dr,rpy,platform,fs,bhiwaves,ahiwaves,Rwaves,G,L);

%*********************************************
%  Then sonics
%*********************************************
[euler,dr] = anglesclimodeyaw(ahi,bhi,fs,platform,deg_rate,gyro,its,goodcompass,G,L);    % euler angles are right-handed
[acc, uvwplat, nope] = accelsclimode(bhi,ahi,fs,platform,euler,G,L);
[uvw,uvwr,uvwrot] = sonic(sonics,dr,euler,uvwplat,Rvec,L);

edge = fix(1 * 20 * fs);
tot=length(uvw)-edge*2;
incr1=1+edge;
incr2=incr1+tot-1;
incr=incr1:incr2;

UVW=uvw(1:3,incr);
UVWraw=sonics(1:3,incr);
Um=mean(UVW');
Umean=mean(UVWraw');
Ts=Tv(incr);
rollpitchyaw=euler(1:3,incr); %*180/pi;
Tilts=mean(rollpitchyaw');
% UNITS#10 Cal mean roll and pitch in radians
Pitch=mean(pitch);
Roll=mean(roll);
Pitchstd = std(pitch);
Rollstd=std(roll);
Pitchmax = max(pitch);
Rollmax=max(roll);
Pitchmin = min(pitch);
Rollmin=min(roll);

Gravs=gcomp;
Gstds=gstd;
Drates=dcomp;
Dstds=dstd;
heave = xyzplat(3,incr);
heaveavg = mean(heave);
Heave=detrend(heave);
waveheight = 4*std(Heave);

% UNITS#11 gyroSave is in radians
gyroSave=-euler(3,incr); %*180/pi;

Tm=mean(Ts);
Tmot=mean(tmot);
U = mean(UVW(1,1:tot));
V = mean(UVW(2,1:tot));
W = mean(UVW(3,1:tot));
Uearth=sqrt(U.*U+V.*V+W.*W);

% UNITS#12 wdir is in radians
wdir = atan2(V,-U);  %*180/pi;
if wdir<0
    wdir=wdir+2*pi;
end

[u, alpha, beta] = alignwind(UVW);

% Relative to Earth
wspd=mean(u(1,1:tot));
Uearth=wspd;
u=u';
uh=sqrt(u(1:tot,1).*u(1:tot,1)+u(1:tot,2).*u(1:tot,2));
u=detrend(u);

Ts=detrend(Ts');

fluxes=zeros(1,3);
uwavg=mean(u(1:tot,3).*u(1:tot,1));
vwavg=mean(u(1:tot,3).*u(1:tot,2));
wTavg=mean(u(1:tot,3).*Ts);
%JBE Fixed a bug here by replacing mean with std

Ucorstd = std(u(1:tot,1));
Vcorstd = mean(u(1:tot,2).*u(1:tot,2));
Wcorstd = mean(u(1:tot,3).*u(1:tot,3));

%HACK-JDW
formatString4 = ['FLUX' char(10)  char(0)];
%coder.ceval('printf',formatString4);

fluxes(1)=uwavg;
fluxes(2)=vwavg;
fluxes(3)=wTavg;

fid = fopen('flux_out','w');
if fid ~= -1
%fprintf(fid,'%u.',tsec(1));  % 1
%fprintf(fid,'%03u,',tmsec(1)); % 1 remainder
fprintf(fid,'%9.2f,',Mtime); % 1
fprintf(fid,'%3.1f,',version_number); %2
fprintf(fid,'%06u,',status_val); %3
fprintf(fid,'%6.2f,',Uavg);  %4
fprintf(fid,'%6.2f,',Vavg);  %5
fprintf(fid,'%6.2f,',Wavg);  %6
fprintf(fid,'%6.2f,',Tavg);  %7
fprintf(fid,'%6.2f,',Ustd);  %8
fprintf(fid,'%6.2f,',Vstd);  %9
fprintf(fid,'%6.2f,',Wstd);  %10
fprintf(fid,'%6.2f,',Tstd);  %11

fprintf(fid,'%6.2f,',Umax);  %12
fprintf(fid,'%6.2f,',Vmax);  %13
fprintf(fid,'%6.2f,',Wmax);  %14
fprintf(fid,'%6.2f,',Tvmax);  %15
fprintf(fid,'%6.2f,',Umin);  %16
fprintf(fid,'%6.2f,',Vmin);  %17
fprintf(fid,'%6.2f,',Wmin);  %18
fprintf(fid,'%6.2f,',Tvmin);  %19

fprintf(fid,'%6.2f,',Axavg);  %20
fprintf(fid,'%6.2f,',Ayavg); %21
fprintf(fid,'%6.2f,',Azavg); %22
fprintf(fid,'%6.2f,',Axstd); %23
fprintf(fid,'%6.2f,',Aystd); %24
fprintf(fid,'%6.2f,',Azstd);  %25

fprintf(fid,'%6.2f,',Axmax);  %26
fprintf(fid,'%6.2f,',Aymax); %27
fprintf(fid,'%6.2f,',Azmax); %28
fprintf(fid,'%6.2f,',Axmin); %29
fprintf(fid,'%6.2f,',Aymin); %30
fprintf(fid,'%6.2f,',Azmin);  %31

fprintf(fid,'%6.2f,',Phiavg); %32
fprintf(fid,'%6.2f,',Thetaavg); %33
fprintf(fid,'%6.2f,',Psiavg); %34
fprintf(fid,'%6.2f,',Phistd); %35
fprintf(fid,'%6.2f,',Thetastd); %36
fprintf(fid,'%6.2f,',Psistd); %37

fprintf(fid,'%6.2f,',Phimax); %38
fprintf(fid,'%6.2f,',Thetamax); %39
fprintf(fid,'%6.2f,',Psimax); %40
fprintf(fid,'%6.2f,',Phimin); %41
fprintf(fid,'%6.2f,',Thetamin); %42
fprintf(fid,'%6.2f,',Psimin); %43

fprintf(fid,'%6.2f,',compavg); %44
fprintf(fid,'%6.2f,',Pitch); %45
fprintf(fid,'%6.2f,',Roll); %46
fprintf(fid,'%6.2f,',compstd); %47
fprintf(fid,'%6.2f,',Pitchstd); %48
fprintf(fid,'%6.2f,',Rollstd); %49

fprintf(fid,'%6.2f,',compmax); %50
fprintf(fid,'%6.2f,',Pitchmax); %51
fprintf(fid,'%6.2f,',Rollmax); %52
fprintf(fid,'%6.2f,',compmin); %53
fprintf(fid,'%6.2f,',Pitchmin); %54
fprintf(fid,'%6.2f,',Rollmin); %55

fprintf(fid,'%6.2f,',U);   %56
fprintf(fid,'%6.2f,',V); %57
fprintf(fid,'%6.2f,',W); %58
fprintf(fid,'%6.2f,',Ucorstd); %59
fprintf(fid,'%6.2f,',Vcorstd); %60
fprintf(fid,'%6.2f,',Wcorstd); %61

fprintf(fid,'%6.2f,',wspd); %62
fprintf(fid,'%6.2f,',uwavg); %63
fprintf(fid,'%6.2f,',vwavg); %64
fprintf(fid,'%6.2f,',wTavg); %65
fprintf(fid,'%6.2f\n,',waveheight); %66
%fprintf(fid,'%6.2f,',wavedir); %67
%fprintf(fid,'%6.2f\n',waveperiod); %68
fclose(fid);
end
end

% *************************************************************************************************
%  Function calls
% *************************************************************************************************

function [euler, dr] = anglesclimodeyaw(ahi,bhi,sf,accm,ratem,gyro,its,goodcompass,gravity,L)
%# codegen
% Function from EDDYCORR toolbox
% 
% Sept 2018  Made compass right-handed by changing
%
%  psislow= -gyro - filtfilter(bhi2,ahi2,-gyro) to
%  psislow=  gyro - filtfilter(bhi2,ahi2, gyro)
%
% Sept 2000     Replaced integrations with cumtrapz function
%
% May 16 1997 - modified to remove the first estimate of the euler
% 	angles in the nonlinear euler angle update matrix, F^-1 matrix
% 	is approximated by the identity matrix. still uses trapezoidal
% 	intetgration
%
% INPUT
%
%    ahi,bhi - filter coefficients
%    sf    - sampling frequency
%    accm  - (3xN) array of recalibrated linear accelerations,accx,accy,accz
%    ratem - (3XN) array of recalibrated angular rates, ratex, ratey, ratez
%    gyro  - (1XN) array of gyro signal
%    its   - number of interations
%
% OUTPUT
%
%    euler    - (3XN) array of the euler angles (phi, theta, psi) in radians.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% THE ANGLES ARE  ESTIMATED FROM
%
% angle = slow_angle (from accelerometers) + fast_angle (integrated rate sensors)
%
% CALCULATE GRAVITY
gravxyz=zeros(1,3);
gravxyz(1) = mean(accm(1,1:L));
gravxyz(2) = mean(accm(2,1:L));
gravxyz(3) = mean(accm(3,1:L));
%gravity = sqrt( sum(gravxyz.^2) );

% We now use a constant value of gravity to avoid unrealistic values
% disp(gravity)
% Unwrap compass
gyro = unwrap(gyro);

% REMOVE MEAN FROM RATE SENSORS

ratem = detrend(ratem')';

% LOW FREQUENCY ANGLES FROM ACCELEROMETERS AND GYRO
% SLOW ROLL FROM GRAVITY EFFECTS ON HORIZONTAL ACCELERATIONS. LOW PASS
% FILTER SINCE HIGH FREQUENCY HORIZONTAL ACCELERATIONS MAY BE 'REAL'
%
% PITCH
accm1grav = -accm(1,1:L)./gravity;
thetath = min(accm1grav,1);               %Use small angles
theta = max(thetath,-1);                   %Use small angles
i=find(abs(accm(1,1:L))<gravity);
theta(i) = asin(accm1grav(1,i));
thetaslow = theta - filtfilter(bhi,ahi,theta);

% ROLL
accm2grav = accm(2,1:L)./gravity;
phith = min(accm2grav,1);               %Use small angles
phi = max(phith,-1);                   %Use small angles
accmgravcosthetaslow = accm(2,1:L)./gravity./cos(thetaslow);
i = find(abs(accmgravcosthetaslow)<1);
phi(i) = asin(accmgravcosthetaslow(i));
phislow = phi - filtfilter(bhi,ahi,phi);

% YAW
% HERE, WE ESTIMATE THE SLOW HEADING. THE 'FAST HEADING' IS NOT NEEDED
% FOR THE EULER ANGLE UPDATE MATRIX.
%
% BECAUSE THE COMPASS IS HAVING ISSUES I AM GOING TO EXTEND THE FILTER
% SO THE INTEGRATED YAW GOES OUT TO LOW FREQUENCIES
% I KEPT THIS APPROACH EVEN THOUGH THE NEW COMPASS IS MUCH BETTER
% NO, I'VE REMOVED THIS BECAUSE THE MICROSTRAIN COMPASS IS MUCH BETTER
%fc=1/240;
%ahi2=[1.000000000000000  -3.996744577840595   5.990239031362109  -3.990244324153397   0.996749870634288];
%bhi2=[0.998373612749399  -3.993494450997597   5.990241676496396  -3.993494450997597   0.998373612749399];

%THE COMPASS IS PASSED RIGHT HANDED SO IT CAN BE TREATED LIKE PITCH AND ROLL

bhi2=bhi;
ahi2=ahi;
if goodcompass
    psislow = gyro - filtfilter(bhi2,ahi2,gyro);
else
    psislow = median(gyro)*ones(size(phi));
end

% USE SLOW ANGLES AS FIRST GUESS

euler  = [phislow; thetaslow; psislow];
rates = update(ratem,euler,L);
% INTEGRATE AND FILTER ANGLE RATES, AND ADD TO SLOW ANGLES

for i = 1:its
    phi_int   = 1/sf*cumtrapz(rates(1,1:L));
    phi       = phislow   + filtfilter(bhi,ahi,phi_int); 
    theta_int = 1/sf*cumtrapz(rates(2,1:L));
    theta     = thetaslow + filtfilter(bhi,ahi,theta_int); 
    psi_int = 1/sf*cumtrapz(rates(3,1:L));
    
    if goodcompass
        psi   = psislow   +  filtfilter(bhi2,ahi2,psi_int); 
    else
%        psi   = psislow   + psi_int;
        psi   = psislow + filtfilter(bhi2,ahi2,psi_int);
    end
   
    euler  = [phi; theta; psi];
    rates = update(ratem,euler,L);
    rates(1:2,1:L) = detrend(rates(1:2,1:L)','constant')';
    rates(3,1:L) = detrend(rates(3,1:L)','constant')';
end
dr = ratem;
end

function [Y] = despikesimple(Y,L);
%#codegen
%Remove Outliers

[col, N]=size(Y);
t=1:L;
DS=4;
for tot=1:col
    for iter=1:3         %Do it threetime
        X=Y(tot,1:L);
        M=median(X, "omitnan");
        S=std(X, "omitnan");
        j=find(X<M+DS*S & X>M-DS*S);
        k=length(j);
        if k>0
            Y(tot,1:L)=interp1(t(j),X(j),t,'nearest',M);
        end
    end
end
end

function [uvwplat,xyzplat] = heavecalc(omegam,euler,accm,sf,bhi,ahi,R,gravity,L)
%
% Function from EDDYCORR toolbox
%
% CORRECT SONIC ANEMOMETER COMPONENTS FOR PLATFORM MOTION AND ORIENTATION.
%
% INPUTS:
%
%    omegam     - (3XN) measured angular rate 'vector' in platform frame
%    euler      - (3XN) array of euler angles (phi, theta, psi)
%    accm       - (3XN) array of platform accelerations
%    R          - vector distance from motionPak to wave sensor
% 
% OUTPUTS:
%
%   uvwplat - platform velocity at sensor location
%   xyzplat - platform displacement at sensor location
%

Rvec = [R(1); R(2); R(3)] * ones(1,L);
uvwrot = cross(omegam,Rvec);
uvwrot  = trans(uvwrot,euler,0,L);

% gravxyz = mean(accm');
% %gravity = sqrt( sum(gravxyz.^2) );

acc = trans(accm,euler,0,L);            % first rotate
acc(3,1:L) = acc(3,1:L) - gravity;      % remove gravity

uvwplat = zeros(size(accm));

for i=1:3
    uvwplat(i,1:L) = cumtrapz(acc(i,1:L))/sf + uvwrot(i,1:L);
    uvwplat(i,1:L) = filtfilter(bhi,ahi,uvwplat(i,1:L));
end

% INTEGRATE AGAIN TO GET DISPLACEMENTS
xyzplat = zeros(size(accm));
for i=1:3
    xyzplat(i,1:L) = cumtrapz(uvwplat(i,1:L))/sf;
    xyzplat(i,1:L) = filtfilter(bhi,ahi,xyzplat(i,1:L));
end
end

function [acc,uvwplat,xyzplat] = accelsclimode(bhi,ahi,sf,accm,euler,gravity,L)
%#codegen
% Function from EDDYCORR toolbox
%
% 2008          Allows filter to have different cutoff from angular filter
%
% Sept 2000     Replaced integrations with cumtrapz function
%
% Mar 3 1998	Redesigned the high pass filter (see below).
%
% Revised: June 10, 1997 - high pass filter with higher cutoff 
%			   frequency than previous version
% 
% Integrate linear accelerations to get platform velocity
% and displacement. After each integration, signals are 
% high pass filtered to remove low frequency effects.
%
% INPUT
%
%    bhigh,ahigh - high pass filter coefficients
%    sf 	 - sampling frequency
%    accm 	 - calibrated linear accelerations (output from recal.m)
%    euler	 - (3xN) Euler angles phi,theta,psi 
%
% OUTPUT:
%
%    acc     - (3XN) linear accelerations in FLIP/Earth reference 
%    uvwplat - (3XN) linear velocities at the point of motion measurement
%    xyzplat - (3XN) platform displacements from mean position

% DEFINE ARRAY SIZES UP FRONT

gravxyz=zeros(1,3);
gravxyz(1) = mean(accm(1,1:L));
gravxyz(2) = mean(accm(2,1:L));
gravxyz(3) = mean(accm(3,1:L));
%gravity = sqrt( sum(gravxyz.^2) );

acc      = trans(accm,euler,0,L);       % first rotate
acc(3,1:L) = acc(3,1:L) - gravity;      % remove gravity

% INTEGRATE ACCELERATIONS TO GET PLATFORM VELOCITIES
uvwplat = zeros(size(accm));
for i=1:3
    uvwplat(i,1:L) = cumtrapz(acc(i,1:L))/sf;
    uvwplat(i,1:L) = filtfilter(bhi,ahi,uvwplat(i,1:L));
end

% INTEGRATE AGAIN TO GET DISPLACEMENTS

xyzplat = zeros(size(accm));
for i=1:3
    xyzplat(i,1:L) = cumtrapz(uvwplat(i,1:L))/sf;
    xyzplat(i,1:L) = filtfilter(bhi,ahi,xyzplat(i,1:L));
end
end

function [uvw,uvwr,uvwrot] = sonic(sonics,omegam,euler,uvwplat,R,L)
%#codegen
% Function from EDDYCORR toolbox
%
% CORRECT SONIC ANEMOMETER COMPONENTS FOR PLATFORM MOTION AND ORIENTATION.
%
% INPUTS:
%
%    Sonics     - row of integers corre to sonic numbers which are to be 
%	  	  corrected
%    omegam     - (3XN) measured angular rate 'vector' in platform frame
%    euler      - (3XN) array of euler angles (phi, theta, psi)
%    uvwplat    - (3XN) array of platform velocities (output from accels_.m)
% 
% OUTPUTS:
%
%    uvw        - (MXN) array of corrected sonic anemometer components, in the 
% 		            fixed earth reference frame  (North-West-up)
%

% CALCULATE WINDS IN EARTH BASED FRAME. THE ANGULAR VELOCITY IS CALCULATED AS
% THE CROSS PRODUCT BETWEEN THE ANGULAR RATE VECTOR AND POSITION VECTOR.THE 
% MEASURED AND ANGULAR VELOCITIES ARE IN THE PLATFORM FRAME AND MUST BE
% ROTATED INTO THE EARTH FRAME. THE PLATFORM VELOCITY IS ALREADY IN THE EARTH
% FRAME (FROM ACCELS.M), SO HERE THEY CAN JUST BE ADDED.
%
%  UVW =  MEASURED VELOCITY + ANGULAR RATE INDUCED VELOCITIES + 
%	  INTEGRATED ACCELEROMETERS  

Rvec = [R(1); R(2); R(3)] * ones(1,L);
uvwrot = cross(omegam,Rvec);
   
uvw  = trans(sonics + uvwrot,euler,0,L) + uvwplat;
uvwr = trans(sonics + uvwrot,euler,0,L);
end

function OUT = trans(IN,ANGLES,IFLAG,L)
%#codegen
if nargin==2
    IFLAG=0;
    L=12000;
end
sinp = zeros(1,L);
cosp = zeros(1,L);
sint = zeros(1,L);
cost = zeros(1,L);
sinps = zeros(1,L);
cosps = zeros(1,L);

sinp  = sin(ANGLES(1,1:L));
cosp  = cos(ANGLES(1,1:L));
sint  = sin(ANGLES(2,1:L));
cost  = cos(ANGLES(2,1:L));
sinps = sin(ANGLES(3,1:L));
cosps = cos(ANGLES(3,1:L));

up = IN(1,1:L);
vp = IN(2,1:L);
wp = IN(3,1:L);

if IFLAG          	% =1, from xyz to x'y'z'
    
    u = up.*cost.*cosps                           + vp.*cost.*sinps                           - wp.*sint;
    v = up.*(sinp.*sint.*cosps-cosp.*sinps) + vp.*(sinp.*sint.*sinps+cosp.*cosps) + wp.*(cost.*sinp);
    w = up.*(cosp.*sint.*cosps+sinp.*sinps) + vp.*(cosp.*sint.*sinps-sinp.*cosps) + wp.*(cost.*cosp);
    
else		% =0, from x'y'z' to xyz
    
    u = up.*cost.*cosps + vp.*(sinp.*sint.*cosps-cosp.*sinps) + wp.*(cosp.*sint.*cosps+sinp.*sinps);
    v = up.*cost.*sinps + vp.*(sinp.*sint.*sinps+cosp.*cosps) + wp.*(cosp.*sint.*sinps-sinp.*cosps);
    w = up.*(-sint)       + vp.*(cost.*sinp)                          + wp.*(cost.*cosp);
    
end;

OUT = [u;v;w];
end

function OUT = update(IN,ANGLES,L)
%#codegen
% Function from EDDYCORR toolbox
%
%  This function computes the angular update matrix
%  as described in Edson et al. (1998) and Thwaites
%  (1995) page 50.

p  = ANGLES(1,1:L);
t  = ANGLES(2,1:L);
ps = ANGLES(3,1:L);

up = IN(1,1:L);
vp = IN(2,1:L);
wp = IN(3,1:L);

u = up  + vp.*sin(p).*tan(t) + wp.*cos(p).*tan(t);
v =  0  + vp.*cos(p)         - wp.*sin(p);
w =  0  + vp.*sin(p)./cos(t) + wp.*cos(p)./cos(t);

OUT = [u;v;w];
end

function [u, alpha, beta] = alignwind(U)
%#codegen
% Function from EDDYCORR toolbox
%
Ub = mean(U(1,:));
Vb = mean(U(2,:));
Wb = mean(U(3,:));
Sb = sqrt(Ub^2+Vb^2);
beta  = atan2(Wb,Sb);
alpha = atan2(Vb,Ub);
Ur =  U(1,:)*cos(alpha)*cos(beta) + U(2,:)*sin(alpha)*cos(beta) + U(3,:)*sin(beta);
Vr = -U(1,:)*sin(alpha)           + U(2,:)*cos(alpha);
Wr = -U(1,:)*cos(alpha)*sin(beta) - U(2,:)*sin(alpha)*sin(beta) + U(3,:)*cos(beta);

% predefine u for coder
nU=length(U);
u=zeros(3,nU);

u(1,:) = Ur;
u(2,:) = Vr;
u(3,:) = Wr;

beta  = beta*180/pi;
alpha = alpha*180/pi;
end

function g=grv(lat)
%#codegen
if lat<0 | lat>90
    lat=45.0;
end
gamma=9.7803267715;
c1=0.0052790414;
c2=0.0000232718;
c3=0.0000001262;
c4=0.0000000007;

phi=lat*pi/180;
x=sin(phi);
g=gamma.*(1+c1*x.^2+c2*x.^4+c3*x.^6+c4*x.^8);
end

function y=medfilt(x,n)
%#codegen
% This version of medfilt can only hand one time series at a time
blksz = []; 
DIM = [];

% Check if the input arguments are valid
if isempty(n)
    n = 3;
end

[x, nshifts] = shiftdim(x);

% Verify that the block size is valid.
siz = size(x);
blksz = siz(1); % siz(1) is the number of rows of x (default)

% Initialize y with the correct dimension
y = zeros(siz);

% Call medfilt1D (vector)
for i = 1:prod(siz(2:end)),
    y(:,i) = medfilt1D(x(:,i),n,blksz);
end

% Convert y to the original shape of x
    y = shiftdim(y, -nshifts);
end

function y = medfilt1D(x,n,blksz)
%#codegen
%MEDFILT1D  One dimensional median filter.
%
% Inputs:
%   x     - vector
%   n     - order of the filter
%   blksz - block size

nx = length(x);
if rem(n,2)~=1    % n even
    m = n/2;
else
    m = (n-1)/2;
end
X = [zeros(m,1); x; zeros(m,1)];
y = zeros(nx,1);

% Work in chunks to save memory
indr = (0:n-1)';
indc = 1:nx;
blksz=1;
for i=1:blksz:nx
    ind = indc(ones(1,n),i:min(i+blksz-1,nx)) + ...
    indr(:,ones(1,min(i+blksz-1,nx)-i+1));
    xx = reshape(X(ind),n,min(i+blksz-1,nx)-i+1);
    y(i:min(i+blksz-1,nx)) = median(xx,1);
end
end

function yout = filtfilter(b,a,xin)
%#codegen
% x must be a collumn vector for this to work
%FILTFILT Zero-phase forward and reverse digital IIR filtering.
%   Y = FILTFILT(B, A, X) filters the data in vector X with the filter
%   described by vectors A and B to create the filtered data Y.  The
%   filter is described by the difference equation:
%
%     a(1)*y(n) = b(1)*x(n) + b(2)*x(n-1) + ... + b(nb+1)*x(n-nb)
%                           - a(2)*y(n-1) - ... - a(na+1)*y(n-na)
%
%   The length of the input X must be more than three times
%   the filter order, defined as max(length(B)-1,length(A)-1). 
%
%   References:
%     [1] Sanjit K. Mitra, Digital Signal Processing, 2nd ed.,
%         McGraw-Hill, 2001
%     [2] Fredrik Gustafsson, Determining the initial states in forward-
%         backward filtering, IEEE Transactions on Signal Processing,
%         pp. 988-992, April 1996, Volume 44, Issue 4

%   Copyright 1988-2010 The MathWorks, Inc.
%   $Revision: 1.7.4.8 $  $Date: 2011/05/13 18:07:25 $

% If input data is a row vector, convert it to a column
isRowVec = size(xin,1)==1;
if isRowVec
    x = xin(:);
else
    x=xin; 
end
[Npts,Nchans] = size(x);

%----------------------------------------------------------------------
% Parse coefficients vectors and determine initial conditions
% b and a are vectors that define the transfer function of the filter
%----------------------------------------------------------------------
[L,nfilt] = size(b);
% Check coefficients
b = b(:);
a = a(:);
nfact = 3*(nfilt-1);  % length of edge transientsl

% The non-sparse solution to zi may be computed using:
zi=(eye(nfilt-1) - [-a(2:nfilt), [eye(nfilt-2); zeros(1,nfilt-2)]]);
zi=zi \ (b(2:nfilt) - b(1)*a(2:nfilt));

% Filter the data
y = [2*x(1)-x(nfact+1:-1:2); x; 2*x(end)-x(end-1:-1:end-nfact)];

% filter, reverse data, filter again, and reverse data again
y = filter(b,a,y,zi*y(1));
y = y(end:-1:1);
y = filter(b,a,y,zi*y(1));

% retain reversed central section of y
y = y(end-nfact:-1:nfact+1);
if isRowVec
    
    yout = y.';   % convert back to row if necessary
else 
    yout = y;
end
end
