%% Generate spiral wave breakup using Fenton-Karma model of cardiac action potentials
% Fenton FH and Karma A AD. Vortex dynamics in three-dimensional continuous myocardium with 
% fiber rotation: Filament instability and fibrillationa. Chaos 8: 20-47, 1998
clear variables
program=mfilename('fullpath');

[status,output]=system('hostname');
if status==0
    host=strtrim(output);
else
    error('Hostname not found')
end

if strcmp(host,'LAPTOP-8JGJ5SSI')
    access_to_display=true;
else
    access_to_display=false;
end


% Model geometry
ncols = 200;                                % Number of columns; e.g. 500unit x 0.025mm/unit = 12.5cm
nrows = 200;                                % Number of rows; e.g. 500unit x 0.025mm/unit = 12.5cm
h = 0.025;                                  % Grid spacing [mm/unit]; e.g. 12.5 x 12.5 cm lattice
h2 = h^2;
Lvert=nrows*h;
Lhorz=ncols*h;

% Model parameters
%Cm = 1;                                     % Capacitance (uF_per_cm2)

global u_c u_v tau_d tau_v1_minus tau_v2_minus tau_v_plus tau_0 tau_r 
global tau_si u_csi k_fk tau_w_minus tau_w_plus ksm
global lap_u lap_v lap_w gh

u_c = 0.13;                                 % (dimensionless)
u_v = 0.04;                                 % (dimensionless)
tau_d = 0.395;                              % Fast_inward_current (ms)
tau_v1_minus = 9;                           % Fast_inward_current_v_gate (ms)
tau_v2_minus = 8;                           % Fast_inward_current_v_gate (ms)
tau_v_plus = 3.33;                          % Fast_inward_current_v_gate (ms)
tau_0 = 9;                                  % Slow_outward_current (ms)
tau_r = 33.33;                              % Slow_outward_current (ms)
tau_si = 29;                                % Slow_inward_current (ms)
u_csi = 0.50;                               % Slow_inward_current (dimensionless)
k_fk = 15;                                     % Slow_inward_current (dimensionless)
tau_w_minus = 60;                           % Slow_inward_current_w_gate (ms)
tau_w_plus = 250;                           % Slow_inward_current_w_gate (ms)
Du = 0.001;                                 % u diffusivity (cm^2/ms)
Dv = 1e-5;
Dw = 1e-5;

% Parameter in smoothed step function
ksm=100;

% Integration Parameters
dt = 0.1;                                   % Duration of each time step = 0.1 ms 
final_time=50; %ms
%T = 0:dt:time_units; T(end) = [];           % Time vector (ms)
si = 10/dt;                                 % Final sampling interval; 10/0.1 = 100time steps = 100 x 0.1ms = 10ms/frame
                                            % Final sampling rate = 1,000ms/10ms ~ 100Hz

% Laplacian in matrix
% active=0;
% dead=1;
% C(1:nrows,1:ncols)=active;
% lap=fun_create_laplacian_matrix(C,active,dead);
%load lap_file_200_N_holes_400.mat lap active_index ghost_lr
%load lap_no_hole_200.mat lap active_index
load('../data/lap_Cfile_2nd_order_grad_200_N_holes_400.mat','lap','active_index','ghost_lr')

lap_u=Du/h2*lap;
lap_v=Dv/h2*lap;
lap_w=Dw/h2*lap;

bx=(active_index > 0);
index=(1:nrows*ncols)';
general_index=index(bx); % previously called reverse_index

load('../data/Vuw_lap_matrix_1.mat','V','u','w')
% swtiching variable names
v=u;
u=V;

u=u(general_index);
v=v(general_index);
w=w(general_index);

gh=Du*2/h*ghost_lr;
n = length(u);
%E=0;
nt=float_integer(final_time/dt);
tt=(0:nt)*dt;
Et = smooth_biphasic_pulse(tt);
% vidObj=VideoWriter('movie.mp4','MPEG-4');
% open(vidObj)


y=[u;v;w];
nvar=3;
% below, will store solution at every time step
yt=zeros(nvar*n,nt+1);
yt(:,1)=y;

if access_to_display
    nf=3;
    figure(nf)
    clf
end

% Generating interpolant for electric field
pp=spline(tt,Et);
global E_fun
E_fun=@(t) ppval(pp,t);

% Forward evolution
tic
for kt = 1:nt
    t0 = tt(kt);
    t1 = tt(kt+1);
        
    k1=fk_lap_ion_t( y, t0 );
    k2=fk_lap_ion_t( y + dt/2*k1, t0+dt/2 );
    k3=fk_lap_ion_t( y + dt/2*k2, t0+dt/2 );
    k4=fk_lap_ion_t( y + dt*k3, t1 );
    
    y = y + dt/6*(k1 + 2*k2 + 2*k3 + k4);
    
    yt(:,kt+1)=y;
    
    if access_to_display
        if rem(kt,si) == 0
            umat=nan(nrows,ncols);
            u=y(1:n);%y(2*n+1:3*n);
            umat(general_index)=u;
            
            imagesc([0,Lvert],[0,Lhorz],umat, [0.1,1.2])
            xlabel('cm')
            ylabel('cm')
            title(sprintf('t=%.2f ms',kt*dt))
            colorbar
            drawnow
        end
    end
    
end
toc

