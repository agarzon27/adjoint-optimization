import socket
import scipy.io
import numpy as np
import scipy.interpolate
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import time

import utils
import smooth_biphasic_pulse as pulse
import fk_lap_ion_t as fk


program="__file__ not working in emacs"
host = socket.gethostname()

if host == 'LAPTOP-8JGJ5SSI':
    access_to_display=True
else:
    access_to_display=False

# Model geometry
ncols = 200                                # Number of columns; e.g. 500unit x 0.025mm/unit = 12.5cm
nrows = 200                                # Number of rows; e.g. 500unit x 0.025mm/unit = 12.5cm
h = 0.025                                  # Grid spacing [mm/unit]; e.g. 12.5 x 12.5 cm lattice
h2 = h**2
Lvert=nrows*h
Lhorz=ncols*h

# Model parameters
#Cm = 1                                     # Capacitance (uF_per_cm2)

#global u_c u_v tau_d tau_v1_minus tau_v2_minus tau_v_plus tau_0 tau_r 
#global tau_si u_csi k_fk tau_w_minus tau_w_plus ksm
#global lap_u lap_v lap_w gh

param = {}
param['u_c'] = 0.13                                 # (dimensionless)
param['u_v'] = 0.04                                 # (dimensionless)
param['tau_d'] = 0.395                              # Fast_inward_current (ms)
param['tau_v1_minus'] = 9                           # Fast_inward_current_v_gate (ms)
param['tau_v2_minus'] = 8                           # Fast_inward_current_v_gate (ms)
param['tau_v_plus'] = 3.33                          # Fast_inward_current_v_gate (ms)
param['tau_0'] = 9                                  # Slow_outward_current (ms)
param['tau_r'] = 33.33                              # Slow_outward_current (ms)
param['tau_si'] = 29                                # Slow_inward_current (ms)
param['u_csi'] = 0.50                               # Slow_inward_current (dimensionless)
param['k_fk'] = 15                                     # Slow_inward_current (dimensionless)
param['tau_w_minus'] = 60                           # Slow_inward_current_w_gate (ms)
param['tau_w_plus'] = 250                           # Slow_inward_current_w_gate (ms)

Du = 0.001                                 # u diffusivity (cm^2/ms)
Dv = 1e-5
Dw = 1e-5

param['Du'] = Du
param['Dv'] = Dv
param['Dw'] = Dw

# Parameter in smoothed step function
param['ksm']=100

# Integration Parameters
dt = 0.1                                   # Duration of each time step = 0.1 ms 
final_time=50 #ms
#T = 0:dt:time_units; T(end) = [];           # Time vector (ms)
si = 10/dt                                 # Final sampling interval; 10/0.1 = 100time steps = 100 x 0.1ms = 10ms/frame
                                            # Final sampling rate = 1,000msp/10ms ~ 100Hz

# Laplacian in mat5orix
# active=0;
# dead=1;
# C(1:nrows,1:ncols)=active;
# lap=fun_create_laplacian_matrix(C,active,dead);
#load lap_file_200_N_holes_400.mat lap active_index ghost_lr
#load lap_no_hole_200.mat lap active_index
mat = scipy.io.loadmat('../data/lap_Cfile_2nd_order_grad_200_N_holes_400.mat')
active_index = mat['active_index']
ghost_lr = mat['ghost_lr']
lap = sparse.csr_matrix(mat['lap']) # compressed sparse row matrix, more efficient matrix-vector products
#lap = mat['lap'] # compressed sparse row matrix, more efficient matrix-vector products

param['lap_u']=Du/h2*lap
param['lap_v']=Dv/h2*lap
param['lap_w']=Dw/h2*lap

# squeeze below, produces an array that has a shape which is the
# shape of the original array (a tuple) with all ones
# ("single-dimensional entries) removed
# Operationally: An array is a set of nested lists. If at any level, a list has a single element, the list gets replaced by its contents, i.e., the square brackets are removed.

active_index = np.squeeze(active_index)

bx=(active_index > 0)
index=np.arange(nrows*ncols)
#general_index=index[bx] # previously called reverse_index

mat = scipy.io.loadmat('../data/Vuw_lap_matrix_1.mat')
u=mat['V'].flatten(order='F')
v=mat['u'].flatten(order='F')
w=mat['w'].flatten(order='F')

u=u[bx]
v=v[bx]
w=w[bx]

ghost_lr = np.squeeze(ghost_lr)
param['gh']=Du*2/h*ghost_lr
n = len(u)
#E=0
nt=utils.float_integer(final_time/dt)
tt=dt*np.arange(nt)

Et = pulse.smooth_biphasic_pulse(tt)

y=np.concatenate((u,v,w))
nvar=3

yt=np.zeros((nvar*n,nt+1))
yt[:,0]=y

# Generating interpolant for electric field
pp=scipy.interpolate.CubicSpline(tt,Et,bc_type='natural')

# pp=spline(tt,Et);
# global E_fun
param['E_fun']=pp

fk.set_parameters(param)

start_time= time.time()

for kt in range(nt):
    t0 = tt[kt]
    t1 = tt[kt]

    k1=fk.fk_lap_ion_t( y, t0 );
    k2=fk.fk_lap_ion_t( y + dt/2*k1, t0+dt/2 );
    k3=fk.fk_lap_ion_t( y + dt/2*k2, t0+dt/2 );
    k4=fk.fk_lap_ion_t( y + dt*k3, t1 );
    
    y = y + dt/6*(k1 + 2*k2 + 2*k3 + k4);

    
    #y = y + dt*fk.fk_lap_ion_t(y,t0) # Euler step
    #print(kt)
    yt[:,kt+1]=y
    
    if (kt+1) % si == 0:
        uvec = np.full(nrows*ncols,np.nan)
        uvec[bx]=y[0:n]
        umat = uvec.reshape((nrows,ncols),order='F')

        plt.imshow(umat)
        plt.title(f't={(kt+1)*dt}')
        plt.pause(1)
        print(kt)

print("---------%s seconds--------" % (time.time() - start_time))


    
