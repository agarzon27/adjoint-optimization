import utils
import numpy as np

def set_parameters(param):
    global u_c, u_v, tau_d, tau_v1_minus, tau_v2_minus, tau_v_plus
    global tau_0, tau_r, tau_si, u_csi, k_fk, tau_w_minus, tau_w_plus
    global ksm, lap_u, lap_v, lap_w, gh, E_fun
    
    u_c = param['u_c']                                 # (dimensionless)
    u_v = param['u_v']                                 # (dimensionless)
    tau_d = param['tau_d']                              # Fast_inward_current (ms)
    tau_v1_minus = param['tau_v1_minus']                           # Fast_inward_current_v_gate (ms)
    tau_v2_minus = param['tau_v2_minus']                           # Fast_inward_current_v_gate (ms)
    tau_v_plus = param['tau_v_plus']                          # Fast_inward_current_v_gate (ms)
    tau_0 = param['tau_0']                                  # Slow_outward_current (ms)
    tau_r = param['tau_r']                              # Slow_outward_current (ms)
    tau_si = param['tau_si']                                # Slow_inward_current (ms)
    u_csi = param['u_csi']                               # Slow_inward_current (dimensionless)
    k_fk = param['k_fk']                                     # Slow_inward_current (dimensionless)
    tau_w_minus = param['tau_w_minus']                           # Slow_inward_current_w_gate (ms)
    tau_w_plus = param['tau_w_plus']                           # Slow_inward_current_w_gate (ms)

    ksm = param['ksm']
    lap_u = param['lap_u']
    lap_v = param['lap_v']
    lap_w = param['lap_w']
    gh = param['gh']
    E_fun = param['E_fun']
    

def fk_lap_ion_t(y,t):

    n3=len(y)
    n=utils.float_integer(n3/3)

    u=y[0:n]
    v=y[n:2*n]
    w=y[2*n:3*n]

    """
    global u_c u_v tau_d tau_v1_minus tau_v2_minus tau_v_plus tau_0 tau_r 
    global tau_si u_csi k_fk tau_w_minus tau_w_plus ksm
    global lap_u lap_v lap_w gh
    """

    p = 0.5*( 1 + np.tanh(ksm*(u-u_c)) )                 # (dimensionless)
    q = 0.5*( 1 + np.tanh(ksm*(u-u_v)) )                 # q (dimensionless)
    J_fi = - v*p*(1 - u)*(u - u_c)/tau_d # fast_inward_current (per_ms)
    J_so = u*(1 - p)/tau_0 + p/tau_r      # slow_outward_current (per_ms)
    tau_v_minus =  q * tau_v1_minus + (1 - q) * tau_v2_minus # fast_inward_current_v_gate (ms)
    J_si  = - w*(1 + np.tanh(k_fk*(u - u_csi)))/(2*tau_si) # slow_inward_current (per_ms)

    udot =  - (J_fi + J_so + J_si)
 
    # dv/dt v fast_inward_current_v_gate (dimensionless)
    vdot = ((1 - p)*(1 - v)/tau_v_minus) - (p*v)/tau_v_plus

    # dw/dt w slow_inward_current_w_gate (dimensionless)
    wdot = ((1 - p)*(1 - w)/tau_w_minus) - (p*w)/tau_w_plus

    E=E_fun(t) # function that computes electric field

    udot = udot + lap_u.dot(u) + E*gh # CORRECT MATRIX-VECTOR PRODUCT
    vdot = vdot + lap_v.dot(v)
    wdot = wdot + lap_w.dot(w)

    return np.concatenate((udot, vdot, wdot))


