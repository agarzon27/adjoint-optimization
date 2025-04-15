function out = fk_lap_ion_t(y,t)

n3=length(y);
n=float_integer(n3/3);

u=y(1:n);
v=y(n+1:2*n);
w=y(2*n+1:3*n);

global u_c u_v tau_d tau_v1_minus tau_v2_minus tau_v_plus tau_0 tau_r 
global tau_si u_csi k_fk tau_w_minus tau_w_plus ksm
global lap_u lap_v lap_w gh

p = 0.5*( 1 + tanh(ksm*(u-u_c)) );                 % (dimensionless)
q = 0.5*( 1 + tanh(ksm*(u-u_v)) );                 % q (dimensionless)
J_fi = - v.*p.*(1 - u).*(u - u_c)/tau_d; % fast_inward_current (per_ms)
J_so = u.*(1 - p)/tau_0 + p/tau_r;      % slow_outward_current (per_ms)
tau_v_minus =  q * tau_v1_minus + (1 - q) * tau_v2_minus; % fast_inward_current_v_gate (ms)
J_si  = - w.*(1 + tanh(k_fk*(u - u_csi)))/(2*tau_si); % slow_inward_current (per_ms)

udot =  - (J_fi + J_so + J_si);

% dv/dt v fast_inward_current_v_gate (dimensionless)
vdot = ((1 - p).*(1 - v)./tau_v_minus) - (p.*v)/tau_v_plus;

% dw/dt w slow_inward_current_w_gate (dimensionless)
wdot = ((1 - p).*(1 - w)./tau_w_minus) - (p.*w)/tau_w_plus;

global E_fun % function that computes electric field
E=E_fun(t);

udot = udot + lap_u*u + E*gh;
vdot = vdot + lap_v*v;
wdot = wdot + lap_w*w;

out = [udot; vdot; wdot];
