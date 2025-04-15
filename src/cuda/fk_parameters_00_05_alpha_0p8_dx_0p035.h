// Parameters for the Fenton-Karma model
#define u_c 0.13                                 // (dimensionless)
#define u_v 0.04                                 // (dimensionless)
#define tau_d 0.388                              // Fast_inward_current (ms)
#define tau_v1_minus 7.6                           // Fast_inward_current_v_gate (ms)
#define tau_v2_minus 8.8                           // Fast_inward_current_v_gate (ms)
#define tau_v_plus 3.324                          // Fast_inward_current_v_gate (ms)
#define tau_0 8.2                                  // Slow_outward_current (ms)
#define tau_r 33.264                              // Slow_outward_current (ms)
#define tau_si 29.0                                // Slow_inward_current (ms)
#define u_csi 0.54                               // Slow_inward_current (dimensionless)
#define k_fk 15.0                                     // Slow_inward_current (dimensionless)
#define tau_w_minus 68.0                           // Slow_inward_current_w_gate (ms)
#define tau_w_plus 399.99999999999994                           // Slow_inward_current_w_gate (ms)

#define Du 1.0e-3
#define Dv 1.0e-5
#define Dw 1.0e-5

// Parameter in smoothed step function
#define ksm 100.0

constexpr double h = 0.035; // spatial mesh width
constexpr int nvar = 3;


// File written by interpolate_fk_parameters.py