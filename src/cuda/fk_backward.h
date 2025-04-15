__global__ void lapT_func(const double *z, double *lapT, const int nx_g, const int ny_g);
__device__ void lapT_func_dev(const double *z, const int nx_g, const int ny_g,
			      double *lapT_th, double *z_th, int *k_g);
__device__ void jacT(const double *y, double *out);
__device__ void fk_lap_ion_E_dev(const double *y, const double *gh, const double E,
			    const int nx, const int ny, double *y_th, double *fy,
			    int *k_g);
__device__ double hermite3(const double x0, const double x1, const double m0, const double m1);
__device__ double hermite4_1(const double x0, const double x1, const double m0, const double m1);
__global__ void interp_midpoint_1(const double *y0, const double *y1, const double *gh,
			     const double E0, const double E1,
			     const double dt, const int nx, const int ny,
			     double *y_3rd, double *y_4th_1);
__global__ void interp_midpoint_2(const double *y_3rd, const double *y_4th_1,
			     const double *gh, const double E_3rd, const double dt,
			     const int nx, const int ny, double *yh);
__device__ void adj_vfield(const double *z, const double *y, const int nx, const int ny,
			   double *fz, double *z_out, int *kPtr);
__global__ void fk_adj_rk4_1(const double *z, const double *y, const double dt, const int nx, const int ny,
			double *Z, double *dz);
__global__ void fk_adj_rk4_2(const double *z, const double *y, const double *z_,
			     const double dt, const int nx, const int ny,
			     double *Z, double *dz);
__global__ void fk_adj_rk4_3(const double *z, const double *y, const double *z_,
			     const double dt, const int nx, const int ny,
			     double *Z, double *dz);
__global__ void fk_adj_rk4_4(const double *z, const double *y, const double dt, const int nx, const int ny,
			     const double *dz, double *z_);
__global__ void zgh_dot( const double *a, const double *b, const int ny_g, double *c);
__global__ void zgh_dot_partial( const double *a, const double *b, const int ny_g, double *c);
