int __device__ getIndex(const int i, const int j, const int width);
void __device__ F_ion(double const u, double const v, double const w,
		      double *udotPtr, double *vdotPtr, double *wdotPtr);
__global__ void fk_lap_ion_E(const double *u, const double *v, const double *w, const double *gh, double E,
			     double *fu, double *fv, double *fw,
			     const int nx, const int ny);
__device__ void lap_func(const double *u, const double *v, const double *w, const int nx, const int ny,
		       int *kPtr, double *uPtr, double *vPtr, double *wPtr, 
		      double *lap_uPtr, double *lap_vPtr, double *lap_wPtr);
__global__ void fk_lap_ion_E_1(const double *u, const double *v, const double *w, const double *gh, double E,
			     double *fu, double *fv, double *fw,
			     const int nx, const int ny) ;
__global__ void fk_lap_ion_rk4_1(const double *u, const double *v, const double *w, const double *gh,
				 const double E, const double dt,
				 double *U, double *V, double *W, double *du, double *dv, double *dw,
				 const int nx, const int ny) ;
__global__ void fk_lap_ion_rk4_23(double *u, double *v, double *w, const double *gh,
				  const double E, const double dt, const int istep,
				  const double *u_, const double *v_, const double *w_ ,
				  double *Utemp, double *Vtemp, double *Wtemp,
				  double *du, double *dv, double *dw,
				  const int nx, const int ny) ;
__global__ void fk_lap_ion_rk4_4(const double *u, const double *v, const double *w, const double *gh,
				 const double E, const double dt,
				 double *u_, double *v_, double *w_ , const double *du, const double *dv, const double *dw,
				 const int nx, const int ny) ;
