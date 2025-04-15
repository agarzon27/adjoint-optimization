#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "mat.h"
#include "matrix.h"

#include "fk_parameters_00_05_alpha_0p8_dx_0p035.h" // __PARAMETERS_FILE__ text will be (was) substituted by external sed command
#include "block_size_16_16.h" // __BLOCK_SIZE_FILE__ text will be (was) substituted by external sed command
#include "fk_forward.h"
//#include "fk_backward.h"

# include "spline.h"

#include "fk_parameters_string_00_05_alpha_0p8_dx_0p035.cpp" // __PARAMETERS_STRING_FILE__ text will be (was) substituted by external sed command

//------------------------- forward dynamics --------------------------------

int __device__ getIndex(const int i, const int j, const int width)
{
    return i*width + j;
}

void __device__ F_ion(double const u, double const v, double const w,
		      double *udotPtr, double *vdotPtr, double *wdotPtr){
  double p = 0.5*( 1 + tanh(ksm*(u-u_c)) );                 // (dimensionless)
  double q = 0.5*( 1 + tanh(ksm*(u-u_v)) );                 // q (dimensionless)
  double J_fi = - v*p*(1 - u)*(u - u_c)/tau_d; // fast_inward_current (per_ms)
  double J_so = u*(1 - p)/tau_0 + p/tau_r;      // slow_outward_current (per_ms)
  double tau_v_minus =  q * tau_v1_minus + (1 - q) * tau_v2_minus; // fast_inward_current_v_gate (ms)
  double J_si  = - w*(1 + tanh(k_fk*(u - u_csi)))/(2*tau_si); // slow_inward_current (per_ms)

  *udotPtr =  - (J_fi + J_so + J_si);

  // dv/dt v fast_inward_current_v_gate (dimensionless)
  *vdotPtr = ((1 - p)*(1 - v)/tau_v_minus) - (p*v)/tau_v_plus;

  // dw/dt w slow_inward_current_w_gate (dimensionless)
  *wdotPtr = ((1 - p)*(1 - w)/tau_w_minus) - (p*w)/tau_w_plus;
}

__global__ void fk_lap_ion_E(const double *u, const double *v, const double *w, const double *gh, double E,
			     double *fu, double *fv, double *fw,
			     const int nx, const int ny) {

  __shared__ double s_u[(BLOCK_SIZE_X + 2)*(BLOCK_SIZE_Y + 2)];
  __shared__ double s_v[(BLOCK_SIZE_X + 2)*(BLOCK_SIZE_Y + 2)];
  __shared__ double s_w[(BLOCK_SIZE_X + 2)*(BLOCK_SIZE_Y + 2)];
  
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;

  int s_i = threadIdx.x + 1;
  int s_j = threadIdx.y + 1;
  int s_ny = BLOCK_SIZE_Y + 2;

  // Load data into shared memory
  // Central square
  int k = getIndex(i, j, ny);
  double uij = u[k];
  double vij = v[k];
  double wij = w[k];

  int s_k = getIndex(s_i, s_j, s_ny);
  s_u[s_k] = uij;
  s_v[s_k] = vij;
  s_w[s_k] = wij;

  // Top border
  if (s_i == 1 && i != 0){
    int s_kb = getIndex(0, s_j, s_ny);
    int kb = getIndex(blockIdx.x*blockDim.x - 1, j, ny);
    
    s_u[s_kb] = u[kb];
    s_v[s_kb] = v[kb];
    s_w[s_kb] = w[kb];
  }
  // Bottom border
  if (s_i == BLOCK_SIZE_X && i != nx - 1){
    int s_kb =getIndex(BLOCK_SIZE_X + 1, s_j, s_ny);
    int kb = getIndex((blockIdx.x + 1)*blockDim.x, j, ny);

    s_u[s_kb] = u[kb];
    s_v[s_kb] = v[kb];
    s_w[s_kb] = w[kb];
  }
  // Left border
  if (s_j == 1 && j != 0){
    int s_kb = getIndex(s_i, 0, s_ny);
    int kb = getIndex(i, blockIdx.y*blockDim.y - 1, ny);

    s_u[s_kb] = u[kb];
    s_v[s_kb] = v[kb];
    s_w[s_kb] = w[kb];
  }
  // Right border
  if (s_j == BLOCK_SIZE_Y && j != ny - 1){
    int s_kb = getIndex(s_i, BLOCK_SIZE_Y + 1, s_ny);
    int kb =getIndex(i, (blockIdx.y + 1)*blockDim.y, ny); 

    s_u[s_kb] = u[kb];
    s_v[s_kb] = v[kb];
    s_w[s_kb] = w[kb];    
  }

  // Make sure all the data is loaded before computing
  __syncthreads();

  // Calculating Laplacian
  if(i < nx){
    if(j < ny){
            
      if(!isnan(uij)){
	double uim1j, uip1j, uijm1, uijp1;
	double vim1j, vip1j, vijm1, vijp1;
	double wim1j, wip1j, wijm1, wijp1;

	// top and bottom neighbor
	if(i==0) { // top row
	  int kn = getIndex(s_i+1, s_j, s_ny);
	  
	  uip1j = s_u[kn];
	  uim1j = uip1j;

	  vip1j = s_v[kn];
	  vim1j = vip1j;

	  wip1j = s_w[kn];
	  wim1j = wip1j;

	}
	else if(i==nx-1) { // bottom row
	  int kn = getIndex(s_i-1, s_j, s_ny);
	  
	  uim1j = s_u[kn];
	  uip1j = uim1j;
	  
	  vim1j = s_v[kn];
	  vip1j = vim1j;
	  
	  wim1j = s_w[kn];
	  wip1j = wim1j;
	  
	}
	else{
	  int kt = getIndex(s_i-1, s_j, s_ny);
	  int kb = getIndex(s_i+1, s_j, s_ny);

	  uim1j = s_u[kt];
	  uip1j = s_u[kb];

	  vim1j = s_v[kt];
	  vip1j = s_v[kb];

	  wim1j = s_w[kt];
	  wip1j = s_w[kb];

	  if(isnan(uim1j)){
	    uim1j = uip1j;
	    vim1j = vip1j;
	    wim1j = wip1j;
	  }
	  else if(isnan(uip1j)){
	    uip1j = uim1j;
	    vip1j = vim1j;
	    wip1j = wim1j;
	  }
	}

	// left and right neighbor
	if(j==0){ // left column
	  int kr = getIndex(s_i, s_j+1, s_ny);

	  uijp1 = s_u[kr];
	  uijm1 = uijp1;

	  vijp1 = s_v[kr];
	  vijm1 = vijp1;

	  wijp1 = s_w[kr];
	  wijm1 = wijp1;
	}
	else if(j==ny-1){ // right column
	  int kl = getIndex(s_i, s_j-1, s_ny);
	  
	  uijm1 = s_u[kl];
	  uijp1 = uijm1;

	  vijm1 = s_v[kl];
	  vijp1 = vijm1;

	  wijm1 = s_w[kl];
	  wijp1 = wijm1;

	}
	else{
	  int kl = getIndex(s_i, s_j-1, s_ny);
	  int kr = getIndex(s_i, s_j+1, s_ny);

	  uijm1 = s_u[kl];
	  uijp1 = s_u[kr];

	  vijm1 = s_v[kl];
	  vijp1 = s_v[kr];

	  wijm1 = s_w[kl];
	  wijp1 = s_w[kr];

	  if(isnan(uijm1)){
	    uijm1 = uijp1;
	    vijm1 = vijp1;
	    wijm1 = wijp1;
	  }
	  else if(isnan(uijp1)){
	    uijp1 = uijm1;
	    vijp1 = vijm1;
	    wijp1 = wijm1;
	  }
	}

	double lap_u = Du/(h*h)*(uim1j + uip1j + uijm1 + uijp1 -4.0*uij);
	double lap_v = Dv/(h*h)*(vim1j + vip1j + vijm1 + vijp1 -4.0*vij);
	double lap_w = Dw/(h*h)*(wim1j + wip1j + wijm1 + wijp1 -4.0*wij);

	// Evaluating reaction term
	double udot, vdot, wdot;
	F_ion(uij, vij, wij, &udot, &vdot, &wdot);
	
	fu[k] = lap_u + udot + E*gh[k];
	fv[k] = lap_v + vdot;
	fw[k] = lap_w + wdot;
	
      }
    }
  }
}

__device__ void lap_func(const double *u, const double *v, const double *w, const int nx, const int ny,
		       int *kPtr, double *uPtr, double *vPtr, double *wPtr, 
		      double *lap_uPtr, double *lap_vPtr, double *lap_wPtr){
  
  __shared__ double s_u[(BLOCK_SIZE_X + 2)*(BLOCK_SIZE_Y + 2)];
  __shared__ double s_v[(BLOCK_SIZE_X + 2)*(BLOCK_SIZE_Y + 2)];
  __shared__ double s_w[(BLOCK_SIZE_X + 2)*(BLOCK_SIZE_Y + 2)];
  
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;

  int s_i = threadIdx.x + 1;
  int s_j = threadIdx.y + 1;
  int s_ny = BLOCK_SIZE_Y + 2;

  // Load data into shared memory
  // Central square
  int k = getIndex(i, j, ny);
  double uij = u[k];
  double vij = v[k];
  double wij = w[k];

  int s_k = getIndex(s_i, s_j, s_ny);
  s_u[s_k] = uij;
  s_v[s_k] = vij;
  s_w[s_k] = wij;

  // Top border
  if (s_i == 1 && i != 0){
    int s_kb = getIndex(0, s_j, s_ny);
    int kb = getIndex(blockIdx.x*blockDim.x - 1, j, ny);
    
    s_u[s_kb] = u[kb];
    s_v[s_kb] = v[kb];
    s_w[s_kb] = w[kb];
  }
  // Bottom border
  if (s_i == BLOCK_SIZE_X && i != nx - 1){
    int s_kb =getIndex(BLOCK_SIZE_X + 1, s_j, s_ny);
    int kb = getIndex((blockIdx.x + 1)*blockDim.x, j, ny);

    s_u[s_kb] = u[kb];
    s_v[s_kb] = v[kb];
    s_w[s_kb] = w[kb];
  }
  // Left border
  if (s_j == 1 && j != 0){
    int s_kb = getIndex(s_i, 0, s_ny);
    int kb = getIndex(i, blockIdx.y*blockDim.y - 1, ny);

    s_u[s_kb] = u[kb];
    s_v[s_kb] = v[kb];
    s_w[s_kb] = w[kb];
  }
  // Right border
  if (s_j == BLOCK_SIZE_Y && j != ny - 1){
    int s_kb = getIndex(s_i, BLOCK_SIZE_Y + 1, s_ny);
    int kb =getIndex(i, (blockIdx.y + 1)*blockDim.y, ny); 

    s_u[s_kb] = u[kb];
    s_v[s_kb] = v[kb];
    s_w[s_kb] = w[kb];    
  }

  // Make sure all the data is loaded before computing
  __syncthreads();

  // Calculating Laplacian
  if(!isnan(uij)){
      if(i < nx){
	  if(j < ny){
            
	      double uim1j, uip1j, uijm1, uijp1;
	      double vim1j, vip1j, vijm1, vijp1;
	      double wim1j, wip1j, wijm1, wijp1;

	      // top and bottom neighbor
	      if(i==0) { // top row
		  int kn = getIndex(s_i+1, s_j, s_ny);
	  
		  uip1j = s_u[kn];
		  uim1j = uip1j;

		  vip1j = s_v[kn];
		  vim1j = vip1j;

		  wip1j = s_w[kn];
		  wim1j = wip1j;

	      }
	      else if(i==nx-1) { // bottom row
		  int kn = getIndex(s_i-1, s_j, s_ny);
	  
		  uim1j = s_u[kn];
		  uip1j = uim1j;
	  
		  vim1j = s_v[kn];
		  vip1j = vim1j;
	  
		  wim1j = s_w[kn];
		  wip1j = wim1j;
	  
	      }
	      else{
		  int kt = getIndex(s_i-1, s_j, s_ny);
		  int kb = getIndex(s_i+1, s_j, s_ny);

		  uim1j = s_u[kt];
		  uip1j = s_u[kb];

		  vim1j = s_v[kt];
		  vip1j = s_v[kb];

		  wim1j = s_w[kt];
		  wip1j = s_w[kb];

		  if(isnan(uim1j)){
		      uim1j = uip1j;
		      vim1j = vip1j;
		      wim1j = wip1j;
		  }
		  else if(isnan(uip1j)){
		      uip1j = uim1j;
		      vip1j = vim1j;
		      wip1j = wim1j;
		  }
	      }

	      // left and right neighbor
	      if(j==0){ // left column
		  int kr = getIndex(s_i, s_j+1, s_ny);

		  uijp1 = s_u[kr];
		  uijm1 = uijp1;

		  vijp1 = s_v[kr];
		  vijm1 = vijp1;

		  wijp1 = s_w[kr];
		  wijm1 = wijp1;
	      }
	      else if(j==ny-1){ // right column
		  int kl = getIndex(s_i, s_j-1, s_ny);
	  
		  uijm1 = s_u[kl];
		  uijp1 = uijm1;

		  vijm1 = s_v[kl];
		  vijp1 = vijm1;

		  wijm1 = s_w[kl];
		  wijp1 = wijm1;

	      }
	      else{
		  int kl = getIndex(s_i, s_j-1, s_ny);
		  int kr = getIndex(s_i, s_j+1, s_ny);

		  uijm1 = s_u[kl];
		  uijp1 = s_u[kr];

		  vijm1 = s_v[kl];
		  vijp1 = s_v[kr];

		  wijm1 = s_w[kl];
		  wijp1 = s_w[kr];

		  if(isnan(uijm1)){
		      uijm1 = uijp1;
		      vijm1 = vijp1;
		      wijm1 = wijp1;
		  }
		  else if(isnan(uijp1)){
		      uijp1 = uijm1;
		      vijp1 = vijm1;
		      wijp1 = wijm1;
		  }
	      }

	      *lap_uPtr = Du/(h*h)*(uim1j + uip1j + uijm1 + uijp1 -4.0*uij);
	      *lap_vPtr = Dv/(h*h)*(vim1j + vip1j + vijm1 + vijp1 -4.0*vij);
	      *lap_wPtr = Dw/(h*h)*(wim1j + wip1j + wijm1 + wijp1 -4.0*wij);

	  }
      }
  }else{
      *lap_uPtr = nan("");
      *lap_vPtr = nan("");
      *lap_wPtr = nan("");
  }
  
  *uPtr = uij;
  *vPtr = vij;
  *wPtr = wij;

  *kPtr = k;
}

__global__ void fk_lap_ion_E_1(const double *u, const double *v, const double *w, const double *gh, double E,
			     double *fu, double *fv, double *fw,
			     const int nx, const int ny) {
  int k;
  double uij, vij, wij;
  double lap_u, lap_v, lap_w;
  lap_func(u, v, w, nx, ny, &k, &uij, &vij, &wij, &lap_u, &lap_v, &lap_w);
  
  // Evaluating reaction term
  double udot, vdot, wdot;
  F_ion(uij, vij, wij, &udot, &vdot, &wdot);
  
  fu[k] = lap_u + udot + E*gh[k];
  fv[k] = lap_v + vdot;
  fw[k] = lap_w + wdot;
	
}

__global__ void fk_lap_ion_rk4_1(const double *u, const double *v, const double *w, const double *gh,
				 const double E, const double dt,
				 double *U, double *V, double *W, double *du, double *dv, double *dw,
				 const int nx, const int ny) {

  int k;
  double uij, vij, wij;
  double lap_u, lap_v, lap_w;
  lap_func(u, v, w, nx, ny, &k, &uij, &vij, &wij, &lap_u, &lap_v, &lap_w);
  
  // Evaluating reaction term
  double udot, vdot, wdot;
  F_ion(uij, vij, wij, &udot, &vdot, &wdot);
	
  double fu = lap_u + udot + E*gh[k];
  double fv = lap_v + vdot;
  double fw = lap_w + wdot;

  //preparing U, V, W for next RK4 step
  U[k] = uij + dt*0.5*fu;
  V[k] = vij + dt*0.5*fv;
  W[k] = wij + dt*0.5*fw;

  /* U[k] =  dt*0.5*fu; */
  /* V[k] =  dt*0.5*fv; */
  /* W[k] =  dt*0.5*fw; */

	
  // setting du, dv, dw
  du[k] = 1.0/6.0*dt*fu;
  dv[k] = 1.0/6.0*dt*fv;
  dw[k] = 1.0/6.0*dt*fw;
  
}

__global__ void fk_lap_ion_rk4_23(double *u, double *v, double *w, const double *gh,
				  const double E, const double dt, const int istep,
				  const double *u_, const double *v_, const double *w_ ,
				  double *Utemp, double *Vtemp, double *Wtemp,
				  double *du, double *dv, double *dw,
				  const int nx, const int ny) {

  int k;
  double uij, vij, wij;
  double lap_u, lap_v, lap_w;
  lap_func(u, v, w, nx, ny, &k, &uij, &vij, &wij, &lap_u, &lap_v, &lap_w);
  
  // Evaluating reaction term
  double udot, vdot, wdot;
  F_ion(uij, vij, wij, &udot, &vdot, &wdot);
	
  double fu = lap_u + udot + E*gh[k];
  double fv = lap_v + vdot;
  double fw = lap_w + wdot;

  // preparing U, V, W for next RK4 step
  // inside this function U is called u, V is called v, W is called w.
  double fac;
  if (istep==2)
    fac=0.5;
  else if(istep==3)
    fac=1.0;
  else{
    printf("Unknown istep value\n");
  }
	
  Utemp[k] = u_[k] + dt*fac*fu;
  Vtemp[k] = v_[k] + dt*fac*fv;
  Wtemp[k] = w_[k] + dt*fac*fw;
	
  /* Utemp[k] = dt*fac*fu; */
  /* Vtemp[k] = dt*fac*fv; */
  /* Wtemp[k] = dt*fac*fw; */
	
  // setting du, dv, dw
  du[k] = du[k] + 1.0/3.0*dt*fu;
  dv[k] = dv[k] + 1.0/3.0*dt*fv;
  dw[k] = dw[k] + 1.0/3.0*dt*fw;
}

__global__ void fk_lap_ion_rk4_4(const double *u, const double *v, const double *w, const double *gh,
				 const double E, const double dt,
				 double *u_, double *v_, double *w_ , const double *du, const double *dv, const double *dw,
				 const int nx, const int ny) {

  int k;
  double uij, vij, wij;
  double lap_u, lap_v, lap_w;
  lap_func(u, v, w, nx, ny, &k, &uij, &vij, &wij, &lap_u, &lap_v, &lap_w);
  
  // Evaluating reaction term
  double udot, vdot, wdot;
  F_ion(uij, vij, wij, &udot, &vdot, &wdot);
	
  double fu = lap_u + udot + E*gh[k];
  double fv = lap_v + vdot;
  double fw = lap_w + wdot;

  // inside this function U is called u, V is called v, W is called w.
  u_[k] = u_[k] + du[k] + dt*1.0/6.0*fu;
  v_[k] = v_[k] + dv[k] + dt*1.0/6.0*fv;
  w_[k] = w_[k] + dw[k] + dt*1.0/6.0*fw;

}


//------------------------ utility functions ----------------------------------
// From https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api/14038590#14038590
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


// verifies if a float has integer value
int float_integer(double r){
    if(r != 0){
	double rn = round(r);
	if( fabs(r-rn) < 5e-16*r )
	    return (int) rn;
	else {
	    fprintf(stderr, "--------------Not a float integer-------------------");
	    exit(EXIT_FAILURE);
	}
    }
    else
	return 0;
}

void csc_mv_mult(const mwIndex *Ir, const mwIndex *Jc, const double *S,
		 const double *vec, const size_t n_col, double *z){
    
    for(int i=0; i<n_col; i++) z[i]=0.0;

    for(int j=0; j<n_col; j++)
	for(int k=Jc[j]; k<Jc[j+1]; k++)
	    z[Ir[k]] += S[k]*vec[j];
}

double dot_prod(const double *a, const double *b, const int n){
    double sum = 0.0;
    
    for(int i=0; i<n; i++)
	sum += a[i]*b[i];

    return sum;
}

void num2nan(const double *u_num, const int *active_index, const int nn, double* u_nan){
    // nn is the size of the array with NaNs
    for(int i=0; i<nn; i++){
	int j = active_index[i];
	if(j>0) // below, subtracting 1 to j to change from Matlab to C indexing
	    u_nan[i] = u_num[j-1];
	else
	    u_nan[i] = nan("");
    }
}

void nan2num(const double *u_nan, const int *active_index, const int nn, double* u_num){
    // nn is the size of the array with NaNs
    for(int i=0; i<nn; i++){
	int j = active_index[i];
	if(j>0)// below, subtracting 1 to j to change from Matlab to C indexing
	    u_num[j-1] = u_nan[i];
    }
}

double vec_sum(const double *vec, const int n){
    
    double sum = 0.0;
    
    for(int i=0; i<n; i++)
	sum += vec[i];

    return sum;
}


// ---------------- COMPUTATIONAL ROUTINE --------------------------


int fk_forward_Et_comp(const double dt, const double *Et, const int nt,
		    const double *y_ini, const int *active_index, const int nrows, const int ncols, const int n_num,
		       const double *gh_num, const int nstep, const int ns, double *ys, double *yf, char **p_strPtr){

    int n = nrows*ncols;
    
    // Splitting y_ini into 3 arrays with NaNs
    double *u = (double *)malloc( sizeof(double)*n );
    double *v = (double *)malloc( sizeof(double)*n );
    double *w = (double *)malloc( sizeof(double)*n );
    
    num2nan(y_ini, active_index, n, u);
    num2nan(y_ini + n_num, active_index, n, v);
    num2nan(y_ini + 2*n_num, active_index, n, w);

    /* for(int i=0; i<10; i++) */
    /* 	printf("u[%d]=%.16e, v[%d]=%.16e, w[%d]=%.16e\n", i, u[i], i, v[i], i, w[i]); */

    double *gh = (double *)malloc( sizeof(double)*n );
    for(int i=0; i<n; i++){
	int j = active_index[i];
	if(j>0) // below, subtracting 1 to j to change from Matlab to C indexing
	    gh[i] = Du*2/h*gh_num[j-1];
	else
	    gh[i] = nan("");
    }
    
    // Allocate device memory
    double *d_u, *d_v, *d_w, *d_gh;//, *d_fu, *d_fv, *d_fw;
    double *d_U, *d_V, *d_W, *d_du, *d_dv, *d_dw;

    cudaMalloc((void**)&d_u, sizeof(double) * n);
    cudaMalloc((void**)&d_v, sizeof(double) * n);
    cudaMalloc((void**)&d_w, sizeof(double) * n);

    /* cudaMalloc((void**)&d_fu, sizeof(double) * n); */
    /* cudaMalloc((void**)&d_fv, sizeof(double) * n); */
    /* cudaMalloc((void**)&d_fw, sizeof(double) * n); */
  
    cudaMalloc((void**)&d_U, sizeof(double) * n);
    cudaMalloc((void**)&d_V, sizeof(double) * n);
    cudaMalloc((void**)&d_W, sizeof(double) * n);

    cudaMalloc((void**)&d_du, sizeof(double) * n);
    cudaMalloc((void**)&d_dv, sizeof(double) * n);
    cudaMalloc((void**)&d_dw, sizeof(double) * n);
  
    cudaMalloc((void**)&d_gh, sizeof(double) * n);
  
    // Transfer data from host to device
    cudaMemcpy(d_u, u, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w, sizeof(double) * n, cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_U, u, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, v, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, w, sizeof(double) * n, cudaMemcpyHostToDevice);
    
    cudaMemset(d_du, 0, sizeof(double) * n);
    cudaMemset(d_dv, 0, sizeof(double) * n);
    cudaMemset(d_dw, 0, sizeof(double) * n);

    double *d_Utemp, *d_Vtemp, *d_Wtemp;
    cudaMalloc((void**)&d_Utemp, sizeof(double) * n);
    cudaMalloc((void**)&d_Vtemp, sizeof(double) * n);
    cudaMalloc((void**)&d_Wtemp, sizeof(double) * n);

    cudaMemcpy(d_Utemp, u, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Vtemp, v, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wtemp, w, sizeof(double) * n, cudaMemcpyHostToDevice);

    cudaMemcpy(d_gh, gh, sizeof(double) * n, cudaMemcpyHostToDevice);

    // Release host memory
    free(gh);
    
    // below, transfering the NaNs to state after Euler step
    //cudaMemcpy(d_u1, u, sizeof(double) * n, cudaMemcpyHostToDevice);

    /* double *U = (double*)malloc( sizeof(double) * n ); */
    /* double *V = (double*)malloc( sizeof(double) * n ); */
    /* double *W = (double*)malloc( sizeof(double) * n ); */

    /* double *du = (double*)malloc( sizeof(double) * n ); */
    /* double *dv = (double*)malloc( sizeof(double) * n ); */
    /* double *dw = (double*)malloc( sizeof(double) * n ); */


    /* mxArray *ytMx= mxCreateDoubleMatrix( (mwSize)(::nvar*n), (mwSize)(nt+1), mxREAL); */
    /* double *yt = mxGetPr(ytMx); */
    
    /* // Copy initial condition into yt */
    /* memcpy(yt, u, sizeof(double) * n); */
    /* memcpy(yt + n, v, sizeof(double) * n); */
    /* memcpy(yt + 2*n, w, sizeof(double) * n); */

    int num_blocks_x = float_integer( ((double)nrows) / BLOCK_SIZE_X );
    printf("num_blocks_x=%d\n",num_blocks_x);
    
    int num_blocks_y = float_integer( ((double)ncols) / BLOCK_SIZE_Y );
    printf("num_blocks_y=%d\n",num_blocks_y);
 
    dim3 numBlocks( num_blocks_x, num_blocks_y );
    dim3 threadsPerBlock( BLOCK_SIZE_X, BLOCK_SIZE_Y );

    // setting up spline interpolation
    double *tt = (double *)malloc( sizeof(double) * (nt+1) );
    int nt1 = nt+1;
    for(int i=0; i<nt1; i++)
	tt[i] = i*dt;
    
    double *Etpp = spline_cubic_set(nt1, tt, Et, 3, 0.0, 3, 0.0);

    double  Ep, Epp;//E,
    //double t = 185.0;
    int interv = 1;

    printf("\n");
    
    // time stepping
    for(int i=0; i<nt; i++){
    
	double t_half, E_half;
	t_half = (i+0.5)*dt;

	spline_cubic_val2(nt1, tt, t_half, &interv, Et, Etpp, &E_half, &Ep, &Epp);
    
	fk_lap_ion_rk4_1<<<numBlocks,threadsPerBlock>>>(d_u, d_v, d_w, d_gh, Et[i], dt,
							d_U, d_V, d_W, d_du, d_dv, d_dw, nrows, ncols);

	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
      
  
	fk_lap_ion_rk4_23<<<numBlocks,threadsPerBlock>>>(d_U, d_V, d_W, d_gh, E_half, dt, 2,
							 d_u, d_v, d_w, d_Utemp, d_Vtemp, d_Wtemp,
							 d_du, d_dv, d_dw, nrows, ncols);

	// pointer swap in function call
	fk_lap_ion_rk4_23<<<numBlocks,threadsPerBlock>>>(d_Utemp, d_Vtemp, d_Wtemp, d_gh, E_half, dt, 3,
							 d_u, d_v, d_w, d_U, d_V, d_W,
							 d_du, d_dv, d_dw, nrows, ncols);

	/* fk_lap_ion_E<<<numBlocks,threadsPerBlock>>>(d_Utemp, d_Vtemp, d_Wtemp, d_gh, E_half, */
	/* 						d_U, d_V, d_W, */
	/* 						nrows, ncols); */
	fk_lap_ion_rk4_4<<<numBlocks,threadsPerBlock>>>(d_U, d_V, d_W, d_gh, Et[i+1], dt,
							d_u, d_v, d_w, d_du, d_dv, d_dw, nrows, ncols);
	if( 0==(i+1)%nstep ){

	    int ks = (i+1)/nstep - 1;
	    /* printf("ks=%d\n",ks); */
	    /* printf("::nvar=%d\n",::nvar); */
	    /* printf("ns=%d\n",ns); */

	    if( ks < ns ){
		int offset = ks*::nvar*n;

		/* printf("offset=%d\n", offset); */
		
		cudaMemcpy(ys + offset, d_u, sizeof(double) * n, cudaMemcpyDeviceToHost);
		cudaMemcpy(ys + offset + n, d_v, sizeof(double) * n, cudaMemcpyDeviceToHost);
		cudaMemcpy(ys + offset + 2*n, d_w, sizeof(double) * n, cudaMemcpyDeviceToHost);

		//printf("ks = %d, max = %d\n", ks, ns-1);
		printf("\rks = %d, max = %d", ks, ns-1);
	     
	    }
	}
    }

    printf("\n\n");
	
    cudaMemcpy(yf, d_u, sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(yf + n, d_v, sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(yf + 2*n, d_w, sizeof(double) * n, cudaMemcpyDeviceToHost);

    // Release device memory
    cudaFree(d_u); cudaFree(d_v); cudaFree(d_w);
    cudaFree(d_U); cudaFree(d_V); cudaFree(d_W);
    cudaFree(d_Utemp); cudaFree(d_Vtemp); cudaFree(d_Wtemp);
    cudaFree(d_du); cudaFree(d_dv); cudaFree(d_dw);
    cudaFree(d_gh);
    
    /* for(int i=0; i<10; i++) */
    /* 	printf("u[%d]=%.16e\n", i, u[i]); */

    

    // Release host memory
    //mxDestroyArray(zgh_tMx);
    
    free(u); free(v); free(w);
    free(tt); free(Etpp);

    /* printf("parameters_string[]=%s\n",::parameters_string); */
    int n_str = (int)strlen(::parameters_string);
    /* printf("string length=%d\n",n_str); */
    char *param_str = (char *)mxMalloc( n_str*sizeof(char) );
    memcpy( param_str, ::parameters_string, n_str*sizeof(char) ); // copying parameters string to heap memory
    *p_strPtr = param_str;
    
    return(0);
}

//------------------------ interpolation functions -------------------------------------

// In the functions below, local variable "h" hides global "h"

/******************************************************************************/

double *penta ( int n, double a1[], double a2[], double a3[], double a4[], 
  double a5[], double b[] )

/******************************************************************************/
/*
  Purpose:

    PENTA solves a pentadiagonal system of linear equations.

  Discussion:

    The matrix A is pentadiagonal.  It is entirely zero, except for
    the main diagaonal, and the two immediate sub- and super-diagonals.

    The entries of Row I are stored as:

      A(I,I-2) -> A1(I)
      A(I,I-1) -> A2(I)
      A(I,I)   -> A3(I)
      A(I,I+1) -> A4(I)
      A(I,I-2) -> A5(I)

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    07 June 2013

  Author:

    John Burkardt

  Reference:

    Cheney, Kincaid,
    Numerical Mathematics and Computing,
    1985, pages 233-236.

  Parameters:

    Input, int N, the order of the matrix.

    Input, double A1[N], A2[N], A3[N], A4[N], A5[N], the nonzero
    elements of the matrix.  Note that the data in A2, A3 and A4
    is overwritten by this routine during the solution process.

    Input, double B[N], the right hand side of the linear system.

    Output, double PENTA[N], the solution of the linear system.
*/
{
  int i;
  double *x;
  double xmult;

  x = ( double * ) malloc ( n * sizeof ( double ) );

  for ( i = 1; i < n - 1; i++ )
  {
    xmult = a2[i] / a3[i-1];
    a3[i] = a3[i] - xmult * a4[i-1];
    a4[i] = a4[i] - xmult * a5[i-1];
    b[i] = b[i] - xmult * b[i-1];
    xmult = a1[i+1] / a3[i-1];
    a2[i+1] = a2[i+1] - xmult * a4[i-1];
    a3[i+1] = a3[i+1] - xmult * a5[i-1];
    b[i+1] = b[i+1] - xmult * b[i-1];
  }

  xmult = a2[n-1] / a3[n-2];
  a3[n-1] = a3[n-1] - xmult * a4[n-2];
  x[n-1] = ( b[n-1] - xmult * b[n-2] ) / a3[n-1];
  x[n-2] = ( b[n-2] - a4[n-2] * x[n-1] ) / a3[n-2];
  for ( i = n - 3; 0 <= i; i-- )
  {
    x[i] = ( b[i] - a4[i] * x[i+1] - a5[i] * x[i+2] ) / a3[i];
  }

  return x;
}
/******************************************************************************/

void r8vec_bracket3 ( int n, double t[], double tval, int *left )

/******************************************************************************/
/*
  Purpose:

    R8VEC_BRACKET3 finds the interval containing or nearest a given value.

  Discussion:

    An R8VEC is a vector of R8's.

    The routine always returns the index LEFT of the sorted array
    T with the property that either
    *  T is contained in the interval [ T[LEFT], T[LEFT+1] ], or
    *  T < T[LEFT] = T[0], or
    *  T > T[LEFT+1] = T[N-1].

    The routine is useful for interpolation problems, where
    the abscissa must be located within an interval of data
    abscissas for interpolation, or the "nearest" interval
    to the (extreme) abscissa must be found so that extrapolation
    can be carried out.

    This version of the function has been revised so that the value of
    LEFT that is returned uses the 0-based indexing natural to C++.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    31 May 2009

  Author:

    John Burkardt

  Parameters:

    Input, int N, length of the input array.

    Input, double T[N], an array that has been sorted into ascending order.

    Input, double TVAL, a value to be bracketed by entries of T.

    Input/output, int *LEFT.
    On input, if 0 <= LEFT <= N-2, LEFT is taken as a suggestion for the
    interval [ T[LEFT-1] T[LEFT] ] in which TVAL lies.  This interval
    is searched first, followed by the appropriate interval to the left
    or right.  After that, a binary search is used.
    On output, LEFT is set so that the interval [ T[LEFT], T[LEFT+1] ]
    is the closest to TVAL; it either contains TVAL, or else TVAL
    lies outside the interval [ T[0], T[N-1] ].
*/
{
  int high;
  int low;
  int mid;
/*
  Check the input data.
*/
  if ( n < 2 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "R8VEC_BRACKET3 - Fatal error\n" );
    fprintf ( stderr, "  N must be at least 2.\n" );
    exit ( 1 );
  }
/*
  If *LEFT is not between 0 and N-2, set it to the middle value.
*/
  if ( *left < 0 || n - 2 < *left )
  {
    *left = ( n - 1 ) / 2;
  }
/*
  CASE 1: TVAL < T[*LEFT]:
  Search for TVAL in (T[I],T[I+1]), for I = 0 to *LEFT-1.
*/
  if ( tval < t[*left] )
  {
    if ( *left == 0 )
    {
      return;
    }
    else if ( *left == 1 )
    {
      *left = 0;
      return;
    }
    else if ( t[*left-1] <= tval )
    {
      *left = *left - 1;
      return;
    }
    else if ( tval <= t[1] )
    {
      *left = 0;
      return;
    }
/*
  ...Binary search for TVAL in (T[I],T[I+1]), for I = 1 to *LEFT-2.
*/
    low = 1;
    high = *left - 2;

    for ( ; ; )
    {
      if ( low == high )
      {
        *left = low;
        return;
      }

      mid = ( low + high + 1 ) / 2;

      if ( t[mid] <= tval )
      {
        low = mid;
      }
      else
      {
        high = mid - 1;
      }
    }
  }
/*
  CASE 2: T[*LEFT+1] < TVAL:
  Search for TVAL in (T[I],T[I+1]) for intervals I = *LEFT+1 to N-2.
*/
  else if ( t[*left+1] < tval )
  {
    if ( *left == n - 2 )
    {
      return;
    }
    else if ( *left == n - 3 )
    {
      *left = *left + 1;
      return;
    }
    else if ( tval <= t[*left+2] )
    {
      *left = *left + 1;
      return;
    }
    else if ( t[n-2] <= tval )
    {
      *left = n - 2;
      return;
    }
/*
  ...Binary search for TVAL in (T[I],T[I+1]) for intervals I = *LEFT+2 to N-3.
*/
    low = *left + 2;
    high = n - 3;

    for ( ; ; )
    {

      if ( low == high )
      {
        *left = low;
        return;
      }

      mid = ( low + high + 1 ) / 2;

      if ( t[mid] <= tval )
      {
        low = mid;
      }
      else
      {
        high = mid - 1;
      }
    }
  }
/*
  CASE 3: T[*LEFT] <= TVAL <= T[*LEFT+1]:
  T is just where the user said it might be.
*/
  else
  {
  }

  return;
}

/******************************************************************************/

double *spline_cubic_set ( int n, double t[], const double y[], int ibcbeg, 
  double ybcbeg, int ibcend, double ybcend )

/******************************************************************************/
/*
  Purpose:

    SPLINE_CUBIC_SET computes the second derivatives of a piecewise cubic spline.

  Discussion:

    For data interpolation, the user must call SPLINE_SET to determine
    the second derivative data, passing in the data to be interpolated,
    and the desired boundary conditions.

    The data to be interpolated, plus the SPLINE_SET output, defines
    the spline.  The user may then call SPLINE_VAL to evaluate the
    spline at any point.

    The cubic spline is a piecewise cubic polynomial.  The intervals
    are determined by the "knots" or abscissas of the data to be
    interpolated.  The cubic spline has continous first and second
    derivatives over the entire interval of interpolation.

    For any point T in the interval T(IVAL), T(IVAL+1), the form of
    the spline is

      SPL(T) = A(IVAL)
             + B(IVAL) * ( T - T(IVAL) )
             + C(IVAL) * ( T - T(IVAL) )^2
             + D(IVAL) * ( T - T(IVAL) )^3

    If we assume that we know the values Y(*) and YPP(*), which represent
    the values and second derivatives of the spline at each knot, then
    the coefficients can be computed as:

      A(IVAL) = Y(IVAL)
      B(IVAL) = ( Y(IVAL+1) - Y(IVAL) ) / ( T(IVAL+1) - T(IVAL) )
        - ( YPP(IVAL+1) + 2 * YPP(IVAL) ) * ( T(IVAL+1) - T(IVAL) ) / 6
      C(IVAL) = YPP(IVAL) / 2
      D(IVAL) = ( YPP(IVAL+1) - YPP(IVAL) ) / ( 6 * ( T(IVAL+1) - T(IVAL) ) )

    Since the first derivative of the spline is

      SPL'(T) =     B(IVAL)
              + 2 * C(IVAL) * ( T - T(IVAL) )
              + 3 * D(IVAL) * ( T - T(IVAL) )^2,

    the requirement that the first derivative be continuous at interior
    knot I results in a total of N-2 equations, of the form:

      B(IVAL-1) + 2 C(IVAL-1) * (T(IVAL)-T(IVAL-1))
      + 3 * D(IVAL-1) * (T(IVAL) - T(IVAL-1))^2 = B(IVAL)

    or, setting H(IVAL) = T(IVAL+1) - T(IVAL)

      ( Y(IVAL) - Y(IVAL-1) ) / H(IVAL-1)
      - ( YPP(IVAL) + 2 * YPP(IVAL-1) ) * H(IVAL-1) / 6
      + YPP(IVAL-1) * H(IVAL-1)
      + ( YPP(IVAL) - YPP(IVAL-1) ) * H(IVAL-1) / 2
      =
      ( Y(IVAL+1) - Y(IVAL) ) / H(IVAL)
      - ( YPP(IVAL+1) + 2 * YPP(IVAL) ) * H(IVAL) / 6

    or

      YPP(IVAL-1) * H(IVAL-1) + 2 * YPP(IVAL) * ( H(IVAL-1) + H(IVAL) )
      + YPP(IVAL) * H(IVAL)
      =
      6 * ( Y(IVAL+1) - Y(IVAL) ) / H(IVAL)
      - 6 * ( Y(IVAL) - Y(IVAL-1) ) / H(IVAL-1)

    Boundary conditions must be applied at the first and last knots.  
    The resulting tridiagonal system can be solved for the YPP values.

    Anton Reinhard corrected the assignments:
      a2[i] = ( t[i+1] - t[i]   ) / 6.0;
      a3[i] = ( t[i+1] - t[i-1] ) / 3.0;
      a4[i] = ( t[i]   - t[i-1] ) / 6.0;
    to
      a2[i] = ( t[i]   - t[i-1] ) / 6.0;
      a3[i] = ( t[i+1] - t[i-1] ) / 3.0;
      a4[i] = ( t[i+1] - t[i]   ) / 6.0;

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    17 August 2020

  Author:

    John Burkardt

  Reference:

    Carl deBoor,
    A Practical Guide to Splines,
    Springer, 2001,
    ISBN: 0387953663.

  Input:

    int N, the number of data points.  N must be at least 2.
    In the special case where N = 2 and IBCBEG = IBCEND = 0, the
    spline will actually be linear.

    double T[N], the knot values, that is, the points were data is
    specified.  The knot values should be distinct, and increasing.

    double Y[N], the data values to be interpolated.

    int IBCBEG, left boundary condition flag:
    0: the cubic spline should be a quadratic over the first interval;
    1: the first derivative at the left endpoint should be YBCBEG;
    2: the second derivative at the left endpoint should be YBCBEG;
    3: Not-a-knot: the third derivative is continuous at T(2).

    double YBCBEG, the values to be used in the boundary
    conditions if IBCBEG is equal to 1 or 2.

    int IBCEND, right boundary condition flag:
    0: the cubic spline should be a quadratic over the last interval;
    1: the first derivative at the right endpoint should be YBCEND;
    2: the second derivative at the right endpoint should be YBCEND;
    3: Not-a-knot: the third derivative is continuous at T(N-1).

    double YBCEND, the values to be used in the boundary
    conditions if IBCEND is equal to 1 or 2.

  Output:

    double SPLINE_CUBIC_SET[N], the second derivatives 
    of the cubic spline.
*/
{
  double *a1;
  double *a2;
  double *a3;
  double *a4;
  double *a5;
  double *b;
  int i;
  double *ypp;
/*
  Check.
*/
  if ( n <= 1 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "SPLINE_CUBIC_SET - Fatal error!\n" );
    fprintf ( stderr, "  The number of data points N must be at least 2.\n" );
    fprintf ( stderr, "  The input value is %d.\n", n );
    exit ( 1 );
  }

  for ( i = 0; i < n - 1; i++ )
  {
    if ( t[i+1] <= t[i] )
    {
      fprintf ( stderr, "\n" );
      fprintf ( stderr, "SPLINE_CUBIC_SET - Fatal error!\n" );
      fprintf ( stderr, "  The knots must be strictly increasing, but\n" );
      fprintf ( stderr, "  T(%d) = %g\n", i, t[i] );
      fprintf ( stderr, "  T(%d) = %g\n", i+1, t[i+1] );
      exit ( 1 );
    }
  }
  a1 = ( double * ) malloc ( n * sizeof ( double ) );
  a2 = ( double * ) malloc ( n * sizeof ( double ) );
  a3 = ( double * ) malloc ( n * sizeof ( double ) );
  a4 = ( double * ) malloc ( n * sizeof ( double ) );
  a5 = ( double * ) malloc ( n * sizeof ( double ) );
  b = ( double * ) malloc ( n * sizeof ( double ) );

  for ( i = 0; i < n; i++ )
  {
    a1[i] = 0.0;
    a2[i] = 0.0;
    a3[i] = 0.0;
    a4[i] = 0.0;
    a5[i] = 0.0;
  }
/*
  Set up the first equation.
*/
  if ( ibcbeg == 0 )
  {
    b[0] = 0.0;
    a3[0] = 1.0;
    a4[0] = -1.0;
  }
  else if ( ibcbeg == 1 )
  {
    b[0] = ( y[1] - y[0] ) / ( t[1] - t[0] ) - ybcbeg;
    a3[0] = ( t[1] - t[0] ) / 3.0;
    a4[0] = ( t[1] - t[0] ) / 6.0;
  }
  else if ( ibcbeg == 2 )
  {
    b[0] = ybcbeg;
    a3[0] = 1.0;
    a4[0] = 0.0;
  }
  else if ( ibcbeg == 3 )
  {
    b[0] = 0.0;
    a3[0] = - ( t[2] - t[1] );
    a4[0] =   ( t[2]        - t[0] );
    a5[0] = - (        t[1] - t[0] );
  }
  else
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "SPLINE_CUBIC_SET - Fatal error!\n" );
    fprintf ( stderr, "  IBCBEG must be 0, 1, 2, or 3.\n" );
    fprintf ( stderr, "  The input value is %d.\n", ibcbeg );
    exit ( 1 );
  }
/*
  Set up the intermediate equations.
*/
  for ( i = 1; i < n - 1; i++ )
  {
    b[i] = ( y[i+1] - y[i] ) / ( t[i+1] - t[i] )
      - ( y[i] - y[i-1] ) / ( t[i] - t[i-1] );
    a2[i] = ( t[i]   - t[i-1] ) / 6.0;
    a3[i] = ( t[i+1] - t[i-1] ) / 3.0;
    a4[i] = ( t[i+1] - t[i]   ) / 6.0;
  }
/*
  Set up the last equation.
*/
  if ( ibcend == 0 )
  {
    b[n-1] = 0.0;
    a2[n-1] = -1.0;
    a3[n-1] = 1.0;
  }
  else if ( ibcend == 1 )
  {
    b[n-1] = ybcend - ( y[n-1] - y[n-2] ) / ( t[n-1] - t[n-2] );
    a2[n-1] = ( t[n-1] - t[n-2] ) / 6.0;
    a3[n-1] = ( t[n-1] - t[n-2] ) / 3.0;
  }
  else if ( ibcend == 2 )
  {
    b[n-1] = ybcend;
    a2[n-1] = 0.0;
    a3[n-1] = 1.0;
  }
  else if ( ibcbeg == 3 )
  {
    b[n-1] = 0.0;
    a1[n-1] = - ( t[n-1] - t[n-2] );
    a2[n-1] =   ( t[n-1]          - t[n-3] );
    a3[n-1] = - (          t[n-2] - t[n-3] );
  }
  else
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "SPLINE_CUBIC_SET - Fatal error!\n" );
    fprintf ( stderr, "  IBCEND must be 0, 1, 2 or 3.\n" );
    fprintf ( stderr, "  The input value is %d.\n", ibcend );
    exit ( 1 );
  }
/*
  Solve the linear system.
*/
  if ( n == 2 && ibcbeg == 0 && ibcend == 0 )
  {
    ypp = ( double * ) malloc ( 2 * sizeof ( double ) );

    ypp[0] = 0.0;
    ypp[1] = 0.0;
  }
  else
  {
    ypp = penta ( n, a1, a2, a3, a4, a5, b );
  }
/*
  Free memory.
*/
  free ( a1 );
  free ( a2 );
  free ( a3 );
  free ( a4 );
  free ( a5 );
  free ( b );

  return ypp;
}

/******************************************************************************/

void spline_cubic_val2 ( int n, double t[], double tval, int *left, const double y[], 
  double ypp[], double *yval, double *ypval, double *yppval )

/******************************************************************************/
/*
  Purpose:

    SPLINE_CUBIC_VAL2 evaluates a piecewise cubic spline at a point.

  Discussion:

    This routine is a modification of SPLINE_CUBIC_VAL; it allows the
    user to speed up the code by suggesting the appropriate T interval
    to search first.

    SPLINE_CUBIC_SET must have already been called to define the
    values of YPP.

    In the LEFT interval, let RIGHT = LEFT+1.  The form of the spline is

    SPL(T) =
      A
    + B * ( T - T[LEFT] )
    + C * ( T - T[LEFT] )^2
    + D * ( T - T[LEFT] )^3

    Here:
      A = Y[LEFT]
      B = ( Y[RIGHT] - Y[LEFT] ) / ( T[RIGHT] - T[LEFT] )
        - ( YPP[RIGHT] + 2 * YPP[LEFT] ) * ( T[RIGHT] - T[LEFT] ) / 6
      C = YPP[LEFT] / 2
      D = ( YPP[RIGHT] - YPP[LEFT] ) / ( 6 * ( T[RIGHT] - T[LEFT] ) )

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    24 February 2004

  Author:

    John Burkardt

  Parameters:

    Input, int N, the number of knots.

    Input, double T[N], the knot values.

    Input, double TVAL, a point, typically between T[0] and T[N-1], at
    which the spline is to be evalulated.  If TVAL lies outside
    this range, extrapolation is used.

    Input/output, int *LEFT, the suggested T interval to search.
    LEFT should be between 1 and N-1.  If LEFT is not in this range,
    then its value will be ignored.  On output, LEFT is set to the
    actual interval in which TVAL lies.

    Input, double Y[N], the data values at the knots.

    Input, double YPP[N], the second derivatives of the spline at
    the knots.

    Output, double *YVAL, *YPVAL, *YPPVAL, the value of the spline, and
    its first two derivatives at TVAL.
*/
{
  double dt;
  double h;
  int right;
/*
  Determine the interval [T[LEFT], T[RIGHT]] that contains TVAL.  
  
  What you want from R8VEC_BRACKET3 is that TVAL is to be computed
  by the data in interval [T[LEFT-1], T[RIGHT-1]].  
*/
  r8vec_bracket3 ( n, t, tval, left );
/*
 In the interval LEFT, the polynomial is in terms of a normalized
 coordinate  ( DT / H ) between 0 and 1.
*/
/*  On Sep 01, 2021, Alejandro Garzon changed
 *left - 1 -> *left
 right-1 -> right
*/
  
  right = *left + 1;

  dt = tval - t[*left];
  h = t[right] - t[*left];

  *yval = y[*left]
     + dt * ( ( y[right] - y[*left] ) / h
        - ( ypp[right] / 6.0 + ypp[*left] / 3.0 ) * h
     + dt * ( 0.5 * ypp[*left]
     + dt * ( ( ypp[right] - ypp[*left] ) / ( 6.0 * h ) ) ) );

  *ypval = ( y[right] - y[*left] ) / h
     - ( ypp[right] / 6.0 + ypp[*left] / 3.0 ) * h
     + dt * ( ypp[*left]
     + dt * ( 0.5 * ( ypp[right] - ypp[*left] ) / h ) );

  *yppval = ypp[*left] + dt * ( ypp[right] - ypp[*left] ) / h;

  return;
}

