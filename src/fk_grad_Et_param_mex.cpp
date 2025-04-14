#include "mex.h"

int fk_grad_Et_comp(const double dt, const double *Et, const int nt,
		    const double *y_ini, const int *active_index, const int nrows, const int ncols, const int n_num,
		    const double *gh_num, const double *gamma, const mxArray *SMx, const double alpha,
		    double *L0Ptr, double *LEPtr, double *LcostPtr, double *gradE, char **p_strPtr);



/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    constexpr int nrhs_ = 11;
    if(nrhs != nrhs_) {
	mexErrMsgIdAndTxt("foo:bar:nrhs",
			  "%d inputs required.", nrhs_);
    }

    constexpr int nlhs_ = 5;    
    if(nlhs != nlhs_) {
	mexErrMsgIdAndTxt("foo:bar:nlhs",
			  "%d outputs required.", nlhs_);
    }

    const double *Et = mxGetPr(prhs[0]);
    const double dt = mxGetPr(prhs[1])[0];
    const int nt = (int)(mxGetPr(prhs[2])[0]);
    const double *y_ini = mxGetPr(prhs[3]);
    const int nvar = 3;
    const int n_num = mxGetM(prhs[3])/nvar;
    const int *active_index = (int *)mxGetData(prhs[4]);
    const int ncols = (int)(mxGetPr(prhs[5])[0]);
    const int nrows = (int)(mxGetPr(prhs[6])[0]);
    const double *gh_num =  mxGetPr(prhs[7]);
    const double *gamma = mxGetPr(prhs[8]);
    const mxArray *SMx = prhs[9];
    const double alpha = mxGetPr(prhs[10])[0];

    // for(int i=0; i<10; i++)
    // 	printf("Et[%d]=%.16e\n", i, Et[i]);

    // printf("dt=%f\n", dt);
    // printf("nt=%d\n", nt);
    
    // for(int i=0; i<10; i++)
    // 	printf("y_ini[%d]=%.16e\n", i, y_ini[i]);

    // printf("n_num=%d\n",n_num);
    
    // for(int i=0; i<10; i++)
    // 	printf("active_index[%d]=%d\n", i, active_index[i]);

    // printf("nrows=%d\n", nrows);
    // printf("ncols=%d\n", ncols);

    // for(int i=0; i<10; i++)
    // 	printf("gh_num[%d]=%.16e\n", i, gh_num[i]);
    
    // for(int i=0; i<nvar; i++)
    // 	printf("gamma[%d]=%f\n",i,gamma[i]);
    
    // printf("alpha=%f\n", alpha);


    
    
    // Create output matrices
    plhs[0] = mxCreateDoubleMatrix( (mwSize)nt+1, 1, mxREAL);
    double *gradE = mxGetPr(plhs[0]);

    plhs[1] = mxCreateDoubleMatrix( 1, 1, mxREAL);
    double *LcostPtr = mxGetPr(plhs[1]);
    
    plhs[2] = mxCreateDoubleMatrix( 1, 1, mxREAL);
    double *L0Ptr = mxGetPr(plhs[2]);
    
    plhs[3] = mxCreateDoubleMatrix( 1, 1, mxREAL);
    double *LEPtr = mxGetPr(plhs[3]);

    char *param_str = NULL;
    
    int error_code = fk_grad_Et_comp(dt, Et, nt, y_ini, active_index, nrows, ncols, n_num, gh_num, gamma, SMx, alpha,
				     L0Ptr, LEPtr, LcostPtr, gradE, &param_str);

    //printf("param_str = %s\n", param_str);

    plhs[4] = mxCreateString(param_str);
    mxFree(param_str);
    
    
}


