void computeResultError(float *h_A, float *h_b, float *h_x, unsigned int dim);
void jacobi_gpu(float* h_x, float* h_A, float* h_b,  int dim, int n_it);
void GS_gpu(float* h_x, float* h_A, float* h_b,  int dim, int n_it);