#include <iostream>
#include "matrixkernels.cuh"
#include "helper_cuda.h"
#include "assert.h"

extern "C"
int iDivUp( int a, int b ){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}


extern "C"
unsigned int nextPow2( unsigned int x ) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}


// vector operation: x = fac0*a op fac1*b
__global__ void
_cl_vector_op_( int op, float fac0, float fac1, float *a, float *b, float *x, int dim )
{
	/* TASK 1.1: implement the elementwise vector operations
	
	 	x = fac0 * a (op) fac1 * b
	
		with op = {+,-,*, NONE}.
		NONE means x = fac0 * a   (b might be NULL)

		
		HINT: remember to safeguard the index (the thread id might be larger than the array size)! 
		-> if the thread index is >= dim return!
		
	*/

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while (idx < dim) {
		switch (op) {
		case CL_ADD: x[idx] = fac0*a[idx]+fac1*b[idx]; break;
		case CL_SUB: x[idx] = fac0 * a[idx] - fac1 * b[idx]; break;
		case CL_MULT: x[idx] = fac0 * a[idx] * fac1 * b[idx]; break;
		case NONE: x[idx] = fac0 * a[idx]; break;
		default: assert(false); break;
		}
		idx+= gridDim.x * blockDim.x;
	}
}




// matrix vector multiplication: x = A*b op c

//__global__ void
//_cl_matrix_vector_old(int op, float* A, float* b, float* c, float* x, int dim)
//{
//	/* TASK 1.2: implement the matrix vector multiplication
//
//		x = A * b (op) c
//
//		with op = {+,-,*,NONE}.
//		NONE means x = A * b     (c might be NULL)
//
//		HINT: remember to safeguard the index (the thread id might be larger than the array size)!
//		-> if the thread index is >= dim return!
//	*/
//	// TDL: remove the assumption that dim is a multiple of BLOCK_SIZE
//	__shared__ float cache[BLOCK_SIZE];
//
//	int out_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
//	float out = 0.f;
//
//	for (int m = 0; m < dim / BLOCK_SIZE; m++) {
//		cache[threadIdx.x] = b[m * BLOCK_SIZE + threadIdx.x];
//		__syncthreads();
//		for (int k = 0; k < BLOCK_SIZE; k++) {
//			out += A[out_idx * dim + m * BLOCK_SIZE + k] * cache[k];
//		}
//		__syncthreads();
//	}
//	x[out_idx] = out;
//
//}


__global__ void
_cl_matrix_vector_( int op, float *A, float *b, float *c, float *x, int dim )
{
	/* TASK 1.2: implement the matrix vector multiplication
	
		x = A * b (op) c
	
		with op = {+,-,*,NONE}.
		NONE means x = A * b     (c might be NULL)

		HINT: remember to safeguard the index (the thread id might be larger than the array size)!
		-> if the thread index is >= dim return!
	*/
	// TDL: remove the assumption that dim is a multiple of BLOCK_SIZE
#define min(x,y) x<y?x:y
	extern __shared__ float cache[];
	
	int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	int m = 0;
	float out = 0.f;
	while (m < dim) {

		int items_in_cache = min(blockDim.x, dim - m);

		if (threadIdx.x < items_in_cache) {
			cache[threadIdx.x] = b[m + threadIdx.x];
		}
			
		__syncthreads();
		if (out_idx < dim) {
			for (int k = 0; k < items_in_cache; k++) {
				out += A[out_idx * dim + m + k] * cache[k];
			}
		}
		
		__syncthreads();
		m += blockDim.x;
	}

	if (out_idx < dim) {
		switch (op) {
		case CL_ADD: x[out_idx] = out + c[out_idx]; break;
		case CL_SUB: x[out_idx] = out - c[out_idx]; break;
		case CL_MULT: x[out_idx] = out * c[out_idx]; break;
		case NONE: x[out_idx] = out; break;
		default: assert(false); break;
		}
	}
	

	

}
__global__ void inner_product_partial_reduction(float* a, float* b, float* c, int dim) {
	// TODO:
	extern __shared__ float cache[];

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;
	float temp = 0;
	while (tid < dim){
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}
	cache[cacheIndex] = temp;
	__syncthreads();
	int i = blockDim.x / 2;
	while (i != 0) {
		if (cacheIndex < i) {
			cache[cacheIndex] += cache[cacheIndex + i];
		}
		__syncthreads();
		i /= 2;
	}
	if (threadIdx.x == 0) {
		c[blockIdx.x] = cache[0];
	}
}


// d_x = SUM[d_a * d_b]
float gpuReduceSUM( float* d_a, float *d_b, float* d_x, int dim, int nBlocks, int nThreads ){
	// nThreads should be power of 2
	assert((nThreads & (nThreads - 1)) == 0);
	/* TASK 1.3: implement the vector multiplication and sum reduction

		d_x = SUM[d_a * d_b]
		
		implement reduction as discussed in the lecture using shared memory.
		
	*/
	float* h_x = new float[nBlocks];
	std::fill(h_x, h_x + nBlocks, 0);
	inner_product_partial_reduction << <nBlocks, nThreads, nThreads*sizeof(float) >> > (d_a, d_b, d_x, dim);

	checkCudaErrors(cudaMemcpy(h_x, d_x, nBlocks * sizeof(float), cudaMemcpyDeviceToHost));
	
	float sum = 0;
	for (int i = 0; i < nBlocks; i++) {
		sum += h_x[i];
	}
	
	delete[] h_x;
	return sum;
}

// x = A*a
extern "C" 
void multiplyMatrixVector( float *h_A, float *h_a, float *h_x, int dim )
{
	float *d_A, *d_a, *d_x;

	checkCudaErrors( cudaMalloc( (void**) &d_A, dim * dim * sizeof( float ) ) );
	checkCudaErrors( cudaMalloc( (void**) &d_a, dim * sizeof( float ) ) );
	checkCudaErrors( cudaMalloc( (void**) &d_x, dim * sizeof( float ) ) );

	checkCudaErrors( cudaMemcpy( d_A, h_A, dim * dim * sizeof( float ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy( d_a, h_a, dim * sizeof( float ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy( d_x, h_x, dim * sizeof( float ), cudaMemcpyHostToDevice ) );

	checkCudaErrors( cudaDeviceSynchronize() );

	// x = A*a
	int nThreads = 15;
	int nBlocks = iDivUp( dim, nThreads );
	// TODO: change back
	_cl_matrix_vector_<<< nBlocks, nThreads, nThreads*sizeof(float) >>>( NONE, d_A, d_a, NULL, d_x, dim );
	checkCudaErrors( cudaDeviceSynchronize() );

	// copy solution from device to host
	checkCudaErrors( cudaMemcpy( h_x, d_x, dim * sizeof( float ), cudaMemcpyDeviceToHost ) );

	// release device memory
	checkCudaErrors( cudaFree( d_A ) );
	checkCudaErrors( cudaFree( d_a ) );
	checkCudaErrors( cudaFree( d_x ) );
	
	
}


extern "C" 
void computeConjugateGradient( float *h_A, float *h_b, float *h_x, int dim, float errorTolerance )
{
	printf("errorTolerance: %.6g\n", errorTolerance);
	int nThreads = 128;							// set the number of threads per block to use by default
	int nBlocks = iDivUp( dim, nThreads );
	
	float *d_A, *d_b, *d_x, *d_r, *d_p, *d_q, *d_tmp;
	float alpha, beta, rho = 0;

	//allocate device memory
	checkCudaErrors( cudaMalloc( (void**) &d_A, dim * dim * sizeof( float ) ) );
	checkCudaErrors( cudaMalloc( (void**) &d_b, dim * sizeof( float ) ) );
	checkCudaErrors( cudaMalloc( (void**) &d_x, dim * sizeof( float ) ) );
	checkCudaErrors( cudaMalloc( (void**) &d_r, dim * sizeof( float ) ) );
	checkCudaErrors( cudaMalloc( (void**) &d_p, dim * sizeof( float ) ) );
	checkCudaErrors( cudaMalloc( (void**) &d_q, dim * sizeof( float ) ) );
	checkCudaErrors( cudaMalloc( (void**) &d_tmp, dim * sizeof( float ) ) );
	
	// copy host to device
	checkCudaErrors( cudaMemcpy( d_A, h_A, dim * dim * sizeof( float ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy( d_b, h_b, dim * sizeof( float ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy( d_x, h_x, dim * sizeof( float ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaDeviceSynchronize() );

	

	// init CG
	// ALGORITHM: r_0 = b-Ax_0
	// r_0 = Ax_0 - b
	_cl_matrix_vector_<<< nBlocks, nThreads, nThreads*sizeof(float) >>>( CL_SUB, d_A, d_x, d_b, d_r, dim );
	checkCudaErrors( cudaDeviceSynchronize() );
	
	
	// r_0 = -r_0
	_cl_vector_op_<<< nBlocks, nThreads >>>( NONE, -1.0f, 0.0f, d_r, NULL, d_r, dim );
	checkCudaErrors( cudaDeviceSynchronize() );

	// p_0 = r_0
	_cl_vector_op_<<< nBlocks, nThreads >>>( NONE,  1.0f, 0.0f, d_r, NULL, d_p, dim );
	checkCudaErrors( cudaDeviceSynchronize() );
	
	// CG needs max dim iterations
	int i = 0;
	float minRho = 1000000000;
	for( i = 0; i < dim; i++ ){	
		
		// rho_k = sum(r_k * r_k)
		rho = gpuReduceSUM(d_r, d_r, d_tmp, dim, nBlocks, nThreads );
		checkCudaErrors( cudaDeviceSynchronize() );
		
		if (minRho > rho) {
			minRho = rho;
		}
		
		//printf("iteration #%d, with rho = %f", i, rho);
		std::cout << "iteration #" << i << ", with rho = " << rho << "          " << '\r' << std::flush;
		// check here for criterion
		if( rho < errorTolerance) {
			break;
		}
		
		// q_k = A*p_k
		_cl_matrix_vector_<<< nBlocks, nThreads, nThreads * sizeof(float) >>>( NONE, d_A, d_p, NULL, d_q, dim );
		checkCudaErrors( cudaDeviceSynchronize() );
		
		// alpha_k = rho_k / sum(p_k * q_k)
		alpha = rho / gpuReduceSUM(d_p, d_q, d_tmp, dim, nBlocks, nThreads );
		checkCudaErrors( cudaDeviceSynchronize() );
		
		 // x_(k+1) = x_k + alpha_k * p_k
		_cl_vector_op_<<< nBlocks, nThreads >>>( CL_ADD, 1.0f, alpha, d_x, d_p, d_x, dim );
		checkCudaErrors( cudaDeviceSynchronize() );
		
		// r_(k+1) = r_k + (-alpha_k * q_k)
		_cl_vector_op_<<< nBlocks, nThreads >>>( CL_ADD, 1.0f, -alpha, d_r, d_q, d_r, dim );
		checkCudaErrors( cudaDeviceSynchronize() );

		// beta_k = sum(r_(k+1) * r_(k+1)) / rho_k
		beta = gpuReduceSUM(d_r, d_r, d_tmp, dim, nBlocks, nThreads ) / rho;
		checkCudaErrors( cudaDeviceSynchronize() );
		
		// p_(k+1) = r_(k+1) + beta_k * p_k
		_cl_vector_op_<<< nBlocks, nThreads >>>( CL_ADD, 1.0f, beta, d_r, d_p, d_p, dim );
		checkCudaErrors( cudaDeviceSynchronize() );
	}

	rho = gpuReduceSUM(d_r, d_r, d_tmp, dim, nBlocks, nThreads );

	printf("\nSolution found at iteration #%d, with rho = %f\n", i, rho);
	printf("\nminrho was %f\n", minRho);
	
	// copy solution from device to host
	checkCudaErrors( cudaMemcpy( h_x, d_x, dim * sizeof( float ), cudaMemcpyDeviceToHost ) );

	// release device memory
	checkCudaErrors( cudaFree( d_A ) );
	checkCudaErrors( cudaFree( d_b ) );
	checkCudaErrors( cudaFree( d_x ) );
	checkCudaErrors( cudaFree( d_r ) );
	checkCudaErrors( cudaFree( d_p ) );
	checkCudaErrors( cudaFree( d_q ) );
	checkCudaErrors( cudaFree( d_tmp ) );
}
