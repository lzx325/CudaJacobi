#include "helper_cuda.h"
#include "iterative_solver_gpu.cuh"
#include "matrixkernels.cuh"
#include <iostream>
void computeResultError(float *h_A, float *h_b, float *h_x, unsigned int dim)
{
	float* h_e0 = new float[dim];
	multiplyMatrixVector(h_A, h_x, h_e0, dim);

	float sum_abs_diff_0 = 0;
	float max_abs_diff_0 = 0;

	for (unsigned int i = 0; i < dim; i++) {

		float abs_diff_0 = h_e0[i] - h_b[i];
		abs_diff_0 = abs_diff_0 > 0.0f ? abs_diff_0 : -abs_diff_0;

		if (abs_diff_0 > max_abs_diff_0)
		{
			max_abs_diff_0 = abs_diff_0;
		}
		sum_abs_diff_0 += abs_diff_0;
	}
	
	std::cout << "errors:\n";
	
	std::cout << "sum  | " << sum_abs_diff_0 << "\navgs | " << sum_abs_diff_0 / dim << "\nmax  | " << max_abs_diff_0 << std::endl << std::endl;

	delete[] h_e0;
}

__global__ void jacobi_one_iter_gpu(float* dx, float* A, float* b, float*x, int dim){
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	while(i<dim){
		dx[i]=b[i];
		for(int j=0;j<dim;j++){
			dx[i]-=A[i*dim+j]*x[j];
		}
		dx[i]/=A[i*dim+i];
		i+=blockDim.x*gridDim.x;
	}
}

void jacobi_gpu(float* h_x, float* h_A, float* h_b,  int dim, int n_it){
	float *d_x, *d_A, *d_b, *d_dx;
	checkCudaErrors( cudaMalloc( (void**) &d_x, dim * sizeof( float ) ) );
	checkCudaErrors( cudaMalloc( (void**) &d_A, dim * dim * sizeof( float ) ) );
	checkCudaErrors( cudaMalloc( (void**) &d_b, dim * sizeof( float ) ) );
	checkCudaErrors( cudaMalloc( (void**) &d_dx, dim * sizeof( float ) ) );

	checkCudaErrors( cudaMemcpy( d_x, h_x, dim * sizeof( float ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy( d_A, h_A, dim * dim * sizeof( float ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy( d_b, h_b, dim * sizeof( float ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaDeviceSynchronize() );

	
	int nThreads = 128;							
	int nBlocks = iDivUp( dim, nThreads );
	for(int iter=0;iter<n_it;iter++){
		jacobi_one_iter_gpu <<< nThreads, nBlocks >>> (d_dx,d_A,d_b,d_x,dim);
		checkCudaErrors( cudaDeviceSynchronize() );
	
		_cl_vector_op_ <<< nThreads, nBlocks >>>( CL_ADD, 1, 1, d_x, d_dx, d_x, dim );
		checkCudaErrors( cudaDeviceSynchronize() );
	}
	checkCudaErrors( cudaMemcpy( h_x, d_x, dim * sizeof( float ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaFree( d_A ) );
	checkCudaErrors( cudaFree( d_b ) );
	checkCudaErrors( cudaFree( d_x ) );
}

__global__ void GS_one_iter_gpu(float* dx, float* A, float* b, float*x, int dim){
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	while(i<dim){
		dx[i]=b[i];
		for(int j=0;j<dim;j++){
			dx[i]-=A[i*dim+j]*x[j];
		}
		i+=blockDim.x*gridDim.x;
	}
}

void GS_one_iter_cpu_remain(float* dx, float* A, float* b,int dim){
	for(int i=0;i<dim;i++){
        for(int j=0;j<i;j++){
            dx[i]-=A[i*dim+j]*(dx[j]);
        }
        dx[i]/=A[i*dim+i];
    }
}

void GS_gpu(float* h_x, float* h_A, float* h_b,  int dim, int n_it){
    float* h_dx= new float[dim];
	float *d_x, *d_A, *d_b, *d_dx;
	checkCudaErrors( cudaMalloc( (void**) &d_x, dim * sizeof( float ) ) );
	checkCudaErrors( cudaMalloc( (void**) &d_A, dim * dim * sizeof( float ) ) );
	checkCudaErrors( cudaMalloc( (void**) &d_b, dim * sizeof( float ) ) );
	checkCudaErrors( cudaMalloc( (void**) &d_dx, dim * sizeof( float ) ) );

	checkCudaErrors( cudaMemcpy( d_x, h_x, dim * sizeof( float ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy( d_A, h_A, dim * dim * sizeof( float ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaMemcpy( d_b, h_b, dim * sizeof( float ), cudaMemcpyHostToDevice ) );
	checkCudaErrors( cudaDeviceSynchronize() );

	
	int nThreads = 128;							
	int nBlocks = iDivUp( dim, nThreads );
	for(int iter=0;iter<n_it;iter++){
		GS_one_iter_gpu <<< nThreads, nBlocks >>> (d_dx,d_A,d_b,d_x,dim);
		checkCudaErrors( cudaMemcpy( h_dx, d_dx, dim * sizeof( float ), cudaMemcpyDeviceToHost ) );
		GS_one_iter_cpu_remain(h_dx, h_A, h_b, dim);
		checkCudaErrors( cudaMemcpy( d_dx, h_dx, dim * sizeof( float ), cudaMemcpyHostToDevice ) );

		_cl_vector_op_ <<< nThreads, nBlocks >>> ( CL_ADD, 1, 1, d_x, d_dx, d_x, dim );
		checkCudaErrors( cudaDeviceSynchronize() );
	}
	checkCudaErrors( cudaMemcpy( h_x, d_x, dim * sizeof( float ), cudaMemcpyDeviceToHost ) );
	checkCudaErrors( cudaFree( d_A ) );
	checkCudaErrors( cudaFree( d_b ) );
	checkCudaErrors( cudaFree( d_x ) );
	delete[] h_dx;
}