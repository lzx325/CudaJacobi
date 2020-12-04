#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include <sstream>
#include <algorithm> 
#include <array>
#include <assert.h>
#include <fstream>
#include <chrono>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "helper_cuda.h"

#include "matrixkernels.cuh"
#include "iterative_solver_cpu.h"
#include "iterative_solver_gpu.cuh"

enum MAJORITY {
	ROW_MAJOR = 0,
	COLUMN_MAJOR = 1
};

bool readMatrix(char* filename, float* &matrix, unsigned int *dim = NULL, int majority = ROW_MAJOR)
{
	unsigned int w, h, x, y, num_entries;

	float val;

	std::ifstream file(filename);

	if (file)
	{
		file >> h >> w >> num_entries;
		std::cout << w << " " << h << " " << num_entries << "\n";

		assert(w == h || w == 1 || h == 1);

		if (dim != NULL) {
			*dim = w;
			if (h > w) { *dim = h; }
		}

		matrix = new float[w * h];
		memset(matrix, 0, num_entries * sizeof(float));

		unsigned int i;
		for (i = 0; i < num_entries; i++) {

			if (file.eof()) break;

			file >> y >> x >> val;

			if (majority == ROW_MAJOR) {

				matrix[w * y + x] = val;

			}
			else if (majority == COLUMN_MAJOR) {

				matrix[h * x + y] = val;
			}
		}
		file.close();

		if (i == num_entries)
			std::cout << "\nFile read successfully\n";
		else
			std::cout << "\nFile read successfully but seems defective:\n num entries read = " << i << ", entries epected = " << num_entries << "\n";
	}
	else {
		std::cout << "Unable to open file\n";
		return false;
	}

	return true;
}
int main(){
    // float A[] = {
    //     8,-3,2,
    //     4,11,-1,
    //     6,3,12
    // };

    // float b[] = {
    //     20,33,36
    // };

    // float x[3] ={};
    float *A, *b,*x;
    unsigned int dim;
    bool success=false;
    success = readMatrix((char *)"matrices/dd-A-512x512.txt", A, &dim, ROW_MAJOR);
	success = success && readMatrix((char *)"matrices/dd-b-512x1.txt", b);
    assert(success);
    x=new float[dim];
    std::fill(x,x+dim,0);
    auto start1 = std::chrono::high_resolution_clock::now();
	GS_gpu(x,A,b,(int) dim,1000);
	auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff1 = end1-start1;
    std::cout<<"Time elapsed: "<<std::chrono::duration_cast<std::chrono::milliseconds>(diff1).count()/1000.f<<std::endl;
    
    computeResultError(A,b,x,(int) dim);

    delete[] x;
    return 0;
}