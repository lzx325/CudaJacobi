set -e
nvcc -o test main.cpp matrixkernels.cu iterative_solver_cpu.cpp iterative_solver_gpu.cu
./test 