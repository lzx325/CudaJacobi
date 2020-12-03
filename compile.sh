set -e
nvcc -o test jacobi.cu
./test -f matrix_512.txt