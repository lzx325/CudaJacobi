import numpy as np
import sys
def print_coo_list(mat,file):
    n_rows,n_cols=mat.shape
    print(f"{n_rows} {n_cols} {n_rows*n_cols}",file=file)
    for i in range(n_rows):
        for j in range(n_cols):
            print(f"{i} {j} {mat[i,j]:.3f}",file=file)
if __name__=="__main__":
    dim=int(sys.argv[1])
    A=np.random.rand(dim,dim)
    A[range(dim),range(dim)]+=A.sum(axis=1)
    b=np.random.rand(dim,1)
    with open("./matrices/dd-A-%dx%d.txt"%(dim,dim),'w') as f:
        print_coo_list(A,f)
    with open("./matrices/dd-b-%dx1.txt"%(dim),'w') as f:
        print_coo_list(b,f)