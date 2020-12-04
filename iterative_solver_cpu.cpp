#include <cstdio>
#include <algorithm>

void jacobi_one_iter(float* dx, float* A, float* b, float*x, int dim){
    for(int i=0;i<dim;i++){
        dx[i]=b[i];
        for(int j=0;j<dim;j++){
            dx[i]-=A[i*dim+j]*x[j];
            
        }
        

        dx[i]/=A[i*dim+i];
        
    }

    
}

void add(float* z,float *x, float *y,  int sz){
    for(int i=0;i<sz;i++){
        z[i]=x[i]+y[i];
    }
}

void abs_sum(float * res, float *x, int sz){
    *res=0.f;
    for(int i=0;i<sz;i++){
        *res+=fabsf(x[i]);
    }
}
void jacobi_cpu(float* x, float* A, float* b, int dim, int maxit, float eps=1e-4){
    float* dx= new float[dim];
    float sum=0;
    bool convergence=false;
    
    for(int k=0;k<maxit;k++){
        jacobi_one_iter(dx,A,b,x,dim);
        add(x,x,dx,dim);
        abs_sum(&sum,dx,dim);
        // if(sum<eps){
        //     printf("Exit due to convergence. n_iter=%d\n",k);
        //     convergence=true;
        //     break;
        // }
    }
    
    if(!convergence){
        printf("Exit due to maxit. sum=%g\n",sum);
    }
    delete[] dx;
}

void GS_one_iter(float* dx, float* A, float* b, float*x, int dim){
    for(int i=0;i<dim;i++){
        dx[i]=b[i];
        for(int j=0;j<dim;j++){
            dx[i]-=A[i*dim+j]*(x[j]+(j<i)*dx[j]);
        }
        dx[i]/=A[i*dim+i];
    }
}

void GS_cpu(float* x, float* A, float* b, int dim, int maxit, float eps=1e-4){
    float* dx= new float[dim];
    float sum=0;
    bool convergence=false;
    for(int k=0;k<maxit;k++){
        GS_one_iter(dx,A,b,x,dim);
        add(x,x,dx,dim);
        abs_sum(&sum,dx,dim);
        // if(sum<eps){
        //     printf("Exit due to convergence. n_iter=%d\n",k);
        //     convergence=true;
        //     break;
        // }
    }
    if(!convergence){
        printf("Exit due to maxit. sum=%g\n",sum);
    }

    delete[] dx;
}