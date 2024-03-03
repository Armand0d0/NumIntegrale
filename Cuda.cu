#include <iostream>
#include "Cuda.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h> 
#include <stdio.h>
#include <fstream>
#include <cmath>
using namespace std;

Cuda::Cuda(){
	/*int devID;
    cudaDeviceProp deviceProps;

    devID = findCudaDevice(0, 0);

    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
    printf("CUDA device [%s]\n", deviceProps.name);//*/ 
}

double eval(Cuda::func* fun, double x, double  (*F[])(double)){////////////////////////////////////////////////////////////////////////
	if(fun == NULL){
		printf("gggoog\n"); cout<<"errr§§§!!!!!"<<endl;
		return -1;
	}
    if(fun->func1 != NULL && fun->func2 != NULL){ 
        if(fun->opType == COMP){
           return eval(fun->func1,   eval(fun->func2,x,F)   ,F);

        }else if(fun->opType == MUL){
            return eval(fun->func1,x,F) * eval(fun->func2,x,F);
        }else if(fun->opType == SUM){
            return  eval(fun->func1,x,F) + eval(fun->func2,x,F);
        }
    }else if(fun->func2 != NULL){
        if(fun->opType == COMP){
           return F[fun->funcId1](  eval(fun->func2,  x  ,F)   );

        }else if(fun->opType == MUL){
            return  F[fun->funcId1](x) * eval(fun->func2,x,F);
        }else if(fun->opType == SUM){
            return  F[fun->funcId1](x) + eval(fun->func2,x,F);
        }
    }else{
        if(fun->opType == COMP){
           return F[fun->funcId1](   F[fun->funcId2](x)   );

        }else if(fun->opType == MUL){
            return F[fun->funcId1](x) * F[fun->funcId2](x);
        }else if(fun->opType == SUM){
            return F[fun->funcId1](x) + F[fun->funcId2](x);
        }
    }*/	

    return -1;
}
void printFunc(Cuda::func* f){
	char c=0;
    if(f->opType==COMP){
      c = 'o';
    }else if(f->opType==MUL){
        c='*';
    }else if(f->opType==SUM){
        c='+';
    }
    if(f->func1!=NULL && f->func2!=NULL){
        cout<<"(";
        printFunc(f->func1);
        cout<<") "<<c<<" (";
        printFunc(f->func2);
        cout<<")";
    }else if(f->func2 != NULL){
        cout<<" (f"<<f->funcId1<<c<<"(";
        printFunc(f->func2);
        cout<<"))";
    }else{
        cout<<"(f"<<f->funcId1<<c<<"f"<<f->funcId2<<")";
    }
}/*
double Cuda::show2(Cuda::func* fun[], int n,double* b, int nb,double  (*FU[])(double) ,double goal,double dist_goal,int iter){////////
    double near=0;
    double min=INFINITY;
    Cuda::func *fx;
    int b1 = -1,b2 = -1;
    for(int i=0;i<n;i++){
    // cout<<d[i]<<"  ";
    for(int j=0;j<nb;j++){
    //    for(int k=j+1;k<nb;k++){
            double ev = abs(eval(fun[i],b[j+1],FU)-eval(fun[i],b[j],FU));
                double di = abs( ev - goal);
                if(di<min){
                    min = di;
                    near = ev;
                    fx = fun[i];
                    b1 = j;
                    b2 = j+1;//k
                }
               
       // }
    }
    }if(min < 1e-7){
		cout<<"near : "<<near<< " iter : "<< iter <<endl;
	}
	if(min < dist_goal){
		cout<<endl<<"near : "<<near<<"  fonction : "; printFunc(fx);
		cout<<endl<<" bornes : "<<b1<<"  " << b2<<endl;
		cout<<"dist : "<<min<<endl;
	}
	return min;
}*/
double Cuda::show3(Cuda::func* fun,double* b, int nb,double  (*FU[])(double) ,double goal,double dist_goal/*,int iter*/){////////
  double near=0;
    double min=INFINITY;
    int b1 = -1,b2 = -1;
     for(int j=0;j<nb-1;j++){
    //    for(int k=j+1;k<nb;k++){

            double ev =abs(eval(fun,b[j+1],FU)-eval(fun,b[j],FU));

                double di = abs( ev - goal);
                if(di<min){
                    min = di;
                    near = ev;
                    b1 = j;
                    b2 = j+1;//k
                }
               
       // }
			}


    if(min < 1e-7){
		cout<<"near : "<<near <<endl;
	}
	if(min < dist_goal){
		cout<<endl<<"near : "<<near<<"  fonction : "; printFunc(fun);
		cout<<endl<<" bornes : "<<b1<<"  " << b2<<endl;
		cout<<"dist : "<<min<<endl;
	}
	return 0;
}
/*
__global__ void evalCuKern(Cuda::func* fun,double x,double  (*FU[])(double), double* ret){
	//int tid = blockDim.x * blockIdx.x + threadIdx.x;
		double r1,r2;
	if(fun->func1 != NULL && fun->func2 != NULL){
        if(fun->opType == 0){
			evalCuKern<<<1,1,0,0>>>(fun->func2,x,FU,&r1);
            evalCuKern<<<1,1,0,0>>>(fun->func1,   r1   ,FU,ret);

        }else if(fun->opType == 1){
            evalCuKern<<<1,1,0,0>>>(fun->func1,x,FU,&r1);
			evalCuKern<<<1,1,0,0>>>(fun->func2,x,FU,&r2);
			*ret = r1*r2;
        }else if(fun->opType == 2){
             evalCuKern<<<1,1,0,0>>>(fun->func1,x,FU,&r1);
			 evalCuKern<<<1,1,0,0>>>(fun->func2,x,FU,&r2);
			 *ret = r1+r2;
        }
    }else if(fun->func2 != NULL){
        if(fun->opType == 0){
			evalCuKern<<<1,1,0,0>>>(fun->func2,  x  ,FU,&r1);
           *ret = FU[fun->funcId1](  r1   );

        }else if(fun->opType == 1){
			evalCuKern<<<1,1,0,0>>>(fun->func2,x,FU,&r1);
            *ret = r1;
			*ret *=  FU[fun->funcId1](x);
        }else if(fun->opType == 2){
			evalCuKern<<<1,1,0,0>>>(fun->func2,x,FU,&r1);
            *ret =  FU[fun->funcId1](x) + r1 ;
        }
    }else{
        if(fun->opType == 0){
           *ret = FU[fun->funcId1](   FU[fun->funcId2](x)   );

        }else if(fun->opType == 1){
            *ret = FU[fun->funcId1](x) * FU[fun->funcId2](x);
        }else if(fun->opType == 2){
            *ret = FU[fun->funcId1](x) + FU[fun->funcId2](x);
        }
    }
	
}

double Cuda::evalCu(Cuda::func* fun, double x, double  (*FU[])(double)){

	//size_t sizeFun = sizeof(Cuda::func)*n;
	//size_t sizeB = sizeof(double)*nb;
	//float* d_f;	checkCudaErrors(cudaMalloc(&d_f,sizeFun));
	//float* F;   checkCudaErrors(cudaMallocHost(&F,sizeFun));
	//float* d_b;	checkCudaErrors(cudaMalloc(&d_b,sizeB));
	//float* B;   checkCudaErrors(cudaMallocHost(&B,sizeB));
	//checkCudaErrors(cudaMemcpyAsync(d_f,fun, sizeFun, cudaMemcpyHostToDevice,0));
	//checkCudaErrors(cudaMemcpyAsync(d_b,b, sizeB, cudaMemcpyHostToDevice,0));
		double ret = -1;
			evalCuKern<<<1,1,0,0>>>(fun,x,FU,&ret);
	
	//checkCudaErrors(cudaMemcpyAsync(F,d_f, sizeFun, cudaMemcpyDeviceToHost, 0));
	//checkCudaErrors(cudaMemcpyAsync(F,d_f, sizeFun, cudaMemcpyDeviceToHost, 0));
	//checkCudaErrors( cudaPeekAtLastError() );
	//checkCudaErrors(cudaDeviceSynchronize());
		
			//	checkCudaErrors(cudaFreeHost(A));
			//	checkCudaErrors(cudaFree(d_f));
	return ret;
}
/*
__global__ void SumArrKernel(float* A,float* S,int N){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
		if(tid<N/2){
		S[tid] += A[tid*2]+A[tid*2+1];
		}
	
}
/*
void showArrCuda(float* f,int l){
	for(int i=0;i<l;i++){
				//cout<< "i="<<i<<"  "<<f[i]<<endl;
				cout<<f[i]<<"   ";
				if(i%10==0)cout<<endl;
			}
		cout<<endl<<endl;
	}
__global__ void SumArrKernel(float* A,int N){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid<N){
	A[tid] = A[tid*2]+A[tid*2+1];
	}
}
float Cuda::SumArr(float* a,int N){
	
	size_t size = sizeof(float)*N;
	float* d_a;	checkCudaErrors(cudaMalloc(&d_a,size));
	float* A;   checkCudaErrors(cudaMallocHost(&A,sizeof(float)));
	checkCudaErrors(cudaMemcpyAsync(d_a,a, size, cudaMemcpyHostToDevice,0));////////
	while(N!=0){
		SumArrKernel<<<5,256,0,0>>>(d_a,N);
		if(N%2 != 0){  a[0]+=a[N-1];  N--; }		N/=2;

	}
	checkCudaErrors(cudaMemcpyAsync(A,d_a, sizeof(float), cudaMemcpyDeviceToHost, 0));
	checkCudaErrors( cudaPeekAtLastError() );
	checkCudaErrors(cudaDeviceSynchronize());

	float res = A[0];
		checkCudaErrors(cudaFreeHost(A));
		checkCudaErrors(cudaFree(d_a));

	return res;

}//*/
/*
__global__ void SumArrKernel(float* A,float* S,int N){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
		if(tid<N/2){
		S[tid] += A[tid*2]+A[tid*2+1];
		}
	
}
float Cuda::SumArr(float* a,int N){
	
	size_t size = sizeof(float)*N;
	float* d_a;	checkCudaErrors(cudaMalloc(&d_a,size));
	float* A;   checkCudaErrors(cudaMallocHost(&A,size));
	checkCudaErrors(cudaMemcpyAsync(d_a,a, size, cudaMemcpyHostToDevice,0));
	while(N>0){
		SumArrKernel<<<5,256,0,0>>>(d_a,A,N);
		if(N%2 != 0){  a[0]+=a[N-1];  N--; } N/=2;
	}
	checkCudaErrors(cudaMemcpyAsync(A,d_a, size, cudaMemcpyDeviceToHost, 0));
	checkCudaErrors( cudaPeekAtLastError() );
	checkCudaErrors(cudaDeviceSynchronize());

	float res = A[0];
		checkCudaErrors(cudaFreeHost(A));
		checkCudaErrors(cudaFree(d_a));

	return res;

}
__global__ void Normalize(unsigned char* in,float* out,float mean,int N){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i<N){
		out[i] = (((float)in[i]-mean)/255.0f);
	}

}


float*  Cuda::normalize(unsigned char* data){
	int N = 784;
size_t sizein = sizeof(unsigned char)*N;
size_t sizeout = sizeof(float)*N;
float *res ;			checkCudaErrors(cudaMallocHost(&res,sizeout)); //(float*)malloc(sizeout);
unsigned char *d_data;		checkCudaErrors(cudaMalloc(&d_data,sizein));
float *d_res;				checkCudaErrors(cudaMalloc(&d_res,sizeout));
		float mean=255/2;	//	for(int i=0;i<N;i++) {mean+=data[i];} 	mean/=((float)N);
	  checkCudaErrors(cudaMemcpyAsync(d_data,data, sizein, cudaMemcpyHostToDevice,0));
        Normalize<<<4, 256,0,0>>>(d_data, d_res,mean,N);//--------------------------
	  checkCudaErrors(cudaMemcpyAsync(res, d_res, sizeout, cudaMemcpyDeviceToHost,0));
	  checkCudaErrors( cudaPeekAtLastError() );
	  checkCudaErrors(cudaDeviceSynchronize());
		
      checkCudaErrors(cudaFree(d_data));
	  checkCudaErrors(cudaFree(d_res));
	  return res;
}
__global__ void SquareKernel(float* A,int N){///////////////////////////////
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid<N){
	A[tid]=A[tid]*A[tid];
	}
}
float* Cuda::Square(float* a,int N){
	size_t size = sizeof(float)*N;
	float* d_a;	checkCudaErrors(cudaMalloc(&d_a,size));
	float* A;   checkCudaErrors(cudaMallocHost(&A,size));

	checkCudaErrors(cudaMemcpyAsync(d_a,a, size, cudaMemcpyHostToDevice,0));
	SquareKernel<<<5,256,0,0>>>(d_a,N);
	checkCudaErrors(cudaMemcpyAsync(A,d_a, size, cudaMemcpyDeviceToHost, 0));
	checkCudaErrors( cudaPeekAtLastError() );
	checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaFree(d_a));
		return A;
}
__global__ void MatrixVectorMulKernel(Cuda::Matrix A, Cuda::Matrix B, Cuda::Matrix C,int maxThreads){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
		if(tid<maxThreads){
			float acc = 0;
			for(int i=0;i<A.w;i++ ){
				acc+=A.arr[tid*A.w+i]*B.arr[i];

			}
			C.arr[tid]=acc;
		}

}

float* Cuda::MatrixVectorMul(Cuda::Matrix A,Cuda::Matrix B){
	int maxThreads = A.h;
	size_t sizeA = sizeof(float)*A.h*A.w;
	size_t sizeB = sizeof(float)*B.h*1;
	size_t sizeC = sizeof(float)*A.h*1;
	Cuda::Matrix d_A ;		d_A.w = A.w; d_A.h = A.h;	checkCudaErrors(cudaMalloc(&d_A.arr,sizeA));
	Cuda::Matrix d_B ;		d_B.w = 1; d_B.h = B.h;	checkCudaErrors(cudaMalloc(&d_B.arr,sizeB));
	Cuda::Matrix d_C ;		d_C.w = 1; d_C.h = A.h;	checkCudaErrors(cudaMalloc(&d_C.arr,sizeC));
	Cuda::Matrix C ;		C.w = 1;   C.h = A.h;		checkCudaErrors(cudaMallocHost(&C.arr,sizeC));
	checkCudaErrors(cudaMemcpyAsync(d_A.arr,A.arr, sizeA, cudaMemcpyHostToDevice,0));
	checkCudaErrors(cudaMemcpyAsync(d_B.arr,B.arr, sizeB, cudaMemcpyHostToDevice,0));
			MatrixVectorMulKernel<<<1,256,0,0>>>(d_A,d_B,d_C,maxThreads);
	checkCudaErrors(cudaMemcpyAsync(C.arr, d_C.arr, sizeC, cudaMemcpyDeviceToHost,0));
		checkCudaErrors( cudaPeekAtLastError() );
	 	checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaFree(d_A.arr));
	  	checkCudaErrors(cudaFree(d_B.arr));
		checkCudaErrors(cudaFree(d_C.arr));
		return C.arr;  
}
__global__ void VecSumKernel(float* A,float* B,float* C,int N){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid<N){
		C[tid]=A[tid]+B[tid];
	}
}
float* Cuda::VecSum(float* a,float* b,int N){
	size_t size = N*sizeof(float);
	float* d_a; checkCudaErrors(cudaMalloc(&d_a,size));
	float* d_b; checkCudaErrors(cudaMalloc(&d_b,size));
	float* d_c; checkCudaErrors(cudaMalloc(&d_c,size));
	float* C; checkCudaErrors(cudaMallocHost(&C,size));

	checkCudaErrors(cudaMemcpyAsync(d_a,a, size, cudaMemcpyHostToDevice,0));
	checkCudaErrors(cudaMemcpyAsync(d_b,b, size, cudaMemcpyHostToDevice,0));
			VecSumKernel<<<2,256,0,0>>>(d_a,d_b,d_c,N);
	checkCudaErrors(cudaMemcpyAsync(C, d_c, size, cudaMemcpyDeviceToHost,0));
		checkCudaErrors( cudaPeekAtLastError() );
		checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaFree(d_a));
		checkCudaErrors(cudaFree(d_b));
		checkCudaErrors(cudaFree(d_c));
	return C;
}
__global__ void VectorScalarMulKernel(float* A,float B,float* C,int N){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid<N){
		C[tid]=A[tid]*B;
	}
}
float* Cuda::VectorScalarMul(float* a,float b,int N){
	size_t size = N*sizeof(float);
	float* d_a; checkCudaErrors(cudaMalloc(&d_a,size));
	float* d_c; checkCudaErrors(cudaMalloc(&d_c,size));
	float* C; checkCudaErrors(cudaMallocHost(&C,size));

	checkCudaErrors(cudaMemcpyAsync(d_a,a, size, cudaMemcpyHostToDevice,0));
		VectorScalarMulKernel<<<2,256,0,0>>>(d_a,b,d_c,N);
	checkCudaErrors(cudaMemcpyAsync(C, d_c, size, cudaMemcpyDeviceToHost,0));
		checkCudaErrors( cudaPeekAtLastError() );
		checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaFree(d_a));
		checkCudaErrors(cudaFree(d_c));
	return C;
}

__global__ void ReluKernel(float* A,int N){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid<N){
	A[tid]=max(A[tid],0.0f);
	}
}
float* Cuda::Relu(float* in,int N){
	size_t size = sizeof(float)*N;
	float* d_a;	checkCudaErrors(cudaMalloc(&d_a,size));
	float* A;   checkCudaErrors(cudaMallocHost(&A,size));

	checkCudaErrors(cudaMemcpyAsync(d_a,in, size, cudaMemcpyHostToDevice,0));
		ReluKernel<<<5,256,0,0>>>(d_a,N);
	checkCudaErrors(cudaMemcpyAsync(A,d_a, size, cudaMemcpyDeviceToHost, 0));
	checkCudaErrors( cudaPeekAtLastError() );
	checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaFree(d_a));
	return A;
}
#define Beta 0.01
__global__ void SoftMaxKernel(float* A,float sum,int N){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid<N){
		A[tid]= exp(A[tid]*Beta)/sum;
	}
}
__global__ void ExponentialKernel(float* A,float* B,int N){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid<N){
		B[tid] = exp(A[tid]*Beta);
	}
}
float* Cuda::SoftMax(float* a,int N){
	size_t size = sizeof(float)*N;
	float* d_a;	checkCudaErrors(cudaMalloc(&d_a,size));
	float* d_e;	checkCudaErrors(cudaMalloc(&d_e,size));
	float* A;   checkCudaErrors(cudaMallocHost(&A,size));
	float* E;   checkCudaErrors(cudaMallocHost(&E,size));

	checkCudaErrors(cudaMemcpyAsync(d_a,a, size, cudaMemcpyHostToDevice,0));
		ExponentialKernel<<<5,256,0,0>>>(d_a,d_e,N);
	checkCudaErrors(cudaMemcpyAsync(E,d_e, size, cudaMemcpyDeviceToHost, 0));
			float s = SumArr(E,N); 
		SoftMaxKernel<<<5,256,0,0>>>(d_a,s,N);
	checkCudaErrors(cudaMemcpyAsync(A,d_a, size, cudaMemcpyDeviceToHost, 0));
	
	checkCudaErrors( cudaPeekAtLastError() );
	checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaFreeHost(E));
		checkCudaErrors(cudaFree(d_a));
		checkCudaErrors(cudaFree(d_e));
	return A;
}

__global__ void CLLWNG(float* S,float* A,float* y,float* res,int K,int J,int maxThreads){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid<maxThreads){
	int k = tid/K, j = tid%K; //tid = K * k + j
	float s = S[j];
	float sum = 0;//for i!=j (when this weight act on neurons as denominator in softmax function)
	for(int i=0;i<J;i++){																		//
	sum += 2*(s-y[j])*	(-Beta)*S[i]*S[j]  *  A[k];												//																								
	}																							//
	//^^^TODO : int value =2*(s-y[j])*	(-Beta)*S[j]  *  A[k]; for(...){sum += value*S[i];}	
	float deriv = 2*(s-y[j]) * Beta * s * ( 1 - s ) * A[k];//for i=j (when this weight act on neurons as numerator in softmax function)
	res[tid] =    -( 	deriv  +   sum   );
	}
}
float* Cuda::computeLastLayerWeightsNegGradient(float* z,float* a,float* bias4,float* l,int K,int J){
int sizeJ = sizeof(float)*J;
int sizeK = sizeof(float)*K;
int sizeR = sizeof(float)*J*K;
float* d_z; 	checkCudaErrors(cudaMalloc(&d_z,sizeJ));
float* d_e;		checkCudaErrors(cudaMalloc(&d_e,sizeJ));
float* d_al_1;  checkCudaErrors(cudaMalloc(&d_al_1,sizeK));
float* d_y;		checkCudaErrors(cudaMalloc(&d_y,sizeJ));
float* d_res;	checkCudaErrors(cudaMalloc(&d_res,sizeR));
float* E;		checkCudaErrors(cudaMallocHost(&E,sizeJ));
float* R; 		checkCudaErrors(cudaMallocHost(&R,sizeR));
float* Z;		checkCudaErrors(cudaMallocHost(&Z,sizeJ));
int maxThreads = J*K;
checkCudaErrors(cudaMemcpyAsync(d_z,z, sizeJ, cudaMemcpyHostToDevice,0));
	ExponentialKernel<<<2,256,0,0>>>(d_z,d_e,J);
checkCudaErrors(cudaMemcpyAsync(E,d_e, sizeJ, cudaMemcpyDeviceToHost, 0));
			float sum = SumArr(E,J);
SoftMaxKernel<<<2,256>>>(d_z,sum,J);//
checkCudaErrors(cudaMemcpyAsync(Z,d_z, sizeJ, cudaMemcpyDeviceToHost, 0));
	showArrCuda(Z,J);
checkCudaErrors(cudaMemcpyAsync(d_al_1,a, sizeK, cudaMemcpyHostToDevice,0));
checkCudaErrors(cudaMemcpyAsync(d_y,l, sizeJ, cudaMemcpyHostToDevice,0));
	CLLWNG<<<5,512,0,0>>>(d_z,d_al_1,d_y,d_res,K,J,maxThreads);
checkCudaErrors(cudaMemcpyAsync(R,d_res, sizeR, cudaMemcpyDeviceToHost, 0));
	checkCudaErrors( cudaPeekAtLastError() );
	checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaFreeHost(Z));
		checkCudaErrors(cudaFreeHost(E));
		checkCudaErrors(cudaFree(d_z));
		checkCudaErrors(cudaFree(d_e));
		checkCudaErrors(cudaFree(d_al_1));
		checkCudaErrors(cudaFree(d_y));
		checkCudaErrors(cudaFree(d_res));

return R;
}
__global__ void CLLWNG2(float* S,float* A,float* y,float* res,int K,int maxThreads){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid<maxThreads){
	int k = tid/K, j = tid%K; //tid = K * k + j
	float s = S[j];
	float dS = s*(1-s);
	res[tid] =    -( 2*Beta*Beta * A[k]*A[k] * dS*( dS + ( (s-y[j])*(1-(2*s)) ) ) 	);
	}
}
float* Cuda::computeLastLayerWeightsNegGradient2(float* z,float* a,float* bias4,float* l,int K,int J){
	int sizeJ = sizeof(float)*J;
	int sizeK = sizeof(float)*K;
	int sizeR = sizeof(float)*J*K;
	float* d_z; 	checkCudaErrors(cudaMalloc(&d_z,sizeJ));
	float* d_e;		checkCudaErrors(cudaMalloc(&d_e,sizeJ));
	float* d_al_1;  checkCudaErrors(cudaMalloc(&d_al_1,sizeK));
	float* d_y;		checkCudaErrors(cudaMalloc(&d_y,sizeJ));
	float* d_res;	checkCudaErrors(cudaMalloc(&d_res,sizeR));
	float* E;		checkCudaErrors(cudaMallocHost(&E,sizeJ));
	float* R; 		checkCudaErrors(cudaMallocHost(&R,sizeR));
	int maxThreads = J*K;
	checkCudaErrors(cudaMemcpyAsync(d_z,z, sizeJ, cudaMemcpyHostToDevice,0));
		ExponentialKernel<<<2,256,0,0>>>(d_z,d_e,J);
	checkCudaErrors(cudaMemcpyAsync(E,d_e, sizeJ, cudaMemcpyDeviceToHost, 0));
				float sum = SumArr(E,J);
	SoftMaxKernel<<<2,256>>>(d_z,sum,J);
	checkCudaErrors(cudaMemcpyAsync(d_al_1,a, sizeK, cudaMemcpyHostToDevice,0));
	checkCudaErrors(cudaMemcpyAsync(d_y,l, sizeJ, cudaMemcpyHostToDevice,0));
		CLLWNG2<<<5,512,0,0>>>(d_z,d_al_1,d_y,d_res,K,maxThreads);
	checkCudaErrors(cudaMemcpyAsync(R,d_res, sizeR, cudaMemcpyDeviceToHost, 0));
		checkCudaErrors( cudaPeekAtLastError() );
		checkCudaErrors(cudaDeviceSynchronize());
	
			checkCudaErrors(cudaFreeHost(E));
			checkCudaErrors(cudaFree(d_z));
			checkCudaErrors(cudaFree(d_e));
			checkCudaErrors(cudaFree(d_al_1));
			checkCudaErrors(cudaFree(d_y));
			checkCudaErrors(cudaFree(d_res));
	
	return R;
	}
	__global__ void VecMulKernel(float* A,float* B,float* C,int N){
		int tid = blockDim.x * blockIdx.x + threadIdx.x;
		if(tid<N){
			C[tid]=A[tid]*B[tid];
		}
	}
float* Cuda::VecMul(float* a, float* b,int N){
	size_t size = N*sizeof(float);
	float* d_a; checkCudaErrors(cudaMalloc(&d_a,size));
	float* d_b; checkCudaErrors(cudaMalloc(&d_b,size));
	float* d_c; checkCudaErrors(cudaMalloc(&d_c,size));
	float* C; checkCudaErrors(cudaMallocHost(&C,size));

	checkCudaErrors(cudaMemcpyAsync(d_a,a, size, cudaMemcpyHostToDevice,0));
	checkCudaErrors(cudaMemcpyAsync(d_b,b, size, cudaMemcpyHostToDevice,0));
			VecMulKernel<<<2,256,0,0>>>(d_a,d_b,d_c,N);
	checkCudaErrors(cudaMemcpyAsync(C, d_c, size, cudaMemcpyDeviceToHost,0));
		checkCudaErrors( cudaPeekAtLastError() );
		checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaFree(d_a));
		checkCudaErrors(cudaFree(d_b));
		checkCudaErrors(cudaFree(d_c));
	return C;

}
void Cuda::clean(float* f){
	checkCudaErrors(cudaFreeHost(f));
	checkCudaErrors( cudaPeekAtLastError() );
}
void Cuda::clean(float** f,int n){
	for(int i =0;i<n;i++){
	checkCudaErrors(cudaFreeHost(f[i]));
	}
	checkCudaErrors( cudaPeekAtLastError() );
}//*/