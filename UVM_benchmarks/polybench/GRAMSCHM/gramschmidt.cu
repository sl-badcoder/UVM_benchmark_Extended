/**
 * gramschmidt.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cuda.h>

//#define PREF
//#define MEMADVISE

using index_t = size_t;
typedef float DATA_TYPE;

__host__ __device__ static inline index_t IDX(index_t i, index_t ld, index_t j) {
    return i * ld + j;
}

#include "../../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Problem size */
//#define M 2048
//#define N 2048

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void gramschmidt(DATA_TYPE* A, DATA_TYPE* R, DATA_TYPE* Q, index_t M, index_t N)
{
	index_t i,j,k;
	DATA_TYPE nrm;
	for (k = 0; k < N; k++)
	{
		nrm = 0;
		for (i = 0; i < M; i++)
		{
			nrm += A[IDX(i,N ,k)] * A[IDX(i,N ,k)];
		}
		
		R[IDX(k,N , k)] = sqrt(nrm);
		for (i = 0; i < M; i++)
		{
			Q[IDX(i,N , k)] = A[IDX(i,N , k)] / R[IDX(k,N ,k)];
		}
		
		for (j = k + 1; j < N; j++)
		{
			R[IDX(k,N , j)] = 0;
			for (i = 0; i < M; i++)
			{
				R[IDX(k,N , j)] += Q[IDX(i,N , k)] * A[IDX(i,N , j)];
			}
			for (i = 0; i < M; i++)
			{
				A[IDX(i,N ,j)] = A[IDX(i,N , j)] - Q[IDX(i,N , k)] * R[IDX(k,N , j)];
			}
		}
	}
}


void init_array(DATA_TYPE* A, DATA_TYPE* A_gpu, index_t M, index_t N)
{
	index_t i, j;

	for (i = 0; i < M; i++)
	{
		for (j = 0; j < N; j++)
		{
			A[IDX(i,N, j)] = ((DATA_TYPE) (i+1)*(j+1)) / (M+1);
			A_gpu[IDX(i,N , j)] = ((DATA_TYPE) (i+1)*(j+1)) / (M+1);
		}
	}
}


void compareResults(DATA_TYPE* A, DATA_TYPE* A_outputFromGpu, index_t M, index_t N)
{
	index_t i, j, fail;
	fail = 0;

	for (i=0; i < M; i++) 
	{
		for (j=0; j < N; j++) 
		{
			if (percentDiff(A[IDX(i,N ,j)], A_outputFromGpu[IDX(i,N , j)]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{				
				fail++;
				printf("i: %zu j: %zu \n1: %f\n 2: %f\n", i, j, A[IDX(i,N , j)], A_outputFromGpu[IDX(i,N , j)]);
			}
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %zu\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );	
	return;
}

static index_t parse_index_arg(const char* value, const char* name)
{
	errno = 0;
	char* end = NULL;
	unsigned long long parsed = strtoull(value, &end, 10);
	if (errno != 0 || end == value || *end != '\0' || parsed == 0) {
		fprintf(stderr, "Invalid %s: %s\n", name, value);
		exit(EXIT_FAILURE);
	}
	return (index_t)parsed;
}


__global__ void gramschmidt_kernel1(DATA_TYPE *a, DATA_TYPE *r, DATA_TYPE *q, index_t M, index_t N, index_t k)
{
	index_t tid = (index_t)blockIdx.x * blockDim.x + threadIdx.x;

	if(tid==0)
	{
		DATA_TYPE nrm = 0.0;
		index_t i;
		for (i = 0; i < M; i++)
		{
			nrm += a[IDX(i , N , k)] * a[IDX(i , N , k)];
		}
      		r[IDX(k , N , k)] = sqrt(nrm);
	}
}


__global__ void gramschmidt_kernel2(DATA_TYPE *a, DATA_TYPE *r, DATA_TYPE *q, index_t M, index_t N, index_t k)
{
	index_t i = (index_t)blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < M)
	{	
		q[IDX(i,N,k)] = a[IDX(i ,N ,k)] / r[IDX(k , N , k)];
	}
}


__global__ void gramschmidt_kernel3(DATA_TYPE *a, DATA_TYPE *r, DATA_TYPE *q, index_t M, index_t N, index_t k)
{
	index_t j = (index_t)blockIdx.x * blockDim.x + threadIdx.x;

	if ((j > k) && (j < N))
	{
		r[k*N + j] = 0.0;

		index_t i;
		for (i = 0; i < M; i++)
		{
			r[IDX(k,N ,j)] += q[IDX(i,N, k)] * a[IDX(i,N,j)];
		}
		
		for (i = 0; i < M; i++)
		{
			a[IDX(i,N,j)] -= q[IDX(i,N,k)] * r[IDX(k,N,j)];
		}
	}
}


void gramschmidtCuda(DATA_TYPE* A_gpu, DATA_TYPE* R_gpu, DATA_TYPE* Q_gpu, index_t M, index_t N)
{
	double t_start, t_end;

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 gridKernel1(1, 1);
	dim3 gridKernel2((size_t)ceil(((float)M) / ((float)DIM_THREAD_BLOCK_X)), 1);
	dim3 gridKernel3((size_t)ceil(((float)N) / ((float)DIM_THREAD_BLOCK_X)), 1);
	
	t_start = rtclock();
	index_t k;
	for (k = 0; k < N; k++)
	{
		gramschmidt_kernel1<<<gridKernel1,block>>>(A_gpu, R_gpu, Q_gpu, M, N, k);
		cudaDeviceSynchronize();
		gramschmidt_kernel2<<<gridKernel2,block>>>(A_gpu, R_gpu, Q_gpu, M, N, k);
		cudaDeviceSynchronize();
		gramschmidt_kernel3<<<gridKernel3,block>>>(A_gpu, R_gpu, Q_gpu, M, N, k);
		cudaDeviceSynchronize();
	}
	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
  
}


int main(int argc, char *argv[])
{
	double t_start, t_end;

	index_t M = 49000;
	index_t N = 49000;
	if (argc == 3) {
		M = parse_index_arg(argv[1], "M");
		N = parse_index_arg(argv[2], "N");
	} else if (argc != 1) {
		fprintf(stderr, "Usage: %s [M N]\n", argv[0]);
		return 1;
	}

	size_t elems = (size_t)M * (size_t)N;
	if (N != 0 && elems / N != M) {
		fprintf(stderr, "M * N overflows size_t\n");
		return 1;
	}
	size_t bytes = elems * sizeof(DATA_TYPE);

	DATA_TYPE* A = (DATA_TYPE*)malloc(bytes);
	DATA_TYPE* R = (DATA_TYPE*)malloc(bytes);
	DATA_TYPE* Q = (DATA_TYPE*)malloc(bytes);
	if (!A || !R || !Q) {
		fprintf(stderr,"Host allocation failed\n");
		return 1;
	}
	DATA_TYPE* A_gpu = nullptr;
	DATA_TYPE* R_gpu = nullptr;
	DATA_TYPE* Q_gpu = nullptr;

	cudaError_t err = cudaMallocManaged(&A_gpu, bytes);
	if (err != cudaSuccess) {
		fprintf(stderr,"cudaMallocManaged failed: %s\n", cudaGetErrorString(err));
		return 1;
	}
	err = cudaMallocManaged(&R_gpu, bytes);
	if (err != cudaSuccess) {
		fprintf(stderr,"cudaMallocManaged failed: %s\n", cudaGetErrorString(err));
		return 1;
	}
	err = cudaMallocManaged(&Q_gpu, bytes);
	if (err != cudaSuccess) {
		fprintf(stderr,"cudaMallocManaged failed: %s\n", cudaGetErrorString(err));
		return 1;
	}
	init_array(A, A_gpu, M, N);

#ifdef MEMADVISE
	(cudaMemAdvise(A_gpu, bytes, cudaMemAdviseSetAccessedBy, 0));
	(cudaMemAdvise(A_gpu, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
	(cudaMemAdvise(R_gpu, bytes, cudaMemAdviseSetAccessedBy, 0));
	(cudaMemAdvise(R_gpu, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
	(cudaMemAdvise(Q_gpu, bytes, cudaMemAdviseSetAccessedBy, 0));
	(cudaMemAdvise(Q_gpu, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
#endif
#ifdef PREF
	size_t total_m, free_m;
	cudaMemGetInfo(&free_m, &total_m);
	(cudaMemPrefetchAsync(A_gpu, (size_t) (free_m * 0.8) < (size_t) bytes? (free_m * 0.8) : bytes, 0, 0));
	cudaMemGetInfo(&free_m, &total_m);
	(cudaMemPrefetchAsync(R_gpu, (size_t) (free_m * 0.8) < (size_t) bytes? (free_m * 0.8) : bytes, 0, 0));
	cudaMemGetInfo(&free_m, &total_m);
	(cudaMemPrefetchAsync(Q_gpu, (size_t) (free_m * 0.8) < (size_t) bytes? (free_m * 0.8) : bytes, 0, 0));
#endif
	
	GPU_argv_init();
	gramschmidtCuda(A_gpu, R_gpu, Q_gpu, M, N);
	
	free(A);
	free(R);
	free(Q);  
	cudaFree(A_gpu);
	cudaFree(R_gpu);
	cudaFree(Q_gpu);

    	return 0;
}
