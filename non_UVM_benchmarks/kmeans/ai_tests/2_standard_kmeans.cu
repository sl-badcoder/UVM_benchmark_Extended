#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <chrono>
#include <algorithm>

#define CUDA_CHECK(call)                                                                 \
  do {                                                                                   \
    cudaError_t _e = (call);                                                             \
    if (_e != cudaSuccess) {                                                             \
      std::cerr << "CUDA error: " << cudaGetErrorString(_e)                              \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;                  \
      std::exit(EXIT_FAILURE);                                                          \
    }                                                                                    \
  } while (0)

// ---------------- ATOMIC DOUBLE SUPPORT ----------------
__device__ double atomicAdd_double(double* addr, double val) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(addr, val);
#else
  unsigned long long* ull = (unsigned long long*)addr;
  unsigned long long old = *ull, assumed;
  do {
    assumed = old;
    double sum = __longlong_as_double(assumed) + val;
    old = atomicCAS(ull, assumed, __double_as_longlong(sum));
  } while (assumed != old);
  return __longlong_as_double(old);
#endif
}

__device__ inline float dist2(float x1, float y1, float x2, float y2) {
  float dx = x1 - x2;
  float dy = y1 - y2;
  return dx * dx + dy * dy;
}

// ---------------- KERNELS ----------------

__global__ void zero_kernel(double* sx, double* sy, uint64_t* cnt, int k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < k) {
    sx[i] = 0.0;
    sy[i] = 0.0;
    cnt[i] = 0;
  }
}

__global__ void assign_accumulate(
    const float* __restrict__ x,
    const float* __restrict__ y,
    size_t n,
    const float* __restrict__ mx,
    const float* __restrict__ my,
    int k,
    double* sx,
    double* sy,
    uint64_t* cnt)
{
  // Shared memory for partial block reductions to minimize global atomic contention
  extern __shared__ unsigned char smem[];
  double* s_sx = (double*)smem;
  double* s_sy = s_sx + k;
  uint64_t* s_cnt = (uint64_t*)(s_sy + k);

  for (int c = threadIdx.x; c < k; c += blockDim.x) {
    s_sx[c] = 0.0;
    s_sy[c] = 0.0;
    s_cnt[c] = 0;
  }
  __syncthreads();

  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    float px = x[i], py = y[i];
    float best_dist = FLT_MAX;
    int best_c = 0;

    for (int c = 0; c < k; ++c) {
      float d = dist2(px, py, mx[c], my[c]);
      if (d < best_dist) { 
        best_dist = d; 
        best_c = c; 
      }
    }

    atomicAdd_double(&s_sx[best_c], (double)px);
    atomicAdd_double(&s_sy[best_c], (double)py);
    atomicAdd((unsigned long long*)&s_cnt[best_c], 1ULL);
  }
  __syncthreads();

  // Finalize block results into global memory
  for (int c = threadIdx.x; c < k; c += blockDim.x) {
    if (s_cnt[c] > 0) {
      atomicAdd_double(&sx[c], s_sx[c]);
      atomicAdd_double(&sy[c], s_sy[c]);
      atomicAdd((unsigned long long*)&cnt[c], s_cnt[c]);
    }
  }
}

__global__ void update_centroids(float* mx, float* my, const double* sx, const double* sy, const uint64_t* cnt, int k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < k && cnt[i] > 0) {
    mx[i] = (float)(sx[i] / (double)cnt[i]);
    my[i] = (float)(sy[i] / (double)cnt[i]);
  }
}

// ---------------- HELPERS ----------------

static inline uint32_t mix32(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352dU;
  x ^= x >> 15; x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

void init_points(float* x, float* y, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    uint32_t a = mix32((uint32_t)i);
    uint32_t b = mix32((uint32_t)i ^ 0x9e3779b9);
    x[i] = (a >> 8) * (1.0f / 16777216.0f);
    y[i] = (b >> 8) * (1.0f / 16777216.0f);
  }
}

int main(int argc, const char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <N_points> <K_clusters> [iters]\n";
    return 1;
  }

  size_t N = std::stoull(argv[1]);
  int K = std::stoi(argv[2]);
  int iters = (argc > 3) ? std::stoi(argv[3]) : 100;

  int device;
  CUDA_CHECK(cudaGetDevice(&device));

  // 1. Unified Memory Allocation
  float *x, *y, *mx, *my;
  double *sx, *sy;
  uint64_t *cnt;

  CUDA_CHECK(cudaMallocManaged(&x, N * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&y, N * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&mx, K * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&my, K * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&sx, K * sizeof(double)));
  CUDA_CHECK(cudaMallocManaged(&sy, K * sizeof(double)));
  CUDA_CHECK(cudaMallocManaged(&cnt, K * sizeof(uint64_t)));

  // 2. Initialize on CPU
  init_points(x, y, N);
  for(int i=0; i<K; ++i) { mx[i] = x[i]; my[i] = y[i]; }

  // 3. Optimization Hints
  // Points are read-only for the duration of the kernel
  CUDA_CHECK(cudaMemAdvise(x, N * sizeof(float), cudaMemAdviseSetReadMostly, device));
  CUDA_CHECK(cudaMemAdvise(y, N * sizeof(float), cudaMemAdviseSetReadMostly, device));
  
  // Prefetch data to GPU to avoid page faults during the first iteration
  CUDA_CHECK(cudaMemPrefetchAsync(x, N * sizeof(float), device));
  CUDA_CHECK(cudaMemPrefetchAsync(y, N * sizeof(float), device));
  CUDA_CHECK(cudaMemPrefetchAsync(mx, K * sizeof(float), device));
  CUDA_CHECK(cudaMemPrefetchAsync(my, K * sizeof(float), device));

  // 4. Execution Config
  int blockSize = 256;
  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device);
  int gridSize = numSMs * 32; // Standard heuristic to fill the GPU

  size_t shmemSize = (2 * K * sizeof(double)) + (K * sizeof(uint64_t));
  
  // Guard against exceeding shared memory limits
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  if (shmemSize > prop.sharedMemPerBlock) {
      std::cerr << "K is too large for shared memory optimization. Reducing K or using global atomics only.\n";
      return 1;
  }

  std::cout << "Starting K-Means: N=" << N << ", K=" << K << ", Iters=" << iters << std::endl;
  auto t0 = std::chrono::high_resolution_clock::now();

  for (int it = 0; it < iters; ++it) {
    zero_kernel<<<(K + 255) / 256, 256>>>(sx, sy, cnt, K);
    
    assign_accumulate<<<gridSize, blockSize, shmemSize>>>(
        x, y, N, mx, my, K, sx, sy, cnt);
    
    update_centroids<<<(K + 255) / 256, 256>>>(mx, my, sx, sy, cnt, K);
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  auto t1 = std::chrono::high_resolution_clock::now();
  double secs = std::chrono::duration<double>(t1 - t0).count();

  std::cout << "Finalized in: " << secs << "s" << std::endl;

  // Cleanup (Unified Memory)
  cudaFree(x); cudaFree(y);
  cudaFree(mx); cudaFree(my);
  cudaFree(sx); cudaFree(sy);
  cudaFree(cnt);

  return 0;
}