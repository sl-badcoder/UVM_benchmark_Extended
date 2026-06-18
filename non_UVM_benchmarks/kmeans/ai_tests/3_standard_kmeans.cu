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

__global__ void assign_accumulate(const float* x, const float* y, size_t n, const float* mx, const float* my, int k, double* sx, double* sy, uint64_t* cnt) {
    extern __shared__ char smem[];
    double* s_sx = (double*)smem;
    double* s_sy = s_sx + k;
    uint64_t* s_cnt = (uint64_t*)(s_sy + k);

    for (int c = threadIdx.x; c < k; c += blockDim.x) {
        s_sx[c] = 0; s_sy[c] = 0; s_cnt[c] = 0;
    }
    __syncthreads();

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float px = x[i], py = y[i];
        float best = FLT_MAX;
        int best_c = 0;
        for (int c = 0; c < k; ++c) {
            float d = (px-mx[c])*(px-mx[c]) + (py-my[c])*(py-my[c]);
            if (d < best) { best = d; best_c = c; }
        }
        atomicAdd_double(&s_sx[best_c], px);
        atomicAdd_double(&s_sy[best_c], py);
        atomicAdd((unsigned long long*)&s_cnt[best_c], 1ULL);
    }
    __syncthreads();

    for (int c = threadIdx.x; c < k; c += blockDim.x) {
        if (s_cnt[c]) {
            atomicAdd_double(&sx[c], s_sx[c]);
            atomicAdd_double(&sy[c], s_sy[c]);
            atomicAdd((unsigned long long*)&cnt[c], s_cnt[c]);
        }
    }
}

__global__ void update(float* mx, float* my, double* sx, double* sy, uint64_t* cnt, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k && cnt[i] > 0) {
        mx[i] = sx[i] / (double)cnt[i];
        my[i] = sy[i] / (double)cnt[i];
    }
}

__global__ void zero(double* sx, double* sy, uint64_t* cnt, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) { sx[i] = 0; sy[i] = 0; cnt[i] = 0; }
}

int main(int argc, const char* argv[]) {
    // Basic Parsing
    size_t N=3221225472; 
    int k = 100;
    int iters = 10;
    int deviceId;
    CUDA_CHECK(cudaGetDevice(&deviceId));

    // 1. Check VRAM limits for safety
    size_t free_byte, total_byte;
    CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte));
    // Reserve 10% for system overhead
    size_t vram_limit = (size_t)(free_byte * 0.9); 

    // 2. Managed Allocations
    float *x, *y, *mx, *my;
    double *sx, *sy;
    uint64_t *cnt;

    size_t bytes_n = N * sizeof(float);
    size_t bytes_k_f = k * sizeof(float);
    size_t bytes_k_d = k * sizeof(double);
    size_t bytes_k_u = k * sizeof(uint64_t);

    CUDA_CHECK(cudaMallocManaged(&x, bytes_n));
    CUDA_CHECK(cudaMallocManaged(&y, bytes_n));
    CUDA_CHECK(cudaMallocManaged(&mx, bytes_k_f));
    CUDA_CHECK(cudaMallocManaged(&my, bytes_k_f));
    CUDA_CHECK(cudaMallocManaged(&sx, bytes_k_d));
    CUDA_CHECK(cudaMallocManaged(&sy, bytes_k_d));
    CUDA_CHECK(cudaMallocManaged(&cnt, bytes_k_u));

    // 3. Initialize Points (on CPU)
    init_points(x, y, N);
    std::copy(x, x + k, mx);
    std::copy(y, y + k, my);

    // 4. Memory Advises (As requested: Preferred=CPU, AccessedBy=GPU)
    auto apply_hints = [&](void* ptr, size_t size) {
        CUDA_CHECK(cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
        CUDA_CHECK(cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, deviceId));
    };

    apply_hints(x, bytes_n);   apply_hints(y, bytes_n);
    apply_hints(mx, bytes_k_f); apply_hints(my, bytes_k_f);
    apply_hints(sx, bytes_k_d); apply_hints(sy, bytes_k_d);
    apply_hints(cnt, bytes_k_u);

    // 5. Strategic Prefetching
    // Priority 1: Centroids and Accumulators (Small, high frequency)
    size_t current_prefetched = bytes_k_f * 2 + bytes_k_d * 2 + bytes_k_u;
    CUDA_CHECK(cudaMemPrefetchAsync(mx, bytes_k_f, deviceId));
    CUDA_CHECK(cudaMemPrefetchAsync(my, bytes_k_f, deviceId));
    CUDA_CHECK(cudaMemPrefetchAsync(sx, bytes_k_d, deviceId));
    CUDA_CHECK(cudaMemPrefetchAsync(sy, bytes_k_d, deviceId));
    CUDA_CHECK(cudaMemPrefetchAsync(cnt, bytes_k_u, deviceId));

    // Priority 2: Data Points (Large, limited by VRAM)
    size_t remaining_vram = (vram_limit > current_prefetched) ? vram_limit - current_prefetched : 0;
    size_t prefetch_pts_bytes = std::min(remaining_vram / 2, bytes_n);
    
    if (prefetch_pts_bytes > 0) {
        CUDA_CHECK(cudaMemPrefetchAsync(x, prefetch_pts_bytes, deviceId));
        CUDA_CHECK(cudaMemPrefetchAsync(y, prefetch_pts_bytes, deviceId));
    }

    // 6. Main Loop
    auto t0 = std::chrono::high_resolution_clock::now();
    size_t shmem = 2 * bytes_k_d + bytes_k_u;

    for (int it = 0; it < iters; ++it) {
        zero<<<(k+255)/256, 256>>>(sx, sy, cnt, k);

        // Even with UM, tiling is good if N > VRAM.
        // If prefetch_pts_bytes < bytes_n, UM will page fault the rest in automatically.
        assign_accumulate<<<480, 256, shmem>>>(x, y, N, mx, my, k, sx, sy, cnt);
        
        update<<<(k+255)/256, 256>>>(mx, my, sx, sy, cnt, k);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cerr << "Time: " << std::chrono::duration<double>(t1-t0).count() << "s\n";

    // Cleanup
    cudaFree(x); cudaFree(y); cudaFree(mx); cudaFree(my);
    cudaFree(sx); cudaFree(sy); cudaFree(cnt);

    return 0;
}