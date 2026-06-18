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
      std::exit(EXIT_FAILURE);                                                           \
    }                                                                                    \
  } while (0)

// ---------------- SAFE MATH & HELPERS ----------------
static inline bool mul_overflow_size(size_t a, size_t b, size_t* out) {
    if (a == 0 || b == 0) { *out = 0; return false; }
    if (a > std::numeric_limits<size_t>::max() / b) return true;
    *out = a * b;
    return false;
}

static inline size_t div_ceil(size_t a, size_t b) {
    return (a + b - 1) / b;
}

static inline uint32_t mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static void init_points(float* x, float* y, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        uint32_t a = mix32((uint32_t)i);
        uint32_t b = mix32((uint32_t)i ^ 0x9e3779b9);
        x[i] = (a >> 8) * (1.0f / 16777216.0f);
        y[i] = (b >> 8) * (1.0f / 16777216.0f);
    }
}

// ---------------- ATOMIC & KERNELS ----------------
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

__device__ float dist2(float x1, float y1, float x2, float y2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    return dx * dx + dy * dy;
}

__global__ void assign_accumulate(
    const float* x, const float* y, size_t n,
    const float* mx, const float* my, int k,
    double* sx, double* sy, uint64_t* cnt)
{
    extern __shared__ unsigned char smem[];
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
            float d = dist2(px, py, mx[c], my[c]);
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
    if (i >= k) return;
    if (cnt[i] > 0) {
        mx[i] = sx[i] / (double)cnt[i];
        my[i] = sy[i] / (double)cnt[i];
    }
}

__global__ void zero(double* sx, double* sy, uint64_t* cnt, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) { sx[i] = 0; sy[i] = 0; cnt[i] = 0; }
}

// ---------------- MAIN ----------------
int main(int argc, const char* argv[]) {
    if (argc < 4) {
        std::cerr << "usage: --random-n N k [iters]\n";
        return 1;
    }

    size_t N = std::stoull(argv[2]);
    int k = std::atoi(argv[3]);
    int iters = (argc >= 5) ? std::atoi(argv[4]) : 300;

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    // ---------------- MANAGED ALLOCATIONS ----------------
    float *x, *y, *mx, *my;
    double *sx, *sy;
    uint64_t *cnt;
    size_t bytes_N, bytes_K_f, bytes_K_d, bytes_K_u;

    mul_overflow_size(N, sizeof(float), &bytes_N);
    mul_overflow_size(k, sizeof(float), &bytes_K_f);
    mul_overflow_size(k, sizeof(double), &bytes_K_d);
    mul_overflow_size(k, sizeof(uint64_t), &bytes_K_u);

    CUDA_CHECK(cudaMallocManaged(&x, bytes_N));
    CUDA_CHECK(cudaMallocManaged(&y, bytes_N));
    CUDA_CHECK(cudaMallocManaged(&mx, bytes_K_f));
    CUDA_CHECK(cudaMallocManaged(&my, bytes_K_f));
    CUDA_CHECK(cudaMallocManaged(&sx, bytes_K_d));
    CUDA_CHECK(cudaMallocManaged(&sy, bytes_K_d));
    CUDA_CHECK(cudaMallocManaged(&cnt, bytes_K_u));

    // Initialize on CPU
    init_points(x, y, N);
    for(int i=0; i<k; ++i) { mx[i] = x[i]; my[i] = y[i]; }

    // PREFETCH to GPU to avoid page faults during first iteration
    CUDA_CHECK(cudaMemPrefetchAsync(x, bytes_N, device));
    CUDA_CHECK(cudaMemPrefetchAsync(y, bytes_N, device));
    CUDA_CHECK(cudaMemPrefetchAsync(mx, bytes_K_f, device));
    CUDA_CHECK(cudaMemPrefetchAsync(my, bytes_K_f, device));

    auto t0 = std::chrono::high_resolution_clock::now();

    int threads = 256;
    int blocks = std::min((size_t)65535, div_ceil(N, (size_t)threads));
    size_t shmem = (2 * k * sizeof(double)) + (k * sizeof(uint64_t));

    for (int it = 0; it < iters; ++it) {
        zero<<<(k + 255) / 256, 256>>>(sx, sy, cnt, k);
        cudaGetLastError();
        assign_accumulate<<<blocks, threads, shmem>>>(x, y, N, mx, my, k, sx, sy, cnt);
        cudaGetLastError();
        update<<<(k + 255) / 256, 256>>>(mx, my, sx, sy, cnt, k);
        cudaGetLastError();
    }
    cudaGetLastError();
    // Must synchronize before CPU reads managed memory
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cerr << "Time: " << std::chrono::duration<double>(t1 - t0).count() << "s\n";

    // Cleanup (cudaFree works for Managed Memory)
    cudaFree(x); cudaFree(y);
    cudaFree(mx); cudaFree(my);
    cudaFree(sx); cudaFree(sy);
    cudaFree(cnt);

    return 0;
}