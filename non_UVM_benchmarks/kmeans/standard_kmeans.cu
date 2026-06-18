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

// ---------------- SAFE MATH ----------------
static inline bool mul_overflow_size(size_t a, size_t b, size_t* out) {
  if (a == 0 || b == 0) { *out = 0; return false; }
  if (a > std::numeric_limits<size_t>::max() / b) return true;
  *out = a * b;
  return false;
}

static inline bool mul_overflow_u64(uint64_t a, uint64_t b, uint64_t* out) {
  if (a == 0 || b == 0) { *out = 0; return false; }
  if (a > std::numeric_limits<uint64_t>::max() / b) return true;
  *out = a * b;
  return false;
}

static inline size_t div_ceil(size_t a, size_t b) {
  return (a + b - 1) / b;
}

// ---------------- SAFE PARSING ----------------
static int parse_int(const char* s) {
  char* end = nullptr;
  errno = 0;
  long v = std::strtol(s, &end, 10);
  if (errno || *end != '\0' || v <= 0 || v > INT32_MAX)
    throw std::runtime_error("invalid int");
  return (int)v;
}

static uint64_t parse_u64(const char* s) {
  char* end = nullptr;
  errno = 0;
  unsigned long long v = std::strtoull(s, &end, 10);
  if (errno || *end != '\0' || v == 0)
    throw std::runtime_error("invalid u64");
  return (uint64_t)v;
}

// ---------------- RANDOM ----------------
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

// ---------------- ATOMIC DOUBLE ----------------
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

// ---------------- KERNEL ----------------
__global__ void assign_accumulate(
    const float* x,
    const float* y,
    size_t n,
    const float* mx,
    const float* my,
    int k,
    double* sx,
    double* sy,
    uint64_t* cnt)
{
  extern __shared__ unsigned char smem[];

  double* s_sx = (double*)smem;
  double* s_sy = s_sx + k;
  uint64_t* s_cnt = (uint64_t*)(s_sy + k);

  for (int c = threadIdx.x; c < k; c += blockDim.x) {
    s_sx[c] = 0;
    s_sy[c] = 0;
    s_cnt[c] = 0;
  }
  __syncthreads();

  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += blockDim.x * gridDim.x)
  {
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

__global__ void update(float* mx, float* my,
                       double* sx, double* sy,
                       uint64_t* cnt, int k)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= k) return;

  uint64_t c = cnt[i];
  if (c > 0) {
    mx[i] = sx[i] / (double)c;
    my[i] = sy[i] / (double)c;
  }
}

__global__ void zero(double* sx, double* sy, uint64_t* cnt, int k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < k) {
    sx[i] = 0;
    sy[i] = 0;
    cnt[i] = 0;
  }
}

// ---------------- MAIN ----------------
int main(int argc, const char* argv[]) {
  if (argc < 4) {
    std::cerr << "usage: --random-n N k [iters] [tile_gib]\n";
    return 1;
  }

  const std::string mode = argv[1];
  size_t N = 0;
  int k = 0;
  int iters = 300;
  uint64_t tile_gib = 1; // default tile size for x+y (GiB)

  if (mode == "--random-gib") {
    uint64_t gib_xy = std::stoull(argv[2]);
    k = std::atoi(argv[3]);
    if (argc >= 5) iters = std::atoi(argv[4]);
    if (argc >= 6) tile_gib = std::stoull(argv[5]);

    uint64_t bytes_xy = 0;
    if (mul_overflow_u64(gib_xy, 1024ULL * 1024ULL * 1024ULL, &bytes_xy)) {
      std::cerr << "error: GiB_xy too large (overflow)\n";
      return EXIT_FAILURE;
    }
    // x+y = 2 floats = 8 bytes per point
    N = (size_t)(bytes_xy / (2ULL * sizeof(float)));

  } else if (mode == "--random-n") {
    N = (size_t)std::stoull(argv[2]);
    long double gib_xy = (long double)N * 2.0L * (long double)sizeof(float) /
                       (1024.0L * 1024.0L * 1024.0L);
                       std::cout << "size: " << gib_xy << "GiB"<< std::endl;
    k = std::atoi(argv[3]);
    if (argc >= 5) iters = std::atoi(argv[4]);
    if (argc >= 6) tile_gib = std::stoull(argv[5]);
  } else {
    std::cerr << "error: unknown mode '" << mode << "'\n";
    return EXIT_FAILURE;
  }

  // ---------------- HOST PINNED ----------------
  float *h_x, *h_y;
  size_t bytes = 0;
  if (mul_overflow_size(N, sizeof(float), &bytes))
    throw std::overflow_error("host alloc overflow");

  CUDA_CHECK(cudaMallocHost(&h_x, bytes));
  CUDA_CHECK(cudaMallocHost(&h_y, bytes));

  init_points(h_x, h_y, N);

  // ---------------- DEVICE ----------------
  float *d_x, *d_y, *d_mx, *d_my;
  double *d_sx, *d_sy;
  uint64_t *d_cnt;

  CUDA_CHECK(cudaMalloc(&d_x, bytes));
  CUDA_CHECK(cudaMalloc(&d_y, bytes));
  CUDA_CHECK(cudaMalloc(&d_mx, k*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_my, k*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_sx, k*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_sy, k*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_cnt, k*sizeof(uint64_t)));
  auto t0 = std::chrono::high_resolution_clock::now();

  CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_mx, h_x, k*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_my, h_y, k*sizeof(float), cudaMemcpyHostToDevice));

  // ---------------- TILE ----------------
  uint64_t tile_bytes = 0;
  mul_overflow_u64(tile_gib, 1024ULL*1024ULL*1024ULL, &tile_bytes);
  size_t tile_pts = tile_bytes / (2*sizeof(float));
  if (tile_pts == 0) tile_pts = 1;
  if (tile_pts > N) tile_pts = N;

  int threads = 256;
  int blocks = std::min((size_t)65535, div_ceil(tile_pts, (size_t)threads));

  size_t shmem = 0;
  size_t tmp1, tmp2;
  mul_overflow_size(2*k, sizeof(double), &tmp1);
  mul_overflow_size(k, sizeof(uint64_t), &tmp2);
  shmem = tmp1 + tmp2;
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  for (int it = 0; it < iters; ++it) {

    zero<<<(k+255)/256,256, 0, stream>>>(d_sx,d_sy,d_cnt,k);

    for (size_t base = 0; base < N; base += tile_pts) {
      size_t n = std::min(tile_pts, N-base);

      assign_accumulate<<<blocks,threads,shmem, stream>>>(
        d_x+base, d_y+base, n,
        d_mx, d_my, k,
        d_sx, d_sy, d_cnt
      );
    }
    {
      update<<<(k+255)/256,256, 0, stream>>>(d_mx,d_my,d_sx,d_sy,d_cnt,k);
    }
    cudaStreamSynchronize(stream);
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  auto t1 = std::chrono::high_resolution_clock::now();
  double secs = std::chrono::duration<double>(t1-t0).count();

  std::cerr << "Time: " << secs << "s\n";

  // cleanup
  cudaFree(d_x); cudaFree(d_y);
  cudaFree(d_mx); cudaFree(d_my);
  cudaFree(d_sx); cudaFree(d_sy);
  cudaFree(d_cnt);
  cudaFreeHost(h_x); cudaFreeHost(h_y);

  return 0;
}