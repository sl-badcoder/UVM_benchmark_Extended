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

//#define MEMADVISE
//#define PREF

#define CUDA_CHECK(call)                                                                 \
  do {                                                                                   \
    cudaError_t _e = (call);                                                             \
    if (_e != cudaSuccess) {                                                             \
      std::cerr << "CUDA error: " << cudaGetErrorString(_e)                              \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;                  \
      std::exit(EXIT_FAILURE);                                                          \
    }                                                                                    \
  } while (0)

static inline bool mul_overflow_u64(uint64_t a, uint64_t b, uint64_t* out) {
  if (a == 0 || b == 0) { *out = 0; return false; }
  if (a > (std::numeric_limits<uint64_t>::max() / b)) return true;
  *out = a * b;
  return false;
}

static inline bool mul_overflow_size(size_t a, size_t b, size_t* out) {
  if (a == 0 || b == 0) { *out = 0; return false; }
  if (a > (std::numeric_limits<size_t>::max() / b)) return true;
  *out = a * b;
  return false;
}

static inline size_t div_ceil(size_t a, size_t b) { return (a + b - 1) / b; }

// -----------------------------------------
// Random generator (GPU) for points
// -----------------------------------------
static inline uint32_t mix32_host(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

static void init_points_cpu(float* x, float* y, size_t n, uint32_t seed) {
  for (size_t i = 0; i < n; ++i) {
    // Using only low 32 bits; repeats beyond 2^32, but no overflow/UB.
    uint32_t a = mix32_host((uint32_t)i ^ seed);
    uint32_t b = mix32_host((uint32_t)i ^ (seed * 747796405u + 2891336453u));
    x[i] = (a >> 8) * (1.0f / 16777216.0f);
    y[i] = (b >> 8) * (1.0f / 16777216.0f);
  }
}
// -----------------------------------------
// Portable atomicAdd(double) fallback
// -----------------------------------------
__device__ __forceinline__ double atomicAdd_double(double* addr, double val) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(addr, val);
#else
  // CAS-based atomic add for double (works on older arch; slower)
  unsigned long long int* ull = reinterpret_cast<unsigned long long int*>(addr);
  unsigned long long int old = *ull, assumed;
  do {
    assumed = old;
    double sum = __longlong_as_double(assumed) + val;
    old = atomicCAS(ull, assumed, __double_as_longlong(sum));
  } while (assumed != old);
  return __longlong_as_double(old);
#endif
}

__device__ __forceinline__ float squared_l2_distance(float x1, float y1, float x2, float y2) {
  float dx = x1 - x2;
  float dy = y1 - y2;
  return dx * dx + dy * dy;
}

// -----------------------------------------
// Data container (managed x/y)
// -----------------------------------------
struct Data {
  Data() = default;
  explicit Data(size_t n) { allocate(n); }

  Data(const Data&) = delete;
  Data& operator=(const Data&) = delete;

  Data(Data&& o) noexcept { move_from(o); }
  Data& operator=(Data&& o) noexcept {
    if (this != &o) { release(); move_from(o); }
    return *this;
  }

  ~Data() { release(); }

  float* x{nullptr};
  float* y{nullptr};
  size_t size{0};
  size_t bytes{0};

private:
  void allocate(size_t n) {
    size = n;
    size_t tmp = 0;
    if (mul_overflow_size(n, sizeof(float), &tmp)) throw std::overflow_error("n*sizeof(float) overflow");
    bytes = tmp;
    CUDA_CHECK(cudaMallocManaged(&x, bytes));
    CUDA_CHECK(cudaMallocManaged(&y, bytes));
    std::memset(x, 0, bytes);
    std::memset(y, 0, bytes);
  }
  void release() {
    if (x) CUDA_CHECK(cudaFree(x));
    if (y) CUDA_CHECK(cudaFree(y));
    x = nullptr; y = nullptr; size = 0; bytes = 0;
  }
  void move_from(Data& o) noexcept {
    x = o.x; y = o.y; size = o.size; bytes = o.bytes;
    o.x = nullptr; o.y = nullptr; o.size = 0; o.bytes = 0;
  }
};

// -----------------------------------------
// Plan A kernel: assign + block-local accumulate + global atomic flush
// -----------------------------------------
__global__ void assign_accumulate_tile(
    const float* __restrict__ data_x,
    const float* __restrict__ data_y,
    size_t n_tile,
    const float* __restrict__ means_x,
    const float* __restrict__ means_y,
    int k,
    double* __restrict__ sum_x,
    double* __restrict__ sum_y,
    uint64_t* __restrict__ count)
{
  extern __shared__ unsigned char smem[];
  double* s_sumx = reinterpret_cast<double*>(smem);
  double* s_sumy = s_sumx + k;
  uint64_t* s_cnt = reinterpret_cast<uint64_t*>(s_sumy + k);

  // init shared accumulators
  for (int c = threadIdx.x; c < k; c += blockDim.x) {
    s_sumx[c] = 0.0;
    s_sumy[c] = 0.0;
    s_cnt[c]  = 0;
  }
  __syncthreads();

  // grid-stride over tile points
  for (size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
       i < n_tile;
       i += (size_t)blockDim.x * (size_t)gridDim.x)
  {
    float px = data_x[i];
    float py = data_y[i];

    float best = FLT_MAX;
    int best_c = 0;
    for (int c = 0; c < k; ++c) {
      float d = squared_l2_distance(px, py, means_x[c], means_y[c]);
      if (d < best) { best = d; best_c = c; }
    }

    // shared atomics (still contention if k tiny; but far better than global-per-point atomics)
    atomicAdd_double(&s_sumx[best_c], (double)px);
    atomicAdd_double(&s_sumy[best_c], (double)py);
    atomicAdd(reinterpret_cast<unsigned long long*>(&s_cnt[best_c]), 1ULL);
  }

  __syncthreads();

  // flush per-block partials to global
  for (int c = threadIdx.x; c < k; c += blockDim.x) {
    uint64_t ccount = s_cnt[c];
    if (ccount) {
      atomicAdd_double(&sum_x[c], s_sumx[c]);
      atomicAdd_double(&sum_y[c], s_sumy[c]);
      atomicAdd(reinterpret_cast<unsigned long long*>(&count[c]), (unsigned long long)ccount);
    }
  }
}

__global__ void update_means_from_sums(
    float* means_x,
    float* means_y,
    const double* sum_x,
    const double* sum_y,
    const uint64_t* count,
    int k)
{
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= k) return;

  uint64_t cnt = count[c];
  if (cnt > 0) {
    means_x[c] = (float)(sum_x[c] / (double)cnt);
    means_y[c] = (float)(sum_y[c] / (double)cnt);
  }
}

// Utility: set sums/count to zero (device)
__global__ void zero_accumulators(double* sum_x, double* sum_y, uint64_t* count, int k) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= k) return;
  sum_x[c] = 0.0;
  sum_y[c] = 0.0;
  count[c] = 0;
}

// -----------------------------------------
// main
// -----------------------------------------
int main(int argc, const char* argv[]) {
  if (argc < 4) {
    std::cerr
      << "usage:\n"
      << "  k-means --random-gib <GiB_xy> <k> [iterations] [tile_gib]\n"
      << "  k-means --random-n   <N>      <k> [iterations] [tile_gib]\n";
    return EXIT_FAILURE;
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

  if (k <= 0) { std::cerr << "error: k must be > 0\n"; return EXIT_FAILURE; }
  if (N == 0)  { std::cerr << "error: N must be > 0\n"; return EXIT_FAILURE; }
  if (tile_gib == 0) { std::cerr << "error: tile_gib must be > 0\n"; return EXIT_FAILURE; }

  // Device info / limits checks
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  int maxShared = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&maxShared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  if (maxShared == 0) CUDA_CHECK(cudaDeviceGetAttribute(&maxShared, cudaDevAttrMaxSharedMemoryPerBlock, device));

  // Shared mem needed per block for Plan A:
  // 2*k doubles + k uint64
  size_t shmem_bytes = 0;
  {
    size_t part1 = 0, part2 = 0;
    if (mul_overflow_size((size_t)(2 * k), sizeof(double), &part1)) {
      std::cerr << "error: shared mem size overflow (2*k*sizeof(double))\n";
      return EXIT_FAILURE;
    }
    if (mul_overflow_size((size_t)k, sizeof(uint64_t), &part2)) {
      std::cerr << "error: shared mem size overflow (k*sizeof(uint64_t))\n";
      return EXIT_FAILURE;
    }
    if (part1 > std::numeric_limits<size_t>::max() - part2) {
      std::cerr << "error: shared mem size overflow (sum)\n";
      return EXIT_FAILURE;
    }
    shmem_bytes = part1 + part2;
  }

  if ((int)shmem_bytes > maxShared) {
    std::cerr
      << "error: k too large for shared memory in Plan A.\n"
      << "k=" << k << " needs " << shmem_bytes << " bytes shared/block, device allows ~" << maxShared << "\n";
    return EXIT_FAILURE;
  }

  // Allocate dataset in Unified Memory
  Data d_data(N);

  // Generate random points on GPU
  init_points_cpu(d_data.x, d_data.y, N, 12345u);
  // Means arrays (managed, small)
  Data d_means((size_t)k);
  std::memcpy(d_means.x, d_data.x, (size_t)k * sizeof(float));
  std::memcpy(d_means.y, d_data.y, (size_t)k * sizeof(float));

  // Global accumulators on device
  double* d_sumx = nullptr;
  double* d_sumy = nullptr;
  uint64_t* d_cnt = nullptr;
  CUDA_CHECK(cudaMalloc(&d_sumx, (size_t)k * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_sumy, (size_t)k * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_cnt,  (size_t)k * sizeof(uint64_t)));

  #ifdef MEMADVISE
    CUDA_CHECK(cudaMemAdvise(d_data.x, d_data.size * sizeof(float), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    CUDA_CHECK(cudaMemAdvise(d_data.y, d_data.size * sizeof(float), cudaMemAdviseSetAccessedBy, 0));
    CUDA_CHECK(cudaMemAdvise(d_means.x, (size_t)k * sizeof(float), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    CUDA_CHECK(cudaMemAdvise(d_means.y, (size_t)k * sizeof(float), cudaMemAdviseSetAccessedBy, 0));
  #endif


  auto t0 = std::chrono::high_resolution_clock::now();

  // Tile sizing: tile_gib refers to x+y bytes in GiB.
  uint64_t tile_bytes_xy = 0;
  if (mul_overflow_u64(tile_gib, 1024ULL * 1024ULL * 1024ULL, &tile_bytes_xy)) {
    std::cerr << "error: tile_gib too large (overflow)\n";
    return EXIT_FAILURE;
  }
  size_t tile_points = (size_t)(tile_bytes_xy / (2ULL * sizeof(float)));
  if (tile_points == 0) tile_points = 1;
  if (tile_points > N) tile_points = N;

  // Kernel launch params
  const int threads = 256;
  int blocks = 0;
  {
    // Don't scale blocks with N; keep it “reasonable” and rely on grid-stride.
    // Pick a few thousand blocks max.
    size_t blocks_needed = div_ceil(tile_points, (size_t)threads);
    blocks = (int)blocks_needed;
    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;
  }

  // Optional: for UM tests, you probably want to prefetch each tile to device.
  // We do it by default; if you hate it, remove the cudaMemPrefetchAsync calls.
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  size_t free_m1, total_m, free_m2, free_m3, free_m4;


    #ifdef PREF
    int iter =0;
    // Prefetch tile to GPU for better UM behavior; safe even if already resident.
    if(iter == 0)cudaMemGetInfo(&free_m1, &total_m);
    // Prefetch means and accumulators (tiny)
    CUDA_CHECK(cudaMemPrefetchAsync(d_data.x, min(d_data.size * sizeof(float), free_m1), device, stream));
    if(iter == 0)cudaMemGetInfo(&free_m2, &total_m);
    CUDA_CHECK(cudaMemPrefetchAsync(d_data.y, min(d_data.size * sizeof(float), free_m2), device, stream));
    if(iter == 0)cudaMemGetInfo(&free_m3, &total_m);
    CUDA_CHECK(cudaMemPrefetchAsync(d_means.x, min((size_t)k * sizeof(float), free_m3), device, stream));
    if(iter == 0)cudaMemGetInfo(&free_m4, &total_m);
    CUDA_CHECK(cudaMemPrefetchAsync(d_means.y, min((size_t)k * sizeof(float), free_m4), device, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    #endif

  for (int iter = 0; iter < iters; ++iter) {

    // zero accumulators (device)
    {
      int zb = (k + 255) / 256;
      zero_accumulators<<<zb, 256, 0, stream>>>(d_sumx, d_sumy, d_cnt, k);
      CUDA_CHECK(cudaGetLastError());
    }



    // stream over tiles
    for (size_t base = 0; base < N; base += tile_points) {
      size_t n_tile = tile_points;
      if (base + n_tile > N) n_tile = N - base;

      float* x_tile = d_data.x + base;
      float* y_tile = d_data.y + base;


      assign_accumulate_tile<<<blocks, threads, shmem_bytes, stream>>>(
        x_tile, y_tile, n_tile,
        d_means.x, d_means.y,
        k,
        d_sumx, d_sumy, d_cnt
      );
      CUDA_CHECK(cudaGetLastError());
    }

    // update means
    {
      int ub = (k + 255) / 256;
      update_means_from_sums<<<ub, 256, 0, stream>>>(
        d_means.x, d_means.y, d_sumx, d_sumy, d_cnt, k
      );
      CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaDeviceSynchronize();
  }
  cudaDeviceSynchronize();
  auto t1 = std::chrono::high_resolution_clock::now();
  double secs = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();

  // Report dataset size
  long double gib_xy = (long double)N * 2.0L * (long double)sizeof(float) /
                       (1024.0L * 1024.0L * 1024.0L);

  std::cerr << "Plan A (scalable) k-means took: " << secs
            << " s for N=" << N << " points (x+y ~ " << (double)gib_xy
            << " GiB), k=" << k << ", iters=" << iters
            << ", tile_gib=" << tile_gib << "\n";

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_sumx));
  CUDA_CHECK(cudaFree(d_sumy));
  CUDA_CHECK(cudaFree(d_cnt));
  return 0;
}