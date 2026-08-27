#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <chrono>
#include <iostream>
#include <cstring>
#include <cerrno>
#include <climits>

#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define STR_SIZE 256
#define DEVICE 0
#define HALO 1 // halo width along one direction when advancing to the next iteration

//#define BENCH_PRINT
//#define PREF
//#define MEMADVISE

using std::cout;
using std::endl;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

void run(int argc, char** argv);

double diff_in_second(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) +
           (double)(end.tv_nsec - start.tv_nsec) / 1e9;
}

size_t rows, cols;
#define M_SEED 9
int pyramid_height;

static inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

static size_t parse_size_t(const char* s, const char* name) {
    errno = 0;
    char* end = nullptr;
    unsigned long long v = strtoull(s, &end, 10);

    if (errno != 0 || end == s || *end != '\0') {
        fprintf(stderr, "Invalid %s: %s\n", name, s);
        exit(EXIT_FAILURE);
    }

    return static_cast<size_t>(v);
}

static int parse_int(const char* s, const char* name) {
    errno = 0;
    char* end = nullptr;
    long v = strtol(s, &end, 10);

    if (errno != 0 || end == s || *end != '\0' || v < INT_MIN || v > INT_MAX) {
        fprintf(stderr, "Invalid %s: %s\n", name, s);
        exit(EXIT_FAILURE);
    }

    return static_cast<int>(v);
}

static void validate_inputs() {
    if (rows < 2) {
        fprintf(stderr, "rows must be >= 2\n");
        exit(EXIT_FAILURE);
    }
    if (cols < 1) {
        fprintf(stderr, "cols must be >= 1\n");
        exit(EXIT_FAILURE);
    }
    if (pyramid_height <= 0) {
        fprintf(stderr, "pyramid_height must be > 0\n");
        exit(EXIT_FAILURE);
    }

    int smallBlockCol = BLOCK_SIZE - pyramid_height * HALO * 2;
    if (smallBlockCol <= 0) {
        fprintf(stderr, "Invalid pyramid_height=%d, requires BLOCK_SIZE - 2*pyramid_height > 0\n",
                pyramid_height);
        exit(EXIT_FAILURE);
    }

    if (cols != 0 && rows > SIZE_MAX / cols) {
        fprintf(stderr, "rows * cols overflows size_t\n");
        exit(EXIT_FAILURE);
    }

    size_t total_elements = rows * cols;
    if (total_elements > SIZE_MAX / sizeof(int)) {
        fprintf(stderr, "Allocation size overflows size_t\n");
        exit(EXIT_FAILURE);
    }

    // Kernel/grid logic still uses int in a few places
    if (cols > static_cast<size_t>(INT_MAX)) {
        fprintf(stderr, "cols too large for current kernel/indexing code\n");
        exit(EXIT_FAILURE);
    }
}

void fatal(char *s) {
    fprintf(stderr, "error: %s\n", s);
}

#define IN_RANGE(x, min, max)   ((x) >= (min) && (x) <= (max))
#define CLAMP_RANGE(x, min, max) x = (x < (min)) ? (min) : ((x > (max)) ? (max) : x)
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

__global__ void dynproc_kernel(
    int iteration,
    int *gpuWall,
    int *gpuSrc,
    int *gpuResults,
    size_t cols,
    size_t rows,
    int startStep,
    int border)
{
    __shared__ int prev[BLOCK_SIZE];
    __shared__ int result[BLOCK_SIZE];

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int small_block_cols = BLOCK_SIZE - iteration * HALO * 2;

    int blkX = small_block_cols * bx - border;
    int blkXmax = blkX + BLOCK_SIZE - 1;

    int xidx = blkX + tx;

    int validXmin = (blkX < 0) ? -blkX : 0;
    int validXmax = (blkXmax > (int)cols - 1)
                        ? BLOCK_SIZE - 1 - (blkXmax - (int)cols + 1)
                        : BLOCK_SIZE - 1;

    int W = tx - 1;
    int E = tx + 1;

    W = (W < validXmin) ? validXmin : W;
    E = (E > validXmax) ? validXmax : E;

    bool isValid = IN_RANGE(tx, validXmin, validXmax);

    if (IN_RANGE(xidx, 0, (int)cols - 1)) {
        prev[tx] = gpuSrc[xidx];
    }
    __syncthreads();

    bool computed = false;
    for (int i = 0; i < iteration; i++) {
        computed = false;
        if (IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) && isValid) {
            computed = true;
            int left = prev[W];
            int up = prev[tx];
            int right = prev[E];
            int shortest = MIN(left, up);
            shortest = MIN(shortest, right);
            long long index = (long long)cols * (startStep + i) + xidx;
            result[tx] = shortest + gpuWall[index];
        }
        __syncthreads();

        if (i == iteration - 1)
            break;

        if (computed)
            prev[tx] = result[tx];

        __syncthreads();
    }

    if (computed) {
        gpuResults[xidx] = result[tx];
    }
}

/*
   compute N time steps
*/
int calc_path(int *gpuWall, int *gpuResult[2], size_t rows, size_t cols,
              int pyramid_height, int blockCols, int borderCols)
{
    size_t total_elements = rows * cols;

#ifdef PREF

    cudaStream_t stream1, stream2, stream3, stream4;
    checkCuda(cudaStreamCreate(&stream1), "cudaStreamCreate(stream1)");
    checkCuda(cudaStreamCreate(&stream2), "cudaStreamCreate(stream2)");
    checkCuda(cudaStreamCreate(&stream3), "cudaStreamCreate(stream3)");
    checkCuda(cudaStreamCreate(&stream4), "cudaStreamCreate(stream4)");

    size_t free_t, total_t;
    checkCuda(cudaMemGetInfo(&free_t, &total_t), "cudaMemGetInfo");

    checkCuda(cudaMemPrefetchAsync( gpuWall, ((rows * cols - cols) * sizeof(int)) > free_t ? free_t : ((rows * cols - cols) * sizeof(int)), 0, stream1), "cudaMemPrefetchAsync(gpuWall)");
    checkCuda(cudaMemGetInfo(&free_t, &total_t), "cudaMemGetInfo");
    checkCuda(cudaMemPrefetchAsync( gpuResult[0], (sizeof(int) * cols) > free_t ? free_t : (sizeof(int) * cols), 0, stream2),  "cudaMemPrefetchAsync(gpuResult[0])");
    checkCuda(cudaMemGetInfo(&free_t, &total_t), "cudaMemGetInfo");
    checkCuda(cudaMemPrefetchAsync(  gpuResult[1], (sizeof(int) * cols) > free_t ? free_t : (sizeof(int) * cols),0, stream3), "cudaMemPrefetchAsync(gpuResult[1])");

    checkCuda(cudaStreamSynchronize(stream1), "cudaStreamSynchronize(stream1)");
    checkCuda(cudaStreamSynchronize(stream2), "cudaStreamSynchronize(stream2)");
    checkCuda(cudaStreamSynchronize(stream3), "cudaStreamSynchronize(stream3)");

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(blockCols);

    auto start = high_resolution_clock::now();

    int src = 1, dst = 0;
    for (int t = 0; t < (int)rows - 1; t += pyramid_height) {
        int temp = src;
        src = dst;
        dst = temp;
        dynproc_kernel<<<dimGrid, dimBlock, 0, stream4>>>(
            MIN(pyramid_height, (int)rows - t - 1),
            gpuWall, gpuResult[src], gpuResult[dst],
            cols, rows, t, borderCols);
    }

    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    auto end = high_resolution_clock::now();
    auto dur = end - start;
    cout << "Runtime: "
         << duration_cast<std::chrono::microseconds>(dur).count()
         << " us" << endl;

    printf("Logical Input Size: %.2f GiB\n",
           (double)total_elements * sizeof(int) / (1024.0 * 1024.0 * 1024.0));

    printf("Managed Allocation Footprint: %.2f GiB\n",
           (double)(total_elements + cols) * sizeof(int) / (1024.0 * 1024.0 * 1024.0));

    return dst;
#else
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(blockCols);

    auto start = high_resolution_clock::now();

    int src = 1, dst = 0;
    for (int t = 0; t < (int)rows - 1; t += pyramid_height) {
        int temp = src;
        src = dst;
        dst = temp;
        dynproc_kernel<<<dimGrid, dimBlock>>>(
            MIN(pyramid_height, (int)rows - t - 1),
            gpuWall, gpuResult[src], gpuResult[dst],
            cols, rows, t, borderCols);
    }

    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    auto end = high_resolution_clock::now();
    auto dur = end - start;
    cout << "Runtime: "
         << duration_cast<std::chrono::microseconds>(dur).count()
         << " us" << endl;

    printf("Logical Input Size: %.2f GiB\n",
           (double)total_elements * sizeof(int) / (1024.0 * 1024.0 * 1024.0));

    printf("Managed Allocation Footprint: %.2f GiB\n",
           (double)(total_elements + cols) * sizeof(int) / (1024.0 * 1024.0 * 1024.0));

    return dst;
#endif
}

int main(int argc, char** argv)
{
    int num_devices = 0;
    checkCuda(cudaGetDeviceCount(&num_devices), "cudaGetDeviceCount");
    if (num_devices > 1) {
        checkCuda(cudaSetDevice(DEVICE), "cudaSetDevice");
    }
    run(argc, argv);
    return EXIT_SUCCESS;
}

void run(int argc, char** argv)
{
    if (argc != 4) {
        printf("Usage: dynproc row_len col_len pyramid_height\n");
        exit(EXIT_FAILURE);
    }

    cols = parse_size_t(argv[1], "cols");
    rows = parse_size_t(argv[2], "rows");
    pyramid_height = parse_int(argv[3], "pyramid_height");

    validate_inputs();

    /* --------------- pyramid parameters --------------- */
    int borderCols = pyramid_height * HALO;
    int smallBlockCol = BLOCK_SIZE - pyramid_height * HALO * 2;
    int blockCols = (int)(cols / smallBlockCol + ((cols % smallBlockCol == 0) ? 0 : 1));

    printf("pyramidHeight: %d\ngridSize: [%zu]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",
           pyramid_height, cols, borderCols, BLOCK_SIZE, blockCols, smallBlockCol);

    size_t total_elements = rows * cols;
    size_t wall_elements = total_elements - cols;

    int *gpuWall = nullptr;
    int *gpuResult[2] = {nullptr, nullptr};

    checkCuda(cudaMallocManaged(&gpuResult[0], sizeof(int) * cols),
              "cudaMallocManaged(gpuResult[0])");
    checkCuda(cudaMallocManaged(&gpuResult[1], sizeof(int) * cols),
              "cudaMallocManaged(gpuResult[1])");
    checkCuda(cudaMallocManaged(&gpuWall, sizeof(int) * wall_elements),
              "cudaMallocManaged(gpuWall)");

    // Initialize directly in managed memory
    srand(M_SEED);

    for (size_t j = 0; j < cols; j++) {
        gpuResult[0][j] = rand() % 10;
    }

    // Optional, but keeps the buffer deterministic
    memset(gpuResult[1], 0, sizeof(int) * cols);

    for (size_t i = 0; i < wall_elements; i++) {
        gpuWall[i] = rand() % 10;
    }

    int device = 0;
    checkCuda(cudaGetDevice(&device), "cudaGetDevice");

#ifdef MEMADVISE
    checkCuda(cudaMemAdvise((void*)gpuResult[0], sizeof(int) * cols,  cudaMemAdviseSetAccessedBy, device), "cudaMemAdvise(gpuResult[0], device)");
    checkCuda(cudaMemAdvise((void*)gpuResult[0], sizeof(int) * cols,  cudaMemAdviseSetAccessedBy, cudaCpuDeviceId), "cudaMemAdvise(gpuResult[0], cpu)");
    checkCuda(cudaMemAdvise((void*)gpuResult[1], sizeof(int) * cols,  cudaMemAdviseSetAccessedBy, device), "cudaMemAdvise(gpuResult[1], device)");
    checkCuda(cudaMemAdvise((void*)gpuResult[1], sizeof(int) * cols, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId), "cudaMemAdvise(gpuResult[1], cpu)");
    checkCuda(cudaMemAdvise((void*)gpuWall, sizeof(int) * wall_elements, cudaMemAdviseSetAccessedBy, device), "cudaMemAdvise(gpuWall, device)");
    checkCuda(cudaMemAdvise((void*)gpuWall, sizeof(int) * wall_elements, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId), "cudaMemAdvise(gpuWall, cpu)");
#endif

    int final_ret = -1;
    for (int i = 0; i < 1; i++) {
        final_ret = calc_path(gpuWall, gpuResult, rows, cols,
                              pyramid_height, blockCols, borderCols);
    }

#ifdef BENCH_PRINT
    for (size_t i = 0; i < cols; i++)
        printf("%d ", gpuResult[final_ret][i]);
    printf("\n");
#endif

    checkCuda(cudaFree(gpuWall), "cudaFree(gpuWall)");
    checkCuda(cudaFree(gpuResult[0]), "cudaFree(gpuResult[0])");
    checkCuda(cudaFree(gpuResult[1]), "cudaFree(gpuResult[1])");
}