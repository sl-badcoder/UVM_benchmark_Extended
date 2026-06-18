#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <chrono>
#include <iostream>
#include <cstring>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <limits>

#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define STR_SIZE 256
#define DEVICE 0
#define HALO 1

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

// Global variables for managed memory
size_t rows = 0, cols = 0;
int* data = nullptr;     // Managed: Holds the full matrix
int** wall = nullptr;    // Managed: Row pointers
int* result = nullptr;   // Managed: Final output
int* gpuResult[2] = {nullptr, nullptr}; // Managed: Ping-pong buffers
int pyramid_height = 0;

#define M_SEED 9
#define IN_RANGE(x, min, max)   ((x) >= (min) && (x) <= (max))
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

static inline void fatal(const char* s) {
    fprintf(stderr, "error: %s\n", s);
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err__));             \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// Helper to apply requested memory advice for zero-copy hints
void apply_zero_copy_advice(void* ptr, size_t size) {
    if (size == 0) return;
    CUDA_CHECK(cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    CUDA_CHECK(cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, DEVICE));
}

static bool checked_mul_size_t(size_t a, size_t b, size_t& out) {
    if (a == 0 || b == 0) { out = 0; return true; }
    if (a > std::numeric_limits<size_t>::max() / b) return false;
    out = a * b;
    return true;
}

static size_t parse_size_t_arg(const char* s, const char* name) {
    char* end = nullptr;
    unsigned long long v = strtoull(s, &end, 10);
    return static_cast<size_t>(v);
}

static int parse_int_arg(const char* s, const char* name, int min_value, int max_value) {
    char* end = nullptr;
    long v = strtol(s, &end, 10);
    return static_cast<int>(v);
}

__global__ void dynproc_kernel(
    int iteration,
    const int* gpuWall,
    const int* gpuSrc,
    int* gpuResults,
    size_t cols,
    size_t rows,
    size_t startStep,
    int border)
{
    __shared__ int prev[BLOCK_SIZE];
    __shared__ int result_sh[BLOCK_SIZE];

    const int bx = blockIdx.x;
    const int tx = threadIdx.x;
    const int small_block_cols = BLOCK_SIZE - iteration * HALO * 2;
    const int blkX = small_block_cols * bx - border;
    const int blkXmax = blkX + BLOCK_SIZE - 1;
    const int xidx_i = blkX + tx;
    const int cols_i = static_cast<int>(cols);

    const int validXmin = (blkX < 0) ? -blkX : 0;
    const int validXmax = (blkXmax > cols_i - 1) ? BLOCK_SIZE - 1 - (blkXmax - cols_i + 1) : BLOCK_SIZE - 1;

    int W = tx - 1;
    int E = tx + 1;
    W = (W < validXmin) ? validXmin : W;
    E = (E > validXmax) ? validXmax : E;

    const bool isValid = IN_RANGE(tx, validXmin, validXmax);

    if (IN_RANGE(xidx_i, 0, cols_i - 1)) {
        prev[tx] = gpuSrc[xidx_i];
    }
    __syncthreads();

    bool computed = false;
    for (int i = 0; i < iteration; ++i) {
        computed = false;
        if (IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) && isValid) {
            computed = true;
            int shortest = MIN(prev[W], prev[tx]);
            shortest = MIN(shortest, prev[E]);

            const size_t row_index = startStep + static_cast<size_t>(i);
            // gpuWall here points to data + cols
            result_sh[tx] = shortest + gpuWall[row_index * cols + xidx_i];
        }
        __syncthreads();
        if (i == iteration - 1) break;
        if (computed) prev[tx] = result_sh[tx];
        __syncthreads();
    }

    if (computed) {
        gpuResults[xidx_i] = result_sh[tx];
    }
}

void prefetch_data_to_gpu(size_t rows, size_t cols) {
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    
    // Safety margin (10% of VRAM)
    size_t available_vram = free_mem * 0.9;
    
    size_t result_bytes = cols * sizeof(int);
    size_t wall_bytes = (rows - 1) * cols * sizeof(int);

    // Priority 1: gpuResult buffers (Crucial for ping-pong)
    if (available_vram >= result_bytes * 2) {
        CUDA_CHECK(cudaMemPrefetchAsync(gpuResult[0], result_bytes, DEVICE));
        CUDA_CHECK(cudaMemPrefetchAsync(gpuResult[1], result_bytes, DEVICE));
        available_vram -= (result_bytes * 2);
        printf("Prefetched Ping-Pong buffers to GPU.\n");
    }

    // Priority 2: gpuWall (The matrix data)
    // gpuWall starts at data + cols
    if (available_vram >= wall_bytes) {
        CUDA_CHECK(cudaMemPrefetchAsync(data + cols, wall_bytes, DEVICE));
        printf("Prefetched full matrix data to GPU.\n");
    } else if (available_vram > 0) {
        // Prefetch as many rows as possible if the whole thing doesn't fit
        size_t partial_rows = available_vram / (cols * sizeof(int));
        if (partial_rows > 0) {
            CUDA_CHECK(cudaMemPrefetchAsync(data + cols, partial_rows * cols * sizeof(int), DEVICE));
            printf("Prefetched %zu rows of matrix to GPU (VRAM limit).\n", partial_rows);
        }
    }
}

int calc_path(int* gpuWall, int* gpuResult[2], size_t rows, size_t cols,
              int pyramid_height, int blockCols, int borderCols)
{
    auto start = high_resolution_clock::now();

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(static_cast<unsigned int>(blockCols));
    int src = 1, dst = 0;

    for (size_t t = 0; t + 1 < rows; t += static_cast<size_t>(pyramid_height)) {
        const int temp = src;
        src = dst;
        dst = temp;

        const size_t remaining = rows - t - 1;
        const int iter = MIN(pyramid_height, static_cast<int>(remaining));

        dynproc_kernel<<<dimGrid, dimBlock>>>(
            iter, gpuWall, gpuResult[src], gpuResult[dst], 
            cols, rows, t, borderCols
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = high_resolution_clock::now();
    cout << "Runtime: " << duration_cast<std::chrono::microseconds>(end - start).count() << " us" << endl;

    return dst;
}

void run(int argc, char** argv) {
    if (argc != 4) fatal("Usage: dynproc col_len row_len pyramid_height");

    cols = parse_size_t_arg(argv[1], "cols");
    rows = parse_size_t_arg(argv[2], "rows");
    pyramid_height = parse_int_arg(argv[3], "pyramid_height", 1, BLOCK_SIZE / 2 - 1);

    size_t total_elements = rows * cols;
    size_t data_bytes = total_elements * sizeof(int);
    size_t wall_ptr_bytes = rows * sizeof(int*);
    size_t res_bytes = cols * sizeof(int);

    // 1. Allocate Managed Memory
    CUDA_CHECK(cudaMallocManaged(&data, data_bytes));
    CUDA_CHECK(cudaMallocManaged(&wall, wall_ptr_bytes));
    CUDA_CHECK(cudaMallocManaged(&result, res_bytes));
    CUDA_CHECK(cudaMallocManaged(&gpuResult[0], res_bytes));
    CUDA_CHECK(cudaMallocManaged(&gpuResult[1], res_bytes));

    // 2. Set Memory Advice (Zero-Copy Hints)
    apply_zero_copy_advice(data, data_bytes);
    apply_zero_copy_advice(wall, wall_ptr_bytes);
    apply_zero_copy_advice(result, res_bytes);
    apply_zero_copy_advice(gpuResult[0], res_bytes);
    apply_zero_copy_advice(gpuResult[1], res_bytes);

    // Initialize wall pointers and data on CPU
    for (size_t n = 0; n < rows; ++n) wall[n] = data + (n * cols);
    srand(M_SEED);
    for (size_t i = 0; i < total_elements; ++i) data[i] = rand() % 10;
    
    // Copy first row to starting result buffer (Initial state)
    memcpy(gpuResult[0], data, res_bytes);

    // Calculate grid logic
    const int borderCols = pyramid_height * HALO;
    const int smallBlockCol = BLOCK_SIZE - pyramid_height * HALO * 2;
    const int blockCols = (cols + smallBlockCol - 1) / smallBlockCol;

    // 3. VRAM-aware Prefetching
    // We prioritize the result buffers, then the rest of the wall data
    prefetch_data_to_gpu(rows, cols);

    int final_idx = calc_path(data + cols, gpuResult, rows, cols, pyramid_height, blockCols, borderCols);

    // Copy final result to CPU result buffer
    memcpy(result, gpuResult[final_idx], res_bytes);

    // Cleanup
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(wall));
    CUDA_CHECK(cudaFree(result));
    CUDA_CHECK(cudaFree(gpuResult[0]));
    CUDA_CHECK(cudaFree(gpuResult[1]));
}

int main(int argc, char** argv) {
    CUDA_CHECK(cudaSetDevice(DEVICE));
    run(argc, argv);
    return EXIT_SUCCESS;
}