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

// Global pointers now point to Managed Memory
size_t rows = 0, cols = 0;
int* data = nullptr;   // Unified Memory for the whole grid
int* result = nullptr; // Unified Memory for the final result
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

// Helper for overflow-safe size calculations
static size_t checked_bytes_for_ints(size_t count) {
    size_t out;
    if (count == 0) return 0;
    if (count > std::numeric_limits<size_t>::max() / sizeof(int)) {
        fatal("size overflow while computing byte count");
    }
    return count * sizeof(int);
}

__global__ void dynproc_kernel(
    int iteration,
    const int* gpuWall,
    int* gpuSrc,
    int* gpuDst,
    size_t cols,
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
            int left = prev[W];
            int up = prev[tx];
            int right = prev[E];

            int shortest = MIN(left, up);
            shortest = MIN(shortest, right);

            // Accessing the 'Wall' portion of managed memory
            const size_t row_index = startStep + static_cast<size_t>(i);
            const size_t index = row_index * cols + xidx_i;
            result_sh[tx] = shortest + gpuWall[index];
        }
        __syncthreads();

        if (i == iteration - 1) break;
        if (computed) prev[tx] = result_sh[tx];
        __syncthreads();
    }

    if (computed) {
        gpuDst[xidx_i] = result_sh[tx];
    }
}

void run(int argc, char** argv) {
    if (argc != 4) fatal("Usage: dynproc col_len row_len pyramid_height");

    cols = std::stoull(argv[1]);
    rows = std::stoull(argv[2]);
    pyramid_height = std::stoi(argv[3]);

    size_t total_elements = rows * cols;
    size_t data_bytes = checked_bytes_for_ints(total_elements);
    size_t result_bytes = checked_bytes_for_ints(cols);

    // 1. Allocate Managed Memory
    CUDA_CHECK(cudaMallocManaged(&data, data_bytes));
    CUDA_CHECK(cudaMallocManaged(&result, result_bytes));

    // Initialize data on CPU
    srand(M_SEED);
    for (size_t i = 0; i < total_elements; ++i) {
        data[i] = rand() % 10;
    }

    // 2. Optimization: Memory Advice
    // The "Wall" data (everything after the first row) is read-only for the GPU
    if (rows > 1) {
        CUDA_CHECK(cudaMemAdvise(data + cols, data_bytes - result_bytes, cudaMemAdviseSetReadMostly, DEVICE));
    }

    // 3. Optimization: Prefetch to GPU
    CUDA_CHECK(cudaMemPrefetchAsync(data, data_bytes, DEVICE));

    // We need a second buffer for the ping-pong swap in the kernel
    int* gpuTempResult;
    CUDA_CHECK(cudaMallocManaged(&gpuTempResult, result_bytes));

    int* src_ptr = data; // First row of data is our initial source
    int* dst_ptr = gpuTempResult;

    int grid_size = (cols + (BLOCK_SIZE - pyramid_height * HALO * 2) - 1) / (BLOCK_SIZE - pyramid_height * HALO * 2);
    int borderCols = pyramid_height * HALO;

    auto start = high_resolution_clock::now();

    // Kernel Loop
    for (size_t t = 0; t + 1 < rows; t += static_cast<size_t>(pyramid_height)) {
        const size_t remaining = rows - t - 1;
        const int iter = MIN(pyramid_height, static_cast<int>(remaining));

        // Use index 'cols' as the start of the "Wall" (the rows to be added)
        dynproc_kernel<<<grid_size, BLOCK_SIZE>>>(
            iter,
            data + cols, 
            src_ptr,
            dst_ptr,
            cols,
            t,
            borderCols
        );
        
        // Ping-pong pointers
        int* temp = src_ptr;
        src_ptr = dst_ptr;
        dst_ptr = temp;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = high_resolution_clock::now();

    // 4. Optimization: Prefetch result back to CPU for access
    CUDA_CHECK(cudaMemPrefetchAsync(src_ptr, result_bytes, cudaCpuDeviceId));
    
    // Copy the final ping-pong result to our 'result' pointer
    memcpy(result, src_ptr, result_bytes);

    cout << "Runtime: " << duration_cast<std::chrono::microseconds>(end - start).count() << " us" << endl;

    // Cleanup
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(result));
    CUDA_CHECK(cudaFree(gpuTempResult));
}

int main(int argc, char** argv) {
    run(argc, argv);
    return 0;
}