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
int* data = nullptr;    // Holds the main wall data
int** wall = nullptr;   // Array of pointers for 2D host access
int* result = nullptr;  // Buffer for the final output
int* gpuResult[2] = {nullptr, nullptr}; // Ping-pong buffers for DP

#define M_SEED 9
int pyramid_height = 0;

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

static bool checked_mul_size_t(size_t a, size_t b, size_t& out) {
    if (a == 0 || b == 0) { out = 0; return true; }
    if (a > std::numeric_limits<size_t>::max() / b) return false;
    out = a * b;
    return true;
}

static size_t checked_bytes_for_ints(size_t count) {
    size_t bytes = 0;
    if (!checked_mul_size_t(count, sizeof(int), bytes)) fatal("size overflow");
    return bytes;
}

// Argument parsing remains the same...
static size_t parse_size_t_arg(const char* s, const char* name) {
    char* end = nullptr;
    unsigned long long v = strtoull(s, &end, 10);
    if (v == 0 || *end != '\0') fatal("Invalid size_t arg");
    return static_cast<size_t>(v);
}

static int parse_int_arg(const char* s, const char* name, int min_val, int max_val) {
    char* end = nullptr;
    long v = strtol(s, &end, 10);
    if (v < min_val || v > max_val || *end != '\0') fatal("Invalid int arg");
    return static_cast<int>(v);
}

void init(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: dynproc row_len col_len pyramid_height\n");
        std::exit(EXIT_FAILURE);
    }

    cols = parse_size_t_arg(argv[1], "cols");
    rows = parse_size_t_arg(argv[2], "rows");
    pyramid_height = parse_int_arg(argv[3], "pyramid_height", 1, BLOCK_SIZE / 2 - 1);

    size_t total_elements;
    checked_mul_size_t(rows, cols, total_elements);

    // ALLOCATION: Use Managed Memory instead of cudaMallocHost or cudaMalloc
    CUDA_CHECK(cudaMallocManaged(&data, checked_bytes_for_ints(total_elements)));
    CUDA_CHECK(cudaMallocManaged(&result, checked_bytes_for_ints(cols)));
    CUDA_CHECK(cudaMallocManaged(&wall, rows * sizeof(int*)));
    CUDA_CHECK(cudaMallocManaged(&gpuResult[0], checked_bytes_for_ints(cols)));
    CUDA_CHECK(cudaMallocManaged(&gpuResult[1], checked_bytes_for_ints(cols)));

    // Setup row pointers for CPU-side initialization
    for (size_t n = 0; n < rows; ++n) {
        wall[n] = data + (n * cols);
    }

    srand(M_SEED);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            wall[i][j] = rand() % 10;
        }
    }

    // Initialize the first buffer of the ping-pong with the first row of data
    memcpy(gpuResult[0], data, cols * sizeof(int));
}

__global__ void dynproc_kernel(
    int iteration,
    const int* gpuWall,
    const int* gpuSrc,
    int* gpuResults,
    size_t cols,
    size_t startStep)
{
    __shared__ int prev[BLOCK_SIZE];
    __shared__ int result_sh[BLOCK_SIZE];

    const int bx = blockIdx.x;
    const int tx = threadIdx.x;
    const int border = iteration * HALO; 
    const int small_block_cols = BLOCK_SIZE - iteration * HALO * 2;

    const int blkX = small_block_cols * bx - border;
    const int blkXmax = blkX + BLOCK_SIZE - 1;
    const int xidx_i = blkX + tx;
    const int cols_i = static_cast<int>(cols);

    const int validXmin = (blkX < 0) ? -blkX : 0;
    const int validXmax = (blkXmax > cols_i - 1) ? BLOCK_SIZE - 1 - (blkXmax - cols_i + 1) : BLOCK_SIZE - 1;

    int W = (tx - 1 < validXmin) ? validXmin : tx - 1;
    int E = (tx + 1 > validXmax) ? validXmax : tx + 1;

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

            // Note: gpuWall here is the offset pointer 'data + cols'
            size_t index = (startStep + i) * cols + (size_t)xidx_i;
            result_sh[tx] = shortest + gpuWall[index];
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

int calc_path(int* gpuWall, size_t rows, size_t cols, int pyramid_height, int blockCols)
{
    auto start = high_resolution_clock::now();

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(blockCols);

    int src = 1, dst = 0;

    for (size_t t = 0; t + 1 < rows; t += static_cast<size_t>(pyramid_height)) {
        // Ping-pong switch
        int temp = src; src = dst; dst = temp;

        const int iter = MIN(pyramid_height, (int)(rows - t - 1));

        dynproc_kernel<<<dimGrid, dimBlock>>>(
            iter,
            gpuWall, // This points to the start of the data (effectively row 1 onwards)
            gpuResult[src],
            gpuResult[dst],
            cols,
            t
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize()); // Essential for Managed Memory before CPU reads

    auto end = high_resolution_clock::now();
    cout << "Runtime: " << duration_cast<std::chrono::microseconds>(end - start).count() << " us" << endl;

    return dst;
}

int main(int argc, char** argv) {
    CUDA_CHECK(cudaSetDevice(DEVICE));
    
    init(argc, argv);

    const int smallBlockCol = BLOCK_SIZE - pyramid_height * HALO * 2;
    const int blockCols = (cols + smallBlockCol - 1) / smallBlockCol;

    // In managed memory, we don't need a separate gpuWall. 
    // We just pass the 'data' pointer starting from the second row (index 'cols').
    // However, the kernel math expects row 0 of gpuWall to be the first row of processing.
    // In the original code, gpuWall was 'data + cols'. We'll pass the base pointer.
    int final_ret = calc_path(data + cols, rows, cols, pyramid_height, blockCols);

    // Final result is already in Managed Memory, just copy to 'result' or use directly
    memcpy(result, gpuResult[final_ret], cols * sizeof(int));

    // Cleanup
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(wall));
    CUDA_CHECK(cudaFree(result));
    CUDA_CHECK(cudaFree(gpuResult[0]));
    CUDA_CHECK(cudaFree(gpuResult[1]));

    return EXIT_SUCCESS;
}