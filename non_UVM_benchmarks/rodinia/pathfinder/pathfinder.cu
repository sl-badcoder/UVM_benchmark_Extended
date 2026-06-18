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

void run(int argc, char** argv);

size_t rows = 0, cols = 0;
int* data = nullptr;
int** wall = nullptr;
int* result = nullptr;
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
    if (a == 0 || b == 0) {
        out = 0;
        return true;
    }
    if (a > std::numeric_limits<size_t>::max() / b) {
        return false;
    }
    out = a * b;
    return true;
}

static bool checked_add_size_t(size_t a, size_t b, size_t& out) {
    if (a > std::numeric_limits<size_t>::max() - b) {
        return false;
    }
    out = a + b;
    return true;
}

static size_t checked_bytes_for_ints(size_t count) {
    size_t bytes = 0;
    if (!checked_mul_size_t(count, sizeof(int), bytes)) {
        fatal("size overflow while computing byte count for int buffer");
    }
    return bytes;
}

static size_t checked_bytes_for_int_ptrs(size_t count) {
    size_t bytes = 0;
    if (!checked_mul_size_t(count, sizeof(int*), bytes)) {
        fatal("size overflow while computing byte count for int* buffer");
    }
    return bytes;
}

static size_t parse_size_t_arg(const char* s, const char* name) {
    if (s == nullptr || *s == '\0') {
        fprintf(stderr, "Invalid %s: empty string\n", name);
        std::exit(EXIT_FAILURE);
    }

    errno = 0;
    char* end = nullptr;
    unsigned long long v = strtoull(s, &end, 10);

    if (errno == ERANGE || end == s || *end != '\0') {
        fprintf(stderr, "Invalid %s: '%s'\n", name, s);
        std::exit(EXIT_FAILURE);
    }

    if (v == 0) {
        fprintf(stderr, "%s must be > 0\n", name);
        std::exit(EXIT_FAILURE);
    }

    if (v > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
        fprintf(stderr, "%s is too large for this platform\n", name);
        std::exit(EXIT_FAILURE);
    }

    return static_cast<size_t>(v);
}

static int parse_int_arg(const char* s, const char* name, int min_value, int max_value) {
    if (s == nullptr || *s == '\0') {
        fprintf(stderr, "Invalid %s: empty string\n", name);
        std::exit(EXIT_FAILURE);
    }

    errno = 0;
    char* end = nullptr;
    long v = strtol(s, &end, 10);

    if (errno == ERANGE || end == s || *end != '\0') {
        fprintf(stderr, "Invalid %s: '%s'\n", name, s);
        std::exit(EXIT_FAILURE);
    }

    if (v < min_value || v > max_value) {
        fprintf(stderr, "%s must be in [%d, %d]\n", name, min_value, max_value);
        std::exit(EXIT_FAILURE);
    }

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

    if (rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
        fatal("rows exceeds supported range for host-side loops using int");
    }
    if (cols > static_cast<size_t>(std::numeric_limits<int>::max())) {
        fatal("cols exceeds supported range for current kernel/block math");
    }

    size_t total_elements = 0;
    if (!checked_mul_size_t(rows, cols, total_elements)) {
        fatal("rows * cols overflow");
    }

    const size_t data_bytes = checked_bytes_for_ints(total_elements);
    const size_t wall_ptr_bytes = checked_bytes_for_int_ptrs(rows);
    const size_t result_bytes = checked_bytes_for_ints(cols);

    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&data), data_bytes));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&wall), wall_ptr_bytes));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&result), result_bytes));

    cout << static_cast<double>(data_bytes) / (1024.0 * 1024.0 * 1024.0) << endl;
    cout << static_cast<double>(wall_ptr_bytes) / (1024.0 * 1024.0 * 1024.0) << endl;

    for (size_t n = 0; n < rows; ++n) {
        size_t row_offset = 0;
        if (!checked_mul_size_t(cols, n, row_offset)) {
            fatal("cols * row index overflow");
        }
        wall[n] = data + row_offset;
    }

    srand(M_SEED);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            wall[i][j] = rand() % 10;
        }
    }
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
    (void)rows;

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
        const size_t xidx = static_cast<size_t>(xidx_i);
        prev[tx] = gpuSrc[xidx];
    }
    __syncthreads();

    bool computed = false;

    for (int i = 0; i < iteration; ++i) {
        computed = false;

        if (IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) && isValid) {
            computed = true;

            const int left = prev[W];
            const int up = prev[tx];
            const int right = prev[E];

            int shortest = MIN(left, up);
            shortest = MIN(shortest, right);

            const size_t xidx = static_cast<size_t>(xidx_i);
            const size_t row_index = startStep + static_cast<size_t>(i);
            const size_t index = row_index * cols + xidx;

            result_sh[tx] = shortest + gpuWall[index];
        }

        __syncthreads();

        if (i == iteration - 1) {
            break;
        }

        if (computed) {
            prev[tx] = result_sh[tx];
        }

        __syncthreads();
    }

    if (computed) {
        const size_t xidx = static_cast<size_t>(xidx_i);
        gpuResults[xidx] = result_sh[tx];
    }
}

int calc_path(int* gpuWall, int* gpuResult[2], size_t rows, size_t cols,
              int pyramid_height, int blockCols, int borderCols)
{
    size_t total_elements = 0;
    if (!checked_mul_size_t(rows, cols, total_elements)) {
        fatal("rows * cols overflow in calc_path");
    }

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
            iter,
            gpuWall,
            gpuResult[src],
            gpuResult[dst],
            cols,
            rows,
            t,
            borderCols
        );
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = high_resolution_clock::now();
    auto dur = end - start;

    cout << "Runtime: "
         << duration_cast<std::chrono::microseconds>(dur).count()
         << " us" << endl;

    size_t footprint_elems = 0;
    if (!checked_add_size_t(total_elements, cols, footprint_elems)) {
        fatal("allocation footprint overflow");
    }

    printf("Explicit Allocation Footprint: %.2f GiB\n",
           static_cast<double>(footprint_elems) * sizeof(int) /
           (1024.0 * 1024.0 * 1024.0));

    return dst;
}

int main(int argc, char** argv) {
    int num_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    if (num_devices > 1) {
        CUDA_CHECK(cudaSetDevice(DEVICE));
    }
    run(argc, argv);
    return EXIT_SUCCESS;
}

void run(int argc, char** argv) {
    init(argc, argv);

    const int borderCols = pyramid_height * HALO;
    const int smallBlockCol = BLOCK_SIZE - pyramid_height * HALO * 2;

    if (smallBlockCol <= 0) {
        fatal("pyramid_height is too large; smallBlockCol must be > 0");
    }

    if (cols > static_cast<size_t>(std::numeric_limits<int>::max())) {
        fatal("cols too large for current block/grid calculation");
    }

    const size_t blockCols_sz =
        cols / static_cast<size_t>(smallBlockCol) +
        ((cols % static_cast<size_t>(smallBlockCol) == 0) ? 0 : 1);

    if (blockCols_sz > static_cast<size_t>(std::numeric_limits<int>::max())) {
        fatal("blockCols exceeds int range");
    }

    const int blockCols = static_cast<int>(blockCols_sz);

    printf("pyramidHeight: %d\ngridSize: [%zu]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",
           pyramid_height, cols, borderCols, BLOCK_SIZE, blockCols, smallBlockCol);

    int* gpuWall = nullptr;
    int* gpuResult[2] = {nullptr, nullptr};

    size_t size = 0;
    if (!checked_mul_size_t(rows, cols, size)) {
        fatal("rows * cols overflow in run");
    }

    cout << size << endl;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gpuResult[0]), checked_bytes_for_ints(cols)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gpuResult[1]), checked_bytes_for_ints(cols)));

    CUDA_CHECK(cudaMemcpy(
        gpuResult[0],
        data,
        checked_bytes_for_ints(cols),
        cudaMemcpyHostToDevice
    ));

    if (rows > 1) {
        const size_t wall_elems = size - cols;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gpuWall), checked_bytes_for_ints(wall_elems)));
        CUDA_CHECK(cudaMemcpy(
            gpuWall,
            data + cols,
            checked_bytes_for_ints(wall_elems),
            cudaMemcpyHostToDevice
        ));
    } else {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gpuWall), 0));
    }

    int final_ret = 0;
    final_ret = calc_path(gpuWall, gpuResult, rows, cols, pyramid_height, blockCols, borderCols);

    CUDA_CHECK(cudaMemcpy(
        result,
        gpuResult[final_ret],
        checked_bytes_for_ints(cols),
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaFree(gpuWall));
    CUDA_CHECK(cudaFree(gpuResult[0]));
    CUDA_CHECK(cudaFree(gpuResult[1]));

    CUDA_CHECK(cudaFreeHost(data));
    CUDA_CHECK(cudaFreeHost(result));
    CUDA_CHECK(cudaFreeHost(wall));
}