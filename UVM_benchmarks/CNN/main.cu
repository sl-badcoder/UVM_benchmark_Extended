#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "layer.h"
#include "mnist.h"

#include <cstdio>
#include <cuda.h>
#include <time.h>
#include <chrono>

static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

// Define layers of CNN
static Layer l_input = Layer(0, 0, 28 * 28);
static Layer l_c1 = Layer(5 * 5, 6, 24 * 24 * 6);
static Layer l_s1 = Layer(4 * 4, 1, 6 * 6 * 6);
static Layer l_f = Layer(6 * 6 * 6, 10, 10);

static void learn();
static unsigned int classify(double data[28][28]);
static void test();
static double forward_pass(double* device_data_ptr);
static double back_pass();
__global__ void k_NormalizeAndCast(double* incoming, float* outgoing, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Since the loader already divided by 255.0, we just cast to float
        outgoing[i] = (float)incoming[i];
    }
}
static inline void loaddata() {
  mnist_load("data/massive-images.idx3-ubyte", "data/massive-labels.idx1-ubyte",
             &train_set, &train_cnt);
  mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
             &test_set, &test_cnt);
  printf("Data loading done!\n");
}

int main(int argc, const char **argv) {
  srand(time(NULL));

  CUresult err = cuInit(0);
  if (err != CUDA_SUCCESS) {
    fprintf(stderr, "CUDA initialisation failed with error code - %d\n", err);
    return 1;
  }

  loaddata();
  int deviceId = 0;
  cudaGetDevice(&deviceId);
  size_t free_m, total_m;
  cudaMemGetInfo(&free_m, &total_m);

  size_t total_train_size = (sizeof(mnist_data) * (size_t)train_cnt);
  size_t sz = total_train_size / (1024.0 * 1024.0 * 1024.0);
  printf("%llu\n", sz);
  cudaMemAdvise(train_set, total_train_size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
  cudaMemAdvise(train_set, total_train_size, cudaMemAdviseUnsetAccessedBy, 0);

  cudaMemPrefetchAsync(train_set, total_train_size, deviceId, NULL);
  cudaDeviceSynchronize();
  learn();
  cudaMemGetInfo(&free_m, &total_m);

  size_t total_test_size = (sizeof(mnist_data) * (size_t)test_cnt) % free_m;
  cudaMemPrefetchAsync(train_set, total_test_size, deviceId, NULL);
  test();
  cudaDeviceSynchronize();

  return 0;
}

// Forward propagation of a single row in dataset
// Change the signature to take a pointer
static double forward_pass(double* device_data_ptr) {
  l_input.clear();
  l_c1.clear();
  l_s1.clear();
  l_f.clear();

  clock_t start, end;
  start = clock();

  k_NormalizeAndCast<<<2, 392>>>(device_data_ptr, (float*)l_input.output, 784);

  fp_preact_c1<<<64, 64>>>((float(*)[28])l_input.output,
                           (float(*)[24][24])l_c1.preact,
                           (float(*)[5][5])l_c1.weight);
  fp_bias_c1<<<64, 64>>>((float(*)[24][24])l_c1.preact, l_c1.bias);
  apply_step_function<<<64, 64>>>(l_c1.preact, l_c1.output, l_c1.O);

  fp_preact_s1<<<64, 64>>>((float(*)[24][24])l_c1.output,
                           (float(*)[6][6])l_s1.preact,
                           (float(*)[4][4])l_s1.weight);
  fp_bias_s1<<<64, 64>>>((float(*)[6][6])l_s1.preact, l_s1.bias);
  apply_step_function<<<64, 64>>>(l_s1.preact, l_s1.output, l_s1.O);

  fp_preact_f<<<64, 64>>>((float(*)[6][6])l_s1.output, l_f.preact,
                          (float(*)[6][6][6])l_f.weight);
  fp_bias_f<<<64, 64>>>(l_f.preact, l_f.bias);
  apply_step_function<<<64, 64>>>(l_f.preact, l_f.output, l_f.O);

  cudaDeviceSynchronize();
  end = clock();
  return ((double)(end - start)) / CLOCKS_PER_SEC;
}

// Back propagation to update weights
static double back_pass() {
  clock_t start, end;

  start = clock();

  bp_weight_f<<<64, 64>>>((float(*)[6][6][6])l_f.d_weight, l_f.d_preact,
                          (float(*)[6][6])l_s1.output);
  bp_bias_f<<<64, 64>>>(l_f.bias, l_f.d_preact);

  bp_output_s1<<<64, 64>>>((float(*)[6][6])l_s1.d_output,
                           (float(*)[6][6][6])l_f.weight, l_f.d_preact);
  bp_preact_s1<<<64, 64>>>((float(*)[6][6])l_s1.d_preact,
                           (float(*)[6][6])l_s1.d_output,
                           (float(*)[6][6])l_s1.preact);
  bp_weight_s1<<<64, 64>>>((float(*)[4][4])l_s1.d_weight,
                           (float(*)[6][6])l_s1.d_preact,
                           (float(*)[24][24])l_c1.output);
  bp_bias_s1<<<64, 64>>>(l_s1.bias, (float(*)[6][6])l_s1.d_preact);

  bp_output_c1<<<64, 64>>>((float(*)[24][24])l_c1.d_output,
                           (float(*)[4][4])l_s1.weight,
                           (float(*)[6][6])l_s1.d_preact);
  bp_preact_c1<<<64, 64>>>((float(*)[24][24])l_c1.d_preact,
                           (float(*)[24][24])l_c1.d_output,
                           (float(*)[24][24])l_c1.preact);
  bp_weight_c1<<<64, 64>>>((float(*)[5][5])l_c1.d_weight,
                           (float(*)[24][24])l_c1.d_preact,
                           (float(*)[28])l_input.output);
  bp_bias_c1<<<64, 64>>>(l_c1.bias, (float(*)[24][24])l_c1.d_preact);

  apply_grad<<<64, 64>>>(l_f.weight, l_f.d_weight, l_f.M * l_f.N);
  apply_grad<<<64, 64>>>(l_s1.weight, l_s1.d_weight, l_s1.M * l_s1.N);
  apply_grad<<<64, 64>>>(l_c1.weight, l_c1.d_weight, l_c1.M * l_c1.N);
  cudaDeviceSynchronize();

  end = clock();
  return ((double)(end - start)) / CLOCKS_PER_SEC;
}

static void unfold_input(double input[28][28],
                         double unfolded[24 * 24][5 * 5]) {
  int a = 0;
  (void)unfold_input;

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) {
      int b = 0;
      for (int x = i; x < i + 2; ++x)
        for (int y = j; y < j + 2; ++y)
          unfolded[a][b++] = input[x][y];
      a++;
    }
}

static void learn() {
  static cublasHandle_t blas;
  cublasCreate(&blas);
  float err;
  int iter = 1;
  double time_taken = 0.0;

  fprintf(stdout, "Learning\n");
  auto total_start = std::chrono::high_resolution_clock::now();
  while (iter-- > 0) {
    err = 0.0f;
    for (int i = 0; i < train_cnt; ++i) {
      if (i % 500 == 0 || i == train_cnt - 1) {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = now - total_start;
        
        double img_per_sec = (i + 1) / elapsed.count();
        double seconds_left = (train_cnt - (i + 1)) / img_per_sec;
        float progress = (float)(i + 1) / train_cnt;

        // Simple Progress Bar
        int barWidth = 30;
        fprintf(stdout, "\r%3.0f%% [", progress * 100.0);
        int pos = barWidth * progress;
        for (int j = 0; j < barWidth; ++j) {
            if (j < pos) fprintf(stdout, "=");
            else if (j == pos) fprintf(stdout, ">");
            else fprintf(stdout, " ");
        }
        
        // ETA and Speed
        fprintf(stdout, "] %u/%u | %.1f img/s | ETA: ", i + 1, train_cnt, img_per_sec);
        
        if (seconds_left > 3600) 
            fprintf(stdout, "%dh %dm", (int)seconds_left/3600, ((int)seconds_left%3600)/60);
        else if (seconds_left > 60)
            fprintf(stdout, "%dm %ds", (int)seconds_left/60, (int)seconds_left%60);
        else
            fprintf(stdout, "%ds", (int)seconds_left);
            
        fflush(stdout);
      }
      float tmp_err;

      time_taken += forward_pass((double*)train_set[i].data);

      l_f.bp_clear();
      l_s1.bp_clear();
      l_c1.bp_clear();

      makeError<<<10, 1>>>(l_f.d_preact, l_f.output, train_set[i].label, 10);
      cublasSnrm2(blas, 10, l_f.d_preact, 1, &tmp_err);
      err += tmp_err;

      time_taken += back_pass();
    }
    fprintf(stdout, "error: %e, time_on_gpu: %lf\n", err / train_cnt, time_taken);
  }
}
static unsigned int classify(double data[28][28]) {
  float res[10];

  forward_pass((double*)data);

  unsigned int max = 0;

  // cudaMemcpy(res, l_f.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);
  //   memcpy(res, l_f.output, sizeof(float) * 10);

  for (int i = 1; i < 10; ++i) {
    // if (res[max] < res[i]) {
    if (l_f.output[max] < l_f.output[i]) {
      max = i;
    }
  }

  return max;
}

// Perform forward propagation of test data
static void test() {
  fprintf(stdout, "Testing!\n");
  int error = 0;
  //test_cnt = 1;
  for (int i = 0; i < test_cnt; ++i) {
    if (classify(test_set[i].data) != test_set[i].label) {
      ++error;
    }
  }

  fprintf(stdout, "Error Rate: %.2lf%%\n",
          double(error) / double(test_cnt) * 100.0);
}
