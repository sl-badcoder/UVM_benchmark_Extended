#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <limits.h>
#include <cuda_runtime.h>
inline size_t getGPUFreeMemory(){
    size_t free_m;
    size_t free_t,total_t;

    cudaMemGetInfo(&free_t,&total_t);
    //free_m = free_t/(size_t)(1048576.0);

    return free_t;
}
// includes, kernels
#include "ordergraph_kernel.cu"

static inline void checkCuda(cudaError_t e, const char* file, int line) {
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(e));
    exit(1);
  }
}
#define CHECK_CUDA(x) checkCuda((x), __FILE__, __LINE__)

const int HIGHEST = 3;
int taskperthr = 1;
size_t sizepernode = 0;   // was int
int ITER = 100;

// global var
float preScore = -99999999999.0f;
float score = 0.0f;
float maxScore[HIGHEST] = {-999999999.0f};

// These are O(NODE_N^2); unchanged as requested.
bool orders[NODE_N][NODE_N];
bool preOrders[NODE_N][NODE_N];
bool preGraph[NODE_N][NODE_N];
bool bestGraph[HIGHEST][NODE_N][NODE_N];
bool graph[NODE_N][NODE_N];

float *U_LG, *localscore, *scores;
float *U_localscore, *U_scores;
bool *U_parent;
int *U_resP, *parents;
int *U_data;

void initial();  // initial orders and data
int genOrders(); // swap
int ConCore();   // discard new order or not
bool getparent(int *bit, int *pre, int posN, int *parent, int *parN, int time);
void incr(int *bit, int n);
void incrS(int *bit, int n);
bool getState(int parN, int *state, int time);
float findBestGraph();
void genScore();
void sortGraph();
void swap(int a, int b);
void Pre_logGamma();
uint64_t findindex(int *arr, int size);

// ---- overflow-safer combinatorics and integer power ----
static inline uint64_t gcd_u64(uint64_t a, uint64_t b) {
  while (b) { uint64_t t = a % b; a = b; b = t; }
  return a;
}

// exact, overflow-checked nCk for small k (your code uses k<=4)
static inline uint64_t C_u64(uint64_t n, uint64_t k) {
  if (k > n) return 0;
  if (k == 0) return 1;
  if (k > n - k) k = n - k;

  uint64_t res = 1;
  for (uint64_t i = 1; i <= k; i++) {
    uint64_t num = n - k + i;
    uint64_t den = i;

    uint64_t g = gcd_u64(num, den);
    num /= g; den /= g;

    g = gcd_u64(res, den);
    res /= g; den /= g;

    if (den != 1) {
      // Should not happen with exact reduction
      return 0;
    }
    if (num != 0 && res > UINT64_MAX / num) return 0;
    res *= num;
  }
  return res;
}

// compute base^exp in uint64 with saturation on overflow
static inline uint64_t pow_u64_sat(uint64_t base, int exp) {
  uint64_t r = 1;
  for (int i = 0; i < exp; i++) {
    if (base != 0 && r > UINT64_MAX / base) return UINT64_MAX;
    r *= base;
  }
  return r;
}

// Proper float init for U_scores (cudaMemset cannot set float values correctly)
__global__ void initFloatKernel(float* a, size_t n, float v) {
  size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (tid < n) a[tid] = v;
}

FILE *fpout;

int main() {
  // print bytes needed
  printf("NODE_N=%d DATA_N=%d\n", NODE_N, DATA_N);
  printf("U_data bytes: %zu\n", ((size_t)DATA_N * (size_t)NODE_N * sizeof(int) )/ (1024*1024*1024));
  printf("U_LG bytes: %zu\n",( (size_t)(DATA_N + 2) * sizeof(float)) / (1024*1024*1024));

  // Managed allocation + correct memcpy byte count
  size_t total_elements = (size_t)DATA_N * (size_t)NODE_N;

  // 2. Allocate Managed Memory for CUDA
  CHECK_CUDA(cudaMallocManaged(&U_data, total_elements * sizeof(int)));
  CHECK_CUDA(cudaMemAdvise(U_data,total_elements * sizeof(int), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
  CHECK_CUDA(cudaMemAdvise(U_data,total_elements * sizeof(int), cudaMemAdviseSetAccessedBy,  0));

  // 3. Automatically fill with random 0s and 1s
  printf("Generating %zu random data points...\n", total_elements);
  srand((unsigned)time(NULL)); 
  for (size_t i = 0; i < total_elements; i++) {
      U_data[i] = rand() % 2; // Generates 0 or 1
  }

  // 4. Prefetch to GPU to speed up the first kernel launch
  if( total_elements * sizeof(int) < getGPUFreeMemory())CHECK_CUDA(cudaMemPrefetchAsync(U_data, total_elements * sizeof(int), 0, 0));
  
  int i, j, c = 0, tmp, a, b;
  float tmpd;
  fpout = fopen("out.txt", "w");
  if (!fpout) {
    fprintf(stderr, "Failed to open output file\n");
    return 1;
  }

  clock_t start, finish, total = 0, pre1, pre2;
  CHECK_CUDA(cudaDeviceSynchronize());

  printf("NODE_N=%d\nInitialization...\n", NODE_N);
  pre1 = clock();

  srand((unsigned)time(NULL));
  initial();
  genScore();
  pre2 = clock();
  printf("OK, begin to generate orders.\n");

  i = 0;
  while (i != ITER) {
    start = clock();

    i++;
    score = 0.0f;

    for (a = 0; a < NODE_N; a++) {
      for (j = 0; j < NODE_N; j++) {
        orders[a][j] = preOrders[a][j];
      }
    }

    tmp = rand() % 6;
    for (j = 0; j < tmp; j++) genOrders();

    score = findBestGraph();

    finish = clock();
    total += finish - start;

    ConCore();

    // store the top HIGHEST highest orders
    if (c < HIGHEST) {
      tmp = 1;
      for (j = 0; j < c; j++) {
        if (maxScore[j] == preScore) tmp = 0;
      }
      if (tmp != 0) {
        maxScore[c] = preScore;
        for (a = 0; a < NODE_N; a++) {
          for (b = 0; b < NODE_N; b++) {
            bestGraph[c][a][b] = preGraph[a][b];
          }
        }
        c++;
      }
    } else if (c == HIGHEST) {
      sortGraph();
      c++;
    } else {
      tmp = 1;
      for (j = 0; j < HIGHEST; j++) {
        if (maxScore[j] == preScore) { tmp = 0; break; }
      }
      if (tmp != 0 && preScore > maxScore[HIGHEST - 1]) {
        maxScore[HIGHEST - 1] = preScore;
        for (a = 0; a < NODE_N; a++) {
          for (b = 0; b < NODE_N; b++) {
            bestGraph[HIGHEST - 1][a][b] = preGraph[a][b];
          }
        }
        b = HIGHEST - 1;
        for (a = HIGHEST - 2; a >= 0; a--) {
          if (maxScore[b] > maxScore[a]) {
            swap(a, b);
            tmpd = maxScore[a];
            maxScore[a] = maxScore[b];
            maxScore[b] = tmpd;
            b = a;
          }
        }
      }
    }
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaFree(U_data));
  CHECK_CUDA(cudaFree(U_localscore));
  CHECK_CUDA(cudaFree(U_parent));
  CHECK_CUDA(cudaFree(U_scores));
  CHECK_CUDA(cudaFree(U_resP));

  fprintf(fpout, "Duration per interation is %f seconds.\n",
          ((float)total / (float)ITER) / (float)CLOCKS_PER_SEC);
  fprintf(fpout, "Total duration is %f seconds.\n",
          (float)(pre2 - pre1 + total) / (float)CLOCKS_PER_SEC);
  fprintf(fpout, "Preprocessing duration is %f seconds.\n",
          (float)(pre2 - pre1) / (float)CLOCKS_PER_SEC);

  printf("Duration per interation is %f seconds.\n",
         ((float)total / (float)ITER) / (float)CLOCKS_PER_SEC);
  printf("Total duration is %f seconds.\n",
         (float)(pre2 - pre1 + total) / (float)CLOCKS_PER_SEC);
  printf("Preprocessing duration is %f seconds.\n",
         (float)(pre2 - pre1) / (float)CLOCKS_PER_SEC);

  fclose(fpout);
  return 0;
}

void sortGraph() {
  float max = -99999999999999.0f;
  int maxi, i, j;
  float tmp;

  for (j = 0; j < HIGHEST - 1; j++) {
    max = maxScore[j];
    maxi = j;
    for (i = j + 1; i < HIGHEST; i++) {
      if (maxScore[i] > max) {
        max = maxScore[i];
        maxi = i;
      }
    }
    swap(j, maxi);
    tmp = maxScore[j];
    maxScore[j] = max;
    maxScore[maxi] = tmp;
  }
}

void swap(int a, int b) {
  int i, j;
  bool tmp;
  for (i = 0; i < NODE_N; i++) {
    for (j = 0; j < NODE_N; j++) {
      tmp = bestGraph[a][i][j];
      bestGraph[a][i][j] = bestGraph[b][i][j];
      bestGraph[b][i][j] = tmp;
    }
  }
}

void initial() {
  int i, j, a, b, r;
  bool tmpd;

  uint64_t tmp64 = 1;
  for (i = 1; i <= 4; i++) {
    tmp64 += C_u64((uint64_t)NODE_N - 1ULL, (uint64_t)i);
  }
  sizepernode = (size_t)tmp64;

  printf("U_localscore bytes: %zu\n", ((size_t)sizepernode * (size_t)NODE_N * sizeof(float)) / (1024*1024*1024));

  uint64_t entries64 = tmp64 * (uint64_t)NODE_N;
  if (entries64 > (uint64_t)(SIZE_MAX / sizeof(float))) {
    fprintf(stderr, "Overflow: localscore allocation too large.\n");
    exit(1);
  }
  size_t entries = (size_t)entries64;

  CHECK_CUDA(cudaMallocManaged(&U_localscore, entries * sizeof(float)));
  CHECK_CUDA(cudaMemAdvise(U_localscore, entries * sizeof(float), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
  CHECK_CUDA(cudaMemAdvise(U_localscore, entries * sizeof(float), cudaMemAdviseSetAccessedBy,  0));
  if(entries * sizeof(int) < getGPUFreeMemory())CHECK_CUDA(cudaMemPrefetchAsync(U_localscore, entries * sizeof(int), 0, 0));

  for (size_t t = 0; t < entries; t++) U_localscore[t] = 0.0f;

  for (i = 0; i < NODE_N; i++) {
    for (j = 0; j < NODE_N; j++) orders[i][j] = 0;
  }
  for (i = 0; i < NODE_N; i++) {
    for (j = 0; j < i; j++) orders[i][j] = 1;
  }

  r = rand() % 10000;
  for (i = 0; i < r; i++) {
    a = rand() % NODE_N;
    b = rand() % NODE_N;

    for (j = 0; j < NODE_N; j++) {
      tmpd = orders[j][a];
      orders[j][a] = orders[j][b];
      orders[j][b] = tmpd;
    }
    for (j = 0; j < NODE_N; j++) {
      tmpd = orders[a][j];
      orders[a][j] = orders[b][j];
      orders[b][j] = tmpd;
    }
  }

  for (i = 0; i < NODE_N; i++) {
    for (j = 0; j < NODE_N; j++) preOrders[i][j] = orders[i][j];
  }
}

// generate random order
int genOrders() {
  int a, b, j;
  bool tmp;
  a = rand() % NODE_N;
  b = rand() % NODE_N;

  for (j = 0; j < NODE_N; j++) {
    tmp = orders[a][j];
    orders[a][j] = orders[b][j];
    orders[b][j] = tmp;
  }
  for (j = 0; j < NODE_N; j++) {
    tmp = orders[j][a];
    orders[j][a] = orders[j][b];
    orders[j][b] = tmp;
  }
  return 1;
}

// decide leave or discard an order
int ConCore() {
  int i, j;
  float tmp;
  tmp = log((rand() % 100000) / 100000.0f);
  if (tmp < (score - preScore)) {
    for (i = 0; i < NODE_N; i++) {
      for (j = 0; j < NODE_N; j++) {
        preOrders[i][j] = orders[i][j];
        preGraph[i][j] = graph[i][j];
      }
    }
    preScore = score;
    return 1;
  }
  return 0;
}

void genScore() {
  // grid.x is 32-bit, so clamp/check before casting
  size_t grid_x = sizepernode / 256u + 1u;
  if (grid_x > 0x7fffffffU) {
    fprintf(stderr, "Grid too large: %zu\n", grid_x);
    exit(1);
  }

  Pre_logGamma();

  // Kernel signature likely expects int sizepernode; check before cast
  if (sizepernode > (size_t)INT_MAX) {
    fprintf(stderr, "sizepernode exceeds INT_MAX; kernel interface would overflow.\n");
    exit(1);
  }
  uint64_t total_tasks = (uint64_t)NODE_N * sizepernode;

  int device;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  // Occupancy-based dynamic setting
  // We want enough blocks to keep all SMs busy (usually 2-4 blocks per SM)
  int blocksPerGrid = prop.multiProcessorCount * 4; 
  int threadsPerBlock = prop.maxThreadsPerBlock; // Usually 1024
  genScoreKernel<<<blocksPerGrid, threadsPerBlock>>>((size_t)sizepernode, U_localscore, U_data, U_LG);
  CHECK_CUDA(cudaDeviceSynchronize());
  cudaGetLastError();
  CHECK_CUDA(cudaFree(U_LG));

  size_t blocks = sizepernode / (256u * (unsigned)taskperthr) + 1u;
  if (blocks > 0x7fffffffU) {
    fprintf(stderr, "blocks too large: %zu\n", blocks);
    exit(1);
  }

  CHECK_CUDA(cudaMallocManaged(&U_scores, blocks * sizeof(float)));
  CHECK_CUDA(cudaMallocManaged(&U_parent, (size_t)NODE_N * sizeof(bool)));
  CHECK_CUDA(cudaMallocManaged(&U_resP, blocks * 4u * sizeof(int)));
  
  // add memAdvise
  CHECK_CUDA(cudaMemAdvise(U_scores, blocks * sizeof(float), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
  CHECK_CUDA(cudaMemAdvise(U_scores, blocks * sizeof(float), cudaMemAdviseSetAccessedBy,  0));
  CHECK_CUDA(cudaMemAdvise(U_parent, (size_t)NODE_N * sizeof(bool), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
  CHECK_CUDA(cudaMemAdvise(U_parent, (size_t)NODE_N * sizeof(bool), cudaMemAdviseSetAccessedBy,  0));
  CHECK_CUDA(cudaMemAdvise(U_resP, blocks * 4u * sizeof(int), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
  CHECK_CUDA(cudaMemAdvise(U_resP, blocks * 4u * sizeof(int), cudaMemAdviseSetAccessedBy,  0));

  // add prefetch

}

void Pre_logGamma() {
  CHECK_CUDA(cudaMallocManaged(&U_LG, (size_t)(DATA_N + 2) * sizeof(float)));
  CHECK_CUDA(cudaMemAdvise(U_LG, (size_t)(DATA_N + 2) * sizeof(float), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
  CHECK_CUDA(cudaMemAdvise(U_LG, (size_t)(DATA_N + 2) * sizeof(float), cudaMemAdviseSetAccessedBy,  0));
  if( (size_t)(DATA_N + 2) * sizeof(float) < getGPUFreeMemory())CHECK_CUDA(cudaMemPrefetchAsync(U_LG, (size_t)(DATA_N + 2) * sizeof(float), 0, 0));
  U_LG[1] = log(1.0f);
  for (int i = 2; i <= DATA_N + 1; i++) {
    U_LG[i] = U_LG[i - 1] + log((float)i);
  }
}

void incr(int *bit, int n) {
  bit[n]++;
  if (bit[n] >= 2) {
    bit[n] = 0;
    incr(bit, n + 1);
  }
}

void incrS(int *bit, int n) {
  bit[n]++;
  if (bit[n] >= STATE_N) {
    bit[n] = 0;
    incr(bit, n + 1);
  }
}

bool getState(int parN, int *state, int time) {
  // replaces float pow() with integer pow + overflow-safe comparison
  uint64_t lim = pow_u64_sat((uint64_t)STATE_N, parN);
  if (lim == 0) return false;
  if (lim != UINT64_MAX) lim -= 1;

  if ((uint64_t)time > lim) return false;
  if (time >= 1) incrS(state, 0);
  return true;
}

bool getparent(int *bit, int *pre, int posN, int *parent, int *parN, int time) {
  int i;
  uint64_t lim = 1;

  *parN = 0;
  if (time == 0) return true;

  // lim = 2^posN - 1 (saturating)
  for (i = 0; i < posN; i++) {
    if (lim > (UINT64_MAX >> 1)) { lim = UINT64_MAX; break; }
    lim <<= 1;
  }
  if (lim != UINT64_MAX) lim -= 1;

  if ((uint64_t)time > lim) return false;

  incr(bit, 0);

  for (i = 0; i < posN; i++) {
    if (bit[i] == 1) parent[(*parN)++] = pre[i];
  }
  return true;
}

float findBestGraph() {
  float bestls = -99999999.0f;
  int bestparent[5];
  int bestpN;
  int node;
  int pre[NODE_N] = {0};
  int parent[NODE_N] = {0};
  int posN = 0, i, j, parN, tmp, k, l;
  float ls = -99999999999.0f, score_local = 0.0f;
  unsigned int blocknum;

  for (i = 0; i < NODE_N; i++)
    for (j = 0; j < NODE_N; j++)
      graph[i][j] = 0;

  for (node = 0; node < NODE_N; node++) {
    bestls = -99999999.0f;
    posN = 0;

    for (i = 0; i < NODE_N; i++) {
      if (orders[node][i] == 1) pre[posN++] = i;
    }

    // NOTE: original code had `if (posN >= 0)` which is always true; keep logic as-is.
    if (posN >= 0) {
      uint64_t total64 =
          C_u64((uint64_t)posN, 4) + C_u64((uint64_t)posN, 3) + C_u64((uint64_t)posN, 2) +
          (uint64_t)posN + 1ULL;

      taskperthr = 1;
      uint64_t blocks64 = total64 / (uint64_t)(256 * taskperthr) + 1ULL;
      if (blocks64 > 0x7fffffffU) {
        fprintf(stderr, "blocknum too large\n");
        exit(1);
      }
      blocknum = (unsigned int)blocks64;
      CHECK_CUDA(cudaGetLastError());

      if(blocks64 * sizeof(float) < getGPUFreeMemory())CHECK_CUDA(cudaMemPrefetchAsync(U_scores, blocks64 * sizeof(float), 0, 0));
      if(blocks64 * 4u * sizeof(int) < getGPUFreeMemory())CHECK_CUDA(cudaMemPrefetchAsync(U_resP, blocks64 * 4u * sizeof(int), 0, 0));
      // Correct float init: set to -999999.0f using kernel (memset is bytewise)
      {
        uint32_t threads = 256;
        uint32_t gx = (uint32_t)((blocks64 + threads - 1) / threads);
        //unsigned int gx = (unsigned int)((blocks64 + 255ULL) / 256ULL);
        initFloatKernel<<<gx, threads>>>((float*)U_scores, (size_t)blocks64, -999999.0f);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
      }

      for (int jj = 0; jj < NODE_N; jj++) U_parent[jj] = orders[node][jj];
      if((size_t)NODE_N * sizeof(bool) < getGPUFreeMemory())CHECK_CUDA(cudaMemPrefetchAsync(U_parent, (size_t)NODE_N * sizeof(bool), 0, 0));

      // Kernel likely expects ints; check casts
      if (sizepernode > (size_t)INT_MAX || total64 > (uint64_t)INT_MAX) {
        fprintf(stderr, "sizepernode/total exceed int range expected by kernel.\n");
        exit(1);
      }

      computeKernel<<<blocknum, 256, 256 * sizeof(float)>>>(
          taskperthr, (size_t)sizepernode, U_localscore, U_parent, node, (int)total64, U_scores, U_resP);
      CHECK_CUDA(cudaDeviceSynchronize());

      for (size_t bi = 0; bi < (size_t)blocknum; bi++) {
        if (U_scores[bi] > bestls) {
          bestls = U_scores[bi];
          parN = 0;
          for (tmp = 0; tmp < 4; tmp++) {
            if (U_resP[(size_t)bi * 4u + (size_t)tmp] < 0) break;
            bestparent[tmp] = U_resP[(size_t)bi * 4u + (size_t)tmp];
            parN++;
          }
          bestpN = parN;
        }
      }
    } else {
      // Unchanged (dead in practice), kept as requested
      if (posN >= 4) {
        for (i = 0; i < posN; i++) {
          for (j = i + 1; j < posN; j++) {
            for (k = j + 1; k < posN; k++) {
              for (l = k + 1; l < posN; l++) {
                parN = 4;
                parent[1] = (pre[i] > node) ? pre[i] : (pre[i] + 1);
                parent[2] = (pre[j] > node) ? pre[j] : (pre[j] + 1);
                parent[3] = (pre[k] > node) ? pre[k] : (pre[k] + 1);
                parent[4] = (pre[l] > node) ? pre[l] : (pre[l] + 1);

                uint64_t idx64 = findindex(parent, parN) + (uint64_t)sizepernode * (uint64_t)node;
                ls = U_localscore[(size_t)idx64];

                if (ls > bestls) {
                  bestls = ls;
                  bestpN = parN;
                  for (tmp = 0; tmp < parN; tmp++) bestparent[tmp] = parent[tmp + 1];
                }
              }
            }
          }
        }
      }

      if (posN >= 3) {
        for (i = 0; i < posN; i++) {
          for (j = i + 1; j < posN; j++) {
            for (k = j + 1; k < posN; k++) {
              parN = 3;
              parent[1] = (pre[i] > node) ? pre[i] : (pre[i] + 1);
              parent[2] = (pre[j] > node) ? pre[j] : (pre[j] + 1);
              parent[3] = (pre[k] > node) ? pre[k] : (pre[k] + 1);

              uint64_t idx64 = findindex(parent, parN) + (uint64_t)sizepernode * (uint64_t)node;
              ls = U_localscore[(size_t)idx64];

              if (ls > bestls) {
                bestls = ls;
                bestpN = parN;
                for (tmp = 0; tmp < parN; tmp++) bestparent[tmp] = parent[tmp + 1];
              }
            }
          }
        }
      }

      if (posN >= 2) {
        for (i = 0; i < posN; i++) {
          for (j = i + 1; j < posN; j++) {
            parN = 2;
            parent[1] = (pre[i] > node) ? pre[i] : (pre[i] + 1);
            parent[2] = (pre[j] > node) ? pre[j] : (pre[j] + 1);

            uint64_t idx64 = findindex(parent, parN) + (uint64_t)sizepernode * (uint64_t)node;
            ls = U_localscore[(size_t)idx64];

            if (ls > bestls) {
              bestls = ls;
              bestpN = parN;
              for (tmp = 0; tmp < parN; tmp++) bestparent[tmp] = parent[tmp + 1];
            }
          }
        }
      }

      if (posN >= 1) {
        for (i = 0; i < posN; i++) {
          parN = 1;
          parent[1] = (pre[i] > node) ? pre[i] : (pre[i] + 1);

          uint64_t idx64 = findindex(parent, parN) + (uint64_t)sizepernode * (uint64_t)node;
          ls = U_localscore[(size_t)idx64];

          if (ls > bestls) {
            bestls = ls;
            bestpN = parN;
            for (tmp = 0; tmp < parN; tmp++) bestparent[tmp] = parent[tmp + 1];
          }
        }
      }

      parN = 0;
      ls = U_localscore[(size_t)((uint64_t)sizepernode * (uint64_t)node)];
      if (ls > bestls) { bestls = ls; bestpN = 0; }
    }

    if (bestls > -99999999.0f) {
      for (i = 0; i < bestpN; i++) {
        if (bestparent[i] < node) graph[node][bestparent[i] - 1] = 1;
        else graph[node][bestparent[i]] = 1;
      }
      score_local += bestls;
    }
  }

  return score_local;
}

uint64_t findindex(int *arr, int size) {
  // reminder: arr[0] has to be 0 && size == array size-1 && index starts from 0
  if (size <= 0) return 0;
  uint64_t index = 0;

  for (int i = 1; i < size; i++) {
    index += C_u64((uint64_t)NODE_N - 1ULL, (uint64_t)i);
  }

  for (int i = 1; i <= size - 1; i++) {
    for (int j = arr[i - 1] + 1; j <= arr[i] - 1; j++) {
      index += C_u64((uint64_t)NODE_N - 1ULL - (uint64_t)j, (uint64_t)(size - i));
    }
  }

  index += (uint64_t)(arr[size] - arr[size - 1]);
  return index;
}
