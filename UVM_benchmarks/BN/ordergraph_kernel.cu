#ifndef _ORDERGRAPH_KERNEL_H_
#define _ORDERGRAPH_KERNEL_H_

//#include "data50.cu"
#ifndef DATA_CONFIG
#define DATA_CONFIG
const int NODE_N = 50; 
const int STATE_N = 2;
const int DATA_N = 120000000; 
#endif
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>   // size_t
;
char name[20] = "45.out";

// -------------------- overflow-safe helpers --------------------
__device__ __forceinline__ uint64_t D_gcd_u64(uint64_t a, uint64_t b) {
  while (b) { uint64_t t = a % b; a = b; b = t; }
  return a;
}

// Exact nCk in uint64 with overflow check. Returns UINT64_MAX on overflow.
__device__ __forceinline__ uint64_t D_C_u64(uint64_t n, uint64_t k) {
  if (k > n) return 0;
  if (k == 0) return 1;
  if (k > n - k) k = n - k;

  uint64_t res = 1;
  for (uint64_t i = 1; i <= k; i++) {
    uint64_t num = n - k + i;
    uint64_t den = i;

    uint64_t g = D_gcd_u64(num, den);
    num /= g; den /= g;

    g = D_gcd_u64(res, den);
    res /= g; den /= g;

    if (den != 1) return UINT64_MAX; // should not happen after reductions
    if (num != 0 && res > UINT64_MAX / num) return UINT64_MAX;
    res *= num;
  }
  return res;
}

// Saturating power for small STATE_N^parN usage
__device__ __forceinline__ uint64_t D_pow_u64_sat(uint64_t base, int exp) {
  uint64_t r = 1;
  for (int i = 0; i < exp; i++) {
    if (base != 0 && r > UINT64_MAX / base) return UINT64_MAX;
    r *= base;
  }
  return r;
}

// -------------------- device prototypes --------------------
__device__ void Dincr(int *bit, int n);
__device__ void DincrS(int *bit, int n);
__device__ bool D_getState(int parN, int *sta, int time);

__device__ void D_findComb(int *comb, int l, int n);

__device__ uint64_t D_findindex_u64(int *arr, int size);

__device__ int D_C(int n, int a);

__device__ void D_findComb64(int *comb, uint64_t l, int n);

// -------------------- kernels (64-bit safe indexing) --------------------
__global__ void genScoreKernel(size_t sizepernode,
                               float *D_localscore,
                               const int *D_data,
                               const float *D_LG) {
  // grid-stride loop to avoid launch-size limitations
  for (size_t id = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
       id < sizepernode;
       id += (size_t)gridDim.x * (size_t)blockDim.x) {

    int node;
    bool flag;
    int parent[5] = {0};
    int pre[NODE_N] = {0};
    int state[5] = {0};
    int i, j, parN = 0, tmp, t;
    int t1 = 0, t2 = 0;
    float ls = 0.0f;
    int Nij[STATE_N] = {0};

    if (id <= (size_t)INT32_MAX) {
      D_findComb(parent, (int)id, NODE_N - 1);
    } else {
      D_findComb64(parent, (uint64_t)id, NODE_N - 1);
    }

    parN = 0;
    for (i = 0; i < 4; i++) {
      if (parent[i] > 0) parN++;
    }

    for (node = 0; node < NODE_N; node++) {
      j = 1;
      for (i = 0; i < NODE_N; i++) {
        if (i != node) pre[j++] = i;
      }

      for (tmp = 0; tmp < parN; tmp++) state[tmp] = 0;

      size_t index = sizepernode * (size_t)node + id;

      t = 0;
      while (D_getState(parN, state, t++)) {
        ls = 0.0f;
        for (tmp = 0; tmp < STATE_N; tmp++) Nij[tmp] = 0;

        for (t1 = 0; t1 < DATA_N; t1++) {
          flag = true;
          for (t2 = 0; t2 < parN; t2++) {
            if (D_data[(size_t)t1 * (size_t)NODE_N + (size_t)pre[parent[t2]]] != state[t2]) {
              flag = false;
              break;
            }
          }
          if (!flag) continue;
          Nij[D_data[(size_t)t1 * (size_t)NODE_N + (size_t)node]]++;
        }

        tmp = STATE_N - 1;
        for (t1 = 0; t1 < STATE_N; t1++) {
          ls += D_LG[Nij[t1]];
          tmp += Nij[t1];
        }

        ls -= D_LG[tmp];
        ls += D_LG[STATE_N - 1];

        D_localscore[index] += ls;
      }
    }
  }
}

__global__ void computeKernel(int taskperthr,
                              size_t sizepernode,
                              const float *D_localscore,
                              const bool *D_parent,
                              int node,
                              size_t total,
                              float *D_Score,
                              int *D_resP) {
  extern __shared__ float lsinblock[];
  const unsigned int tid = (unsigned int)threadIdx.x;
  const unsigned int bid = (unsigned int)blockIdx.x;

  const size_t thread_global = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;

  int posN = 1, i, tmp;
  int pre[NODE_N] = {0};
  int parN = 0;
  int bestparent[4] = {0}, parent[5] = {-1};
  float bestls = -999999999999999.0f, ls;

  for (i = 0; i < NODE_N; i++) {
    if (D_parent[i] == 1) pre[posN++] = i;
  }

  for (int it = 0; it < taskperthr; it++) {
    size_t rank = thread_global * (size_t)taskperthr + (size_t)it;
    if (rank >= total) break;

    // unrank combination for this rank
    if (rank <= (size_t)INT32_MAX) {
      D_findComb(parent, (int)rank, posN);
    } else {
      D_findComb64(parent, (uint64_t)rank, posN);
    }

    for (parN = 0; parN < 4; parN++) {
      if (parent[parN] < 0) break;
      if (pre[parent[parN]] > node) parent[parN] = pre[parent[parN]];
      else parent[parN] = pre[parent[parN]] + 1;
    }

    for (tmp = parN; tmp > 0; tmp--) parent[tmp] = parent[tmp - 1];
    parent[0] = 0;

    // index = findindex(parent, parN) + sizepernode*node (64-bit)
    uint64_t off = D_findindex_u64(parent, parN);
    // if combinatorics overflowed, skip
    if (off == UINT64_MAX) continue;

    size_t index = sizepernode * (size_t)node + (size_t)off;
    ls = D_localscore[index];

    if (ls > bestls) {
      bestls = ls;
      for (tmp = 0; tmp < 4; tmp++) bestparent[tmp] = parent[tmp + 1];
    }
  }

  lsinblock[tid] = bestls;
  __syncthreads();

  // Reduction logic unchanged, just types cleaned a bit
  for (int step = 128; step >= 1; step >>= 1) {
    if ((int)tid < step) {
      if (lsinblock[tid + step] > lsinblock[tid] && lsinblock[tid + step] < 0) {
        lsinblock[tid] = lsinblock[tid + step];
        lsinblock[tid + step] = (float)(tid + (unsigned)step);
      } else if (lsinblock[tid + step] < lsinblock[tid] && lsinblock[tid] < 0) {
        lsinblock[tid + step] = (float)tid;
      } else if (lsinblock[tid] > 0 && lsinblock[tid + step] < 0) {
        lsinblock[tid] = lsinblock[tid + step];
        lsinblock[tid + step] = (float)(tid + (unsigned)step);
      } else if (lsinblock[tid] < 0 && lsinblock[tid + step] > 0) {
        lsinblock[tid + step] = (float)tid;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    D_Score[bid] = lsinblock[0];
    int t = 0;
    for (int p = 0; p < 7 && t < 128 && t >= 0; p++) {
      int idx = (1 << p) + t; // exact 2^p
      t = (int)lsinblock[idx];
    }
    lsinblock[0] = (float)t;
  }

  __syncthreads();

  if (tid == (unsigned)lsinblock[0]) {
    for (int k = 0; k < 4; k++) D_resP[bid * 4 + k] = bestparent[k];
  }
}

// -------------------- device helpers --------------------
__device__ void Dincr(int *bit, int n) {
  while (n <= NODE_N) {
    bit[n]++;
    if (bit[n] >= 2) { bit[n] = 0; n++; }
    else break;
  }
}

__device__ void DincrS(int *bit, int n) {
  bit[n]++;
  if (bit[n] >= STATE_N) {
    bit[n] = 0;
    Dincr(bit, n + 1);
  }
}

__device__ bool D_getState(int parN, int *sta, int time) {
  uint64_t lim = D_pow_u64_sat((uint64_t)STATE_N, parN);
  if (lim == 0) return false;
  if (lim != UINT64_MAX) lim -= 1;
  if ((uint64_t)time > lim) return false;

  if (time >= 1) DincrS(sta, 0);
  return true;
}

// ---- Combination unranking (legacy int rank) ----
__device__ void D_findComb(int *comb, int l, int n) {
  const int len = 4;
  if (l == 0) {
    for (int i = 0; i < len; i++) comb[i] = -1;
    return;
  }
  int sum = 0;
  int k = 1;

  while (sum < l) sum += D_C(n, k++);
  l -= sum - D_C(n, --k);

  int low = 0;
  int pos = 0;
  while (k > 1) {
    sum = 0;
    int s = 1;
    while (sum < l) sum += D_C(n - s++, k - 1);
    l -= sum - D_C(n - (--s), --k);
    low += s;
    comb[pos++] = low;
    n -= s;
  }
  comb[pos] = low + l;
  for (int i = pos + 1; i < 4; i++) comb[i] = -1;
}

// ---- Combination unranking (64-bit rank) ----
// Same logic, but uses 64-bit combinatorics; clamps when combinatorics overflow.
__device__ void D_findComb64(int *comb, uint64_t l, int n) {
  const int len = 4;
  if (l == 0) {
    for (int i = 0; i < len; i++) comb[i] = -1;
    return;
  }
  uint64_t sum = 0;
  int k = 1;

  while (sum < l) {
    uint64_t c = D_C_u64((uint64_t)n, (uint64_t)k);
    if (c == UINT64_MAX) { // overflow => can't represent; mark invalid
      for (int i = 0; i < len; i++) comb[i] = -1;
      return;
    }
    sum += c;
    k++;
  }

  k--;
  {
    uint64_t c = D_C_u64((uint64_t)n, (uint64_t)k);
    if (c == UINT64_MAX) { for (int i = 0; i < len; i++) comb[i] = -1; return; }
    l -= sum - c;
  }

  int low = 0;
  int pos = 0;
  while (k > 1) {
    sum = 0;
    int s = 1;
    while (sum < l) {
      uint64_t c = D_C_u64((uint64_t)(n - s), (uint64_t)(k - 1));
      if (c == UINT64_MAX) { for (int i = 0; i < len; i++) comb[i] = -1; return; }
      sum += c;
      s++;
    }

    s--;
    {
      uint64_t c = D_C_u64((uint64_t)(n - s), (uint64_t)(k - 1));
      if (c == UINT64_MAX) { for (int i = 0; i < len; i++) comb[i] = -1; return; }
      l -= sum - c;
    }

    k--;
    low += s;
    comb[pos++] = low;
    n -= s;
  }

  // last element
  if (pos < 4) comb[pos] = low + (int)l;
  for (int i = pos + 1; i < 4; i++) comb[i] = -1;
}

// 64-bit safe index computation
__device__ uint64_t D_findindex_u64(int *arr, int size) {
  if (size <= 0) return 0;          // empty parent set -> index 0
  if (size == 1) {                  // one parent: arr[0]=0, arr[1]=...
    return (uint64_t)(arr[1] - arr[0]);
  }

  uint64_t index = 0;
  for (int i = 1; i < size; i++) {
    uint64_t c = D_C_u64((uint64_t)NODE_N - 1ULL, (uint64_t)i);
    if (c == UINT64_MAX) return UINT64_MAX;
    index += c;
  }
  for (int i = 1; i <= size - 1; i++) {
    for (int j = arr[i - 1] + 1; j <= arr[i] - 1; j++) {
      uint64_t c = D_C_u64((uint64_t)NODE_N - 1ULL - (uint64_t)j,
                           (uint64_t)(size - i));
      if (c == UINT64_MAX) return UINT64_MAX;
      index += c;
    }
  }
  index += (uint64_t)(arr[size] - arr[size - 1]);
  return index;
}

// legacy int combinatorics, safe for small k (<=4) by clamping
__device__ int D_C(int n, int a) {
  if (n < 0 || a < 0) return 0;
  uint64_t v = D_C_u64((uint64_t)n, (uint64_t)a);
  if (v == UINT64_MAX) return INT_MAX;
  if (v > (uint64_t)INT_MAX) return INT_MAX;
  return (int)v;
}

#endif
