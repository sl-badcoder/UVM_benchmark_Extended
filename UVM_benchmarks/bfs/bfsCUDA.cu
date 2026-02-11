#include <device_launch_parameters.h>
#include <cstdio>
#include <bfsCUDA.cuh>
#include <cstddef>   // size_t
#include <climits>   // INT_MAX

__global__
void simpleBfs(int N, int level,
               int *d_adjacencyList,
               const size_t *d_edgesOffset,
               const size_t *d_edgesSize,
               int *d_distance,
               size_t *d_parent,
               int *changed) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    int valueChange = 0;

    if (thid < N && d_distance[thid] == level) {
        int u = thid;
        size_t start = d_edgesOffset[u];
        size_t end   = start + static_cast<size_t>(d_edgesSize[u]);

        for (size_t i = start; i < end; i++) {
            int v = d_adjacencyList[i];
            if (level + 1 < d_distance[v]) {
                d_distance[v] = level + 1;
                d_parent[v] = i;
                valueChange = 1;
            }
        }
    }

    if (valueChange) {
        *changed = valueChange;
    }
}

__global__
void queueBfs(int level,
              int *d_adjacencyList,
              const size_t *d_edgesOffset,
              const size_t *d_edgesSize,
              int *d_distance,
              size_t *d_parent,
              int queueSize,
              int *nextQueueSize,
              int *d_currentQueue,
              int *d_nextQueue) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        int u = d_currentQueue[thid];
        size_t start = d_edgesOffset[u];
        size_t end   = start + static_cast<size_t>(d_edgesSize[u]);

        for (size_t i = start; i < end; i++) {
            int v = d_adjacencyList[i];
            if (d_distance[v] == INT_MAX && atomicMin(&d_distance[v], level + 1) == INT_MAX) {
                d_parent[v] = i;
                int position = atomicAdd(nextQueueSize, 1);
                d_nextQueue[position] = v;
            }
        }
    }
}

// Scan bfs
__global__
void nextLayer(int level,
               int *d_adjacencyList,
               const size_t *d_edgesOffset,
               const size_t *d_edgesSize,
               int *d_distance,
               size_t *d_parent,
               int queueSize,
               int *d_currentQueue) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        int u = d_currentQueue[thid];
        size_t start = d_edgesOffset[u];
        size_t end   = start + static_cast<size_t>(d_edgesSize[u]);

        for (size_t i = start; i < end; i++) {
            int v = d_adjacencyList[i];
            if (level + 1 < d_distance[v]) {
                d_distance[v] = level + 1;
                d_parent[v] = i;
            }
        }
    }
}

__global__
void countDegrees(int *d_adjacencyList,
                  const size_t *d_edgesOffset,
                  const size_t *d_edgesSize,
                  const size_t *d_parent,
                  int queueSize,
                  int *d_currentQueue,
                  int *d_degrees) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        int u = d_currentQueue[thid];
        int degree = 0;

        size_t start = d_edgesOffset[u];
        size_t end   = start + static_cast<size_t>(d_edgesSize[u]);

        for (size_t i = start; i < end; i++) {
            int v = d_adjacencyList[i];
            if (d_parent[v] == i && v != u) {
                ++degree;
            }
        }
        d_degrees[thid] = degree;
    }
}

__global__
void scanDegrees(int size, int *d_degrees, int *incrDegrees) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < size) {
        __shared__ int prefixSum[1024];
        int modulo = threadIdx.x;
        prefixSum[modulo] = d_degrees[thid];
        __syncthreads();

        for (int nodeSize = 2; nodeSize <= 1024; nodeSize <<= 1) {
            if ((modulo & (nodeSize - 1)) == 0) {
                if (thid + (nodeSize >> 1) < size) {
                    int nextPosition = modulo + (nodeSize >> 1);
                    prefixSum[modulo] += prefixSum[nextPosition];
                }
            }
            __syncthreads();
        }

        if (modulo == 0) {
            int block = thid >> 10;
            incrDegrees[block + 1] = prefixSum[modulo];
        }

        for (int nodeSize = 1024; nodeSize > 1; nodeSize >>= 1) {
            if ((modulo & (nodeSize - 1)) == 0) {
                if (thid + (nodeSize >> 1) < size) {
                    int next_position = modulo + (nodeSize >> 1);
                    int tmp = prefixSum[modulo];
                    prefixSum[modulo] -= prefixSum[next_position];
                    prefixSum[next_position] = tmp;
                }
            }
            __syncthreads();
        }
        d_degrees[thid] = prefixSum[modulo];
    }
}

__global__
void assignVerticesNextQueue(int *d_adjacencyList,
                             const size_t *d_edgesOffset,
                             const size_t *d_edgesSize,
                             const size_t *d_parent,
                             int queueSize,
                             int *d_currentQueue,
                             int *d_nextQueue,
                             int *d_degrees,
                             int *incrDegrees,
                             int nextQueueSize) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        __shared__ int sharedIncrement;
        if (!threadIdx.x) {
            sharedIncrement = incrDegrees[thid >> 10];
        }
        __syncthreads();

        int sum = 0;
        if (threadIdx.x) {
            sum = d_degrees[thid - 1];
        }

        int u = d_currentQueue[thid];
        size_t counter = 0;

        size_t start = d_edgesOffset[u];
        size_t end   = start + static_cast<size_t>(d_edgesSize[u]);

        for (size_t i = start; i < end; i++) {
            int v = d_adjacencyList[i];
            if (d_parent[v] == i && v != u) {
                size_t nextQueuePlace = static_cast<size_t>(sharedIncrement) + static_cast<size_t>(sum) + counter;
                // d_nextQueue is int*, so this still assumes nextQueuePlace fits in int range.
                // If you want queue indices > 2^31, d_nextQueue must be size_t-indexable too.
                d_nextQueue[nextQueuePlace] = v;
                counter++;
            }
        }
    }
}
