#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <cstring>
#include <cstddef>   // size_t
#include "graph.h"
#include "bfsCPU.h"
#include "bfsCUDA.cuh"

void runCpu(int startVertex, Graph &G, std::vector<int> &distance,
            std::vector<size_t> &parent, std::vector<bool> &visited) {
    printf("Starting sequential bfs.\n");
    auto start = std::chrono::steady_clock::now();
    bfsCPU(startVertex, G, distance, parent, visited);
    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time in milliseconds : %li ms.\n\n", duration);
}

#define checkError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int *u_adjacencyList;
size_t *u_edgesOffset;
size_t *u_edgesSize;
int *u_distance;
size_t *u_parent;
int *u_currentQueue;
int *u_nextQueue;
int *u_degrees;

int *incrDegrees;

void initCuda(Graph &G) {
    checkError(cudaMallocManaged(&u_adjacencyList, G.numEdges * sizeof(int), cudaMemAttachGlobal ));
    checkError(cudaMallocManaged(&u_edgesOffset,  G.numVertices * sizeof(size_t), cudaMemAttachGlobal ));
    checkError(cudaMallocManaged(&u_edgesSize,    G.numVertices * sizeof(size_t), cudaMemAttachGlobal ));
    checkError(cudaMallocManaged(&u_distance,     G.numVertices * sizeof(int) , cudaMemAttachGlobal));
    checkError(cudaMallocManaged(&u_parent,       G.numVertices * sizeof(size_t), cudaMemAttachGlobal));
    checkError(cudaMallocManaged(&u_currentQueue, G.numVertices * sizeof(int) , cudaMemAttachGlobal));
    checkError(cudaMallocManaged(&u_nextQueue,    G.numVertices * sizeof(int) , cudaMemAttachGlobal));
    checkError(cudaMallocManaged(&u_degrees,      G.numVertices * sizeof(int), cudaMemAttachGlobal ));


    // also add cudaMemAdvise
    checkError(cudaMemAdvise(u_adjacencyList, G.numEdges * sizeof(int), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    checkError(cudaMemAdvise(u_adjacencyList, G.numEdges * sizeof(int), cudaMemAdviseSetAccessedBy, 0));
    checkError(cudaMemAdvise(u_edgesOffset, G.numVertices * sizeof(size_t), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    checkError(cudaMemAdvise(u_edgesOffset, G.numVertices * sizeof(size_t), cudaMemAdviseSetAccessedBy, 0));
    checkError(cudaMemAdvise(u_edgesSize, G.numVertices * sizeof(size_t), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    checkError(cudaMemAdvise(u_edgesSize, G.numVertices * sizeof(size_t), cudaMemAdviseSetAccessedBy, 0));
    checkError(cudaMemAdvise(u_distance, G.numVertices * sizeof(int), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    checkError(cudaMemAdvise(u_distance, G.numVertices * sizeof(int), cudaMemAdviseSetAccessedBy, 0));
    checkError(cudaMemAdvise(u_parent, G.numVertices * sizeof(size_t), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    checkError(cudaMemAdvise(u_parent, G.numVertices * sizeof(size_t), cudaMemAdviseSetAccessedBy, 0));
    checkError(cudaMemAdvise(u_currentQueue, G.numVertices * sizeof(int), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    checkError(cudaMemAdvise(u_currentQueue, G.numVertices * sizeof(int), cudaMemAdviseSetAccessedBy, 0));
    checkError(cudaMemAdvise(u_nextQueue, G.numVertices * sizeof(int), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    checkError(cudaMemAdvise(u_nextQueue, G.numVertices * sizeof(int), cudaMemAdviseSetAccessedBy, 0));
    checkError(cudaMemAdvise(u_degrees, G.numVertices * sizeof(int), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    checkError(cudaMemAdvise(u_degrees, G.numVertices * sizeof(int), cudaMemAdviseSetAccessedBy, 0));

    checkError(cudaMallocHost((void **) &incrDegrees, sizeof(int) * G.numVertices));

    memcpy(u_adjacencyList, G.adjacencyList.data(), G.numEdges * sizeof(int));
    memcpy(u_edgesOffset,   G.edgesOffset.data(),   G.numVertices * sizeof(size_t));
    memcpy(u_edgesSize,     G.edgesSize.data(),     G.numVertices * sizeof(size_t));
   // checkError(cudaMemcpy(d_adjacencyList, G.adjacencyList.data(), G.numEdges * sizeof(int), cudaMemcpyHostToDevice));
    // checkError(cudaMemcpy(d_edgesOffset, G.edgesOffset.data(), G.numVertices * sizeof(int), cudaMemcpyHostToDevice));
    // checkError(cudaMemcpy(d_edgesSize, G.edgesSize.data(), G.numVertices * sizeof(int), cudaMemcpyHostToDevice ));
    //also add prefetching: - need to be sure that we dont prefetch too much
    // int max_size = 15 * 1024 * 1024 * 1024;
    // int actual_prefetch = 0;
    // actual_prefetch += G.numEdges * sizeof(int);
    // if(actual_prefetch < max_size) checkError(cudaMemPrefetchAsync(u_adjacencyList, G.numEdges * sizeof(int), 0, 0));
    // actual_prefetch += G.numVertices * sizeof(int);
    // if(actual_prefetch < max_size)checkError(cudaMemPrefetchAsync(u_edgesOffset, G.numVertices * sizeof(size_t), 0, 0));
    // actual_prefetch += G.numVertices * sizeof(int);
    // if(actual_prefetch < max_size)checkError(cudaMemPrefetchAsync(u_edgesSize, G.numVertices * sizeof(size_t), 0, 0));
    
    cudaDeviceSynchronize();
}

void finalizeCuda() {
    checkError(cudaFree(u_adjacencyList));
    checkError(cudaFree(u_edgesOffset));
    checkError(cudaFree(u_edgesSize));
    checkError(cudaFree(u_distance));
    checkError(cudaFree(u_parent));
    checkError(cudaFree(u_currentQueue));
    checkError(cudaFree(u_nextQueue));
    checkError(cudaFree(u_degrees));
    checkError(cudaFreeHost(incrDegrees));
}

void checkOutput(std::vector<int> &distance, std::vector<int> &expectedDistance, Graph &G) {
    checkError(cudaMemPrefetchAsync(u_adjacencyList, G.numEdges * sizeof(int), cudaCpuDeviceId, 0));
    checkError(cudaMemPrefetchAsync(u_edgesOffset,   G.numVertices * sizeof(size_t), cudaCpuDeviceId, 0));
    checkError(cudaMemPrefetchAsync(u_edgesSize,     G.numVertices * sizeof(size_t), cudaCpuDeviceId, 0));
    checkError(cudaMemPrefetchAsync(u_distance,      G.numVertices * sizeof(int), cudaCpuDeviceId, 0));
    checkError(cudaMemPrefetchAsync(u_parent,        G.numVertices * sizeof(size_t), cudaCpuDeviceId, 0));
    checkError(cudaMemPrefetchAsync(u_currentQueue,  G.numVertices * sizeof(int), cudaCpuDeviceId, 0));
    checkError(cudaMemPrefetchAsync(u_nextQueue,     G.numVertices * sizeof(int), cudaCpuDeviceId, 0));
    checkError(cudaMemPrefetchAsync(u_degrees,       G.numVertices * sizeof(int), cudaCpuDeviceId, 0));

    for (size_t i = 0; i < G.numVertices; i++) {
        if (*(u_distance + i) != expectedDistance[i]) {
            printf("%zu %d %d\n", i, distance[i], expectedDistance[i]);
            printf("Wrong output!\n");
            exit(1);
        }
    }

    printf("Output OK!\n\n");
}

void initializeCudaBfs(int startVertex, std::vector<int> &distance, std::vector<size_t> &parent, Graph &G) {
    std::fill(distance.begin(), distance.end(), std::numeric_limits<int>::max());
    std::fill(parent.begin(), parent.end(), (size_t)std::numeric_limits<int>::max());
    distance[startVertex] = 0;
    parent[startVertex] = 0;

    memcpy(u_distance, distance.data(), G.numVertices * sizeof(int));
    memcpy(u_parent,   parent.data(),   G.numVertices * sizeof(size_t));
    // int dev; cudaGetDevice(&dev);
    // cudaMemPrefetchAsync(u_distance,    G.numVertices*sizeof(int), dev);
    // cudaMemPrefetchAsync(u_parent,      G.numVertices*sizeof(int), dev);
    // cudaMemPrefetchAsync(u_currentQueue,G.numVertices*sizeof(int), dev);
    // cudaDeviceSynchronize();
    int firstElementQueue = startVertex;
    *u_currentQueue = firstElementQueue;
}

void finalizeCudaBfs(std::vector<int> &distance, std::vector<size_t> &parent, Graph &G) {
    // no-op
}

void runCudaSimpleBfs(int startVertex, Graph &G, std::vector<int> &distance,
                      std::vector<size_t> &parent) {
    initializeCudaBfs(startVertex, distance, parent, G);

    int *changed;
    checkError(cudaMallocHost((void **) &changed, sizeof(int)));

    printf("Starting simple parallel bfs.\n");
    auto start = std::chrono::steady_clock::now();

    *changed = 1;
    int level = 0;
    while (*changed) {
        *changed = 0;

        simpleBfs<<<(int)(G.numVertices / 1024 + 1), 1024>>>(
            (int)G.numVertices, level, u_adjacencyList, u_edgesOffset, u_edgesSize, u_distance, u_parent, changed);
        cudaDeviceSynchronize();
        level++;
    }

    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time in milliseconds : %li ms.\n", duration);

    finalizeCudaBfs(distance, parent, G);
}

void runCudaQueueBfs(int startVertex, Graph &G, std::vector<int> &distance,
    std::vector<size_t> &parent) {
    initializeCudaBfs(startVertex, distance, parent, G);

    int *nextQueueSize;
    checkError(cudaMallocHost((void **)&nextQueueSize, sizeof(int)));

    printf("Starting queue parallel bfs.\n");
    auto start = std::chrono::steady_clock::now();

    int queueSize = 1;
    *nextQueueSize = 0;
    int level = 0;
    while (queueSize) {

        queueBfs<<<queueSize / 1024 + 1, 1024>>>(
            level, u_adjacencyList, u_edgesOffset, u_edgesSize, u_distance, u_parent, queueSize,
            nextQueueSize, u_currentQueue, u_nextQueue);
        cudaDeviceSynchronize();
        level++;
        queueSize = *nextQueueSize;
        *nextQueueSize = 0;
        std::swap(u_currentQueue, u_nextQueue);
    }

    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time in milliseconds : %li ms.\n", duration);

    finalizeCudaBfs(distance, parent, G);
}

void nextLayer(int level, int queueSize) {
    nextLayer<<<queueSize / 1024 + 1, 1024>>>(
        level, u_adjacencyList, u_edgesOffset, u_edgesSize, u_distance, u_parent, queueSize, u_currentQueue);
    cudaDeviceSynchronize();
}

void countDegrees(int level, int queueSize) {
    countDegrees<<<queueSize / 1024 + 1, 1024>>>(
        u_adjacencyList, u_edgesOffset, u_edgesSize, u_parent, queueSize, u_currentQueue, u_degrees);
    cudaDeviceSynchronize();
}

void scanDegrees(int queueSize) {
    scanDegrees<<<queueSize / 1024 + 1, 1024>>>(queueSize, u_degrees, incrDegrees);
    cudaDeviceSynchronize();

    incrDegrees[0] = 0;
    for (int i = 1024; i < queueSize + 1024; i += 1024) {
        incrDegrees[i / 1024] += incrDegrees[i / 1024 - 1];
    }
}

void assignVerticesNextQueue(int queueSize, int nextQueueSize) {
    assignVerticesNextQueue<<<queueSize / 1024 + 1, 1024>>>(
        u_adjacencyList, u_edgesOffset, u_edgesSize, u_parent, queueSize, u_currentQueue,
        u_nextQueue, u_degrees, incrDegrees, nextQueueSize);
    cudaDeviceSynchronize();
}

void runCudaScanBfs(int startVertex, Graph &G, std::vector<int> &distance,
   std::vector<size_t> &parent) {
    initializeCudaBfs(startVertex, distance, parent, G);

    printf("Starting scan parallel bfs.\n");
    auto start = std::chrono::steady_clock::now();

    int queueSize = 1;
    int nextQueueSize = 0;
    int level = 0;
    while (queueSize) {
        nextLayer(level, queueSize);
        countDegrees(level, queueSize);
        scanDegrees(queueSize);
        nextQueueSize = incrDegrees[(queueSize - 1) / 1024 + 1];
        assignVerticesNextQueue(queueSize, nextQueueSize);

        level++;
        queueSize = nextQueueSize;
        std::swap(u_currentQueue, u_nextQueue);
    }

    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time in milliseconds : %li ms.\n", duration);

    finalizeCudaBfs(distance, parent, G);
}

int main(int argc, char **argv) {
    // read graph from standard input
    Graph G;
    int startVertex = atoi(argv[1]);
    readGraph(G, argc, argv);

    printf("Number of vertices %llu\n", G.numVertices);
    printf("Number of edges %llu\n", G.numEdges);

    //vectors for results
    std::vector<int> distance(G.numVertices, std::numeric_limits<int>::max());
    std::vector<size_t> parent(G.numVertices, std::numeric_limits<int>::max());
    std::vector<bool> visited(G.numVertices, false);

    //run CPU sequential bfs
    //runCpu(startVertex, G, distance, parent, visited);

    //save results from sequential bfs
    std::vector<int> expectedDistance(distance);
    std::vector<size_t> expectedParent(parent);
    auto start = std::chrono::steady_clock::now();
    initCuda(G);
    // //run CUDA queue parallel bfs
    runCudaQueueBfs(startVertex, G, distance, parent);
    cudaGetLastError();

    finalizeCuda();
    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("CUDA queue parallel bfs Elapsed time in milliseconds : %li ms.\n", duration);
    
    start = std::chrono::steady_clock::now();
    initCuda(G);
    //run CUDA simple parallel bfs
    runCudaSimpleBfs(startVertex, G, distance, parent);
    cudaGetLastError();

    finalizeCuda();
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("CUDA simple parallel bfs Elapsed time in milliseconds : %li ms.\n", duration);

    start = std::chrono::steady_clock::now();
    initCuda(G);
    // //run CUDA scan parallel bfs
    runCudaScanBfs(startVertex, G, distance, parent);
    cudaGetLastError();

    finalizeCuda();
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("CUDA scan parallel bfs Elapsed time in milliseconds : %li ms.\n", duration);
    cudaGetLastError();
    return 0;
}
