#ifndef _BFSCUDA_H_
#define _BFSCUDA_H_

#include <cstddef>   // size_t
#ifdef __cplusplus
extern "C" {
#endif
__global__ void simpleBfs(
    int N,
    int level,
    int *d_adjacencyList,
    const size_t *d_edgesOffset,
    const size_t *d_edgesSize,
    int *d_distance,
    size_t *d_parent,
    int *changed);

__global__ void queueBfs(
    int level,
    int *d_adjacencyList,
    const size_t *d_edgesOffset,
    const size_t *d_edgesSize,
    int *d_distance,
    size_t *d_parent,
    int queueSize,
    int *nextQueueSize,
    int *d_currentQueue,
    int *d_nextQueue);

__global__ void nextLayer(
    int level,
    int *d_adjacencyList,
    const size_t *d_edgesOffset,
    const size_t *d_edgesSize,
    int *d_distance,
    size_t *d_parent,
    int queueSize,
    int *d_currentQueue);

__global__ void countDegrees(
    int *d_adjacencyList,
    const size_t *d_edgesOffset,
    const size_t *d_edgesSize,
    const size_t *d_parent,
    int queueSize,
    int *d_currentQueue,
    int *d_degrees);

__global__ void scanDegrees(
    int size,
    int *d_degrees,
    int *incrDegrees);

__global__ void assignVerticesNextQueue(
    int *d_adjacencyList,
    const size_t *d_edgesOffset,
    const size_t *d_edgesSize,
    const size_t *d_parent,
    int queueSize,
    int *d_currentQueue,
    int *d_nextQueue,
    int *d_degrees,
    int *incrDegrees,
    int nextQueueSize);
#ifdef __cplusplus
}
#endif
#endif
