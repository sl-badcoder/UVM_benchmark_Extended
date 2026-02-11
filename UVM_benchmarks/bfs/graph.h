#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstddef>  // size_t

struct Graph {
    std::vector<int> adjacencyList;       // all edges (vertex IDs)
    std::vector<size_t> edgesOffset;      // offset to adjacencyList for every vertex
    std::vector<size_t> edgesSize;        // number of edges for every vertex
    size_t numVertices = 0;
    size_t numEdges = 0;
};

void readGraph(Graph &G, int argc, char **argv);

#endif //BFS_CUDA_GRAPH_H
