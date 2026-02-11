#include <ctime>
#include "graph.h"
#include <cstdint>
#include <random>
#include <limits>
#include <unordered_set>
#include <cstddef>   // size_t

// Simple 64-bit RNG (fast, deterministic)
static inline uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

void makeRandomGraphFixedDegree(Graph &G, size_t n, size_t deg,
                                bool undirected = false,
                                bool noSelfLoops = true,
                                bool tryAvoidDuplicates = false,
                                uint64_t seed = 12345)
{
    // deg is size_t, so "deg < 0" is meaningless; keep only the real check
    if (n == 0) {
        fprintf(stderr, "Invalid n\n");
        std::exit(1);
    }

    G.edgesOffset.clear();
    G.edgesSize.clear();
    G.adjacencyList.clear();

    G.edgesOffset.reserve(n);
    G.edgesSize.reserve(n);

    const uint64_t E_directed = (uint64_t)n * (uint64_t)deg;
    const uint64_t E_total = undirected ? (2ULL * E_directed) : E_directed;

    if (E_total > (uint64_t)std::numeric_limits<int>::max()) {
        fprintf(stderr, "Warning: edge count %llu may overflow 32-bit offsets.\n",
                (unsigned long long)E_total);
    }

    G.adjacencyList.reserve((size_t)E_total);

    uint64_t rng = seed;

    // Build CSR directly
    for (size_t u = 0; u < n; u++) {
        // Do NOT truncate adjacencyList.size()
        G.edgesOffset.push_back((size_t)G.adjacencyList.size());

        G.edgesSize.push_back(deg);

        if (!tryAvoidDuplicates) {
            for (size_t k = 0; k < deg; k++) {
                size_t v;
                do {
                    v = (size_t)(splitmix64(rng) % (uint64_t)n);
                } while (noSelfLoops && v == u);

                // adjacencyList element type may still be int in Graph.
                // Keep vertex IDs within int range unless Graph is updated.
                G.adjacencyList.push_back((int)v);
            }
        } else {
            // Avoid duplicates within this adjacency list (costly for large deg)
            std::unordered_set<size_t> used;
            used.reserve((size_t)deg * 2);

            size_t added = 0;
            while (added < deg) {
                size_t v = (size_t)(splitmix64(rng) % (uint64_t)n);
                if (noSelfLoops && v == u) continue;
                if (used.insert(v).second) {
                    G.adjacencyList.push_back((int)v);
                    added++;
                }
            }
        }
    }

    G.numVertices = n;
    G.numEdges = (size_t)G.adjacencyList.size();

    if (undirected) {
        fprintf(stderr, "Note: undirected=true is not fully implemented in Option A (directed CSR produced).\n");
    }
}

void readGraph(Graph &G, int argc, char **argv) {
    // Usage:
    //   ./bfs <startVertex> <nVertices> <degree>
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <startVertex> <nVertices> <degree>\n", argv[0]);
        std::exit(1);
    }

    size_t n = (size_t)strtoull(argv[2], nullptr, 10);
    size_t deg = (size_t)strtoull(argv[3], nullptr, 10);

    makeRandomGraphFixedDegree(G, n, deg, /*undirected=*/false,
                               /*noSelfLoops=*/true,
                               /*tryAvoidDuplicates=*/false,
                               /*seed=*/12345);
}


// void readGraph(Graph &G, int argc, char **argv) {
//     int n;
//     int m;

//     //If no arguments then read graph from stdin
//     bool fromStdin = argc <= 2;
//     if (fromStdin) {
//         scanf("%d %d", &n, &m);
//     } else {
//         srand(12345);
//         n = atoi(argv[2]);
//         m = atoi(argv[3]);
//     }

//     std::vector<std::vector<int> > adjecancyLists(n);
//     for (int i = 0; i < m; i++) {
//         int u, v;
//         if (fromStdin) {
//             scanf("%d %d", &u, &v);
//             adjecancyLists[u].push_back(v);
//         } else {
//             u = rand() % n;
//             v = rand() % n;
//             adjecancyLists[u].push_back(v);
//             adjecancyLists[v].push_back(u);
//         }
//     }

//     for (int i = 0; i < n; i++) {
//         G.edgesOffset.push_back(G.adjacencyList.size());
//         G.edgesSize.push_back(adjecancyLists[i].size());
//         for (auto &edge: adjecancyLists[i]) {
//             G.adjacencyList.push_back(edge);
//         }
//     }

//     G.numVertices = n;
//     G.numEdges = G.adjacencyList.size();
// }
