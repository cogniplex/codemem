# codemem-graph

Graph engine with petgraph algorithms, centrality caching, and SQLite persistence.

## Overview

Implements the `GraphBackend` trait from `codemem-core`. Maintains an in-memory petgraph directed graph synced to SQLite for persistence.

## Modules

- `traversal.rs` — `GraphBackend` trait impl (BFS, DFS, shortest path, neighbor lookup, multi-hop expansion)
- `algorithms.rs` — PageRank, personalized PageRank, Louvain community detection, betweenness centrality, strongly connected components, topological layers

## Centrality Cache

`recompute_centrality()` precomputes PageRank and betweenness scores on startup and after graph mutations. These cached scores feed into the 9-component hybrid scoring system (graph_strength component).
