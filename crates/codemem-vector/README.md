# codemem-vector

HNSW vector index using usearch for 768-dimensional cosine similarity search.

## Overview

Implements the `VectorBackend` trait from `codemem-core`. Provides fast approximate nearest neighbor search for memory and code symbol embeddings.

## Configuration

| Parameter | Value |
|-----------|-------|
| Dimensions | 768 |
| Metric | Cosine similarity |
| M (connectivity) | 16 |
| efConstruction | 200 |
| efSearch | 100 |

## Key Operations

- `insert(id, embedding)` — Add a vector to the index
- `search(query, k)` — Find k nearest neighbors
- `remove(id)` — Remove a vector
- `save(path)` / `load(path)` — Persist to disk at `~/.codemem/codemem.idx`
