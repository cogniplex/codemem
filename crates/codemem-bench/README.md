# codemem-bench

Criterion benchmarks for Codemem's hot paths.

## Overview

Benchmarks vector search, storage operations, and graph algorithms with a 20% regression threshold enforced in CI.

## Running

```bash
cargo bench                    # Run all benchmarks
cargo bench -- vector          # Filter by name
```

## Benchmark Suites

Benchmarks are defined in `benches/` and cover:
- Vector index insert/search performance
- Storage CRUD operations
- Graph traversal and algorithm performance
