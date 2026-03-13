# ADR-002: Workspace Consolidation (13 Crates to 6)

**Date:** 2026-03-03
**Status:** Accepted

## Context

The initial workspace had 13 crates: `codemem-core`, `codemem-storage`, `codemem-graph`, `codemem-vector`, `codemem-embeddings`, `codemem-hooks`, `codemem-index`, `codemem-engine`, `codemem-mcp`, `codemem-api`, `codemem-cli`, `codemem-bench`, and `codemem-viz`. Many of these had thin APIs and existed as separate crates only because of an early assumption that consumers might want to use them independently.

In practice:
- `codemem-graph` and `codemem-vector` were never used outside `codemem-storage`.
- `codemem-hooks`, `codemem-index`, and `codemem-viz` were never used outside the engine.
- `codemem-mcp`, `codemem-api`, and `codemem-cli` all depended on the same engine and were always built together into a single binary.
- 13 crates meant 13 Cargo.toml files, 13 sets of dependency versions, and a complex CI matrix.

## Decision

Consolidate to **6 crates** with a clear dependency DAG:

```
codemem-core (types, traits, errors, config)
  ↓
codemem-storage (SQLite + petgraph + HNSW)
codemem-embeddings (Candle / Ollama / OpenAI)
  ↓
codemem-engine (domain logic: index, hooks, watch, persistence, enrichment)
  ↓
codemem (binary: CLI + MCP + REST API)

codemem-bench (benchmarks, separate)
```

- Merge `codemem-graph` and `codemem-vector` into `codemem-storage` as submodules.
- Create `codemem-engine` from hooks, watch, index, and MCP domain modules (BM25, compression, patterns, metrics, scoring).
- Merge `codemem-mcp`, `codemem-api`, and `codemem-cli` into a single `codemem` crate with three transport modules.

## Consequences

- Faster builds — fewer crate boundaries means more inlining opportunities and less redundant compilation.
- Simpler dependency management — 6 Cargo.toml files instead of 13.
- The binary crate (`codemem`) is a thin transport layer. All domain logic lives in `codemem-engine`, enforcing that MCP, CLI, and REST API share the same code paths.
- Trade-off: consumers who only want storage or embeddings must pull in those crates individually (still possible), but the engine is not independently usable without storage + embeddings.
