# ADR-007: Lazy Engine Initialization

**Date:** 2026-03-10
**Status:** Accepted

## Context

`CodememEngine::from_db_path()` eagerly loaded all three heavy components on construction:

1. **BERT embedding model** via Candle — ~2.7s on first load (model download + init)
2. **HNSW vector index** — deserialize from SQLite
3. **BM25 index** — rebuild from stored terms

This meant every Claude Code hook invocation (PostToolUse, SessionStart) paid the full ~2.7s initialization cost, even though hooks only need SQLite + graph access. Users reported noticeable lag after every tool call.

## Decision

Use `OnceLock<Mutex<T>>` for vector index, BM25 index, and embeddings provider. `from_db_path()` only opens SQLite and loads the graph (~200ms). Heavy components initialize lazily on first access via `lock_vector()`, `lock_bm25()`, `lock_embeddings()`.

- **Hooks** never touch embeddings/vector/BM25, so they stay fast.
- **`analyze()`** explicitly triggers all lazy inits upfront since it needs everything.
- **`store_memory`** triggers embeddings + vector on first call.
- **Backfill mechanism:** when the embeddings provider first loads, it finds and embeds any memories that were stored without embeddings by hooks.

## Consequences

- Hook latency dropped from ~2.7s to ~200ms.
- First `recall` or `store_memory` call in a session pays the initialization cost. This is acceptable because these are interactive MCP tool calls where the user expects some latency.
- The backfill mechanism ensures no memories are permanently stored without embeddings, even if they were created by hooks before the embeddings provider loaded.
- Code must be careful about which engine methods trigger lazy init and which don't. The accessor pattern (`lock_vector()` etc.) makes this explicit.
