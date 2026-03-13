# ADR-001: Graph-Vector Hybrid Architecture

**Date:** 2026-02-27
**Status:** Accepted

## Context

AI coding assistants (Claude Code, Cursor, Windsurf) re-explore codebases from scratch every session. A developer's agent might read 50-100 files to understand architecture, discover patterns, and map relationships — then discard all of it when the session ends. The next session starts over.

We needed a persistent memory engine that captures both structured relationships (function calls, imports, containment) and semantic similarity (related concepts, analogous patterns). Pure graph databases excel at traversal but can't answer "what's similar to X?". Pure vector stores handle similarity but lose structural relationships.

## Decision

Build codemem as a **graph-vector hybrid** with three retrieval layers:

1. **SQLite** — persistent storage for memories, metadata, sessions, schema migrations
2. **petgraph** — in-memory directed graph for code structure (nodes = symbols/files/memories, edges = typed relationships with temporal validity)
3. **HNSW vector index** (usearch) — approximate nearest-neighbor search over embeddings for semantic recall

All three layers are unified behind a single `Storage` type. The retrieval pipeline (recall) combines BM25 text matching, vector cosine similarity, graph proximity, and temporal decay into a single ranked result set.

The embedding layer supports multiple providers (Candle for local inference, Ollama, OpenAI) with an LRU cache. The default is Candle — pure Rust, no external dependencies beyond a ~440MB model download on first use.

## Consequences

- Recall quality is high because it fuses structural knowledge (graph) with semantic similarity (vectors) and lexical matching (BM25).
- The system is a single binary with SQLite — no external databases or services required for local use.
- The three-layer architecture adds complexity to the persistence pipeline: every `store_memory` must update SQLite, the graph, the vector index, and the BM25 index.
- Graph and vector index are in-memory, rebuilt from SQLite on startup. This keeps queries fast but means startup cost grows with graph size.
