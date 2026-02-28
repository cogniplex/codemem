# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Codemem is a standalone Rust memory engine for AI coding assistants — a single binary that stores code exploration findings so repos don't need re-exploring across sessions. It uses a graph-vector hybrid architecture with contextual embeddings, BM25 scoring, 33 MCP tools, 4 lifecycle hooks (SessionStart, UserPromptSubmit, PostToolUse, Stop), optional LLM observation compression, real-time file watching, and cross-session pattern detection.

## Workspace Structure (12 crates, ~26,500 LOC Rust)

| Crate | LOC | Purpose |
|-------|-----|---------|
| codemem-core | 1,111 | Shared types (`types.rs`), traits (`traits.rs`), errors (`error.rs`): MemoryNode, Edge, Session, DetectedPattern, 7 MemoryTypes, 23 RelationshipTypes, 12 NodeKinds, ScoringWeights, VectorBackend/GraphBackend/StorageBackend traits |
| codemem-storage | 2,252 | rusqlite (bundled), WAL mode; split into `memory.rs` (CRUD), `graph_persistence.rs` (nodes/edges/embeddings), `queries.rs` (stats/sessions/patterns), `backend.rs` (StorageBackend trait impl) |
| codemem-vector | 269 | usearch HNSW index, 768-dim cosine, M=16, efConstruction=200 |
| codemem-graph | 1,771 | petgraph + SQLite; split into `traversal.rs` (GraphBackend impl), `algorithms.rs` (PageRank, Louvain, SCC, betweenness, topological), cached centrality |
| codemem-embeddings | 846 | Pluggable via `from_env()`: Candle (local BERT, default), Ollama, OpenAI-compatible. `CachedProvider` wrapper, BAAI/bge-base-en-v1.5 (768-dim), LRU cache 10K |
| codemem-index | 7,813 | tree-sitter code indexing, 6 languages (Rust, TS, Python, Go, C/C++, Java) |
| codemem-mcp | 6,553 | JSON-RPC stdio server, 33 MCP tools; split into `tools_memory.rs`, `tools_graph.rs`, `tools_recall.rs`, `tools_consolidation.rs`, `scoring.rs`, `types.rs` |
| codemem-hooks | 979 | PostToolUse JSON parser, per-tool extractors, diff-aware memory (semantic summaries) |
| codemem-cli | 2,803 | clap derive, 15 commands; split into `commands_init.rs`, `commands_search.rs`, `commands_data.rs`, `commands_lifecycle.rs`, `commands_consolidation.rs`, `commands_export.rs` |
| codemem-viz | 696 | PCA visualization dashboard |
| codemem-bench | 7 | Criterion benchmarks, 20% regression threshold |
| codemem-watch | 188 | Real-time file watcher via notify (<50ms debounce), .gitignore support |

## Key Design Decisions

- **Embedding engine**: Pluggable via `CODEMEM_EMBED_PROVIDER` env var — Candle (default, pure Rust ML), Ollama (local server), OpenAI (API-compatible, works with Voyage AI/Together/Azure). `from_env()` factory in codemem-embeddings selects provider at runtime. All providers wrapped with LRU cache (10K).
- **Contextual embeddings**: Text enriched with metadata + graph context before embedding
- **7 memory types**: Decision, Pattern, Preference, Style, Habit, Insight, Context
- **23 relationship types**: RELATES_TO, LEADS_TO, PART_OF, REINFORCES, CONTRADICTS, EVOLVED_INTO, DERIVED_FROM, INVALIDATED_BY, DEPENDS_ON, IMPORTS, EXTENDS, CALLS, CONTAINS, SUPERSEDES, BLOCKS, IMPLEMENTS, INHERITS, SIMILAR_TO, PRECEDED_BY, EXEMPLIFIES, EXPLAINS, SHARES_THEME, SUMMARIZES
- **9-component hybrid scoring**: vector_similarity 25%, graph_strength 25% (PageRank 40% + betweenness 30% + degree 20% + cluster 10%), BM25_token_overlap 15%, temporal 10%, tag_matching 10%, importance 5%, confidence 5%, recency 5%
- **BM25 scoring**: Okapi BM25 (k1=1.2, b=0.75) with code-aware tokenizer (camelCase/snake_case splitting)
- **4 lifecycle hooks**: SessionStart (context injection), UserPromptSubmit (prompt capture), PostToolUse (observation capture), Stop (session summary)
- **Observation compression**: Optional LLM-powered compression via Ollama/OpenAI/Anthropic, configured via env vars
- **Impact-aware recall**: Cached PageRank + betweenness centrality wired into scoring, `recall_with_impact` tool returns graph impact data
- **Cross-session patterns**: Detects repeated searches, file hotspots, decision chains, tool preferences across sessions
- **Diff-aware memory**: Computes semantic diff summaries (function additions/removals, import changes, type definitions)
- **File watching**: Real-time notify-based watcher (<50ms debounce) with auto-indexing
- **Session continuity**: Session tracking with start/end/list, auto-started by lifecycle hooks
- **Consolidation**: 4 cycles — Decay (daily), Creative/REM (weekly), Cluster (monthly), Forget (optional)
- **Storage**: Single database at ~/.codemem/codemem.db + ~/.codemem/codemem.idx, namespace-scoped queries

## Key Dependencies

candle-core/nn/transformers (ML inference), usearch (HNSW), rusqlite (bundled SQLite), petgraph (graph), tokenizers (HuggingFace), tree-sitter (code parsing), clap (CLI), serde/serde_json, hf-hub (model download), lru (cache), sha2 (dedup), similar (diffs), notify/notify-debouncer-mini (file watching), reqwest (HTTP for embedding providers + LLM compression), criterion (benchmarks)

## Build & Test

```bash
cargo build                    # Build all crates
cargo test --workspace         # Run all 358 tests
cargo bench                    # Run benchmarks
cargo build --release          # Optimized release binary
```

## Documentation

- [Architecture](docs/architecture.md) — System design with Mermaid diagrams
- [MCP Tools](docs/mcp-tools.md) — All 33 tools reference
- [CLI Reference](docs/cli-reference.md) — All 15 commands
- [Comparison](docs/comparison.md) — vs claude-mem, AgentDB, AutoMem, and more
