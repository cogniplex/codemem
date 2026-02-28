# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Codemem is a standalone Rust memory engine for AI coding assistants — a single binary that stores code exploration findings so repos don't need re-exploring across sessions. It uses a graph-vector hybrid architecture with contextual embeddings, BM25 scoring, 37 MCP tools, 4 lifecycle hooks (SessionStart, UserPromptSubmit, PostToolUse, Stop), optional LLM observation compression, real-time file watching, and cross-session pattern detection.

## Workspace Structure (12 crates)

| Crate | Purpose |
|-------|---------|
| codemem-core | Shared types (`types.rs`), traits (`traits.rs`), errors (`error.rs`), config (`config.rs`): MemoryNode, Edge, Session, DetectedPattern, CodememConfig, 7 MemoryTypes, 23 RelationshipTypes, 12 NodeKinds, ScoringWeights, VectorBackend/GraphBackend/StorageBackend traits |
| codemem-storage | rusqlite (bundled), WAL mode, versioned schema migrations; split into `memory.rs` (CRUD), `graph_persistence.rs` (nodes/edges/embeddings), `queries.rs` (stats/sessions/patterns), `backend.rs` (StorageBackend trait impl), `migrations.rs` (schema versioning) |
| codemem-vector | usearch HNSW index, 768-dim cosine, M=16, efConstruction=200 |
| codemem-graph | petgraph + SQLite; split into `traversal.rs` (GraphBackend impl), `algorithms.rs` (PageRank, Louvain, SCC, betweenness, topological), cached centrality |
| codemem-embeddings | Pluggable via `from_env()`: Candle (local BERT, default), Ollama, OpenAI-compatible. `CachedProvider` wrapper, BAAI/bge-base-en-v1.5 (768-dim), LRU cache 10K. Safe concurrency via `LockPoisoned` error handling |
| codemem-index | tree-sitter code indexing, 6 languages (Rust, TS, Python, Go, C/C++, Java) |
| codemem-mcp | JSON-RPC stdio server, 37 MCP tools; split into `tools_memory.rs`, `tools_graph.rs`, `tools_recall.rs`, `tools_consolidation.rs`, `scoring.rs`, `types.rs`, `compress.rs`, `patterns.rs`. RwLock-based scoring weights, typed lock helpers, temporal edges, self-editing tools (refine/split/merge), LLM-powered consolidation |
| codemem-hooks | PostToolUse JSON parser, per-tool extractors, diff-aware memory (semantic summaries) |
| codemem-cli | clap derive, 15 commands; split into `commands_init.rs`, `commands_search.rs`, `commands_data.rs`, `commands_lifecycle.rs`, `commands_consolidation.rs`, `commands_export.rs` |
| codemem-viz | PCA visualization dashboard |
| codemem-bench | Criterion benchmarks, 20% regression threshold |
| codemem-watch | Real-time file watcher via notify (<50ms debounce), proper .gitignore parsing via `ignore` crate |

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
- **Consolidation**: 5 cycles — Decay (power-law, daily), Creative/REM (vector KNN O(n log n), weekly), Cluster (semantic cosine + union-find, monthly), Summarize (LLM-powered, on-demand), Forget (optional)
- **Self-editing memory**: refine_memory (EVOLVED_INTO provenance), split_memory (PART_OF edges), merge_memories (SUMMARIZES edges) — all with full store pipeline and temporal edge tracking
- **Temporal edges**: Edges have `valid_from`/`valid_to` fields for tracking when relationships are active
- **Pattern confidence**: Log-scaled confidence based on `ln(frequency)/ln(total_sessions)` instead of linear `count/N`
- **Storage**: Single database at ~/.codemem/codemem.db + ~/.codemem/codemem.idx, namespace-scoped queries
- **Persistent config**: TOML config at `~/.codemem/config.toml` with scoring weights, vector/graph tuning, embedding provider, and storage settings. Loaded at startup; partial configs merge with defaults
- **Schema migrations**: Versioned, idempotent SQL migrations tracked in a `schema_version` table. Schema extracted into `.sql` files
- **Safe concurrency**: Zero `.unwrap()` on lock acquisitions; all mutexes/rwlocks use typed lock helpers returning `CodememError::LockPoisoned`. `UnsafeCell` replaced with `RwLock` for scoring weights

## Key Dependencies

candle-core/nn/transformers (ML inference), usearch (HNSW), rusqlite (bundled SQLite), petgraph (graph), tokenizers (HuggingFace), tree-sitter (code parsing), clap (CLI), serde/serde_json, hf-hub (model download), lru (cache), sha2 (dedup), similar (diffs), notify/notify-debouncer-mini (file watching), reqwest (HTTP for embedding providers + LLM compression), criterion (benchmarks), toml (config persistence), ignore (gitignore parsing), tempfile (test fixtures)

## Build & Test

```bash
cargo build                    # Build all crates
cargo test --workspace         # Run all tests
cargo bench                    # Run benchmarks
cargo build --release          # Optimized release binary
```

## Documentation

- [Architecture](docs/architecture.md) — System design with Mermaid diagrams
- [MCP Tools](docs/mcp-tools.md) — All 37 tools reference
- [CLI Reference](docs/cli-reference.md) — All 15 commands
- [Comparison](docs/comparison.md) — vs claude-mem, AgentDB, AutoMem, and more
