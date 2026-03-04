# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Codemem is a standalone Rust memory engine for AI coding assistants — a single binary that stores code exploration findings so repos don't need re-exploring across sessions. It uses a graph-vector hybrid architecture with contextual embeddings, BM25 scoring, 30 MCP tools (+ legacy aliases for backwards compatibility), 4 lifecycle hooks (SessionStart, UserPromptSubmit, PostToolUse, Stop), 14 enrichment types, optional LLM observation compression, real-time file watching, cross-session pattern detection, a REST/SSE API, and a React web UI.

## Workspace Structure (6 crates)

| Crate | Purpose |
|-------|---------|
| codemem-core | Shared types (`types.rs`), traits (`traits.rs`), errors (`error.rs`), config (`config.rs`): MemoryNode, Edge, Session, DetectedPattern, CodememConfig, ChunkingConfig, EnrichmentConfig, 7 MemoryTypes, 24 RelationshipTypes, 13 NodeKinds, ScoringWeights, VectorBackend/GraphBackend/StorageBackend traits |
| codemem-storage | rusqlite (bundled) WAL mode + usearch HNSW vector index + petgraph graph engine. Split into `memory.rs` (CRUD), `graph_persistence.rs` (nodes/edges/embeddings), `queries.rs` (stats/sessions/patterns), `backend.rs` (StorageBackend trait impl), `migrations.rs` (schema versioning), `vector.rs` (HNSW 768-dim cosine, M=16), `graph/` (traversal with BFS/DFS/kind-aware filtering, algorithms: PageRank, Louvain, SCC, betweenness, topological, cached centrality, graph compaction, package nodes) |
| codemem-embeddings | Pluggable via `from_env()`: Candle (local BERT, default), Ollama, OpenAI-compatible. `CachedProvider` wrapper, BAAI/bge-base-en-v1.5 (768-dim), LRU cache 10K. Safe concurrency via `LockPoisoned` error handling |
| codemem-engine | Domain logic engine: `CodememEngine` struct holds all backends. Modules: `index/` (ast-grep code indexing, 14 languages, YAML-driven rules, manifest parsing for Cargo.toml/package.json/go.mod/pyproject.toml), `hooks/` (lifecycle hook handlers for 9 tool types: Read/Glob/Grep/Edit/Write/Bash/WebFetch/WebSearch/Agent/ListDir, trigger-based auto-insights), `watch/` (real-time file watcher, <50ms debounce, .gitignore support), `enrichment.rs` (14 enrichment types), `bm25.rs` (Okapi BM25 scoring with serialization), `scoring.rs` (hybrid scoring helpers), `recall.rs` (unified recall with temporal edge filtering), `patterns.rs` (cross-session pattern detection), `compress.rs` (LLM observation compression), `persistence.rs` (index persistence with cold-start-aware compaction), `metrics.rs` (operational metrics) |
| codemem | Unified binary + library crate with three transport modules: `mcp/` (JSON-RPC stdio + HTTP server, 30 MCP tools + legacy aliases, scoring, types), `api/` (REST/SSE API with Axum, routes for memories/graph/vectors/stats/patterns/insights/agents/config/timeline/namespaces/sessions, PCA point cloud, embedded React UI), `cli/` (clap derive, 18 commands, lifecycle hooks, config management) |
| codemem-bench | Criterion benchmarks, 20% regression threshold |

## Web UI (`ui/`)

React 19 + Vite + TailwindCSS v4 dashboard with three-tab GraphView: Graph (Sigma.js force-directed with relationship filters, focus mode, ego-graph), Explorer (paginated node browser with kind filters, mini ego-graph detail), Vector Space (3D PCA point cloud via React Three Fiber + Three.js). Key stack: Zustand (state), React Router, React Query, Sigma.js (graph visualization), @react-three/fiber + drei + three (3D point cloud), Recharts. Tests via Playwright. Uses Bun as package manager.

```bash
cd ui && bun install            # Install UI dependencies
cd ui && bun run dev            # Dev server (Vite)
cd ui && bun run build          # Production build → ui/dist/
cd ui && bun run tsc --noEmit   # TypeScript check
cd ui && bun run eslint .       # Lint check
cd ui && npx playwright test    # E2E tests
```

## Agent Definitions (`.claude/agents/`)

- **code-mapper** — Agent definition (`.claude/agents/code-mapper.md`) + skill directory (`.claude/skills/code-mapper/`) with 8 supporting files. Uses team-based deep analysis with priority-driven agent assignments. The thin agent definition restricts tools to read-only + memory storage + team orchestration; the skill directory contains phase-by-phase workflow instructions, memory type reference, error handling, human input protocol, and incremental re-analysis guide.

## Key Design Decisions

- **Embedding engine**: Pluggable via `CODEMEM_EMBED_PROVIDER` env var — Candle (default, pure Rust ML), Ollama (local server), OpenAI (API-compatible, works with Voyage AI/Together/Azure). `from_env()` factory in codemem-embeddings selects provider at runtime. All providers wrapped with LRU cache (10K).
- **Contextual embeddings**: Text enriched with metadata + graph context before embedding
- **7 memory types**: Decision, Pattern, Preference, Style, Habit, Insight, Context
- **24 relationship types**: RELATES_TO, LEADS_TO, PART_OF, REINFORCES, CONTRADICTS, EVOLVED_INTO, DERIVED_FROM, INVALIDATED_BY, DEPENDS_ON, IMPORTS, EXTENDS, CALLS, CONTAINS, SUPERSEDES, BLOCKS, IMPLEMENTS, INHERITS, SIMILAR_TO, PRECEDED_BY, EXEMPLIFIES, EXPLAINS, SHARES_THEME, SUMMARIZES, CO_CHANGED
- **9-component hybrid scoring**: vector_similarity 25%, graph_strength 20% (PageRank 40% + betweenness 30% + degree 20% + cluster 10%), BM25_token_overlap 15%, temporal 10%, importance 10%, confidence 10%, tag_matching 5%, recency 5%
- **BM25 scoring**: Okapi BM25 (k1=1.2, b=0.75) with code-aware tokenizer (camelCase/snake_case splitting)
- **4 lifecycle hooks**: SessionStart (context injection + pending analysis), UserPromptSubmit (prompt capture via full persist pipeline), PostToolUse (observation capture from 9 tool types with 5 auto-insight triggers), Stop (session summary + change tracking via full persist pipeline)
- **Change batching**: Stop hook stores `pending-analysis` tagged memories with edited file lists; SessionStart surfaces these for code-mapper agent analysis
- **Static analysis tagging**: All enrichment pipeline outputs tagged `static-analysis` so agents can find, review, refine, or delete them
- **Observation compression**: Optional LLM-powered compression via Ollama/OpenAI/Anthropic, configured via env vars
- **Impact-aware recall**: Cached PageRank + betweenness centrality wired into scoring, unified `recall` tool with `include_impact=true` returns graph impact data. Temporal edge filtering (valid_to < now) applied during graph expansion
- **Cross-session patterns**: Detects repeated searches, file hotspots, decision chains, tool preferences across sessions
- **Diff-aware memory**: Computes semantic diff summaries (function additions/removals, import changes, type definitions)
- **File watching**: Real-time notify-based watcher (<50ms debounce) with auto-indexing
- **Session continuity**: Session tracking with start/end/list, auto-started by lifecycle hooks
- **Consolidation**: 5 cycles — Decay (power-law, daily), Creative/REM (vector KNN O(n log n), weekly), Cluster (semantic cosine + union-find, monthly), Summarize (LLM-powered, on-demand), Forget (optional)
- **Self-editing memory**: refine_memory (EVOLVED_INTO provenance), split_memory (PART_OF edges), merge_memories (SUMMARIZES edges) — all with full store pipeline and temporal edge tracking
- **Temporal edges**: Edges have `valid_from`/`valid_to` fields for tracking when relationships are active
- **Pattern confidence**: Log-scaled confidence based on `ln(frequency)/ln(total_sessions)` instead of linear `count/N`
- **Storage**: Single database at ~/.codemem/codemem.db + ~/.codemem/codemem.idx, namespace-scoped queries
- **13 node kinds**: File, Package, Function, Method, Class, Interface, Type, Constant, Module, Memory, Endpoint, Test, Chunk
- **Graph compaction**: Two-pass pruning after indexing — chunks scored by centrality + structural parent + memory link + content density, symbols scored by call connectivity + visibility + kind + memory link + code size. Cold-start-aware: when no memories exist, memory_link weight is redistributed to other factors. Configurable via `ChunkingConfig`
- **14 enrichment types**: git history, security, performance, complexity (cyclomatic/cognitive), architecture inference, test mapping, API surface, doc coverage, change impact, code smells, hot+complex correlation, blame/ownership, advanced security scanning, quality stratification. All produce Insight memories tagged `static-analysis`
- **Enhanced symbol model**: Symbols now carry `parameters`, `return_type`, `is_async`, `attributes`, `throws`, `generic_params`, `is_abstract` metadata. 6 new symbol kinds: Field, Property, Constructor, EnumVariant, Macro, Decorator
- **Manifest parsing**: Cargo.toml, package.json, go.mod, pyproject.toml (PEP 621 + Poetry)
- **Chunking improvements**: O(log n) parent resolution via SymbolIntervalIndex, configurable overlap_lines, merged chunks preserve comma-separated node_kind labels
- **Reference improvements**: Rust grouped import decomposition (`std::{HashMap, HashSet}`), reference deduplication by (source, target, kind), AST reuse (parse once for both symbols and references)
- **BM25 persistence**: Index serializable via JSON for fast startup without re-indexing
- **MCP tool consolidation**: 30 primary tools (down from 43), with legacy aliases for backwards compatibility. Key merges: `recall` (unified recall/expansion/impact), `consolidate` (unified 6 modes), `codemem_status` (unified stats/health/metrics), `search_code` (unified semantic/text/hybrid), `get_symbol_graph` (unified deps/impact)
- **Auto-linking**: `store_memory` auto-detects code references (CamelCase, backticks, qualified paths, file paths) in content and links to matching graph nodes
- **Enrichment config**: `EnrichmentConfig` with tunable thresholds for git history, performance analysis, insight confidence, and semantic dedup (cosine > 0.90)
- **Persistent config**: TOML config at `~/.codemem/config.toml` with scoring weights, vector/graph tuning, embedding provider, chunking, enrichment, and storage settings. Loaded at startup; partial configs merge with defaults
- **Schema migrations**: Versioned, idempotent SQL migrations tracked in a `schema_version` table. Schema extracted into `.sql` files
- **Safe concurrency**: Zero `.unwrap()` on lock acquisitions; all mutexes/rwlocks use typed lock helpers returning `CodememError::LockPoisoned`. `UnsafeCell` replaced with `RwLock` for scoring weights

## Key Dependencies

candle-core/nn/transformers (ML inference), usearch (HNSW), rusqlite (bundled SQLite), petgraph (graph), tokenizers (HuggingFace), ast-grep-core/ast-grep-language (code parsing via tree-sitter), clap (CLI), serde/serde_json, serde_yaml (rule definitions), hf-hub (model download), lru (cache), sha2 (dedup), similar (diffs), notify/notify-debouncer-mini (file watching), reqwest (HTTP for embedding providers + LLM compression), criterion (benchmarks), toml (config persistence), ignore (gitignore parsing), tempfile (test fixtures), axum/tower-http (REST API), tokio (async runtime), ndarray (PCA)

## Build & Test

```bash
cargo build                    # Build all crates
cargo test --workspace         # Run all tests (531 tests)
cargo bench                    # Run benchmarks
cargo build --release          # Optimized release binary
cargo install --path crates/codemem      # Install CLI binary
cargo fmt --all -- --check     # Check formatting
cargo clippy --workspace --all-targets -- -D warnings  # Lint check
```

## CI Pipeline (`.github/workflows/ci.yml`)

All checks must pass on push to main:

| Job | What it checks |
|-----|---------------|
| Format | `cargo fmt --all -- --check` |
| Clippy | `cargo clippy --workspace --all-targets -- -D warnings` |
| Test | `cargo test --workspace` (ubuntu + macos) |
| Coverage | `cargo llvm-cov` → Codecov |
| UI Lint | `bun run tsc --noEmit` + `bun run eslint .` |
| UI E2E | Build UI, embed in codemem crate, Playwright tests |
| Benchmarks | `cargo bench --workspace --no-run` |

**Important**: CI uses `RUSTFLAGS: -D warnings` — all warnings are errors. Run `cargo clippy --workspace --all-targets -- -D warnings` locally before pushing.

## Documentation

- [Architecture](docs/architecture.md) — System design with Mermaid diagrams
- [MCP Tools](docs/mcp-tools.md) — All 28 tools reference (+ legacy aliases)
- [CLI Reference](docs/cli-reference.md) — All 18 commands
- [Comparison](docs/comparison.md) — vs claude-mem, AgentDB, AutoMem, and more
