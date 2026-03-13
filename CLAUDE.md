# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Build & Test

```bash
cargo build                    # Build all crates
cargo test --workspace         # Run all tests
cargo bench                    # Run benchmarks
cargo fmt --all -- --check     # Check formatting
cargo clippy --workspace --all-targets -- -D warnings  # Lint check
```

UI (uses Bun, not npm):
```bash
cd ui && bun install && bun run dev       # Dev server
cd ui && bun run build                    # Production build → ui/dist/
cd ui && bun run tsc --noEmit             # Type check
cd ui && bun run eslint .                 # Lint
cd ui && npx playwright test              # E2E tests
```

**CI uses `RUSTFLAGS: -D warnings`** — all warnings are errors. Always run clippy before pushing.

## Git Workflow

When working with git, always confirm the current branch before committing or pushing. Never assume you're on the correct branch.

## Code Review

When asked to do code review, do a single thorough pass. Do not flip-flop on whether a bug exists or is fixed — if uncertain, re-read the code before answering.

## General Principles

Do not over-engineer solutions. Apply only the changes requested. Do not add extra fields, workflows, or abstractions unless explicitly asked.

This is primarily a Rust project. Always run `cargo test` after making changes. Run `cargo check` before claiming code compiles cleanly.

When spawning sub-agents or waiting on background tasks, do NOT poll TaskList repeatedly. Check once, do other productive work, then check again after a reasonable interval.

## Workspace Layout

6 crates: `codemem-core` (types/traits/errors/config) → `codemem-storage` (SQLite + HNSW + petgraph) + `codemem-embeddings` (Candle/Ollama/OpenAI) → `codemem-engine` (domain logic) → `codemem` (binary: CLI + MCP + REST API). Plus `codemem-bench` for benchmarks.

## Development Quirks & Conventions

### Storage method shadowing
`Storage` has concrete methods like `list_sessions(namespace)` (1-arg shorthand) AND `StorageBackend` trait methods like `list_sessions(namespace, limit)` (2-arg). Rust resolves concrete methods first. Be careful when code takes `&dyn StorageBackend` vs concrete `Storage` — the available method signatures differ.

### Lock safety
Zero `.unwrap()` on mutex/rwlock acquisitions — all use typed lock helpers returning `CodememError::LockPoisoned`. Follow this pattern for any new lock usage.

### Hook payload: `tool_response` is `serde_json::Value`
Claude Code sends `tool_response` as a JSON object (not a plain string). The `HookPayload::tool_response_text()` method extracts meaningful text: `file.content` for Read, `stdout` for Bash, `text` for text-bearing responses, with JSON fallback. All extractors in `hooks/extractors.rs` must call this method.

### Lifecycle hooks registered by `codemem init`
The init command registers hooks in `.claude/settings.json`. Hook commands read a single-line JSON payload from stdin and write JSON to stdout. The shared helpers `read_hook_payload()` and `extract_hook_context()` in `commands_lifecycle.rs` handle common stdin/cwd/session_id extraction.

### Embedding provider selection
Configured via `CODEMEM_EMBED_PROVIDER` env var. `from_env()` factory in codemem-embeddings. Default is Candle (pure Rust, downloads ~440MB model on first use). All providers wrapped with LRU cache (10K entries).

### SQLite batching
Multi-row INSERT respects SQLite's 999-parameter limit. `StorageBackend` has `begin/commit/rollback_transaction` with `AtomicBool` guard. Batch cascade deletes for consolidation.

### Graph compaction is cold-start-aware
When no memories exist, the memory_link scoring weight is redistributed to other factors. Don't assume memory-linked nodes always exist.

### Enrichment outputs
All enrichment pipeline outputs are tagged `static-analysis`. Agents find, review, refine, or delete these. Semantic dedup (cosine > 0.90) prevents bloat from repeated analysis runs.

### BM25 index
Okapi BM25 with code-aware tokenizer that splits camelCase and snake_case. Index is JSON-serializable for persistence. Lazy-initialized alongside vector index and embeddings provider.

### CodememEngine lazy init
Vector index, BM25, and embeddings provider are initialized on first use (not at construction). This means `from_db_path()` is cheap. Don't assume these are available immediately after construction — they initialize on first `persist_memory`, `recall`, or `search_code` call.

### Test fixtures
Use `tempfile` for test DBs. Storage tests open ephemeral databases. Hook tests construct `HookPayload` structs directly with all fields (including the newer optional ones like `hook_event_name`, `tool_use_id`, etc.).

### Schema migrations
Versioned, idempotent SQL in `.sql` files. `open()` runs migrations; `open_without_migrations()` skips them (used by hook commands for speed). Always use `open_without_migrations` in hook handlers to avoid blocking the assistant.

### Agent definitions
`.claude/agents/` contains agent prompts installed by `codemem init`. The code-mapper is the team lead; specialized agents have restricted tool subsets. Don't give specialized agents `Agent`/`TeamCreate`/`TaskCreate` tools.

### Namespace convention
Namespace = directory basename (not full path). `namespace_from_path()` / `namespace_from_cwd()` in `cli/mod.rs`.

### Config persistence
TOML at `~/.codemem/config.toml`. Partial configs merge with defaults at startup. Scoring weights, vector/graph tuning, embedding provider, chunking, enrichment settings.

## Documentation

- [Architecture](docs/architecture.md) — System design with Mermaid diagrams
- [Index & Enrich Pipeline](docs/pipeline.md) — Data flow from source files to annotated graph
- [MCP Tools](docs/mcp-tools.md) — All MCP tools reference
- [CLI Reference](docs/cli-reference.md) — All CLI commands
- [Comparison](docs/comparison.md) — vs claude-mem, AgentDB, AutoMem, and more
