# Contributing to Codemem

## Prerequisites

- Rust 1.75+ (edition 2021)
- No external runtime dependencies — all native libs are bundled or compiled from source

## Build

```bash
git clone https://github.com/cogniplex/codemem.git
cd codemem
cargo build
```

## Test

```bash
cargo test --workspace       # All tests
cargo bench                  # Criterion benchmarks
```

## Run from Source

```bash
cargo run -- init
cargo run -- search "some query"
cargo run -- stats
cargo run -- serve           # Start MCP server (stdio)
```

## Project Structure

Codemem is a Cargo workspace with 6 crates. See [docs/architecture.md](docs/architecture.md) for the full design.

| Crate | When to modify |
|-------|---------------|
| codemem-core | Adding/changing memory types, traits, scoring weights, shared types, config |
| codemem-storage | Changing SQLite schema, queries, persistence logic, graph algorithms, HNSW parameters, session management |
| codemem-embeddings | Adding embedding providers (Candle/Ollama/OpenAI), cache behavior |
| codemem-engine | Domain logic: indexing, hooks, enrichment, consolidation, recall, search, patterns, compression |
| codemem | Adding/modifying MCP tools, REST API routes, CLI commands, lifecycle hooks |
| codemem-bench | Adding benchmarks, changing performance targets |

## Web UI

The React dashboard lives in `ui/`. Uses Bun as package manager.

```bash
cd ui && bun install            # Install dependencies
cd ui && bun run dev            # Dev server (Vite)
cd ui && bun run build          # Production build
cd ui && bun run tsc --noEmit   # TypeScript check
cd ui && bun run eslint .       # Lint check
```

## Code Style

- Run `cargo fmt` before committing
- Run `cargo clippy --workspace --all-targets -- -D warnings` and address all warnings (CI treats warnings as errors)
- Prefer `Result<T, E>` over panicking — zero `.unwrap()` on locks in production code
- Keep functions focused and under ~50 lines where practical

## Testing

- Unit tests live alongside code (`#[cfg(test)]` modules)
- Integration tests in `crates/*/tests/`
- Benchmarks in `crates/codemem-bench/` using Criterion
- UI E2E tests via Playwright
- New features should include tests

## Performance

- Don't regress benchmarks by more than 20% (CI enforced)
- Run `cargo bench` before submitting changes to hot paths
- Profile with `cargo flamegraph` for performance investigations

## Commit Messages

Use conventional commits:

```
feat(mcp): add recall_with_expansion tool
fix(hooks): handle empty tool_response gracefully
perf(embeddings): switch to candle for pure Rust inference
docs: update architecture diagrams
test(graph): add Louvain community detection coverage
```

## Pull Requests

1. Fork and create a feature branch from `main`
2. Make changes with tests
3. Run `cargo fmt && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`
4. Open a PR with a clear description

## Architecture Decisions

Significant changes should be discussed in an issue first:

- Adding new crates to the workspace
- Changing the storage schema
- Modifying the MCP tool interface
- Adding new dependencies
- Changing embedding models or HNSW parameters
