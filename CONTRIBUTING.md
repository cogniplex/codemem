# Contributing to Codemem

## Prerequisites

- Rust 1.75+ (edition 2021)
- [Bun](https://bun.sh) for UI development
- No external runtime dependencies — all native libs are bundled

## Build & Test

```bash
git clone https://github.com/cogniplex/codemem.git
cd codemem
cargo build
cargo test --workspace
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```

CI treats **all warnings as errors** (`RUSTFLAGS: -D warnings`). Always run clippy before pushing.

UI:
```bash
cd ui && bun install && bun run dev          # Dev server
cd ui && bun run tsc --noEmit && bun run eslint .  # Checks
```

## Project Structure

6-crate Cargo workspace. See [docs/architecture.md](docs/architecture.md) for the full design.

| Crate | When to modify |
|-------|---------------|
| codemem-core | Memory types, traits, scoring weights, shared types, config |
| codemem-storage | SQLite schema, queries, graph algorithms, HNSW parameters |
| codemem-embeddings | Embedding providers (Candle/Ollama/OpenAI), cache behavior |
| codemem-engine | Indexing, hooks, enrichment, consolidation, recall, search |
| codemem | MCP tools, REST API routes, CLI commands, lifecycle hooks |
| codemem-bench | Benchmarks, performance targets |

## Code Conventions

- **No `.unwrap()` on locks.** Use typed lock helpers returning `CodememError::LockPoisoned`.
- **Hook handlers use `open_without_migrations()`** for fast startup.
- **SQLite batching** respects the 999-parameter limit.
- **Enrichment outputs** must be tagged `static-analysis`.
- Run `cargo fmt` before committing.

## Adding a Lifecycle Hook

1. Add the handler in `crates/codemem/src/cli/commands_lifecycle.rs` (use `read_hook_payload()` + `extract_hook_context()`)
2. Add a `Commands` enum variant in `crates/codemem/src/cli/mod.rs`
3. Wire the match arm in `run()`
4. Register the hook in `commands_init.rs` `hook_defs` vector
5. Add tests in `cli/tests/commands_lifecycle_tests.rs`
6. Update `docs/cli-reference.md` and `docs/architecture.md`

## Adding an Enrichment Type

1. Create a new file in `crates/codemem-engine/src/enrichment/`
2. Add it to the dispatch in `enrichment/mod.rs`
3. Tag outputs as `static-analysis` with a `track:<name>` tag
4. Use semantic dedup (cosine > 0.90) before storing insights

## Adding an MCP Tool

1. Add the tool in `crates/codemem/src/mcp/tools/`
2. Register it in `mcp/mod.rs`
3. Document it in `docs/mcp-tools.md`

## Testing

- Unit tests: `#[cfg(test)]` modules alongside source
- Use `tempfile` for test databases
- Hook tests construct `HookPayload` structs directly (include all optional fields)
- UI E2E tests via Playwright
- Don't regress benchmarks by more than 20% (CI enforced)

## Pull Requests

1. Create a branch from `main`
2. Make changes with tests
3. Run all checks (fmt, clippy, test)
4. Open a PR with a clear description

Significant changes (new crates, schema changes, new dependencies) should be discussed in an issue first.

## Commit Messages

Use conventional commits: `feat:`, `fix:`, `docs:`, `chore:`, `perf:`, `refactor:`, `test:`.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
