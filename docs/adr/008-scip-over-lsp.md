# ADR-008: Replace LSP Enrichment with SCIP-Primary, ast-grep Fallback

**Date:** 2026-03-13
**Status:** Accepted
**Supersedes:** LSP enrichment pipeline (added 2026-03-11, removed 2026-03-13)

## Context

Codemem's code graph was originally built in two layers:

1. **ast-grep / OXC** (always runs) — tree-sitter structural extraction producing nodes and heuristic CALLS/IMPORTS edges based on name matching.
2. **LSP enrichment** (optional) — Pyright for Python, tsserver for TypeScript. Spawned as long-running server processes, queried per-symbol to upgrade edges with compiler-grade resolution.

This worked but had significant operational and engineering costs:

- **Process management complexity.** Each LSP server is a stateful, long-running process requiring custom lifecycle management (spawn, initialize workspace, wait for ready, query, shutdown). A hung tsserver could block the entire enrichment pipeline.
- **Per-language bespoke integration.** Every language needed its own adapter (`pyright.rs`, `tsserver.rs`) with protocol-specific handling. Adding Go or Java meant writing another adapter from scratch.
- **Latency.** LSP servers must analyze the project before responding. Pyright on a large Python codebase takes 30-60s to initialize. Each definition call adds roundtrip overhead.
- **Incomplete coverage.** LSP only enriches symbols one at a time. References that ast-grep couldn't identify were never sent to LSP.
- **Duplicate work.** Pyright and scip-typescript use the same underlying compiler engines. Running them as LSP servers adds protocol overhead without additional analytical power.

SCIP (Source Code Intelligence Protocol) is a better fit:

- **Batch, offline, stateless.** Run `scip-python index .` and get a protobuf file with every definition, reference, and relationship. No server to manage.
- **Complete cross-reference graph.** Every occurrence of every symbol with exact source ranges and role classification. Nothing missed.
- **Same compilers, zero protocol overhead.** scip-python wraps Pyright, scip-typescript wraps tsc — same engines, static output.
- **10+ languages from one integration.** The SCIP protobuf schema is language-agnostic. One reader, one graph builder, 10 indexers.

## Decision

Remove LSP enrichment entirely. Use SCIP as the primary code intelligence source. Use ast-grep as the fallback for files and languages where no SCIP indexer is available.

```
SCIP indexers (primary)   → compiler-grade graph (confidence 0.15)
ast-grep / OXC (fallback) → structural graph for uncovered files (confidence 0.10)
Multi-layer fusion         → edges present in both layers get summed confidence
```

For each source file, exactly one primary path runs — SCIP if an indexer covered it, ast-grep otherwise.

### What was removed

| Component | Files | Reason |
|-----------|-------|--------|
| LSP enrichment pipeline | `index/lsp/mod.rs`, `lsp/pyright.rs`, `lsp/tsserver.rs` | Replaced by SCIP reader + graph builder |
| LSP persistence layer | `persistence/lsp.rs` | No longer needed |
| `enrich_codebase` MCP tool | `mcp/tools_enrich.rs` | Enrichment folded into `analyze` pipeline |
| `enrich_git_history` MCP tool | (same file) | Folded into `analyze` pipeline |
| LSP tests | `tests/lsp_tests.rs`, `tests/pyright_tests.rs`, `tests/tsserver_tests.rs` | Replaced by SCIP tests |

### What was added

| Component | Files | Role |
|-----------|-------|------|
| SCIP reader | `index/scip/mod.rs` | Parse `.scip` protobuf into `ScipDefinition` / `ScipReference` |
| SCIP orchestrator | `index/scip/orchestrate.rs` | Auto-detect languages and indexers, run them, produce `.scip` files |
| SCIP graph builder | `index/scip/graph_builder.rs` | Convert SCIP data into codemem graph nodes and edges |
| Definition extent inference | `index/scip/mod.rs` | Compute function body ranges from SCIP identifier-only positions |

### What stays unchanged

| Component | Role |
|-----------|------|
| ast-grep (tree-sitter + YAML rules) | Fallback for languages without SCIP indexers |
| OXC (JS/TS fast parser) | Fallback when scip-typescript unavailable |
| Cross-repo linker | SCIP feeds package info directly into it |
| Enrichment pipeline (git, security, complexity) | Operates on graph regardless of parser source |
| Code chunking for embedding | tree-sitter CST-aware chunking runs for all files |

## Consequences

### Positive

- **Graph accuracy jumps significantly.** Compiler-verified edges instead of heuristic name matching. In testing on a Go + TypeScript + Python monorepo, silent FK constraint failures from missing nodes dropped from ~20K lost edges to zero after stub node creation.
- **10 languages from one code path.** Rust, TypeScript, Python, Go, Java, C#, Ruby, PHP, Dart, C/C++ use the same reader and graph builder.
- **Simpler, faster indexing.** No server lifecycle. Indexers run as subprocesses, output read as files.
- **Removed ~1,200 lines of LSP adapter code.**
- **Multi-layer fusion gives confidence scoring.** ast-grep at 0.10, SCIP at 0.15. Both agreeing sums confidence.

### Negative

- **Requires SCIP indexers on PATH.** Less commonly installed than LSP servers. Mitigated by auto-detection and ast-grep fallback.
- **SCIP identifier-only positions require extent inference.** `infer_definition_extents()` computes body ranges heuristically.
- **Some LSP-specific features lost.** Hover info and signature help were occasionally richer. Rarely used by the graph in practice.

### Risks

- **SCIP indexer availability.** Languages without indexers (Swift, HCL) get ast-grep quality only. Coverage is strong for mainstream languages (10 supported).
- **Third-party indexer bugs.** scip-typescript emitting `Term` suffix for interface fields required a workaround. Similar issues may surface for other indexers.

## Alternatives Considered

### Keep LSP alongside SCIP
Same compilers, double the work. Merging two enrichment sources creates dedup complexity. Rejected.

### ast-grep only with smarter heuristics
Heuristic resolution cannot match compiler accuracy for overloads, type inference, generics. Engineering cost exceeds SCIP integration cost. Rejected.

### Tree-sitter + LSP (no SCIP)
Per-language adapter cost is O(n). Server lifecycle doesn't scale to 10+ languages. Rejected.
