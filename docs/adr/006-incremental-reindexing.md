# ADR-006: Incremental Re-Indexing with Symbol-Level Diff

**Date:** 2026-03-10
**Status:** Accepted

## Context

Full re-indexing of a large codebase (10K+ files) takes minutes: parse every file, extract symbols, resolve edges, embed everything. Most files don't change between indexing runs. Re-indexing unchanged files wastes time and produces identical results.

Additionally, when files are deleted or symbols are renamed, the old nodes and edges remain in the graph as orphans — never cleaned up, polluting traversal results and inflating PageRank scores.

## Decision

Add a `ChangeDetector` that tracks SHA-256 hashes of file contents, scoped per namespace:

1. **File-level skip.** Before parsing a file, check if its content hash matches the stored hash. If unchanged, skip entirely.
2. **Symbol-level diff.** When a file is re-indexed, compare the new symbol set against the old one. Symbols that no longer exist get their graph nodes marked stale and their memory edges redirected to the parent file node.
3. **Orphan detection.** Periodic scan for graph nodes whose source files no longer exist, and edges pointing to deleted nodes. Orphans are flagged for cleanup.
4. **Watcher integration.** The file watcher's `flush_batch` uses the change detector for incremental updates on file save events.

Hashes are persisted in SQLite (`file_hashes` table) and loaded into memory at the start of each indexing run. The `--force` flag bypasses the change detector for full re-indexing.

## Consequences

- Re-indexing a codebase with 5% changed files is ~20x faster than a full re-index.
- Stale symbols are cleaned up automatically, keeping the graph accurate over time.
- The change detector adds a small overhead to every indexing run (hash computation + lookup), but this is negligible compared to parsing and embedding.
- Edge case: if a file's content changes but its hash collides (astronomically unlikely with SHA-256), the change would be missed. Acceptable risk.
