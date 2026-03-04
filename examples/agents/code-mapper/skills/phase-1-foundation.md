# Phase 1: Foundation

The code-mapper runs this phase directly (no agents needed).

## Step 1: Index the codebase (incremental)

```
index_codebase { "path": "/path/to/project" }
```

This is incremental — it skips unchanged files via SHA-256 file hashes. Creates:
- `sym:qualified_name` nodes for every symbol
- `file:path` nodes for every source file
- `pkg:dir/` nodes forming a directory tree above files
- CALLS, IMPORTS, IMPLEMENTS, CONTAINS edges for every resolved reference

After indexing, **auto-compaction** prunes low-value nodes from the graph:
- **Chunk compaction**: Scores chunks by connectivity, symbol parentage, memory links, and code size. Keeps top-K per file.
- **Symbol compaction**: Scores symbols by call connectivity (30%), visibility (20%), kind (15%), memory links (20%), and code size (15%). Prunes private unreferenced symbols while always retaining classes, interfaces, modules, and memory-linked symbols.

Pruned nodes are removed from graph traversal/PageRank but their embeddings remain in the vector index for semantic search. The response includes `chunks_pruned`, `symbols_pruned`, and `packages_created` counts.

### Error recovery
- If `index_codebase` fails: check that the path exists and contains supported source files. Retry once. If it fails again, abort — indexing is required for all subsequent phases.

## Step 2: Run static enrichment (fast)

Run enrichment in a single call. Each analysis is fast (grep-based static analysis) and auto-tags insights with `static-analysis` for later agent review.

```
enrich_codebase { "path": "/path/to/project", "analyses": ["git", "security", "performance"] }
```

- **Git history**: Churn, co-change coupling, author distribution
- **Security**: Unsafe patterns, hardcoded secrets, input validation gaps, trust boundaries
- **Performance**: N+1 queries, missing caching, blocking I/O, large allocations, concurrency issues

### Error recovery
- If `enrich_codebase` fails: log the error and continue. Enrichment is optional — Phase 2-4 can proceed without it, but priority computation (Step 3) will lack git churn data. Use equal weights for missing signals.

## Step 2b: Browse the directory structure

Use `summary_tree` to get a hierarchical overview before diving into details:

```
summary_tree { "start_id": "pkg:src/", "max_depth": 3 }
```

This returns a packages → files → symbols tree (chunks excluded by default). Use it to understand the module layout and identify high-symbol-count files. Add `"include_chunks": true` if you need chunk-level detail.

## Step 3: Compute priorities

Query graph analysis tools to build a priority score per file:

```
find_important_nodes { "top_k": 50, "damping": 0.85 }
find_related_groups { "resolution": 1.0 }
```

Priority per file = weighted combination:

| Signal | Source | Weight | Rationale |
|--------|--------|--------|-----------|
| PageRank | `find_important_nodes` | 0.35 | High dependency = high blast radius |
| Git churn | `enrich_codebase` results / graph node `git_commit_count` | 0.25 | Frequently changed = actively evolved |
| Symbol count | Graph node `symbol_count` payload | 0.15 | More symbols = more to understand |
| Cluster size | `find_related_groups` result | 0.10 | Larger clusters = more interconnected |
| Unanalyzed | `get_node_memories` check per file node | 0.15 | Never-analyzed files get priority boost |

Compute this by querying existing tools and doing arithmetic in your reasoning — no new MCP tool needed.

### Weight calibration note
These weights are starting points. If enrichment failed (no git churn data), redistribute the 0.25 weight equally to PageRank (+0.10) and Unanalyzed (+0.15 → 0.30). If PageRank returns an empty graph, all files get equal priority.

### Error recovery
- If `find_important_nodes` returns empty: the graph may have no edges. Proceed with equal priority for all files.
- If `find_related_groups` returns a single cluster: all files are interconnected. Skip cluster_size weighting.

## Step 4: Check what's already analyzed

Use `get_node_memories` to check if specific files already have attached memories:

```
get_node_memories { "node_id": "file:src/important_file.rs" }
```

For a broader check, use recall with content-based matching (the `recall` tool does not support include-tags filtering):

```
recall { "query": "agent analysis complete", "exclude_tags": ["static-analysis"], "min_importance": 0.1 }
```

For each file, compare the stored `file_hash` in metadata with the current hash from the graph's `file_hashes` table. Files with stale hashes need re-analysis; files with matching hashes can be skipped.

## Step 5: Check for pending file changes

```
recall { "query": "pending analysis file changes", "k": 20 }
```

Look for memories with `pending-analysis` in their tags. If pending-analysis memories exist, elevate those files' priority. After analysis is complete, delete the pending-analysis memories to mark them as processed.
