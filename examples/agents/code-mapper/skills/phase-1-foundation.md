# Phase 1: Foundation

The code-mapper runs this phase directly (no agents needed).

**Task tracking**: Create a Phase 1 task via TaskCreate. Set to `in_progress` when starting, `completed` when all steps finish.

## Prerequisite: Index the codebase from CLI

Before running the agent, index the codebase from the CLI (much faster than via MCP):

```bash
codemem index /path/to/project
```

This is incremental — it skips unchanged files via SHA-256 file hashes. The agent assumes indexing is already done.

## Step 1: Run comprehensive static enrichment

Run ALL relevant enrichment in a single call. Each analysis is fast (grep-based) and auto-tags insights with `static-analysis`.

```
enrich_codebase {
  "path": "/path/to/project",
  "analyses": ["git", "security", "performance", "complexity", "architecture", "api_surface", "test_mapping", "doc_coverage"]
}
```

- **Git history**: Churn, co-change coupling, author distribution
- **Security**: Unsafe patterns, hardcoded secrets, input validation gaps, trust boundaries
- **Performance**: N+1 queries, missing caching, blocking I/O, large allocations
- **Complexity**: Cyclomatic/cognitive complexity per function — identifies functions that need deep analysis
- **Architecture**: Module boundaries, layering, dependency directions
- **API surface**: REST routes, RPC handlers, endpoint definitions — critical for web projects
- **Test mapping**: Which tests cover which modules — identifies coverage gaps
- **Doc coverage**: Undocumented public APIs — identifies documentation debt

### Error recovery
- If `enrich_codebase` fails: log the error and continue. Enrichment is optional. Use equal weights for missing signals in priority computation.

## Step 2: Browse the directory structure

Use `summary_tree` to get a hierarchical overview:

```
summary_tree { "start_id": "pkg:src/", "max_depth": 4 }
```

Use higher `max_depth` for monorepos or deeply nested projects. This returns packages → files → symbols (chunks excluded by default).

Record:
- Total file count
- Total symbol count
- Package structure and nesting depth
- Largest files by symbol count (these need file-size-aware reading strategies)

## Step 3: Compute priorities at SYMBOL level

Query graph analysis tools to build priority scores per **symbol** (not per file):

```
find_important_nodes { "top_k": 100, "damping": 0.85 }
find_related_groups { "resolution": 1.0 }
```

For large repos (500+ symbols), increase `top_k` to 200 or more.

### Priority formula per symbol:

| Signal | Source | Weight | Rationale |
|--------|--------|--------|-----------|
| PageRank | `find_important_nodes` | 0.30 | High dependency = high blast radius |
| Git churn | Enrichment / node `git_commit_count` | 0.20 | Frequently changed = actively evolved |
| Complexity | Enrichment / cyclomatic complexity | 0.15 | Complex code = most value from analysis |
| Cluster centrality | `find_related_groups` position | 0.10 | Central in cluster = key connector |
| Unanalyzed | `get_node_memories` check | 0.15 | Never-analyzed symbols get priority boost |
| Is public API | Node kind = Endpoint or public | 0.10 | External-facing code must be documented |

### Weight calibration
- No git data → redistribute 0.20 to PageRank (+0.10) and Unanalyzed (+0.10)
- No complexity data → redistribute 0.15 to PageRank (+0.10) and Cluster (+0.05)
- Empty PageRank → all symbols get equal priority

## Step 4: Build the symbol inventory

Create a structured inventory for the planner (Phase 2):

### 4a: Tier all symbols

| Tier | Criteria | Count Target |
|------|----------|-------------|
| **Critical** | Top 5% by priority score | Deep analysis (2-4 memories) |
| **Important** | Top 20% by priority score | Focused analysis (1-2 memories) |
| **Standard** | Remaining 80% | File-grouped baseline+ (1 memory per file, reading actual code) |

### 4b: Enumerate API endpoints

```
search_code { "query": "router handler endpoint route", "mode": "hybrid", "top_k": 100 }
graph_traverse { "start_id": "pkg:src/", "max_depth": 5, "include_kinds": ["Endpoint"] }
```

All endpoints are auto-promoted to at least **Important** tier regardless of PageRank.

### 4c: Identify large files needing chunked analysis

Files with 500+ lines or 20+ symbols need special handling — mark them for chunked reading by agents.

### 4d: Check existing coverage

```
node_coverage { "node_ids": [<all critical + important symbol IDs>] }
```

Mark already-covered symbols (with fresh, non-stale memories) so agents can skip them.

## Step 5: Check for pending changes

```
recall { "query": "pending analysis file changes", "k": 20 }
```

If `pending-analysis` memories exist, elevate those files' symbols to at least Important tier.

## Output

Phase 1 produces:
1. **Enrichment results** stored as `static-analysis` tagged memories
2. **Directory structure** understanding
3. **Symbol inventory** with tiers, priorities, coverage status
4. **API endpoint list** with locations
5. **Large file list** needing chunked reading
6. **Pending changes** list

This feeds directly into Phase 2 planning.
