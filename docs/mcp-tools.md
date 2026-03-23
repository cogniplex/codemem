# Codemem MCP Tools API Reference

Codemem exposes 32 tools over JSON-RPC 2.0 (stdio transport). All requests use the
`tools/call` method with `{"name": "<tool>", "arguments": {...}}` as params.

---

## Quick Examples

### store_memory -- request

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "store_memory",
    "arguments": {
      "content": "Team decided to use Axum over Actix for the API layer",
      "memory_type": "decision",
      "importance": 0.8,
      "tags": ["api", "framework", "axum"],
      "namespace": "/Users/dev/myproject"
    }
  }
}
```

### store_memory -- response

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"id\":\"a1b2c3d4-...\",\"status\":\"stored\",\"content_hash\":\"e5f6...\",\"graph_node_created\":true}"
      }
    ],
    "isError": false
  }
}
```

### recall -- request

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "recall",
    "arguments": {
      "query": "which web framework did we choose?",
      "k": 3,
      "namespace": "/Users/dev/myproject"
    }
  }
}
```

### recall -- response

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "[{\"id\":\"a1b2c3d4-...\",\"content\":\"Team decided to use Axum over Actix for the API layer\",\"memory_type\":\"decision\",\"importance\":0.8,\"score\":0.92,\"score_breakdown\":{\"vector_similarity\":0.88,\"graph_strength\":0.0,\"token_overlap\":0.75,...}}]"
      }
    ],
    "isError": false
  }
}
```

---

## Memory CRUD (7 tools)

### store_memory

Store a new memory with auto-embedding, type classification, and graph linking. Automatically links to code nodes mentioned in content.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `content` | string | yes | -- | The memory content to store |
| `memory_type` | string | no | `"context"` | One of: `decision`, `pattern`, `preference`, `style`, `habit`, `insight`, `context` |
| `importance` | number | no | `0.5` | Importance score, 0.0 to 1.0 |
| `tags` | string[] | no | `[]` | Searchable tags |
| `namespace` | string | no | -- | Project scope (e.g. working directory path) |
| `links` | string[] | no | `[]` | IDs of existing graph nodes to create RELATES_TO edges to |
| `auto_link` | boolean | no | `true` | Auto-link to code nodes mentioned in content |
| `expires_at` | string | no | -- | ISO 8601 expiration timestamp (e.g. `"2026-03-21T00:00:00Z"`) |
| `ttl_hours` | integer | no | -- | Time-to-live in hours (alternative to `expires_at`) |
| `git_ref` | string | no | -- | Git ref (branch/tag) to scope this memory to |

```json
{
  "name": "store_memory",
  "arguments": {
    "content": "Use serde_json::Value for dynamic JSON handling",
    "memory_type": "pattern",
    "importance": 0.6,
    "tags": ["serde", "json"],
    "namespace": "/home/dev/project"
  }
}
```

---

### recall

Unified memory search: 9-component hybrid scoring with optional graph expansion and impact analysis.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | yes | -- | Natural language search query |
| `k` | integer | no | `10` | Number of results to return |
| `memory_type` | string | no | -- | Filter by memory type |
| `namespace` | string | no | -- | Filter results to a specific namespace |
| `exclude_tags` | string[] | no | `[]` | Exclude memories with any of these tags |
| `min_importance` | number | no | -- | Only return memories above this importance threshold |
| `min_confidence` | number | no | -- | Only return memories above this confidence threshold |
| `expand` | boolean | no | `false` | Enable graph expansion to discover related memories |
| `expansion_depth` | integer | no | `1` | Max graph hops for expansion (when `expand=true`) |
| `include_impact` | boolean | no | `false` | Include PageRank, centrality, connected decisions, dependent files |
| `git_ref` | string | no | -- | Filter results to memories with this git ref (branch/tag) |

```json
{
  "name": "recall",
  "arguments": {
    "query": "error handling patterns",
    "k": 5,
    "memory_type": "pattern",
    "exclude_tags": ["static-analysis"],
    "expand": true,
    "include_impact": true
  }
}
```

---

### delete_memory

Delete a memory by ID, removing it from the vector index, graph, and storage.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `id` | string | yes | -- | Memory ID (UUID) |

```json
{
  "name": "delete_memory",
  "arguments": {
    "id": "a1b2c3d4-5678-9abc-def0-123456789abc"
  }
}
```

---

### associate_memories

Create a typed relationship between two nodes in the knowledge graph.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source_id` | string | yes | -- | Source memory/node ID |
| `target_id` | string | yes | -- | Target memory/node ID |
| `relationship` | string | yes | -- | One of: `RELATES_TO`, `LEADS_TO`, `PART_OF`, `REINFORCES`, `CONTRADICTS`, `EVOLVED_INTO`, `DERIVED_FROM`, `INVALIDATED_BY`, `DEPENDS_ON`, `IMPORTS`, `EXTENDS`, `CALLS`, `CONTAINS`, `SUPERSEDES`, `BLOCKS`, `IMPLEMENTS`, `INHERITS`, `SIMILAR_TO`, `PRECEDED_BY`, `EXEMPLIFIES`, `EXPLAINS`, `SHARES_THEME`, `SUMMARIZES`, `CO_CHANGED` |
| `weight` | number | no | `1.0` | Edge weight |

```json
{
  "name": "associate_memories",
  "arguments": {
    "source_id": "aaa-111",
    "target_id": "bbb-222",
    "relationship": "LEADS_TO",
    "weight": 0.8
  }
}
```

---

### refine_memory

Refine an existing memory. Default: creates a new version linked via EVOLVED_INTO. With `destructive=true`: updates in-place (replaces the old `update_memory`).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `id` | string | yes | -- | Memory ID to refine |
| `content` | string | no | -- | New refined content (optional unless `destructive=true`) |
| `importance` | number | no | -- | New importance score (0.0-1.0) |
| `tags` | string[] | no | -- | Updated tags |
| `destructive` | boolean | no | `false` | When true, update in-place instead of creating a new version |

```json
{
  "name": "refine_memory",
  "arguments": {
    "id": "a1b2c3d4-5678-9abc-def0-123456789abc",
    "content": "Updated: Use Axum 0.8 with Tower middleware for rate limiting",
    "importance": 0.9
  }
}
```

---

### split_memory

Split a memory into multiple parts, each linked to the original via PART_OF edges. The original memory is preserved.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `id` | string | yes | -- | Memory ID to split |
| `parts` | array | yes | -- | Array of objects with `content` and optional `tags`, `importance` |

```json
{
  "name": "split_memory",
  "arguments": {
    "id": "a1b2c3d4-5678-9abc-def0-123456789abc",
    "parts": [
      { "content": "Use Axum for HTTP routing", "tags": ["routing"] },
      { "content": "Use Tower middleware for rate limiting", "tags": ["middleware"] }
    ]
  }
}
```

---

### merge_memories

Merge multiple memories into a single summary memory linked via SUMMARIZES edges. Source memories are preserved.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source_ids` | string[] | yes | -- | Array of memory IDs to merge (min 2) |
| `content` | string | yes | -- | Content for the merged summary memory |
| `memory_type` | string | no | -- | Type for the merged memory |
| `importance` | number | no | `0.7` | Importance score |
| `tags` | string[] | no | `[]` | Tags for the merged memory |

```json
{
  "name": "merge_memories",
  "arguments": {
    "source_ids": ["aaa-111", "bbb-222", "ccc-333"],
    "content": "API layer uses Axum 0.8 with Tower middleware for routing, rate limiting, and auth",
    "memory_type": "insight",
    "importance": 0.8
  }
}
```

---

## Graph & Structure (10 tools)

### graph_traverse

Multi-hop graph traversal from a start node with optional filtering by node kind and relationship type.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `start_id` | string | yes | -- | Starting node ID |
| `max_depth` | integer | no | `2` | Maximum traversal depth |
| `algorithm` | string | no | `"bfs"` | Traversal algorithm: `bfs` or `dfs` |
| `exclude_kinds` | string[] | no | -- | Node kinds to exclude from results and traversal |
| `include_relationships` | string[] | no | -- | Only follow edges of these relationship types |
| `at_time` | string | no | -- | ISO 8601 timestamp -- filter out nodes/edges not valid at this time |

```json
{
  "name": "graph_traverse",
  "arguments": {
    "start_id": "aaa-111",
    "max_depth": 3,
    "algorithm": "dfs",
    "exclude_kinds": ["chunk"],
    "include_relationships": ["CALLS", "IMPORTS"]
  }
}
```

---

### summary_tree

Hierarchical summary tree (packages -> files -> symbols). Start from a `pkg:` node to browse the directory structure.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `start_id` | string | yes | -- | Node ID to start from (e.g., `pkg:src/`) |
| `max_depth` | integer | no | `3` | Maximum tree depth |
| `include_chunks` | boolean | no | `false` | Include chunk nodes in the tree |

```json
{
  "name": "summary_tree",
  "arguments": {
    "start_id": "pkg:src/",
    "max_depth": 2,
    "include_chunks": false
  }
}
```

---

### codemem_status

Unified status: database stats, health check, and operational metrics. Replaces the former `codemem_stats`, `codemem_health`, and `codemem_metrics` tools.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `include` | string[] | no | all | Sections to include: `"stats"`, `"health"`, `"metrics"` |

```json
{
  "name": "codemem_status",
  "arguments": {
    "include": ["stats", "health"]
  }
}
```

---

### search_code

Search code by meaning or name. Replaces the former `search_symbols` tool (use `mode=text`).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | yes | -- | Search query (natural language for semantic, substring for text) |
| `k` | integer | no | `10` | Number of results |
| `mode` | string | no | `"semantic"` | Search mode: `semantic` (vector search), `text` (symbol name), `hybrid` (both) |
| `kind` | string | no | -- | Filter by symbol kind (text/hybrid modes): `function`, `method`, `class`, `struct`, `enum`, `interface`, `type`, `constant`, `module`, `test` |

```json
{
  "name": "search_code",
  "arguments": {
    "query": "parse JSON config file",
    "k": 5,
    "mode": "hybrid"
  }
}
```

---

### get_symbol_info

Get full details of a symbol by qualified name, optionally including graph dependencies.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `qualified_name` | string | yes | -- | Fully qualified name (e.g. `module::Struct::method`) |
| `include_dependencies` | boolean | no | `false` | Include graph edges (calls, imports, etc.) |

```json
{
  "name": "get_symbol_info",
  "arguments": {
    "qualified_name": "codemem_engine::CodememEngine::recall",
    "include_dependencies": true
  }
}
```

---

### get_symbol_graph

Get symbol dependency graph. Replaces the former `get_dependencies` and `get_impact` tools.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `qualified_name` | string | yes | -- | Fully qualified name of the symbol |
| `depth` | integer | no | `1` | 1 = direct deps, >1 = impact analysis (BFS reachability) |
| `direction` | string | no | `"both"` | Direction: `incoming`, `outgoing`, or `both` |

```json
{
  "name": "get_symbol_graph",
  "arguments": {
    "qualified_name": "codemem_storage::Storage::open",
    "depth": 2,
    "direction": "incoming"
  }
}
```

---

### find_important_nodes

Run PageRank to find the most important/central nodes. When a namespace is specified, PageRank is computed only over nodes in that namespace, preventing cross-project score pollution. Replaces the former `get_pagerank` tool.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `top_k` | integer | no | `20` | Number of top-ranked nodes to return (bounded to max 1000) |
| `damping` | number | no | `0.85` | PageRank damping factor; must be in range (0, 1) |
| `namespace` | string | no | -- | Restrict PageRank to nodes in this namespace; omit for global PageRank |

```json
{
  "name": "find_important_nodes",
  "arguments": {
    "top_k": 10,
    "damping": 0.85,
    "namespace": "codemem"
  }
}
```

---

### find_related_groups

Run Louvain community detection to find clusters of related symbols. Replaces the former `get_clusters` tool.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `resolution` | number | no | `1.0` | Louvain resolution parameter (higher = more, smaller clusters) |

```json
{
  "name": "find_related_groups",
  "arguments": {
    "resolution": 1.5
  }
}
```

---

### get_node_memories

Retrieve all memories connected to a graph node via BFS traversal. Useful for checking what knowledge exists about a specific file, symbol, or package.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `node_id` | string | yes | -- | Graph node ID (e.g. `file:src/main.rs`, `sym:Module::func`) |
| `max_depth` | integer | no | `1` | Max graph hops to search for memories |
| `include_relationships` | string[] | no | -- | Only follow edges of these relationship types |

```json
{
  "name": "get_node_memories",
  "arguments": {
    "node_id": "file:src/main.rs",
    "max_depth": 2
  }
}
```

---

### node_coverage

Batch-check which graph nodes have attached memories. Returns memory count and coverage status for each node. Useful for identifying gaps in knowledge coverage.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `node_ids` | string[] | yes | -- | Array of graph node IDs to check |

```json
{
  "name": "node_coverage",
  "arguments": {
    "node_ids": ["sym:CodememEngine::recall", "file:src/main.rs", "pkg:src/"]
  }
}
```

---

### get_cross_repo

Scan workspace manifests (Cargo.toml, package.json, go.mod, pyproject.toml) and report cross-package dependencies.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `path` | string | no | -- | Path to scan (defaults to the last indexed codebase root) |

```json
{
  "name": "get_cross_repo",
  "arguments": {
    "path": "/Users/dev/monorepo"
  }
}
```

---

## Consolidation & Patterns (3 tools)

### consolidate

Unified consolidation tool. Replaces the former `consolidate_decay`, `consolidate_creative`, `consolidate_cluster`, `consolidate_forget`, `consolidate_summarize`, and `consolidation_status` tools.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `mode` | string | no | `"auto"` | `auto` (runs all cycles), `decay`, `creative`, `cluster`, `forget`, `summarize` |
| `threshold_days` | integer | no | `30` | For decay mode |
| `similarity_threshold` | number | no | `0.92` | For cluster mode |
| `importance_threshold` | number | no | `0.1` | For forget mode |
| `target_tags` | string[] | no | `[]` | For forget mode: only target memories with these tags |
| `max_access_count` | integer | no | `0` | For forget mode |
| `cluster_size` | integer | no | `5` | For summarize mode: minimum cluster size |

```json
{
  "name": "consolidate",
  "arguments": {
    "mode": "forget",
    "importance_threshold": 0.15,
    "target_tags": ["static-analysis"],
    "max_access_count": 1
  }
}
```

---

### detect_patterns

Detect cross-session patterns: repeated searches, file hotspots, decision chains, tool preferences. Replaces the former `detect_patterns` + `pattern_insights` tools.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `min_frequency` | integer | no | `3` | Minimum occurrences before flagging a pattern |
| `namespace` | string | no | -- | Filter by namespace |
| `format` | string | no | `"json"` | Output format: `json`, `markdown`, `both` |

```json
{
  "name": "detect_patterns",
  "arguments": {
    "min_frequency": 2,
    "format": "markdown"
  }
}
```

---

### get_decision_chain

Follow decision evolution through the knowledge graph via EVOLVED_INTO/LEADS_TO/DERIVED_FROM edges.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file_path` | string | no | -- | Filter by file path (at least one of `file_path` or `topic` required) |
| `topic` | string | no | -- | Filter by topic keyword |

```json
{
  "name": "get_decision_chain",
  "arguments": {
    "file_path": "src/auth.rs"
  }
}
```

---

## Namespace Management (3 tools)

### list_namespaces

List all namespaces with inline stats (counts, avg importance, type distribution, date range).

No parameters.

```json
{
  "name": "list_namespaces",
  "arguments": {}
}
```

---

### namespace_stats

Detailed statistics for a specific namespace.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `namespace` | string | yes | -- | Namespace to get stats for |

```json
{
  "name": "namespace_stats",
  "arguments": {
    "namespace": "/Users/dev/myproject"
  }
}
```

---

### delete_namespace

Delete all memories in a namespace (requires `confirm=true`).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `namespace` | string | yes | -- | Namespace to delete |
| `confirm` | boolean | yes | -- | Must be `true` to confirm deletion |

```json
{
  "name": "delete_namespace",
  "arguments": {
    "namespace": "/Users/dev/old-project",
    "confirm": true
  }
}
```

---

## Session & Context (2 tools)

### session_checkpoint

Mid-session progress report with activity summary, pattern detection, and focus areas.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `session_id` | string | yes | -- | Active session ID |
| `namespace` | string | no | -- | Filter by namespace |

```json
{
  "name": "session_checkpoint",
  "arguments": {
    "session_id": "abc-123"
  }
}
```

---

### session_context

Get session context: recent memories, pending analyses, active patterns, and focus areas.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `namespace` | string | no | -- | Filter by namespace |
| `k` | integer | no | `10` | Number of recent memories to include |

```json
{
  "name": "session_context",
  "arguments": {
    "namespace": "/Users/dev/myproject",
    "k": 20
  }
}
```

---

## Code Review (1 tool)

### review_diff

Analyze a unified diff for blast radius: map changed lines to symbols, find direct and transitive dependents, compute risk score, surface relevant memories and potentially missing changes.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `diff` | string | yes | -- | Unified diff text (e.g., output of `git diff`) |
| `depth` | integer | no | `2` | Max graph hops for transitive impact analysis |
| `base_ref` | string | no | -- | Base branch for overlay resolution (e.g., `"main"`) |

```json
{
  "name": "review_diff",
  "arguments": {
    "diff": "diff --git a/src/auth.rs ...",
    "depth": 2,
    "base_ref": "main"
  }
}
```

**Response** includes: `changed_symbols`, `direct_dependents`, `transitive_dependents`, `affected_files`, `affected_modules`, `risk_score`, `missing_changes`, `relevant_memories`.

---

## Temporal Queries (5 tools)

These tools require temporal ingestion to have been run (included in `codemem analyze`). Commits become graph nodes with `ModifiedBy` edges to the files/symbols they changed. All nodes carry `valid_from`/`valid_to` timestamps.

### what_changed

List commits and their affected files/symbols in a time range.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `from` | string | yes | -- | Start of time range (ISO 8601, e.g. `"2026-01-01T00:00:00Z"`) |
| `to` | string | yes | -- | End of time range (ISO 8601) |
| `namespace` | string | no | -- | Filter by namespace |

```json
{
  "name": "what_changed",
  "arguments": {
    "from": "2026-03-01T00:00:00Z",
    "to": "2026-03-17T00:00:00Z"
  }
}
```

---

### graph_at_time

Snapshot of the graph at a point in time: count of live nodes/edges, broken down by kind. Nodes/edges with `valid_to` before the timestamp are excluded.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `at` | string | yes | -- | Point in time (ISO 8601) |

```json
{
  "name": "graph_at_time",
  "arguments": {
    "at": "2026-02-15T00:00:00Z"
  }
}
```

---

### find_stale_files

Find files with high centrality or incoming edges that haven't been modified recently. Useful for identifying tech debt and neglected critical files.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `namespace` | string | no | -- | Filter by namespace |
| `stale_days` | integer | no | `90` | Files not modified in this many days are considered stale |
| `limit` | integer | no | `20` | Max results to return |

```json
{
  "name": "find_stale_files",
  "arguments": {
    "stale_days": 60,
    "limit": 10
  }
}
```

---

### detect_drift

Detect architectural drift between two time periods: new cross-module edges, hotspot files, coupling increases, added/removed files.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `from` | string | yes | -- | Start of period (ISO 8601) |
| `to` | string | yes | -- | End of period (ISO 8601) |
| `namespace` | string | no | -- | Filter by namespace |

```json
{
  "name": "detect_drift",
  "arguments": {
    "from": "2026-01-01T00:00:00Z",
    "to": "2026-03-01T00:00:00Z"
  }
}
```

---

### symbol_history

Get the commit history for a specific symbol or file node. Returns commits that modified it, with affected files and symbols.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `node_id` | string | yes | -- | Graph node ID (e.g. `"sym:MyClass::method"` or `"file:src/main.rs"`) |

```json
{
  "name": "symbol_history",
  "arguments": {
    "node_id": "sym:AuthService::validate"
  }
}
```

---

## Reference Tables

### Relationship Types (25)

| Type | Category | Description |
|------|----------|-------------|
| `RELATES_TO` | General | Generic association |
| `LEADS_TO` | General | Causal or sequential link |
| `PART_OF` | General | Containment/membership |
| `REINFORCES` | Knowledge | Supports or strengthens |
| `CONTRADICTS` | Knowledge | Conflicts with |
| `EVOLVED_INTO` | Knowledge | Superseded version |
| `DERIVED_FROM` | Knowledge | Created from |
| `INVALIDATED_BY` | Knowledge | Made obsolete by |
| `DEPENDS_ON` | Code | Runtime/build dependency |
| `IMPORTS` | Code | Import statement |
| `EXTENDS` | Code | Extension or mixin |
| `CALLS` | Code | Function/method invocation |
| `CONTAINS` | Code | Parent contains child |
| `SUPERSEDES` | Code | Replaces a previous version |
| `BLOCKS` | Code | Prevents or blocks |
| `IMPLEMENTS` | Structural | Implements interface/trait |
| `INHERITS` | Structural | Class inheritance |
| `SIMILAR_TO` | Semantic | Semantically similar |
| `PRECEDED_BY` | Temporal | Came before |
| `EXEMPLIFIES` | Knowledge | Serves as example of |
| `EXPLAINS` | Knowledge | Provides explanation for |
| `SHARES_THEME` | Semantic | High similarity across types (consolidation) |
| `SUMMARIZES` | Knowledge | Summary of |
| `CO_CHANGED` | Temporal | Files that frequently change together in git commits |
| `MODIFIED_BY` | Temporal | Commit modified this file/symbol |

### Memory Types (7)

| Type | Description |
|------|-------------|
| `decision` | A choice or decision made during development |
| `pattern` | A recurring code pattern or practice |
| `preference` | A stated preference (library, style, approach) |
| `style` | Code style or formatting convention |
| `habit` | A repeated workflow or behavioral habit |
| `insight` | A non-obvious finding or realization |
| `context` | General contextual information (default) |

### Hybrid Scoring Components (9)

| Component | Default Weight | Description |
|-----------|---------------|-------------|
| Vector similarity | 0.25 | Cosine similarity between query and memory embeddings |
| Graph strength | 0.20 | Multi-factor: PageRank (40%) + betweenness centrality (30%) + normalized degree (20%) + cluster bonus (10%) |
| BM25 token overlap | 0.15 | Okapi BM25 scoring with code-aware tokenizer (camelCase/snake_case splitting) |
| Scope context | 0.10 | Branch/repo/user awareness -- boosts memories matching current scope |
| Temporal | 0.10 | Temporal alignment with query context |
| Importance | 0.10 | The memory's stored importance score |
| Confidence | 0.10 | The memory's confidence score |
| Tag matching | 0.05 | Overlap between query-derived tags and memory tags |
| Recency | 0.05 | Boost for recently accessed/created memories |

Weights are configurable via `codemem config set scoring.<key> <value>` and persist in `~/.codemem/config.toml`.

### Node Kinds (15)

| Kind | Prefix | Description |
|------|--------|-------------|
| `File` | `file:` | Source file |
| `Function` | `sym:` | Standalone function |
| `Method` | `sym:` | Method on a struct/class |
| `Class` | `sym:` | Class definition |
| `Struct` | `sym:` | Struct definition |
| `Enum` | `sym:` | Enum definition |
| `Interface` | `sym:` | Interface/trait definition |
| `Type` | `sym:` | Type alias |
| `Constant` | `sym:` | Constant or static variable |
| `Module` | `sym:` | Module/namespace |
| `Test` | `sym:` | Test function |
| `Package` | `pkg:` | Package/crate from manifest |
| `Chunk` | `chunk:` | Code chunk for embedding |
| `Commit` | `commit:` | Git commit (temporal layer) |
| `PullRequest` | `pr:` | Pull request (temporal layer) |
