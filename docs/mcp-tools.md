# Codemem MCP Tools API Reference

Codemem exposes 38 tools over JSON-RPC 2.0 (stdio transport). All requests use the
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

### recall_memory -- request

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "recall_memory",
    "arguments": {
      "query": "which web framework did we choose?",
      "k": 3,
      "namespace": "/Users/dev/myproject"
    }
  }
}
```

### recall_memory -- response

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

## Core Memory Tools (8)

### store_memory

Store a new memory with auto-embedding, type classification, and graph linking.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `content` | string | yes | -- | The memory content to store |
| `memory_type` | string | no | `"context"` | One of: `decision`, `pattern`, `preference`, `style`, `habit`, `insight`, `context` |
| `importance` | number | no | `0.5` | Importance score, 0.0 to 1.0 |
| `tags` | string[] | no | `[]` | Searchable tags |
| `namespace` | string | no | -- | Project scope (e.g. working directory path) |
| `links` | string[] | no | `[]` | IDs of existing graph nodes to create RELATES_TO edges to |

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

### recall_memory

Semantic search using 9-component hybrid scoring with graph expansion and bridge discovery.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | yes | -- | Natural language search query |
| `k` | integer | no | `10` | Number of results to return |
| `memory_type` | string | no | -- | Filter by memory type |
| `namespace` | string | no | -- | Filter results to a specific namespace |

```json
{
  "name": "recall_memory",
  "arguments": {
    "query": "error handling patterns",
    "k": 5,
    "memory_type": "pattern"
  }
}
```

---

### update_memory

Update an existing memory's content and re-embed.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `id` | string | yes | -- | Memory ID (UUID) |
| `content` | string | yes | -- | New content (replaces existing, triggers re-embedding) |
| `importance` | number | no | -- | New importance score, 0.0 to 1.0 |

```json
{
  "name": "update_memory",
  "arguments": {
    "id": "a1b2c3d4-5678-9abc-def0-123456789abc",
    "content": "Updated: use Axum 0.8 with Tower middleware",
    "importance": 0.9
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

Create a typed relationship between two memories in the knowledge graph.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source_id` | string | yes | -- | Source memory/node ID |
| `target_id` | string | yes | -- | Target memory/node ID |
| `relationship` | string | yes | -- | One of: `RELATES_TO`, `LEADS_TO`, `PART_OF`, `REINFORCES`, `CONTRADICTS`, `EVOLVED_INTO`, `DERIVED_FROM`, `INVALIDATED_BY`, `DEPENDS_ON`, `IMPORTS`, `EXTENDS`, `CALLS`, `CONTAINS`, `SUPERSEDES`, `BLOCKS`, `IMPLEMENTS`, `INHERITS`, `SIMILAR_TO`, `PRECEDED_BY`, `EXEMPLIFIES`, `EXPLAINS`, `SHARES_THEME`, `SUMMARIZES` |
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

### graph_traverse

Multi-hop graph traversal from a start node for reasoning and bridge discovery.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `start_id` | string | yes | -- | Starting node ID |
| `max_depth` | integer | no | `2` | Maximum traversal depth |
| `algorithm` | string | no | `"bfs"` | Traversal algorithm: `bfs` or `dfs` |

```json
{
  "name": "graph_traverse",
  "arguments": {
    "start_id": "aaa-111",
    "max_depth": 3,
    "algorithm": "dfs"
  }
}
```

---

### codemem_stats

Get database and index statistics: memory count, graph node/edge counts, vector index size, and cache stats.

No parameters.

```json
{
  "name": "codemem_stats",
  "arguments": {}
}
```

---

### codemem_health

Health check across all Codemem subsystems (storage, vector, graph, embeddings).

No parameters.

```json
{
  "name": "codemem_health",
  "arguments": {}
}
```

---

## Structural Index Tools (10)

### index_codebase

Index a codebase directory to extract symbols and references using tree-sitter, populating the structural knowledge graph. Supports Rust, TypeScript, Python, Go, C/C++, and Java.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `path` | string | yes | -- | Absolute path to the codebase directory to index |

```json
{
  "name": "index_codebase",
  "arguments": {
    "path": "/Users/dev/myproject"
  }
}
```

---

### search_symbols

Search indexed code symbols by name substring, optionally filtering by kind.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | yes | -- | Substring to search for in symbol names |
| `kind` | string | no | -- | Filter by kind: `function`, `method`, `class`, `struct`, `enum`, `interface`, `type`, `constant`, `module`, `test` |
| `limit` | integer | no | `20` | Maximum number of results |

```json
{
  "name": "search_symbols",
  "arguments": {
    "query": "handle_request",
    "kind": "method",
    "limit": 10
  }
}
```

---

### get_symbol_info

Get full details of a symbol by qualified name, including signature, file path, doc comment, and parent.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `qualified_name` | string | yes | -- | Fully qualified name (e.g. `module::Struct::method`) |

```json
{
  "name": "get_symbol_info",
  "arguments": {
    "qualified_name": "codemem_mcp::McpServer::handle_request"
  }
}
```

---

### get_dependencies

Get graph edges (calls, imports, extends, etc.) connected to a symbol.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `qualified_name` | string | yes | -- | Fully qualified name of the symbol |
| `direction` | string | no | `"both"` | Direction: `incoming`, `outgoing`, or `both` |

```json
{
  "name": "get_dependencies",
  "arguments": {
    "qualified_name": "codemem_storage::Storage::open",
    "direction": "incoming"
  }
}
```

---

### get_impact

Impact analysis: find all graph nodes reachable from a symbol within N hops. Answers "what breaks if this changes?"

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `qualified_name` | string | yes | -- | Fully qualified name of the symbol to analyze |
| `depth` | integer | no | `2` | Maximum BFS depth for reachability |

```json
{
  "name": "get_impact",
  "arguments": {
    "qualified_name": "codemem_core::MemoryNode",
    "depth": 3
  }
}
```

---

### get_clusters

Run Louvain community detection on the knowledge graph to find clusters of related symbols.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `resolution` | number | no | `1.0` | Louvain resolution parameter (higher = more, smaller clusters) |

```json
{
  "name": "get_clusters",
  "arguments": {
    "resolution": 1.5
  }
}
```

---

### get_cross_repo

Scan for workspace manifests (Cargo.toml, package.json) and report workspace structure and cross-package dependencies.

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

### get_pagerank

Run PageRank on the full knowledge graph to find the most important/central nodes.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `top_k` | integer | no | `20` | Number of top-ranked nodes to return |
| `damping` | number | no | `0.85` | PageRank damping factor |

```json
{
  "name": "get_pagerank",
  "arguments": {
    "top_k": 10,
    "damping": 0.85
  }
}
```

---

### search_code

Semantic search over indexed code symbols using signature embeddings. Finds functions, types, and methods by meaning rather than exact name match.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | yes | -- | Natural language description of the code (e.g. "parse JSON config", "HTTP request handler") |
| `k` | integer | no | `10` | Number of results to return |

```json
{
  "name": "search_code",
  "arguments": {
    "query": "parse JSON config file",
    "k": 5
  }
}
```

---

### set_scoring_weights

Update the 9-component hybrid scoring weights at runtime. Weights are normalized to sum to 1.0. Omitted weights retain their current values.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `vector_similarity` | number | no | `0.25` | Weight for vector cosine similarity |
| `graph_strength` | number | no | `0.25` | Weight for graph relationship strength |
| `token_overlap` | number | no | `0.15` | Weight for content token overlap |
| `temporal` | number | no | `0.10` | Weight for temporal alignment |
| `tag_matching` | number | no | `0.10` | Weight for tag matching |
| `importance` | number | no | `0.05` | Weight for importance score |
| `confidence` | number | no | `0.05` | Weight for memory confidence |
| `recency` | number | no | `0.05` | Weight for recency boost |

```json
{
  "name": "set_scoring_weights",
  "arguments": {
    "vector_similarity": 0.4,
    "graph_strength": 0.3,
    "token_overlap": 0.1
  }
}
```

---

## Export/Import Tools (2)

### export_memories

Export memories as a JSON array with optional namespace and type filters. Returns memory objects with their graph edges.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `namespace` | string | no | -- | Filter by namespace |
| `memory_type` | string | no | -- | Filter by type: `decision`, `pattern`, `preference`, `style`, `habit`, `insight`, `context` |
| `limit` | integer | no | `100` | Maximum number of memories to export |

```json
{
  "name": "export_memories",
  "arguments": {
    "namespace": "/Users/dev/myproject",
    "memory_type": "decision",
    "limit": 50
  }
}
```

---

### import_memories

Import memories from a JSON array. Each object must have at least a `content` field. Auto-deduplicates by content hash.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `memories` | array | yes | -- | Array of memory objects (see schema below) |

Each memory object in the array:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `content` | string | yes | -- | The memory content |
| `memory_type` | string | no | `"context"` | Memory type |
| `importance` | number | no | `0.5` | Importance score, 0.0 to 1.0 |
| `confidence` | number | no | `1.0` | Confidence score, 0.0 to 1.0 |
| `tags` | string[] | no | `[]` | Searchable tags |
| `namespace` | string | no | -- | Namespace scope |
| `metadata` | object | no | -- | Arbitrary key-value metadata |

```json
{
  "name": "import_memories",
  "arguments": {
    "memories": [
      {
        "content": "Always run clippy before committing",
        "memory_type": "habit",
        "importance": 0.7,
        "tags": ["workflow", "rust", "linting"]
      },
      {
        "content": "The auth module uses JWT with RS256",
        "memory_type": "context",
        "namespace": "/Users/dev/myproject"
      }
    ]
  }
}
```

---

## Graph-Expanded Recall & Namespace Tools (4)

### recall_with_expansion

Semantic search with graph expansion: finds memories via vector similarity then expands through the knowledge graph to discover related memories up to N hops away.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | yes | -- | Natural language search query |
| `k` | integer | no | `5` | Number of results to return |
| `expansion_depth` | integer | no | `1` | Maximum graph hops for expansion (0 = no expansion) |
| `namespace` | string | no | -- | Filter results to a specific namespace |

```json
{
  "name": "recall_with_expansion",
  "arguments": {
    "query": "authentication flow",
    "k": 5,
    "expansion_depth": 2,
    "namespace": "/Users/dev/myproject"
  }
}
```

---

### list_namespaces

List all namespaces with their memory counts.

No parameters.

```json
{
  "name": "list_namespaces",
  "arguments": {}
}
```

---

### namespace_stats

Get detailed statistics for a specific namespace: count, average importance/confidence, type distribution, tag frequency, and date range.

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

Delete all memories in a namespace. This is destructive and requires explicit confirmation.

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

## Consolidation Tools (5)

### consolidate_decay

Run decay consolidation: reduce importance by 10% for memories not accessed within the threshold period.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `threshold_days` | integer | no | `30` | Memories not accessed in this many days will decay |

```json
{
  "name": "consolidate_decay",
  "arguments": {
    "threshold_days": 14
  }
}
```

---

### consolidate_creative

Run creative consolidation: find pairs of memories with overlapping tags but different types and create RELATES_TO edges between them. Inspired by REM-sleep creative association.

No parameters.

```json
{
  "name": "consolidate_creative",
  "arguments": {}
}
```

---

### consolidate_cluster

Run cluster consolidation: group memories by content hash prefix, keep the highest-importance memory per group, and delete duplicates.

No parameters.

```json
{
  "name": "consolidate_cluster",
  "arguments": {}
}
```

---

### consolidate_forget

Run forget consolidation: delete memories with importance below the threshold and zero access count.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `importance_threshold` | number | no | `0.1` | Delete memories with importance below this value (0.0 to 1.0) |

```json
{
  "name": "consolidate_forget",
  "arguments": {
    "importance_threshold": 0.15
  }
}
```

---

### consolidation_status

Show the last run timestamp and affected count for each consolidation cycle type.

No parameters.

```json
{
  "name": "consolidation_status",
  "arguments": {}
}
```

---

## Relationship Types Reference

The following relationship types are available for `associate_memories` and appear in graph traversal results:

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
| `IMPORTS` | Code | Import/use relationship |
| `EXTENDS` | Code | Inheritance or trait impl |
| `CALLS` | Code | Function/method invocation |
| `CONTAINS` | Code | Parent contains child |
| `SUPERSEDES` | Code | Replaces a previous version |
| `BLOCKS` | Code | Prevents or blocks |
| `IMPLEMENTS` | Code | Implements an interface/trait |
| `INHERITS` | Code | Class inheritance |
| `SIMILAR_TO` | Semantic | Semantically similar |
| `PRECEDED_BY` | Temporal | Came before |
| `EXEMPLIFIES` | Knowledge | Serves as example of |
| `EXPLAINS` | Knowledge | Provides explanation for |
| `SHARES_THEME` | Semantic | Shares a common theme |
| `SUMMARIZES` | Knowledge | Summary of |

## Memory Types Reference

| Type | Description |
|------|-------------|
| `decision` | A choice or decision made during development |
| `pattern` | A recurring code pattern or practice |
| `preference` | A stated preference (library, style, approach) |
| `style` | Code style or formatting convention |
| `habit` | A repeated workflow or behavioral habit |
| `insight` | A non-obvious finding or realization |
| `context` | General contextual information (default) |

## Hybrid Scoring Components

The 9-component scoring system used by `recall_memory` and `recall_with_expansion`:

| Component | Default Weight | Description |
|-----------|---------------|-------------|
| Vector similarity | 0.25 | Cosine similarity between query and memory embeddings |
| Graph strength | 0.25 | Multi-factor: PageRank (40%) + betweenness centrality (30%) + normalized degree (20%) + cluster bonus (10%) |
| BM25 token overlap | 0.15 | Okapi BM25 scoring with code-aware tokenizer (camelCase/snake_case splitting) |
| Temporal | 0.10 | Temporal alignment with query context |
| Tag matching | 0.10 | Overlap between query-derived tags and memory tags |
| Importance | 0.05 | The memory's stored importance score |
| Confidence | 0.05 | The memory's confidence score |
| Recency | 0.05 | Boost for recently accessed/created memories |

Weights can be adjusted at runtime via `set_scoring_weights`.

---

## Impact & Pattern Tools (4)

### recall_with_impact

Semantic recall enriched with graph impact data. Each result includes PageRank score, betweenness centrality, connected Decision memories, dependent files, and modification count.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | yes | -- | Natural language search query |
| `k` | integer | no | `10` | Number of results |
| `namespace` | string | no | -- | Filter by namespace |

```json
{
  "name": "recall_with_impact",
  "arguments": {
    "query": "authentication middleware",
    "k": 5
  }
}
```

---

### get_decision_chain

Get a chronologically ordered chain of Decision memories for a file or topic, linked through EVOLVED_INTO, LEADS_TO, and DERIVED_FROM edges.

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

### detect_patterns

Detect cross-session patterns: repeated searches, file hotspots, decision chains, and tool preferences.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `min_frequency` | integer | no | `3` | Minimum occurrences before flagging a pattern |
| `namespace` | string | no | -- | Filter by namespace |

```json
{
  "name": "detect_patterns",
  "arguments": {
    "min_frequency": 2,
    "namespace": "/Users/dev/myproject"
  }
}
```

---

### pattern_insights

Generate human-readable markdown insights from detected patterns. Groups findings by type (file hotspots, repeated searches, decision chains, tool preferences).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `min_frequency` | integer | no | `2` | Minimum occurrences before including a pattern |
| `namespace` | string | no | -- | Filter by namespace |

```json
{
  "name": "pattern_insights",
  "arguments": {
    "min_frequency": 2
  }
}
```
