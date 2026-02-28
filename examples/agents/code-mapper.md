---
name: code-mapper
description: Maps a codebase's structural relationships using Codemem's indexing and graph tools. Use after `codemem init` to build a comprehensive knowledge graph.
---

# Code Mapper Agent

Maps a codebase's structural relationships using Codemem's indexing and graph tools.

## When to Use

Run this agent after `codemem init` to build a comprehensive knowledge graph of a codebase. It indexes source code, resolves cross-file references, runs graph analysis, and stores architectural insights as persistent memories.

## Workflow

### Step 1: Index the codebase

Use `index_codebase` to parse all source files with tree-sitter and extract symbols (functions, structs, classes, methods, interfaces, constants) and their references (calls, imports, implements, inherits).

```
index_codebase { "path": "/path/to/project" }
```

This creates graph nodes (`sym:qualified_name`) for every symbol and edges for every resolved reference. Symbols are embedded with contextual enrichment (visibility, file path, parent, graph neighbors).

### Step 2: Identify the most important symbols

Run PageRank to find the highest-impact nodes in the codebase graph.

```
get_pagerank { "top_k": 30, "damping": 0.85 }
```

These are the symbols that the most other code depends on — changing them has the widest blast radius.

### Step 3: Detect architectural clusters

Run Louvain community detection to find natural groupings of related code.

```
get_clusters { "resolution": 1.0 }
```

Each cluster represents a cohesive module or subsystem. Higher resolution values produce more, smaller clusters.

### Step 4: Analyze key dependencies

For each high-PageRank symbol, check what depends on it and what it depends on.

```
get_dependencies { "qualified_name": "module::SymbolName", "direction": "both" }
get_impact { "qualified_name": "module::SymbolName", "depth": 2 }
```

### Step 5: Map cross-package structure

For monorepos or workspaces, scan manifests to understand package boundaries.

```
get_cross_repo { "path": "/path/to/project" }
```

### Step 6: Store architectural insights

Store the findings as persistent memories so they're available in future sessions.

```
store_memory {
  "content": "The auth module (cluster #3) is the most interconnected subsystem with 12 inbound dependencies. Key entry point: auth::middleware::validate_token (PageRank: 0.089).",
  "memory_type": "insight",
  "importance": 0.9,
  "tags": ["architecture", "auth", "dependencies"],
  "namespace": "/path/to/project",
  "links": ["sym:auth::middleware::validate_token"]
}
```

### Step 7: Semantic code search

Use meaning-based search to find code by description rather than name.

```
search_code { "query": "database connection pooling", "k": 5 }
search_symbols { "query": "handle", "kind": "method", "limit": 10 }
```

## What Gets Created

| Artifact | Storage | Description |
|----------|---------|-------------|
| Symbol nodes | `graph_nodes` table | One per function/struct/class/method/etc. ID format: `sym:qualified_name` |
| Reference edges | `graph_edges` table | CALLS, IMPORTS, IMPLEMENTS, INHERITS, DEPENDS_ON between symbols |
| Symbol embeddings | `memory_embeddings` + HNSW index | 768-dim contextual embeddings for semantic code search |
| File hash cache | `file_hashes` table | SHA-256 per file for incremental re-indexing |
| Insight memories | `memories` table | Architectural findings stored for cross-session recall |

## Supported Languages

Rust (.rs), TypeScript (.ts/.tsx), Python (.py), Go (.go), C/C++ (.c/.h/.cpp/.hpp), Java (.java)

## Tips

- Run `get_pagerank` first to orient yourself — the top 10 symbols tell you where the architectural weight is
- Use `get_impact` with depth=2 before refactoring to understand blast radius
- After major refactors, re-run `index_codebase` to update the graph
- `search_code` finds functions by meaning ("parse JSON config") while `search_symbols` finds by name substring ("parse")
- Store `decision` memories when you make architectural choices so future sessions can recall why
