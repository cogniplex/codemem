use serde_json::{json, Value};

pub(super) fn tool_definitions() -> Vec<Value> {
    vec![
        // ── Memory CRUD (7 tools) ──────────────────────────────────────────
        json!({
            "name": "store_memory",
            "description": "Store a new memory with auto-embedding, type classification, and graph linking. Automatically links to code nodes mentioned in content.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": { "type": "string", "description": "The memory content to store" },
                    "memory_type": {
                        "type": "string",
                        "enum": ["decision", "pattern", "preference", "style", "habit", "insight", "context"],
                        "description": "Type of memory (default: context)"
                    },
                    "importance": { "type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.5 },
                    "tags": { "type": "array", "items": { "type": "string" } },
                    "namespace": { "type": "string", "description": "Namespace to scope the memory (e.g. project path)" },
                    "links": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "List of graph node IDs to link this memory to"
                    },
                    "auto_link": { "type": "boolean", "default": true, "description": "Auto-link to code nodes mentioned in content (default: true)" },
                    "expires_at": { "type": "string", "description": "ISO 8601 expiration timestamp (e.g. 2026-03-21T00:00:00Z)" },
                    "ttl_hours": { "type": "integer", "minimum": 1, "description": "Time-to-live in hours (alternative to expires_at)" },
                    "git_ref": { "type": "string", "description": "Git ref (branch/tag) to scope this memory to" }
                },
                "required": ["content"]
            }
        }),
        json!({
            "name": "recall",
            "description": "Unified memory search: 9-component hybrid scoring with optional graph expansion and impact analysis. Use expand=true for graph-expanded recall, include_impact=true for PageRank-enriched results.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Natural language search query" },
                    "k": { "type": "integer", "default": 10, "description": "Number of results" },
                    "memory_type": { "type": "string", "description": "Filter by memory type" },
                    "namespace": { "type": "string", "description": "Filter results to a specific namespace" },
                    "exclude_tags": { "type": "array", "items": { "type": "string" }, "description": "Exclude memories with any of these tags" },
                    "min_importance": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
                    "min_confidence": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
                    "expand": { "type": "boolean", "default": false, "description": "Enable graph expansion to discover related memories" },
                    "expansion_depth": { "type": "integer", "default": 1, "description": "Max graph hops for expansion (when expand=true)" },
                    "include_impact": { "type": "boolean", "default": false, "description": "Include PageRank, centrality, connected decisions, dependent files" },
                    "git_ref": { "type": "string", "description": "Filter results to memories with this git ref (branch/tag)" }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "delete_memory",
            "description": "Delete a memory by ID, removing from vector index, graph, and storage",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "string" }
                },
                "required": ["id"]
            }
        }),
        json!({
            "name": "associate_memories",
            "description": "Create a typed relationship between two nodes in the knowledge graph",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_id": { "type": "string" },
                    "target_id": { "type": "string" },
                    "relationship": {
                        "type": "string",
                        "enum": ["RELATES_TO","LEADS_TO","PART_OF","REINFORCES","CONTRADICTS",
                                 "EVOLVED_INTO","DERIVED_FROM","INVALIDATED_BY","DEPENDS_ON",
                                 "IMPORTS","EXTENDS","CALLS","CONTAINS","SUPERSEDES","BLOCKS",
                                 "IMPLEMENTS","INHERITS","SIMILAR_TO","PRECEDED_BY",
                                 "EXEMPLIFIES","EXPLAINS","SHARES_THEME","SUMMARIZES","CO_CHANGED"]
                    },
                    "weight": { "type": "number", "default": 1.0 }
                },
                "required": ["source_id", "target_id", "relationship"]
            }
        }),
        json!({
            "name": "refine_memory",
            "description": "Refine an existing memory. Default: creates a new version linked via EVOLVED_INTO. With destructive=true: updates in-place.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "ID of the memory to refine" },
                    "content": { "type": "string", "description": "Updated content (optional unless destructive=true)" },
                    "importance": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
                    "tags": { "type": "array", "items": { "type": "string" } },
                    "destructive": { "type": "boolean", "default": false, "description": "When true, update in-place instead of creating a new version" }
                },
                "required": ["id"]
            }
        }),
        json!({
            "name": "split_memory",
            "description": "Split a memory into multiple parts, each linked to the original via PART_OF edges",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "ID of the memory to split" },
                    "parts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": { "type": "string" },
                                "tags": { "type": "array", "items": { "type": "string" } },
                                "importance": { "type": "number", "minimum": 0.0, "maximum": 1.0 }
                            },
                            "required": ["content"]
                        }
                    }
                },
                "required": ["id", "parts"]
            }
        }),
        json!({
            "name": "merge_memories",
            "description": "Merge multiple memories into a single summary memory linked via SUMMARIZES edges",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_ids": { "type": "array", "items": { "type": "string" }, "minItems": 2 },
                    "content": { "type": "string", "description": "Content for the merged summary memory" },
                    "memory_type": { "type": "string", "enum": ["decision", "pattern", "preference", "style", "habit", "insight", "context"] },
                    "importance": { "type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.7 },
                    "tags": { "type": "array", "items": { "type": "string" } }
                },
                "required": ["source_ids", "content"]
            }
        }),
        // ── Graph & Structure (9 tools) ────────────────────────────────────
        json!({
            "name": "graph_traverse",
            "description": "Multi-hop graph traversal from a start node with optional filtering by node kind and relationship type",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "start_id": { "type": "string" },
                    "max_depth": { "type": "integer", "default": 2 },
                    "algorithm": { "type": "string", "enum": ["bfs", "dfs"], "default": "bfs" },
                    "exclude_kinds": { "type": "array", "items": { "type": "string" } },
                    "include_relationships": { "type": "array", "items": { "type": "string" } },
                    "at_time": { "type": "string", "description": "ISO 8601 timestamp — filter out nodes/edges not valid at this time" }
                },
                "required": ["start_id"]
            }
        }),
        json!({
            "name": "summary_tree",
            "description": "Hierarchical summary tree (packages -> files -> symbols)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "start_id": { "type": "string", "description": "Node ID to start from (e.g. 'pkg:src/')" },
                    "max_depth": { "type": "integer", "default": 3 },
                    "include_chunks": { "type": "boolean", "default": false }
                },
                "required": ["start_id"]
            }
        }),
        json!({
            "name": "codemem_status",
            "description": "Unified status: database stats, health check, and operational metrics. Use include=[\"stats\",\"health\",\"metrics\"] to select sections.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "include": {
                        "type": "array",
                        "items": { "type": "string", "enum": ["stats", "health", "metrics"] },
                        "description": "Sections to include (default: all)"
                    }
                }
            }
        }),
        json!({
            "name": "search_code",
            "description": "Search code by meaning or name. mode=semantic (vector search, default), mode=text (symbol name substring), mode=hybrid (both merged).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query (natural language for semantic, substring for text)" },
                    "k": { "type": "integer", "default": 10, "description": "Number of results" },
                    "mode": { "type": "string", "enum": ["semantic", "text", "hybrid"], "default": "semantic" },
                    "kind": { "type": "string", "enum": ["function", "method", "class", "struct", "enum", "interface", "trait", "type", "constant", "module", "test", "field", "constructor", "external", "enum_variant", "type_parameter", "macro", "property"], "description": "Filter by symbol kind (text/hybrid modes)" }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "get_symbol_info",
            "description": "Get full details of a symbol by qualified name. Optionally include graph dependencies.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "qualified_name": { "type": "string", "description": "Fully qualified name (e.g. 'module::Struct::method')" },
                    "include_dependencies": { "type": "boolean", "default": false, "description": "Include graph edges (calls, imports, etc.)" }
                },
                "required": ["qualified_name"]
            }
        }),
        json!({
            "name": "get_symbol_graph",
            "description": "Get symbol dependency graph. depth=1: direct edges (calls, imports). depth>1: full impact analysis (BFS reachability).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "qualified_name": { "type": "string" },
                    "depth": { "type": "integer", "default": 1, "description": "1=direct deps, >1=impact analysis" },
                    "direction": { "type": "string", "enum": ["incoming", "outgoing", "both"], "default": "both" }
                },
                "required": ["qualified_name"]
            }
        }),
        json!({
            "name": "find_important_nodes",
            "description": "Run PageRank to find the most important/central nodes in the knowledge graph",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "top_k": { "type": "integer", "default": 20 },
                    "damping": { "type": "number", "default": 0.85 }
                }
            }
        }),
        json!({
            "name": "find_related_groups",
            "description": "Run Louvain community detection to find clusters of related symbols",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "resolution": { "type": "number", "default": 1.0, "description": "Higher = more clusters" }
                }
            }
        }),
        json!({
            "name": "get_node_memories",
            "description": "Retrieve all memories connected to a graph node via BFS traversal",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "node_id": { "type": "string", "description": "Graph node ID (e.g. 'file:src/main.rs', 'sym:Module::func')" },
                    "max_depth": { "type": "integer", "default": 1, "description": "Max graph hops to search for memories" },
                    "include_relationships": { "type": "array", "items": { "type": "string" }, "description": "Only follow edges of these relationship types" }
                },
                "required": ["node_id"]
            }
        }),
        json!({
            "name": "node_coverage",
            "description": "Batch-check which graph nodes have attached memories. Returns memory count and coverage status for each node.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "node_ids": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Array of graph node IDs to check coverage for"
                    }
                },
                "required": ["node_ids"]
            }
        }),
        json!({
            "name": "get_cross_repo",
            "description": "Scan workspace manifests and report cross-package dependencies",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to scan" }
                }
            }
        }),
        // ── Consolidation & Patterns (3 tools) ─────────────────────────────
        json!({
            "name": "consolidate",
            "description": "Run memory consolidation. mode=auto runs all cycles. Individual modes: decay, creative, cluster, forget, summarize.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["auto", "decay", "creative", "cluster", "forget", "summarize"],
                        "default": "auto"
                    },
                    "threshold_days": { "type": "integer", "description": "For decay mode (default: 30)" },
                    "similarity_threshold": { "type": "number", "description": "For cluster mode (default: 0.92)" },
                    "importance_threshold": { "type": "number", "description": "For forget mode (default: 0.1)" },
                    "target_tags": { "type": "array", "items": { "type": "string" }, "description": "For forget mode" },
                    "max_access_count": { "type": "integer", "description": "For forget mode" },
                    "cluster_size": { "type": "integer", "description": "For summarize mode (default: 5)" }
                }
            }
        }),
        json!({
            "name": "detect_patterns",
            "description": "Detect cross-session patterns. format=json (default), format=markdown (human-readable), format=both.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "min_frequency": { "type": "integer", "minimum": 1, "default": 3 },
                    "namespace": { "type": "string" },
                    "format": { "type": "string", "enum": ["json", "markdown", "both"], "default": "json" }
                }
            }
        }),
        json!({
            "name": "get_decision_chain",
            "description": "Follow decision evolution through the knowledge graph via EVOLVED_INTO/LEADS_TO/DERIVED_FROM edges",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "file_path": { "type": "string" },
                    "topic": { "type": "string" }
                }
            }
        }),
        // ── Namespace Management (3 tools) ──────────────────────────────────
        json!({
            "name": "list_namespaces",
            "description": "List all namespaces with inline stats (counts, avg importance, type distribution, date range)",
            "inputSchema": { "type": "object", "properties": {} }
        }),
        json!({
            "name": "namespace_stats",
            "description": "Detailed statistics for a specific namespace",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "namespace": { "type": "string" }
                },
                "required": ["namespace"]
            }
        }),
        json!({
            "name": "delete_namespace",
            "description": "Delete all memories in a namespace (requires confirm=true)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "namespace": { "type": "string" },
                    "confirm": { "type": "boolean" }
                },
                "required": ["namespace", "confirm"]
            }
        }),
        // ── Session & Context (2 tools) ─────────────────────────────────────
        json!({
            "name": "session_checkpoint",
            "description": "Mid-session progress report with activity summary, pattern detection, and focus areas",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session_id": { "type": "string" },
                    "namespace": { "type": "string" }
                },
                "required": ["session_id"]
            }
        }),
        json!({
            "name": "session_context",
            "description": "Get session context: recent memories, pending analyses, active patterns, and focus areas",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "namespace": { "type": "string" },
                    "k": { "type": "integer", "default": 10, "description": "Number of recent memories" }
                }
            }
        }),
        json!({
            "name": "review_diff",
            "description": "Analyze a unified diff for blast radius: map changed lines to symbols, find direct and transitive dependents, compute risk score, surface relevant memories and potentially missing changes.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "diff": { "type": "string", "description": "Unified diff text (e.g., output of `git diff`)" },
                    "depth": { "type": "integer", "default": 2, "description": "Max graph hops for transitive impact analysis" },
                    "base_ref": { "type": "string", "description": "Base branch for overlay resolution (e.g., 'main')" }
                },
                "required": ["diff"]
            }
        }),
        // ── Temporal Queries (5 tools) ───────────────────────────────────
        json!({
            "name": "what_changed",
            "description": "List commits and their affected files/symbols in a time range. Requires temporal ingestion to have been run.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "from": { "type": "string", "description": "Start of time range (ISO 8601, e.g. '2026-01-01T00:00:00Z')" },
                    "to": { "type": "string", "description": "End of time range (ISO 8601, e.g. '2026-03-17T00:00:00Z')" },
                    "namespace": { "type": "string", "description": "Filter by namespace" }
                },
                "required": ["from", "to"]
            }
        }),
        json!({
            "name": "graph_at_time",
            "description": "Snapshot of the graph at a point in time: count of live nodes/edges, broken down by kind. Nodes/edges with valid_to before the timestamp are excluded.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "at": { "type": "string", "description": "Point in time (ISO 8601, e.g. '2026-02-15T00:00:00Z')" }
                },
                "required": ["at"]
            }
        }),
        json!({
            "name": "find_stale_files",
            "description": "Find files with high centrality or incoming edges that haven't been modified recently. Requires temporal ingestion.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "namespace": { "type": "string", "description": "Filter by namespace" },
                    "stale_days": { "type": "integer", "default": 90, "description": "Files not modified in this many days are stale" },
                    "limit": { "type": "integer", "default": 20, "description": "Max results to return" }
                }
            }
        }),
        json!({
            "name": "detect_drift",
            "description": "Detect architectural drift between two time periods: new cross-module edges, hotspot files, coupling increases, added/removed files.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "from": { "type": "string", "description": "Start of period (ISO 8601)" },
                    "to": { "type": "string", "description": "End of period (ISO 8601)" },
                    "namespace": { "type": "string", "description": "Filter by namespace" }
                },
                "required": ["from", "to"]
            }
        }),
        json!({
            "name": "symbol_history",
            "description": "Get the commit history for a specific symbol or file node. Returns commits that modified it, with affected files and symbols.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "node_id": { "type": "string", "description": "Graph node ID (e.g. 'sym:MyClass::method' or 'file:src/main.rs')" }
                },
                "required": ["node_id"]
            }
        }),
        json!({
            "name": "test_impact",
            "description": "Find tests affected by changes to symbols. BFS callers up to depth 4, splits into direct and transitive.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "symbols": { "type": "array", "items": { "type": "string" }, "description": "Symbol IDs to analyze" },
                    "max_depth": { "type": "integer", "description": "Max BFS depth (default 4)" }
                },
                "required": ["symbols"]
            }
        }),
    ]
}
