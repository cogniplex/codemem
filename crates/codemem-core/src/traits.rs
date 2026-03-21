use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{
    CodememError, Edge, GraphNode, MemoryNode, NodeKind, RawGraphMetrics, RelationshipType,
    Repository, Session, UnresolvedRefData,
};

// ── Data types for trait return values ───────────────────────────────────────

/// A pending unresolved reference stored for deferred cross-namespace linking.
#[derive(Debug, Clone)]
pub struct PendingUnresolvedRef {
    /// ID of the unresolved ref record.
    pub id: String,
    /// Source symbol qualified name.
    pub source_node: String,
    /// The unresolved target name.
    pub target_name: String,
    /// Namespace the source belongs to.
    pub namespace: String,
    /// File path where the reference occurs.
    pub file_path: String,
    /// Line number.
    pub line: usize,
    /// Kind of reference: "call", "import", "inherits", etc.
    pub ref_kind: String,
    /// Package hint extracted from import context.
    pub package_hint: Option<String>,
}

// ── Traits ──────────────────────────────────────────────────────────────────

/// Vector backend trait for HNSW index operations.
pub trait VectorBackend: Send + Sync {
    /// Insert a vector with associated ID.
    fn insert(&mut self, id: &str, embedding: &[f32]) -> Result<(), CodememError>;

    /// Batch insert vectors.
    fn insert_batch(&mut self, items: &[(String, Vec<f32>)]) -> Result<(), CodememError>;

    /// Search for k nearest neighbors. Returns (id, distance) pairs.
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>, CodememError>;

    /// Remove a vector by ID.
    fn remove(&mut self, id: &str) -> Result<bool, CodememError>;

    /// Save the index to disk.
    fn save(&self, path: &std::path::Path) -> Result<(), CodememError>;

    /// Load the index from disk.
    fn load(&mut self, path: &std::path::Path) -> Result<(), CodememError>;

    /// Get index statistics.
    fn stats(&self) -> VectorStats;

    /// Whether the index has accumulated enough ghost entries to warrant compaction.
    fn needs_compaction(&self) -> bool {
        false
    }

    /// Number of ghost entries left by removals.
    fn ghost_count(&self) -> usize {
        0
    }

    /// Rebuild the index from scratch given all current entries.
    fn rebuild_from_entries(
        &mut self,
        _entries: &[(String, Vec<f32>)],
    ) -> Result<(), CodememError> {
        Ok(())
    }
}

/// Statistics about the vector index.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VectorStats {
    pub count: usize,
    pub dimensions: usize,
    pub metric: String,
    pub memory_bytes: usize,
}

/// Graph backend trait for graph operations.
///
/// # Default implementations
///
/// Extended methods (centrality, community detection, etc.) have no-op defaults
/// so that the core CRUD + traversal methods are sufficient for a minimal
/// implementation. Backend authors should override the algorithmic methods
/// relevant to their storage engine. The in-memory `GraphEngine` overrides all
/// of them.
///
/// # `_ref` methods
///
/// `get_node_ref` and `get_edges_ref` return borrowed references into the
/// backend's in-memory storage. They exist for zero-copy hot-path performance
/// in the default `GraphEngine`. Database-backed implementations cannot return
/// references to internal state and should leave the defaults (which return
/// `None` / empty). Callers that need database compatibility should use the
/// owned variants `get_node` / `get_edges` instead.
pub trait GraphBackend: Send + Sync {
    /// Add a node to the graph.
    fn add_node(&mut self, node: GraphNode) -> Result<(), CodememError>;

    /// Get a node by ID.
    fn get_node(&self, id: &str) -> Result<Option<GraphNode>, CodememError>;

    /// Remove a node by ID.
    fn remove_node(&mut self, id: &str) -> Result<bool, CodememError>;

    /// Add an edge between two nodes.
    fn add_edge(&mut self, edge: Edge) -> Result<(), CodememError>;

    /// Get edges from a node.
    fn get_edges(&self, node_id: &str) -> Result<Vec<Edge>, CodememError>;

    /// Remove an edge by ID.
    fn remove_edge(&mut self, id: &str) -> Result<bool, CodememError>;

    /// BFS traversal from a start node up to max_depth.
    fn bfs(&self, start_id: &str, max_depth: usize) -> Result<Vec<GraphNode>, CodememError>;

    /// DFS traversal from a start node up to max_depth.
    fn dfs(&self, start_id: &str, max_depth: usize) -> Result<Vec<GraphNode>, CodememError>;

    /// BFS traversal with filtering: exclude certain node kinds and optionally
    /// restrict to specific relationship types.
    fn bfs_filtered(
        &self,
        start_id: &str,
        max_depth: usize,
        exclude_kinds: &[NodeKind],
        include_relationships: Option<&[RelationshipType]>,
    ) -> Result<Vec<GraphNode>, CodememError>;

    /// DFS traversal with filtering: exclude certain node kinds and optionally
    /// restrict to specific relationship types.
    fn dfs_filtered(
        &self,
        start_id: &str,
        max_depth: usize,
        exclude_kinds: &[NodeKind],
        include_relationships: Option<&[RelationshipType]>,
    ) -> Result<Vec<GraphNode>, CodememError>;

    /// Shortest path between two nodes.
    fn shortest_path(&self, from: &str, to: &str) -> Result<Vec<String>, CodememError>;

    /// Get graph statistics.
    fn stats(&self) -> GraphStats;

    // ── Extended methods (with defaults for backwards compatibility) ──

    /// Get all nodes in the graph.
    fn get_all_nodes(&self) -> Vec<GraphNode> {
        Vec::new()
    }

    /// Zero-copy node lookup. Returns a reference into the backend's internal storage.
    /// Only meaningful for in-memory backends. Database backends should leave the
    /// default (returns `None`) and callers should use `get_node()` instead.
    fn get_node_ref(&self, _id: &str) -> Option<&GraphNode> {
        None
    }

    /// Zero-copy edge lookup. Returns references into the backend's internal storage.
    /// Only meaningful for in-memory backends. Database backends should leave the
    /// default (returns empty) and callers should use `get_edges()` instead.
    fn get_edges_ref(&self, _node_id: &str) -> Vec<&Edge> {
        Vec::new()
    }

    /// Number of nodes in the graph.
    fn node_count(&self) -> usize {
        self.stats().node_count
    }

    /// Number of edges in the graph.
    fn edge_count(&self) -> usize {
        self.stats().edge_count
    }

    /// Recompute all centrality metrics (PageRank + betweenness).
    fn recompute_centrality(&mut self) {}

    /// Recompute centrality, optionally including expensive betweenness calculation.
    fn recompute_centrality_with_options(&mut self, _include_betweenness: bool) {}

    /// Recompute PageRank scoped to a single namespace, updating only that
    /// namespace's scores in the cache. Nodes from other namespaces are unaffected.
    fn recompute_centrality_for_namespace(&mut self, _namespace: &str) {}

    /// Lazily compute betweenness centrality if not yet computed.
    fn ensure_betweenness_computed(&mut self) {}

    /// Compute degree centrality (updates nodes in place).
    fn compute_centrality(&mut self) {}

    /// Get cached PageRank score for a node.
    fn get_pagerank(&self, _node_id: &str) -> f64 {
        0.0
    }

    /// Get cached betweenness centrality score for a node.
    fn get_betweenness(&self, _node_id: &str) -> f64 {
        0.0
    }

    /// Collect graph metrics for a memory node (used in hybrid scoring).
    fn raw_graph_metrics_for_memory(&self, _memory_id: &str) -> Option<RawGraphMetrics> {
        None
    }

    /// Find connected components (treating graph as undirected).
    fn connected_components(&self) -> Vec<Vec<String>> {
        Vec::new()
    }

    /// Compute PageRank scores for all nodes.
    /// Returns a map from node ID to PageRank score.
    fn pagerank(&self, _damping: f64, _iterations: usize, _tolerance: f64) -> HashMap<String, f64> {
        HashMap::new()
    }

    /// Compute PageRank scores for nodes in a single namespace.
    /// Only nodes belonging to `namespace` participate; cross-namespace edges are ignored.
    /// Returns a map from node ID to PageRank score (only for nodes in the namespace).
    /// Default is abstract; implementers must provide namespace-scoped computation.
    fn pagerank_for_namespace(
        &self,
        _namespace: &str,
        _damping: f64,
        _iterations: usize,
        _tolerance: f64,
    ) -> HashMap<String, f64> {
        panic!("pagerank_for_namespace must be implemented; default fallback violates isolation guarantee");
    }

    /// Run Louvain community detection at the given resolution.
    /// Returns groups of node IDs, one group per community.
    fn louvain_communities(&self, _resolution: f64) -> Vec<Vec<String>> {
        Vec::new()
    }

    /// Compute topological layers of the graph.
    /// Returns layers where all nodes in layer i have no dependencies on nodes
    /// in layer i or later.
    fn topological_layers(&self) -> Vec<Vec<String>> {
        Vec::new()
    }

    /// Return node-to-community-ID mapping for Louvain.
    fn louvain_with_assignment(&self, resolution: f64) -> HashMap<String, usize> {
        let communities = self.louvain_communities(resolution);
        let mut assignment = HashMap::new();
        for (idx, community) in communities.into_iter().enumerate() {
            for node_id in community {
                assignment.insert(node_id, idx);
            }
        }
        assignment
    }

    /// Return a top-N subgraph: the highest-centrality nodes plus all edges between them.
    /// Non-structural edges from top-N nodes pull their targets into the result.
    fn subgraph_top_n(
        &self,
        _n: usize,
        _namespace: Option<&str>,
        _kinds: Option<&[NodeKind]>,
    ) -> (Vec<GraphNode>, Vec<Edge>) {
        (Vec::new(), Vec::new())
    }
}

/// Statistics about the graph.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub node_kind_counts: HashMap<String, usize>,
    pub relationship_type_counts: HashMap<String, usize>,
}

// ── Embedding Provider Trait ────────────────────────────────────────────────

/// Trait for pluggable embedding providers.
pub trait EmbeddingProvider: Send + Sync {
    /// Embedding vector dimensions.
    fn dimensions(&self) -> usize;

    /// Embed a single text string.
    fn embed(&self, text: &str) -> Result<Vec<f32>, crate::CodememError>;

    /// Embed a batch of texts (default: sequential).
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, crate::CodememError> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Provider name for display.
    fn name(&self) -> &str;

    /// Cache statistics: (current_size, capacity). Returns (0, 0) if no cache.
    fn cache_stats(&self) -> (usize, usize) {
        (0, 0)
    }
}

// ── Storage Stats & Consolidation Types ─────────────────────────────────

/// Database statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    pub memory_count: usize,
    pub embedding_count: usize,
    pub node_count: usize,
    pub edge_count: usize,
}

/// A single consolidation log entry.
#[derive(Debug, Clone)]
pub struct ConsolidationLogEntry {
    pub cycle_type: String,
    pub run_at: i64,
    pub affected_count: usize,
}

// ── Storage Backend Trait ───────────────────────────────────────────────

/// Pluggable storage backend trait for all persistence operations.
///
/// This trait unifies every persistence concern behind a single interface so
/// that the engine layer (`CodememEngine`) remains backend-agnostic.
///
/// # Method groups
///
/// | Group | Methods | Purpose |
/// |-------|---------|---------|
/// | **Memory CRUD** | `insert_memory`, `get_memory`, `update_memory`, `delete_memory`, `list_memory_ids`, … | Create, read, update, delete memory nodes |
/// | **Embedding persistence** | `store_embedding`, `get_embedding`, `delete_embedding`, `list_all_embeddings` | Persist and retrieve embedding vectors |
/// | **Graph node/edge storage** | `insert_graph_node`, `get_graph_node`, `all_graph_nodes`, `insert_graph_edge`, … | Persist the knowledge graph structure |
/// | **Sessions** | `start_session`, `end_session`, `list_sessions`, `session_count` | Track interaction sessions |
/// | **Consolidation** | `insert_consolidation_log`, `last_consolidation_runs` | Record and query memory consolidation runs |
/// | **Pattern detection** | `get_repeated_searches`, `get_file_hotspots`, `get_tool_usage_stats`, `get_decision_chains` | Cross-session pattern queries |
/// | **Bulk/batch operations** | `insert_memories_batch`, `store_embeddings_batch`, `insert_graph_nodes_batch`, `insert_graph_edges_batch` | Efficient multi-row inserts |
/// | **Decay & forgetting** | `decay_stale_memories`, `find_forgettable`, `get_stale_memories_for_decay`, `batch_update_importance` | Power-law decay and garbage collection |
/// | **Query helpers** | `find_unembedded_memories`, `search_graph_nodes`, `list_memories_filtered`, `find_hash_duplicates` | Filtered searches and dedup |
/// | **File hash tracking** | `load_file_hashes`, `save_file_hashes` | Incremental indexing support |
/// | **Session activity** | `record_session_activity`, `get_session_activity_summary`, `get_session_hot_directories`, … | Fine-grained activity tracking |
/// | **Stats** | `stats` | Database-level statistics |
///
/// Implementations include SQLite (default) and can be extended for
/// SurrealDB, FalkorDB, or other backends.
pub trait StorageBackend: Send + Sync {
    // ── Memory CRUD ─────────────────────────────────────────────────

    /// Insert a new memory. Returns Err(Duplicate) if content hash already exists.
    fn insert_memory(&self, memory: &MemoryNode) -> Result<(), CodememError>;

    /// Get a memory by ID. Updates access_count and last_accessed_at.
    fn get_memory(&self, id: &str) -> Result<Option<MemoryNode>, CodememError>;

    /// Get a memory by ID without updating access_count or last_accessed_at.
    /// Use this for internal/system reads (consolidation checks, stats, batch processing).
    fn get_memory_no_touch(&self, id: &str) -> Result<Option<MemoryNode>, CodememError> {
        // Default: falls back to get_memory for backwards compatibility.
        self.get_memory(id)
    }

    /// Get multiple memories by IDs in a single batch operation.
    fn get_memories_batch(&self, ids: &[&str]) -> Result<Vec<MemoryNode>, CodememError>;

    /// Update a memory's content and optionally its importance. Re-computes content hash.
    fn update_memory(
        &self,
        id: &str,
        content: &str,
        importance: Option<f64>,
    ) -> Result<(), CodememError>;

    /// Delete a memory by ID. Returns true if a row was deleted.
    fn delete_memory(&self, id: &str) -> Result<bool, CodememError>;

    /// Delete a memory and all related data (graph nodes/edges, embeddings) atomically.
    /// Returns true if the memory existed and was deleted.
    /// Default falls back to individual deletes (non-transactional) for backwards compatibility.
    fn delete_memory_cascade(&self, id: &str) -> Result<bool, CodememError> {
        let deleted = self.delete_memory(id)?;
        if deleted {
            let _ = self.delete_graph_edges_for_node(id);
            let _ = self.delete_graph_node(id);
            let _ = self.delete_embedding(id);
        }
        Ok(deleted)
    }

    /// Delete multiple memories and all related data (graph nodes/edges, embeddings) atomically.
    /// Returns the number of memories that were actually deleted.
    /// Default falls back to calling `delete_memory_cascade` per ID for backwards compatibility.
    fn delete_memories_batch_cascade(&self, ids: &[&str]) -> Result<usize, CodememError> {
        let mut count = 0;
        for id in ids {
            if self.delete_memory_cascade(id)? {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Delete all memories whose `expires_at` timestamp is in the past.
    /// Returns the number of memories deleted.
    fn delete_expired_memories(&self) -> Result<usize, CodememError>;

    /// Expire (set `expires_at` to now) all `static-analysis` memories
    /// linked to symbols in the given file path.
    fn expire_memories_for_file(&self, file_path: &str) -> Result<usize, CodememError>;

    /// List all memory IDs, ordered by created_at descending.
    fn list_memory_ids(&self) -> Result<Vec<String>, CodememError>;

    /// List memory IDs scoped to a specific namespace.
    fn list_memory_ids_for_namespace(&self, namespace: &str) -> Result<Vec<String>, CodememError>;

    /// Find memory IDs whose tags contain the given tag value.
    /// Optionally scoped to a namespace. Excludes `exclude_id`.
    fn find_memory_ids_by_tag(
        &self,
        tag: &str,
        namespace: Option<&str>,
        exclude_id: &str,
    ) -> Result<Vec<String>, CodememError>;

    /// List all distinct namespaces.
    fn list_namespaces(&self) -> Result<Vec<String>, CodememError>;

    /// Get total memory count.
    fn memory_count(&self) -> Result<usize, CodememError>;

    // ── Embedding Persistence ───────────────────────────────────────

    /// Store an embedding vector for a memory.
    fn store_embedding(&self, memory_id: &str, embedding: &[f32]) -> Result<(), CodememError>;

    /// Get an embedding by memory ID.
    fn get_embedding(&self, memory_id: &str) -> Result<Option<Vec<f32>>, CodememError>;

    /// Delete an embedding by memory ID. Returns true if a row was deleted.
    fn delete_embedding(&self, memory_id: &str) -> Result<bool, CodememError>;

    /// List all stored embeddings as (memory_id, embedding_vector) pairs.
    fn list_all_embeddings(&self) -> Result<Vec<(String, Vec<f32>)>, CodememError>;

    // ── Graph Node/Edge Persistence ─────────────────────────────────

    /// Insert or replace a graph node.
    fn insert_graph_node(&self, node: &GraphNode) -> Result<(), CodememError>;

    /// Get a graph node by ID.
    fn get_graph_node(&self, id: &str) -> Result<Option<GraphNode>, CodememError>;

    /// Delete a graph node by ID. Returns true if a row was deleted.
    fn delete_graph_node(&self, id: &str) -> Result<bool, CodememError>;

    /// Get all graph nodes.
    fn all_graph_nodes(&self) -> Result<Vec<GraphNode>, CodememError>;

    /// Insert or replace a graph edge.
    fn insert_graph_edge(&self, edge: &Edge) -> Result<(), CodememError>;

    /// Get all edges from or to a node.
    fn get_edges_for_node(&self, node_id: &str) -> Result<Vec<Edge>, CodememError>;

    /// Get all graph edges.
    fn all_graph_edges(&self) -> Result<Vec<Edge>, CodememError>;

    /// Delete a single graph edge by ID. Returns true if a row was deleted.
    fn delete_graph_edge(&self, edge_id: &str) -> Result<bool, CodememError> {
        // Default: fall back to querying all edges and deleting via for_node.
        // Backends should override with a direct DELETE WHERE id = ?1.
        let _ = edge_id;
        Ok(false)
    }

    /// Delete all graph edges connected to a node. Returns count deleted.
    fn delete_graph_edges_for_node(&self, node_id: &str) -> Result<usize, CodememError>;

    /// Delete all graph nodes, edges, and embeddings whose node ID starts with the given prefix.
    /// Returns count of nodes deleted.
    fn delete_graph_nodes_by_prefix(&self, prefix: &str) -> Result<usize, CodememError>;

    // ── Sessions ────────────────────────────────────────────────────

    /// Start a new session.
    fn start_session(&self, id: &str, namespace: Option<&str>) -> Result<(), CodememError>;

    /// End a session with optional summary.
    fn end_session(&self, id: &str, summary: Option<&str>) -> Result<(), CodememError>;

    /// List sessions, optionally filtered by namespace, up to limit.
    fn list_sessions(
        &self,
        namespace: Option<&str>,
        limit: usize,
    ) -> Result<Vec<Session>, CodememError>;

    // ── Consolidation ───────────────────────────────────────────────

    /// Record a consolidation run.
    fn insert_consolidation_log(
        &self,
        cycle_type: &str,
        affected_count: usize,
    ) -> Result<(), CodememError>;

    /// Get the last consolidation run for each cycle type.
    fn last_consolidation_runs(&self) -> Result<Vec<ConsolidationLogEntry>, CodememError>;

    // ── Pattern Detection Queries ───────────────────────────────────

    /// Find repeated search patterns. Returns (pattern, count, memory_ids).
    fn get_repeated_searches(
        &self,
        min_count: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<(String, usize, Vec<String>)>, CodememError>;

    /// Find file hotspots. Returns (file_path, count, memory_ids).
    fn get_file_hotspots(
        &self,
        min_count: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<(String, usize, Vec<String>)>, CodememError>;

    /// Get tool usage statistics. Returns (tool_name, count) pairs.
    fn get_tool_usage_stats(
        &self,
        namespace: Option<&str>,
    ) -> Result<Vec<(String, usize)>, CodememError>;

    /// Find decision chains. Returns (file_path, count, memory_ids).
    fn get_decision_chains(
        &self,
        min_count: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<(String, usize, Vec<String>)>, CodememError>;

    // ── Bulk Operations ─────────────────────────────────────────────

    /// Decay importance of stale memories older than threshold_ts by decay_factor.
    /// Returns count of affected memories.
    fn decay_stale_memories(
        &self,
        threshold_ts: i64,
        decay_factor: f64,
    ) -> Result<usize, CodememError>;

    /// List memories for creative consolidation: (id, memory_type, tags).
    fn list_memories_for_creative(
        &self,
    ) -> Result<Vec<(String, String, Vec<String>)>, CodememError>;

    /// Find near-duplicate memories by content hash prefix matching.
    /// Returns (id1, id2, similarity) pairs. Only catches exact content matches
    /// (hash prefix), not semantic near-duplicates.
    fn find_hash_duplicates(&self) -> Result<Vec<(String, String, f64)>, CodememError>;

    /// Find memories eligible for forgetting (low importance).
    /// Returns list of memory IDs.
    fn find_forgettable(&self, importance_threshold: f64) -> Result<Vec<String>, CodememError>;

    // ── Batch Operations ────────────────────────────────────────────

    /// Insert multiple memories in a single batch. Default impl calls insert_memory in a loop.
    fn insert_memories_batch(&self, memories: &[MemoryNode]) -> Result<(), CodememError> {
        for memory in memories {
            self.insert_memory(memory)?;
        }
        Ok(())
    }

    /// Store multiple embeddings in a single batch. Default impl calls store_embedding in a loop.
    fn store_embeddings_batch(&self, items: &[(&str, &[f32])]) -> Result<(), CodememError> {
        for (id, embedding) in items {
            self.store_embedding(id, embedding)?;
        }
        Ok(())
    }

    /// Insert multiple graph nodes in a single batch. Default impl calls insert_graph_node in a loop.
    fn insert_graph_nodes_batch(&self, nodes: &[GraphNode]) -> Result<(), CodememError> {
        for node in nodes {
            self.insert_graph_node(node)?;
        }
        Ok(())
    }

    /// Insert multiple graph edges in a single batch. Default impl calls insert_graph_edge in a loop.
    fn insert_graph_edges_batch(&self, edges: &[Edge]) -> Result<(), CodememError> {
        for edge in edges {
            self.insert_graph_edge(edge)?;
        }
        Ok(())
    }

    // ── Query Helpers ───────────────────────────────────────────────

    /// Find memories that have no embeddings yet. Returns (id, content) pairs.
    fn find_unembedded_memories(&self) -> Result<Vec<(String, String)>, CodememError>;

    /// Search graph nodes by label (case-insensitive LIKE). Returns matching nodes
    /// sorted by centrality descending, limited to `limit` results.
    fn search_graph_nodes(
        &self,
        query: &str,
        namespace: Option<&str>,
        limit: usize,
    ) -> Result<Vec<GraphNode>, CodememError>;

    /// List memories matching a specific tag, with optional namespace filter.
    fn list_memories_by_tag(
        &self,
        tag: &str,
        namespace: Option<&str>,
        limit: usize,
    ) -> Result<Vec<MemoryNode>, CodememError>;

    /// List memories with optional namespace and memory_type filters.
    fn list_memories_filtered(
        &self,
        namespace: Option<&str>,
        memory_type: Option<&str>,
    ) -> Result<Vec<MemoryNode>, CodememError>;

    /// Fetch stale memories with access metadata for power-law decay.
    /// Returns (id, importance, access_count, last_accessed_at).
    fn get_stale_memories_for_decay(
        &self,
        threshold_ts: i64,
    ) -> Result<Vec<(String, f64, u32, i64)>, CodememError>;

    /// Batch-update importance values. Returns count of updated rows.
    fn batch_update_importance(&self, updates: &[(String, f64)]) -> Result<usize, CodememError>;

    /// Total session count, optionally filtered by namespace.
    fn session_count(&self, namespace: Option<&str>) -> Result<usize, CodememError>;

    // ── File Hash Tracking ──────────────────────────────────────────

    /// Load file hashes for incremental indexing, scoped to a namespace.
    /// Returns path -> hash map.
    fn load_file_hashes(&self, namespace: &str) -> Result<HashMap<String, String>, CodememError>;

    /// Save file hashes for incremental indexing, scoped to a namespace.
    fn save_file_hashes(
        &self,
        namespace: &str,
        hashes: &HashMap<String, String>,
    ) -> Result<(), CodememError>;

    // ── Session Activity Tracking ─────────────────────────────────

    /// Record a session activity event (tool use with context).
    fn record_session_activity(
        &self,
        session_id: &str,
        tool_name: &str,
        file_path: Option<&str>,
        directory: Option<&str>,
        pattern: Option<&str>,
    ) -> Result<(), CodememError>;

    /// Get a summary of session activity counts.
    fn get_session_activity_summary(
        &self,
        session_id: &str,
    ) -> Result<crate::SessionActivitySummary, CodememError>;

    /// Get the most active directories in a session. Returns (directory, count) pairs.
    fn get_session_hot_directories(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<(String, usize)>, CodememError>;

    /// Check whether a particular auto-insight dedup tag already exists for a session.
    fn has_auto_insight(&self, session_id: &str, dedup_tag: &str) -> Result<bool, CodememError>;

    /// Count how many Read events occurred in a directory during a session.
    fn count_directory_reads(
        &self,
        session_id: &str,
        directory: &str,
    ) -> Result<usize, CodememError>;

    /// Check if a file was read in the current session.
    fn was_file_read_in_session(
        &self,
        session_id: &str,
        file_path: &str,
    ) -> Result<bool, CodememError>;

    /// Count how many times a search pattern was used in a session.
    fn count_search_pattern_in_session(
        &self,
        session_id: &str,
        pattern: &str,
    ) -> Result<usize, CodememError>;

    // ── Repository Management ────────────────────────────────────────

    /// List all registered repositories.
    fn list_repos(&self) -> Result<Vec<Repository>, CodememError>;

    /// Add a new repository.
    fn add_repo(&self, repo: &Repository) -> Result<(), CodememError>;

    /// Get a repository by ID.
    fn get_repo(&self, id: &str) -> Result<Option<Repository>, CodememError>;

    /// Remove a repository by ID. Returns true if it existed.
    fn remove_repo(&self, id: &str) -> Result<bool, CodememError>;

    /// Update a repository's status and optionally its last-indexed timestamp.
    fn update_repo_status(
        &self,
        id: &str,
        status: &str,
        indexed_at: Option<&str>,
    ) -> Result<(), CodememError>;

    // ── Stats ───────────────────────────────────────────────────────

    /// Get database statistics.
    fn stats(&self) -> Result<StorageStats, CodememError>;

    // ── Transaction Control ────────────────────────────────────────

    /// Begin an explicit transaction.
    ///
    /// While a transaction is active, individual storage methods (e.g.
    /// `insert_memory`, `insert_graph_node`) participate in it instead of
    /// starting their own. Call `commit_transaction` to persist or
    /// `rollback_transaction` to discard.
    ///
    /// Default implementation is a no-op for backends that don't support
    /// explicit transaction control.
    fn begin_transaction(&self) -> Result<(), CodememError> {
        Ok(())
    }

    /// Commit the active transaction started by `begin_transaction`.
    ///
    /// Default implementation is a no-op.
    fn commit_transaction(&self) -> Result<(), CodememError> {
        Ok(())
    }

    /// Roll back the active transaction started by `begin_transaction`.
    ///
    /// Default implementation is a no-op.
    fn rollback_transaction(&self) -> Result<(), CodememError> {
        Ok(())
    }

    // ── Cross-Repo Persistence ────────────────────────────────────────

    /// Get graph edges where either the source or destination node belongs to the
    /// given namespace (cross-namespace query). When `include_cross_namespace` is false,
    /// this behaves like `graph_edges_for_namespace` (both endpoints in namespace).
    fn graph_edges_for_namespace_with_cross(
        &self,
        _namespace: &str,
        _include_cross_namespace: bool,
    ) -> Result<Vec<Edge>, CodememError> {
        Ok(Vec::new())
    }

    /// Upsert a package into the cross-repo package registry.
    fn upsert_package_registry(
        &self,
        _package_name: &str,
        _namespace: &str,
        _version: &str,
        _manifest: &str,
    ) -> Result<(), CodememError> {
        Ok(())
    }

    /// Store an unresolved reference for future cross-namespace linking.
    #[allow(clippy::too_many_arguments)]
    fn store_unresolved_ref(
        &self,
        _source_qualified_name: &str,
        _target_name: &str,
        _source_namespace: &str,
        _file_path: &str,
        _line: usize,
        _ref_kind: &str,
        _package_hint: Option<&str>,
    ) -> Result<(), CodememError> {
        Ok(())
    }

    /// Batch store unresolved references. Default falls back to per-ref calls.
    fn store_unresolved_refs_batch(
        &self,
        refs: &[UnresolvedRefData],
    ) -> Result<usize, CodememError> {
        let mut count = 0;
        for r in refs {
            self.store_unresolved_ref(
                &r.source_qualified_name,
                &r.target_name,
                &r.namespace,
                &r.file_path,
                r.line,
                &r.ref_kind,
                r.package_hint.as_deref(),
            )?;
            count += 1;
        }
        Ok(count)
    }

    /// List all registered packages. Returns (name, namespace, manifest_path) tuples.
    fn list_registered_packages(&self) -> Result<Vec<(String, String, String)>, CodememError> {
        Ok(Vec::new())
    }

    /// List all pending unresolved refs with full context.
    fn list_pending_unresolved_refs(&self) -> Result<Vec<PendingUnresolvedRef>, CodememError> {
        Ok(Vec::new())
    }

    /// Delete a resolved unresolved ref by ID.
    fn delete_unresolved_ref(&self, _id: &str) -> Result<(), CodememError> {
        Ok(())
    }

    /// Count unresolved refs for a given namespace.
    fn count_unresolved_refs(&self, _namespace: &str) -> Result<usize, CodememError> {
        Ok(0)
    }

    /// List registered packages for a given namespace.
    /// Returns (name, namespace, manifest_path) tuples.
    fn list_registered_packages_for_namespace(
        &self,
        _namespace: &str,
    ) -> Result<Vec<(String, String, String)>, CodememError> {
        Ok(Vec::new())
    }

    /// Store a detected API endpoint.
    fn store_api_endpoint(
        &self,
        _method: &str,
        _path: &str,
        _handler_symbol: &str,
        _namespace: &str,
    ) -> Result<(), CodememError> {
        Ok(())
    }

    /// Store a detected client call.
    fn store_api_client_call(
        &self,
        _library: &str,
        _method: Option<&str>,
        _caller_symbol: &str,
        _namespace: &str,
    ) -> Result<(), CodememError> {
        Ok(())
    }

    /// List detected API endpoints for a namespace.
    /// Returns (method, path, handler_symbol, namespace) tuples.
    fn list_api_endpoints(
        &self,
        _namespace: &str,
    ) -> Result<Vec<(String, String, String, String)>, CodememError> {
        Ok(Vec::new())
    }

    /// Store a detected event channel.
    fn store_event_channel(
        &self,
        channel: &str,
        direction: &str,
        protocol: &str,
        handler: &str,
        namespace: &str,
        description: &str,
    ) -> Result<(), CodememError> {
        let _ = (
            channel,
            direction,
            protocol,
            handler,
            namespace,
            description,
        );
        Ok(())
    }

    /// List event channels for a namespace.
    /// Returns (channel, direction, protocol, handler, description) tuples.
    #[allow(clippy::type_complexity)]
    fn list_event_channels(
        &self,
        namespace: &str,
    ) -> Result<Vec<(String, String, String, String, String)>, CodememError> {
        let _ = namespace;
        Ok(Vec::new())
    }

    /// List all event channels across all namespaces.
    /// Returns (channel, direction, protocol, handler, namespace, description) tuples.
    #[allow(clippy::type_complexity)]
    fn list_all_event_channels(
        &self,
    ) -> Result<Vec<(String, String, String, String, String, String)>, CodememError> {
        Ok(Vec::new())
    }
}
