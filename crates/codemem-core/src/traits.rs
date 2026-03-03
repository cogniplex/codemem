use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{CodememError, Edge, GraphNode, MemoryNode, NodeKind, RelationshipType, Session};

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
    ) -> Result<Vec<GraphNode>, CodememError> {
        // Default implementation falls back to unfiltered BFS
        let _ = (exclude_kinds, include_relationships);
        self.bfs(start_id, max_depth)
    }

    /// DFS traversal with filtering: exclude certain node kinds and optionally
    /// restrict to specific relationship types.
    fn dfs_filtered(
        &self,
        start_id: &str,
        max_depth: usize,
        exclude_kinds: &[NodeKind],
        include_relationships: Option<&[RelationshipType]>,
    ) -> Result<Vec<GraphNode>, CodememError> {
        // Default implementation falls back to unfiltered DFS
        let _ = (exclude_kinds, include_relationships);
        self.dfs(start_id, max_depth)
    }

    /// Shortest path between two nodes.
    fn shortest_path(&self, from: &str, to: &str) -> Result<Vec<String>, CodememError>;

    /// Get graph statistics.
    fn stats(&self) -> GraphStats;
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
/// This trait unifies memory CRUD, embedding persistence, graph node/edge
/// storage, sessions, consolidation, and pattern detection behind a single
/// interface. Implementations include SQLite (default) and can be extended
/// for SurrealDB, FalkorDB, or other backends.
pub trait StorageBackend: Send + Sync {
    // ── Memory CRUD ─────────────────────────────────────────────────

    /// Insert a new memory. Returns Err(Duplicate) if content hash already exists.
    fn insert_memory(&self, memory: &MemoryNode) -> Result<(), CodememError>;

    /// Get a memory by ID. Updates access_count and last_accessed_at.
    fn get_memory(&self, id: &str) -> Result<Option<MemoryNode>, CodememError>;

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

    /// List all memory IDs, ordered by created_at descending.
    fn list_memory_ids(&self) -> Result<Vec<String>, CodememError>;

    /// List memory IDs scoped to a specific namespace.
    fn list_memory_ids_for_namespace(&self, namespace: &str) -> Result<Vec<String>, CodememError>;

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

    /// Find near-duplicate memories by content hash prefix similarity.
    /// Returns (id1, id2, similarity) pairs.
    fn find_cluster_duplicates(&self) -> Result<Vec<(String, String, f64)>, CodememError>;

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

    /// List memories with optional namespace and memory_type filters.
    fn list_memories_filtered(
        &self,
        namespace: Option<&str>,
        memory_type: Option<&str>,
    ) -> Result<Vec<MemoryNode>, CodememError>;

    /// Get edges filtered by namespace (edges where both src and dst nodes have the given namespace).
    fn graph_edges_for_namespace(&self, namespace: &str) -> Result<Vec<Edge>, CodememError>;

    // ── Temporal Edge Queries ───────────────────────────────────────

    /// Get edges active at a specific timestamp. Default: no temporal filtering.
    fn get_edges_at_time(&self, node_id: &str, _timestamp: i64) -> Result<Vec<Edge>, CodememError> {
        self.get_edges_for_node(node_id)
    }

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

    /// Load all file hashes for incremental indexing. Returns path -> hash map.
    fn load_file_hashes(&self) -> Result<HashMap<String, String>, CodememError>;

    /// Save file hashes for incremental indexing.
    fn save_file_hashes(&self, hashes: &HashMap<String, String>) -> Result<(), CodememError>;

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

    // ── Stats ───────────────────────────────────────────────────────

    /// Get database statistics.
    fn stats(&self) -> Result<StorageStats, CodememError>;
}
