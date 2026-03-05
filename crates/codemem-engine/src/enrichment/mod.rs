//! Enrichment logic: store_insight, git history, security, performance,
//! complexity, architecture, test mapping, API surface, doc coverage,
//! change impact, code smells, hot+complex correlation, blame/ownership,
//! enhanced security scanning, and quality stratification.

mod api_surface;
mod architecture;
mod blame;
mod change_impact;
mod code_smells;
mod complexity;
mod doc_coverage;
mod git;
mod hot_complex;
mod performance;
mod quality;
mod security;
mod security_scan;
mod test_mapping;

use crate::scoring::truncate_content;
use crate::CodememEngine;
use codemem_core::{
    Edge, GraphBackend, GraphNode, MemoryNode, MemoryType, NodeKind, RelationshipType,
    VectorBackend,
};
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Resolve a (possibly relative) file path against a project root.
/// If `project_root` is `Some`, joins it with `rel_path` to produce an absolute path.
/// Otherwise returns `rel_path` as-is.
pub(crate) fn resolve_path(rel_path: &str, project_root: Option<&Path>) -> PathBuf {
    match project_root {
        Some(root) => root.join(rel_path),
        None => PathBuf::from(rel_path),
    }
}

/// Result from an enrichment operation.
pub struct EnrichResult {
    pub insights_stored: usize,
    pub details: serde_json::Value,
}

impl CodememEngine {
    /// Store an Insight memory through a 3-phase pipeline:
    /// 1. Semantic dedup check (reject near-duplicates before persisting)
    /// 2. Core persist via `persist_memory_no_save` (storage, BM25, graph node, embedding)
    /// 3. Post-step: RELATES_TO edges to linked nodes + auto-link to code nodes
    ///
    /// Returns the memory ID if inserted, or None if it was a duplicate.
    /// Does NOT call `save_index()` -- callers should batch that at the end.
    pub fn store_insight(
        &self,
        content: &str,
        track: &str,
        tags: &[&str],
        importance: f64,
        namespace: Option<&str>,
        links: &[String],
    ) -> Option<String> {
        let now = chrono::Utc::now();
        let id = uuid::Uuid::new_v4().to_string();
        let mut all_tags: Vec<String> =
            vec![format!("track:{track}"), "static-analysis".to_string()];
        all_tags.extend(tags.iter().map(|t| t.to_string()));

        // ── Phase 1: Semantic dedup check ────────────────────────────────
        // Compute enriched embedding and check for near-duplicates BEFORE persisting.
        let enriched = self.enrich_memory_text(
            content,
            MemoryType::Insight,
            &all_tags,
            namespace,
            Some(&id),
        );
        if let Ok(Some(emb_guard)) = self.lock_embeddings() {
            if let Ok(embedding) = emb_guard.embed(&enriched) {
                drop(emb_guard);
                if let Ok(vec) = self.lock_vector() {
                    let neighbors = vec.search(&embedding, 3).unwrap_or_default();
                    for (neighbor_id, similarity) in &neighbors {
                        if *neighbor_id == id {
                            continue;
                        }
                        if (*similarity as f64) > self.config.enrichment.dedup_similarity_threshold
                        {
                            return None; // Too similar — reject before persisting
                        }
                    }
                }
            }
        }

        // ── Phase 2: Core persist via persist_memory_no_save ─────────────
        let hash = codemem_storage::Storage::content_hash(content);
        let memory = MemoryNode {
            id: id.clone(),
            content: content.to_string(),
            memory_type: MemoryType::Insight,
            importance: importance.clamp(0.0, 1.0),
            confidence: self.config.enrichment.insight_confidence,
            access_count: 0,
            content_hash: hash,
            tags: all_tags,
            metadata: HashMap::from([
                ("track".into(), json!(track)),
                ("generated_by".into(), json!("enrichment_pipeline")),
            ]),
            namespace: namespace.map(String::from),
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        };

        if self.persist_memory_no_save(&memory).is_err() {
            return None; // duplicate or error -- skip silently
        }

        // ── Phase 3: Post-step — RELATES_TO edges to linked nodes ────────
        if !links.is_empty() {
            if let Ok(mut graph) = self.lock_graph() {
                for link_id in links {
                    let edge = Edge {
                        id: format!("{id}-RELATES_TO-{link_id}"),
                        src: id.clone(),
                        dst: link_id.clone(),
                        relationship: RelationshipType::RelatesTo,
                        weight: 0.3,
                        properties: HashMap::new(),
                        created_at: now,
                        valid_from: None,
                        valid_to: None,
                    };
                    let _ = self.storage.insert_graph_edge(&edge);
                    let _ = graph.add_edge(edge);
                }
            }
        }

        // Auto-link to code nodes mentioned in content
        self.auto_link_to_code_nodes(&id, content, links);

        Some(id)
    }

    /// Store a Pattern memory for code smell detection (E7).
    /// Importance is fixed at 0.5 for code smells.
    pub(super) fn store_pattern_memory(
        &self,
        content: &str,
        namespace: Option<&str>,
        links: &[String],
    ) -> Option<String> {
        let hash = codemem_storage::Storage::content_hash(content);
        let now = chrono::Utc::now();
        let id = uuid::Uuid::new_v4().to_string();
        let tags = vec![
            "static-analysis".to_string(),
            "track:code-smell".to_string(),
        ];

        let memory = MemoryNode {
            id: id.clone(),
            content: content.to_string(),
            memory_type: MemoryType::Pattern,
            importance: 0.5,
            confidence: self.config.enrichment.insight_confidence,
            access_count: 0,
            content_hash: hash,
            tags,
            metadata: HashMap::from([
                ("track".into(), json!("code-smell")),
                ("generated_by".into(), json!("enrichment_pipeline")),
            ]),
            namespace: namespace.map(String::from),
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        };

        if self.storage.insert_memory(&memory).is_err() {
            return None;
        }

        // Minimal pipeline: BM25 + graph node + links
        if let Ok(mut bm25) = self.lock_bm25() {
            bm25.add_document(&id, content);
        }

        let graph_node = GraphNode {
            id: id.clone(),
            kind: NodeKind::Memory,
            label: truncate_content(content, 80),
            payload: HashMap::new(),
            centrality: 0.0,
            memory_id: Some(id.clone()),
            namespace: namespace.map(String::from),
        };
        let _ = self.storage.insert_graph_node(&graph_node);
        if let Ok(mut graph) = self.lock_graph() {
            let _ = graph.add_node(graph_node);

            for link_id in links {
                let edge = Edge {
                    id: format!("{id}-RELATES_TO-{link_id}"),
                    src: id.clone(),
                    dst: link_id.clone(),
                    relationship: RelationshipType::RelatesTo,
                    weight: 0.3,
                    properties: HashMap::new(),
                    created_at: now,
                    valid_from: None,
                    valid_to: None,
                };
                let _ = self.storage.insert_graph_edge(&edge);
                let _ = graph.add_edge(edge);
            }
        }

        self.auto_link_to_code_nodes(&id, content, links);

        Some(id)
    }
}
