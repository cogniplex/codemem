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
pub(crate) mod dead_code;
mod doc_coverage;
mod git;
mod hot_complex;
mod performance;
mod quality;
mod security;
mod security_scan;
pub(crate) mod temporal;
mod test_mapping;

use crate::CodememEngine;
use codemem_core::{Edge, MemoryNode, MemoryType, RelationshipType};
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

/// Result from running multiple enrichment analyses.
pub struct EnrichmentPipelineResult {
    /// JSON object with one key per analysis (e.g. "git", "security", etc.).
    pub results: serde_json::Value,
    /// Total number of insights stored across all analyses.
    pub total_insights: usize,
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
        let mut memory = MemoryNode::new(content, MemoryType::Insight);
        memory.id = id.clone();
        memory.importance = importance.clamp(0.0, 1.0);
        memory.confidence = self.config.enrichment.insight_confidence;
        memory.tags = all_tags;
        memory.metadata = HashMap::from([
            ("track".into(), json!(track)),
            ("generated_by".into(), json!("enrichment_pipeline")),
        ]);
        memory.namespace = namespace.map(String::from);

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

    /// Run selected enrichment analyses (or all 14 if `analyses` is empty).
    ///
    /// Parameters:
    /// - `path`: project root (needed for git, blame, change_impact, complexity, code_smells, security_scan)
    /// - `analyses`: which analyses to run; empty = all (except change_impact which needs file_path)
    /// - `days`: git history lookback days
    /// - `namespace`: optional namespace filter
    /// - `file_path`: optional, needed only for change_impact
    pub fn run_enrichments(
        &self,
        path: &str,
        analyses: &[String],
        days: u64,
        namespace: Option<&str>,
        file_path: Option<&str>,
    ) -> EnrichmentPipelineResult {
        let run_all = analyses.is_empty();
        let mut results = json!({});
        let mut total_insights: usize = 0;

        let root = Path::new(path);
        let project_root = Some(root);

        macro_rules! run_analysis {
            ($name:expr, $call:expr) => {
                if run_all || analyses.iter().any(|a| a == $name) {
                    match $call {
                        Ok(r) => {
                            total_insights += r.insights_stored;
                            results[$name] = r.details;
                        }
                        Err(e) => {
                            results[$name] = json!({"error": format!("{e}")});
                        }
                    }
                }
            };
        }

        run_analysis!("git", self.enrich_git_history(path, days, namespace));
        run_analysis!("security", self.enrich_security(namespace));
        run_analysis!("performance", self.enrich_performance(10, namespace));
        run_analysis!(
            "complexity",
            self.enrich_complexity(namespace, project_root)
        );
        run_analysis!(
            "code_smells",
            self.enrich_code_smells(namespace, project_root)
        );
        run_analysis!(
            "security_scan",
            self.enrich_security_scan(namespace, project_root)
        );
        run_analysis!("architecture", self.enrich_architecture(namespace));
        run_analysis!("test_mapping", self.enrich_test_mapping(namespace));
        run_analysis!("api_surface", self.enrich_api_surface(namespace));
        run_analysis!("doc_coverage", self.enrich_doc_coverage(namespace));
        run_analysis!("hot_complex", self.enrich_hot_complex(namespace));
        run_analysis!("blame", self.enrich_blame(path, namespace));
        run_analysis!("quality", self.enrich_quality_stratification(namespace));

        // change_impact requires a file_path, so it is not included in run_all
        if analyses.iter().any(|a| a == "change_impact") {
            let fp = file_path.unwrap_or("");
            if fp.is_empty() {
                results["change_impact"] =
                    json!({"error": "change_impact requires 'file_path' parameter"});
            } else {
                match self.enrich_change_impact(fp, namespace) {
                    Ok(r) => {
                        total_insights += r.insights_stored;
                        results["change_impact"] = r.details;
                    }
                    Err(e) => {
                        results["change_impact"] = json!({"error": format!("{e}")});
                    }
                }
            }
        }

        EnrichmentPipelineResult {
            results,
            total_insights,
        }
    }

    /// Store a Pattern memory for code smell detection (E7).
    /// Importance is fixed at 0.5 for code smells.
    /// Uses the full persist pipeline (storage → BM25 → graph → embedding → vector).
    pub(super) fn store_pattern_memory(
        &self,
        content: &str,
        namespace: Option<&str>,
        links: &[String],
    ) -> Option<String> {
        let id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now();
        let tags = vec![
            "static-analysis".to_string(),
            "track:code-smell".to_string(),
        ];

        let mut memory = MemoryNode::new(content, MemoryType::Pattern);
        memory.id = id.clone();
        memory.confidence = self.config.enrichment.insight_confidence;
        memory.tags = tags;
        memory.metadata = HashMap::from([
            ("track".into(), json!("code-smell")),
            ("generated_by".into(), json!("enrichment_pipeline")),
        ]);
        memory.namespace = namespace.map(String::from);

        if self.persist_memory_no_save(&memory).is_err() {
            return None;
        }

        // Post-step: RELATES_TO edges to linked nodes
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

        self.auto_link_to_code_nodes(&id, content, links);

        Some(id)
    }
}
