//! Analysis domain logic: impact-aware recall, decision chains, session checkpoints.
//!
//! These methods were extracted from the MCP transport layer to keep domain logic
//! in the engine crate.

use crate::CodememEngine;
use codemem_core::{
    CodememError, DetectedPattern, MemoryNode, MemoryType, NodeCoverageEntry, NodeKind,
    RelationshipType, SearchResult,
};
use serde_json::json;
use std::collections::{HashMap, HashSet};

// ── Result Types ─────────────────────────────────────────────────────────────

/// Impact data enrichment for a single search result.
#[derive(Debug, Clone)]
pub struct ImpactResult {
    /// The underlying search result (memory + score).
    pub search_result: SearchResult,
    /// PageRank score for this memory in the graph.
    pub pagerank: f64,
    /// Betweenness centrality score.
    pub centrality: f64,
    /// IDs of connected Decision-type memories.
    pub connected_decisions: Vec<String>,
    /// Labels/paths of connected File-type nodes.
    pub dependent_files: Vec<String>,
}

/// A single decision entry in a decision chain.
#[derive(Debug, Clone)]
pub struct DecisionEntry {
    pub memory: MemoryNode,
    /// Edges connecting this decision to others in the chain.
    pub connections: Vec<DecisionConnection>,
}

/// A connection between two decisions in a chain.
#[derive(Debug, Clone)]
pub struct DecisionConnection {
    pub relationship: String,
    pub source: String,
    pub target: String,
}

/// Result of a decision chain query.
#[derive(Debug, Clone)]
pub struct DecisionChain {
    /// Number of decisions in the chain.
    pub chain_length: usize,
    /// The filter that was used.
    pub file_path: Option<String>,
    /// The topic filter that was used.
    pub topic: Option<String>,
    /// The decisions in chronological order.
    pub decisions: Vec<DecisionEntry>,
}

/// Result of a session checkpoint.
#[derive(Debug, Clone)]
pub struct SessionCheckpointReport {
    /// Number of files read in this session.
    pub files_read: usize,
    /// Number of files edited in this session.
    pub files_edited: usize,
    /// Number of searches in this session.
    pub searches: usize,
    /// Total actions in this session.
    pub total_actions: usize,
    /// Hot directories with their action counts.
    pub hot_dirs: Vec<(String, usize)>,
    /// Patterns detected within this session.
    pub session_patterns: Vec<DetectedPattern>,
    /// Patterns detected across sessions (excluding session-scoped duplicates).
    pub cross_patterns: Vec<DetectedPattern>,
    /// Number of new pattern insights stored during this checkpoint.
    pub stored_pattern_count: usize,
    /// Pre-built markdown report.
    pub report: String,
}

// ── Engine Methods ───────────────────────────────────────────────────────────

impl CodememEngine {
    /// Recall memories enriched with graph impact data (PageRank, centrality,
    /// connected decisions, dependent files).
    pub fn recall_with_impact(
        &self,
        query: &str,
        k: usize,
        namespace: Option<&str>,
    ) -> Result<Vec<ImpactResult>, CodememError> {
        let results = self.recall(&crate::recall::RecallQuery {
            query,
            k,
            namespace_filter: namespace,
            ..crate::recall::RecallQuery::new(query, k)
        })?;

        if results.is_empty() {
            return Ok(vec![]);
        }

        let mut graph = self.lock_graph()?;
        // C1: Ensure betweenness is computed before reading centrality values.
        graph.ensure_betweenness_computed();

        let output: Vec<ImpactResult> = results
            .into_iter()
            .map(|r| {
                let memory_id = &r.memory.id;

                let pagerank = graph.get_pagerank(memory_id);
                let centrality = graph.get_betweenness(memory_id);

                let edges = graph.get_edges(memory_id).unwrap_or_default();

                let connected_decisions: Vec<String> = edges
                    .iter()
                    .filter_map(|e| {
                        let other_id = if e.src == *memory_id { &e.dst } else { &e.src };
                        self.storage
                            .get_memory_no_touch(other_id)
                            .ok()
                            .flatten()
                            .and_then(|m| {
                                if m.memory_type == MemoryType::Decision {
                                    Some(m.id)
                                } else {
                                    None
                                }
                            })
                    })
                    .collect();

                let dependent_files: Vec<String> = edges
                    .iter()
                    .filter_map(|e| {
                        let other_id = if e.src == *memory_id { &e.dst } else { &e.src };
                        graph.get_node(other_id).ok().flatten().and_then(|n| {
                            if n.kind == NodeKind::File {
                                Some(n.label.clone())
                            } else {
                                n.payload
                                    .get("file_path")
                                    .and_then(|v| v.as_str().map(String::from))
                            }
                        })
                    })
                    .collect();

                ImpactResult {
                    search_result: r,
                    pagerank,
                    centrality,
                    connected_decisions,
                    dependent_files,
                }
            })
            .collect();

        Ok(output)
    }

    /// Find Decision-type memories matching a file_path or topic, then follow
    /// EvolvedInto/LeadsTo/DerivedFrom edges via BFS to build a chronological chain.
    pub fn get_decision_chain(
        &self,
        file_path: Option<&str>,
        topic: Option<&str>,
    ) -> Result<DecisionChain, CodememError> {
        if file_path.is_none() && topic.is_none() {
            return Err(CodememError::InvalidInput(
                "Must provide either 'file_path' or 'topic' parameter".to_string(),
            ));
        }

        let graph = self.lock_graph()?;

        let decision_edge_types = [
            RelationshipType::EvolvedInto,
            RelationshipType::LeadsTo,
            RelationshipType::DerivedFrom,
        ];

        // Batch-load all Decision memories in one query
        let all_decisions = self
            .storage
            .list_memories_filtered(None, Some("decision"))?;

        // Hoist lowercased filter values outside the loop
        let filter_lower = file_path.map(|f| f.to_lowercase());
        let topic_lower = topic.map(|t| t.to_lowercase());

        // Collect Decision memories matching the filter
        let mut decision_memories: Vec<MemoryNode> = Vec::new();
        for memory in all_decisions {
            let content_lower = memory.content.to_lowercase();
            let tags_lower: String = memory.tags.join(" ").to_lowercase();

            let matches = if let Some(ref fp) = filter_lower {
                content_lower.contains(fp)
                    || tags_lower.contains(fp)
                    || memory
                        .metadata
                        .get("file_path")
                        .and_then(|v| v.as_str())
                        .map(|v| v.to_lowercase().contains(fp))
                        .unwrap_or(false)
            } else if let Some(ref tl) = topic_lower {
                content_lower.contains(tl) || tags_lower.contains(tl)
            } else {
                false
            };

            if matches {
                decision_memories.push(memory);
            }
        }

        if decision_memories.is_empty() {
            return Ok(DecisionChain {
                chain_length: 0,
                file_path: file_path.map(String::from),
                topic: topic.map(String::from),
                decisions: vec![],
            });
        }

        // Expand through decision-related edges to find the full chain (BFS)
        let mut chain_ids: HashSet<String> = HashSet::new();
        let mut to_explore: Vec<String> = decision_memories.iter().map(|m| m.id.clone()).collect();

        while let Some(current_id) = to_explore.pop() {
            if !chain_ids.insert(current_id.clone()) {
                continue;
            }

            if let Ok(edges) = graph.get_edges(&current_id) {
                for edge in &edges {
                    if decision_edge_types.contains(&edge.relationship) {
                        let other_id = if edge.src == current_id {
                            &edge.dst
                        } else {
                            &edge.src
                        };
                        if !chain_ids.contains(other_id) {
                            // Only follow to other Decision memories
                            if let Ok(Some(m)) = self.storage.get_memory_no_touch(other_id) {
                                if m.memory_type == MemoryType::Decision {
                                    to_explore.push(other_id.clone());
                                }
                            }
                        }
                    }
                }
            }
        }

        // Collect all chain memories and sort by created_at (temporal order)
        let mut chain: Vec<DecisionEntry> = Vec::new();
        for id in &chain_ids {
            if let Ok(Some(memory)) = self.storage.get_memory_no_touch(id) {
                let connections: Vec<DecisionConnection> = graph
                    .get_edges(id)
                    .unwrap_or_default()
                    .iter()
                    .filter(|e| {
                        decision_edge_types.contains(&e.relationship)
                            && (chain_ids.contains(&e.src) && chain_ids.contains(&e.dst))
                    })
                    .map(|e| DecisionConnection {
                        relationship: e.relationship.to_string(),
                        source: e.src.clone(),
                        target: e.dst.clone(),
                    })
                    .collect();

                chain.push(DecisionEntry {
                    memory,
                    connections,
                });
            }
        }

        // Sort chronologically
        chain.sort_by(|a, b| a.memory.created_at.cmp(&b.memory.created_at));

        let chain_length = chain.len();
        Ok(DecisionChain {
            chain_length,
            file_path: file_path.map(String::from),
            topic: topic.map(String::from),
            decisions: chain,
        })
    }

    /// Build a mid-session progress report: activity summary, pattern detection
    /// (session-scoped + cross-session), stores new pattern insights, hot directories,
    /// markdown report.
    pub fn session_checkpoint(
        &self,
        session_id: &str,
        namespace: Option<&str>,
    ) -> Result<SessionCheckpointReport, CodememError> {
        // 1. Get session activity summary
        let activity = self.storage.get_session_activity_summary(session_id)?;

        // 2. Run session-scoped pattern detection (lower thresholds for single session)
        let total_sessions = self.storage.session_count(namespace).unwrap_or(1).max(1);

        let session_patterns = crate::patterns::detect_patterns(
            &*self.storage,
            namespace,
            2, // session-scoped: min_frequency=2
            total_sessions,
        )
        .unwrap_or_default();

        // Cross-session patterns with higher threshold
        let cross_patterns = crate::patterns::detect_patterns(
            &*self.storage,
            namespace,
            3, // cross-session: min_frequency=3
            total_sessions,
        )
        .unwrap_or_default();

        // 3. Store new session patterns as Insight memories (with dedup)
        let mut stored_patterns = 0usize;
        for pattern in &session_patterns {
            let dedup_tag = format!("checkpoint:{}:{}", session_id, pattern.description);
            let already_exists = self
                .storage
                .has_auto_insight(session_id, &dedup_tag)
                .unwrap_or(true);
            if !already_exists && pattern.confidence > 0.3 {
                let mut metadata = HashMap::new();
                metadata.insert("session_id".to_string(), json!(session_id));
                metadata.insert("auto_insight_tag".to_string(), json!(dedup_tag));
                metadata.insert("source".to_string(), json!("session_checkpoint"));
                metadata.insert(
                    "pattern_type".to_string(),
                    json!(pattern.pattern_type.to_string()),
                );

                let mut mem = codemem_core::MemoryNode::new(
                    format!("Session pattern: {}", pattern.description),
                    MemoryType::Insight,
                );
                mem.importance = 0.6;
                mem.confidence = pattern.confidence;
                mem.tags = vec![
                    "session-checkpoint".to_string(),
                    format!("pattern:{}", pattern.pattern_type),
                ];
                mem.metadata = metadata;
                mem.namespace = namespace.map(|s| s.to_string());
                if self.persist_memory_no_save(&mem).is_ok() {
                    stored_patterns += 1;
                }
            }
        }

        // 4. Get hot directories
        let hot_dirs = self
            .storage
            .get_session_hot_directories(session_id, 5)
            .unwrap_or_default();

        // 5. Filter unique cross-session patterns
        let unique_cross: Vec<DetectedPattern> = cross_patterns
            .iter()
            .filter(|p| {
                !session_patterns
                    .iter()
                    .any(|sp| sp.description == p.description)
            })
            .take(5)
            .cloned()
            .collect();

        // 6. Build markdown report
        let report = Self::format_checkpoint_report(
            &activity,
            &hot_dirs,
            &session_patterns,
            &unique_cross,
            stored_patterns,
        );

        // 7. Persist a checkpoint memory with session state metadata
        let memory_count = self.storage.memory_count().unwrap_or(0);
        let now = chrono::Utc::now();
        let checkpoint_content = format!(
            "Session checkpoint for {}: {} actions ({} reads, {} edits, {} searches), {} total memories, {} patterns detected",
            session_id,
            activity.total_actions,
            activity.files_read,
            activity.files_edited,
            activity.searches,
            memory_count,
            session_patterns.len(),
        );
        let mut checkpoint_metadata = HashMap::new();
        checkpoint_metadata.insert("checkpoint_type".to_string(), json!("manual"));
        checkpoint_metadata.insert("session_id".to_string(), json!(session_id));
        checkpoint_metadata.insert("memory_count".to_string(), json!(memory_count));
        checkpoint_metadata.insert("timestamp".to_string(), json!(now.to_rfc3339()));
        checkpoint_metadata.insert("files_read".to_string(), json!(activity.files_read));
        checkpoint_metadata.insert("files_edited".to_string(), json!(activity.files_edited));
        checkpoint_metadata.insert("searches".to_string(), json!(activity.searches));
        checkpoint_metadata.insert("total_actions".to_string(), json!(activity.total_actions));
        checkpoint_metadata.insert("pattern_count".to_string(), json!(session_patterns.len()));
        checkpoint_metadata.insert("cross_pattern_count".to_string(), json!(unique_cross.len()));
        checkpoint_metadata.insert("stored_pattern_count".to_string(), json!(stored_patterns));
        if !hot_dirs.is_empty() {
            let dirs: Vec<&str> = hot_dirs.iter().map(|(d, _)| d.as_str()).collect();
            checkpoint_metadata.insert("hot_directories".to_string(), json!(dirs));
        }

        let mut checkpoint_mem =
            codemem_core::MemoryNode::new(checkpoint_content, MemoryType::Context);
        checkpoint_mem.tags = vec![
            "session-checkpoint".to_string(),
            format!("session:{session_id}"),
        ];
        checkpoint_mem.metadata = checkpoint_metadata;
        checkpoint_mem.namespace = namespace.map(|s| s.to_string());
        checkpoint_mem.session_id = Some(session_id.to_string());
        // Best-effort persist; don't fail the checkpoint if this errors
        let _ = self.persist_memory(&checkpoint_mem);

        Ok(SessionCheckpointReport {
            files_read: activity.files_read,
            files_edited: activity.files_edited,
            searches: activity.searches,
            total_actions: activity.total_actions,
            hot_dirs,
            session_patterns,
            cross_patterns: unique_cross,
            stored_pattern_count: stored_patterns,
            report,
        })
    }

    /// Format the checkpoint data into a markdown report string.
    fn format_checkpoint_report(
        activity: &codemem_core::SessionActivitySummary,
        hot_dirs: &[(String, usize)],
        session_patterns: &[DetectedPattern],
        cross_patterns: &[DetectedPattern],
        stored_patterns: usize,
    ) -> String {
        let mut report = String::from("## Session Checkpoint\n\n");

        // Activity summary
        report.push_str("### Activity Summary\n\n");
        report.push_str(&format!(
            "| Metric | Count |\n|--------|-------|\n\
             | Files read | {} |\n\
             | Files edited | {} |\n\
             | Searches | {} |\n\
             | Total actions | {} |\n\n",
            activity.files_read, activity.files_edited, activity.searches, activity.total_actions,
        ));

        // Focus areas
        if !hot_dirs.is_empty() {
            report.push_str("### Focus Areas\n\n");
            report.push_str("Directories with most activity in this session:\n\n");
            for (dir, count) in hot_dirs {
                report.push_str(&format!("- `{}` ({} actions)\n", dir, count));
            }
            report.push('\n');
        }

        // Session-scoped patterns
        if !session_patterns.is_empty() {
            report.push_str("### Session Patterns\n\n");
            for p in session_patterns.iter().take(10) {
                report.push_str(&format!(
                    "- [{}] {} (confidence: {:.0}%)\n",
                    p.pattern_type,
                    p.description,
                    p.confidence * 100.0,
                ));
            }
            report.push('\n');
        }

        // Cross-session patterns
        if !cross_patterns.is_empty() {
            report.push_str("### Cross-Session Patterns\n\n");
            for p in cross_patterns {
                report.push_str(&format!(
                    "- [{}] {} (confidence: {:.0}%)\n",
                    p.pattern_type,
                    p.description,
                    p.confidence * 100.0,
                ));
            }
            report.push('\n');
        }

        // Suggestions
        report.push_str("### Suggestions\n\n");
        if activity.files_read > 5 && activity.files_edited == 0 {
            report.push_str(
                "- You've read many files but haven't edited any yet. \
                 Consider storing a `decision` memory about what you've learned.\n",
            );
        }
        if activity.searches > 3 {
            report.push_str(
                "- Multiple searches detected. Use `store_memory` to save \
                 key findings so you don't need to search again.\n",
            );
        }
        if stored_patterns > 0 {
            report.push_str(&format!(
                "- {} new pattern insight(s) stored from this checkpoint.\n",
                stored_patterns,
            ));
        }
        if activity.total_actions == 0 {
            report.push_str("- No activity recorded yet for this session.\n");
        }

        report
    }

    /// Check which graph nodes have attached memories (depth-1 only).
    pub fn node_coverage(&self, node_ids: &[&str]) -> Result<Vec<NodeCoverageEntry>, CodememError> {
        let graph = self.lock_graph()?;
        let mut results = Vec::with_capacity(node_ids.len());

        for &node_id in node_ids {
            let edges = graph.get_edges_ref(node_id);
            let memory_count = edges
                .iter()
                .filter(|e| {
                    let other_id = if e.src == node_id { &e.dst } else { &e.src };
                    graph
                        .get_node_ref(other_id)
                        .map(|n| n.kind == NodeKind::Memory)
                        .unwrap_or(false)
                })
                .count();

            results.push(NodeCoverageEntry {
                node_id: node_id.to_string(),
                memory_count,
                has_coverage: memory_count > 0,
            });
        }

        Ok(results)
    }
}
