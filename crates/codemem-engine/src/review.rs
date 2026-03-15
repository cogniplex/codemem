//! Diff-aware review pipeline: parse unified diffs, map changed lines to symbols,
//! compute blast radius via multi-hop graph traversal.

use crate::CodememEngine;
use codemem_core::{CodememError, GraphBackend, MemoryNode, RelationshipType};
use std::collections::{HashMap, HashSet};

// ── Types ────────────────────────────────────────────────────────────────

/// A parsed diff hunk: file path + changed line ranges.
#[derive(Debug, Clone)]
pub struct DiffHunk {
    pub file_path: String,
    pub added_lines: Vec<u32>,
    pub removed_lines: Vec<u32>,
}

/// Mapping from a unified diff to affected symbol IDs in the graph.
#[derive(Debug, Clone, Default)]
pub struct DiffSymbolMapping {
    /// sym:IDs whose definition range overlaps a changed line.
    pub changed_symbols: Vec<String>,
    /// sym:IDs whose body contains changes (parent of a changed symbol).
    pub containing_symbols: Vec<String>,
    /// file:IDs of changed files.
    pub changed_files: Vec<String>,
}

/// Information about a symbol for the blast radius report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SymbolInfo {
    pub id: String,
    pub label: String,
    pub kind: String,
    pub file_path: Option<String>,
    pub line_start: Option<u32>,
    pub pagerank: f64,
}

/// A potentially missing change detected by pattern analysis.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MissingChange {
    pub symbol: String,
    pub reason: String,
}

/// Full blast radius report for a diff.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BlastRadiusReport {
    pub changed_symbols: Vec<SymbolInfo>,
    pub direct_dependents: Vec<SymbolInfo>,
    pub transitive_dependents: Vec<SymbolInfo>,
    pub affected_files: Vec<String>,
    pub affected_modules: Vec<String>,
    pub risk_score: f64,
    pub missing_changes: Vec<MissingChange>,
    pub relevant_memories: Vec<MemorySnippet>,
}

/// Lightweight memory reference for the report (avoids serializing full MemoryNode).
#[derive(Debug, Clone, serde::Serialize)]
pub struct MemorySnippet {
    pub id: String,
    pub content: String,
    pub memory_type: String,
    pub importance: f64,
}

impl From<&MemoryNode> for MemorySnippet {
    fn from(m: &MemoryNode) -> Self {
        Self {
            id: m.id.clone(),
            content: m.content.clone(),
            memory_type: m.memory_type.to_string(),
            importance: m.importance,
        }
    }
}

// ── Diff Parsing ─────────────────────────────────────────────────────────

/// Parse a unified diff into hunks with file paths and changed line numbers.
pub fn parse_diff(diff: &str) -> Vec<DiffHunk> {
    let mut hunks = Vec::new();
    let mut current_file: Option<String> = None;
    let mut added_lines: Vec<u32> = Vec::new();
    let mut removed_lines: Vec<u32> = Vec::new();
    let mut new_line: u32 = 0;
    let mut old_line: u32 = 0;

    for line in diff.lines() {
        if line.starts_with("+++ b/") {
            // Flush previous file
            if let Some(ref file) = current_file {
                if !added_lines.is_empty() || !removed_lines.is_empty() {
                    hunks.push(DiffHunk {
                        file_path: file.clone(),
                        added_lines: std::mem::take(&mut added_lines),
                        removed_lines: std::mem::take(&mut removed_lines),
                    });
                }
            }
            current_file = line.strip_prefix("+++ b/").map(|s| s.to_string());
        } else if line.starts_with("@@ ") {
            // Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
            if let Some((new_start, old_start)) = parse_hunk_header(line) {
                new_line = new_start;
                old_line = old_start;
            }
        } else if current_file.is_some() {
            if line.starts_with('+') && !line.starts_with("+++") {
                added_lines.push(new_line);
                new_line += 1;
            } else if line.starts_with('-') && !line.starts_with("---") {
                removed_lines.push(old_line);
                old_line += 1;
            } else {
                // Context line
                new_line += 1;
                old_line += 1;
            }
        }
    }

    // Flush last file
    if let Some(file) = current_file {
        if !added_lines.is_empty() || !removed_lines.is_empty() {
            hunks.push(DiffHunk {
                file_path: file,
                added_lines,
                removed_lines,
            });
        }
    }

    hunks
}

/// Parse a @@ hunk header, returning (new_start, old_start).
fn parse_hunk_header(line: &str) -> Option<(u32, u32)> {
    // Format: @@ -old_start[,old_count] +new_start[,new_count] @@
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 3 {
        return None;
    }
    let old_part = parts[1].strip_prefix('-')?;
    let new_part = parts[2].strip_prefix('+')?;

    let old_start: u32 = old_part.split(',').next()?.parse().ok()?;
    let new_start: u32 = new_part.split(',').next()?.parse().ok()?;
    Some((new_start, old_start))
}

// ── Diff to Symbols ──────────────────────────────────────────────────────

impl CodememEngine {
    /// Map a unified diff to affected symbol IDs using the graph's line range data.
    pub fn diff_to_symbols(&self, diff: &str) -> Result<DiffSymbolMapping, CodememError> {
        let hunks = parse_diff(diff);
        let graph = self.lock_graph()?;
        let all_nodes = graph.get_all_nodes();

        let mut mapping = DiffSymbolMapping::default();
        let mut seen_symbols: HashSet<String> = HashSet::new();
        let mut seen_files: HashSet<String> = HashSet::new();

        // Build file→symbols index to avoid O(nodes × hunks) scan
        let mut file_symbols: HashMap<&str, Vec<&codemem_core::GraphNode>> = HashMap::new();
        for node in &all_nodes {
            if !node.id.starts_with("sym:") {
                continue;
            }
            if let Some(fp) = node.payload.get("file_path").and_then(|v| v.as_str()) {
                file_symbols.entry(fp).or_default().push(node);
            }
        }

        for hunk in &hunks {
            let file_id = format!("file:{}", hunk.file_path);
            if seen_files.insert(file_id.clone()) {
                mapping.changed_files.push(file_id);
            }

            let changed_lines: HashSet<u32> = hunk
                .added_lines
                .iter()
                .chain(hunk.removed_lines.iter())
                .copied()
                .collect();

            // Only check symbols in this file (indexed lookup)
            if let Some(nodes) = file_symbols.get(hunk.file_path.as_str()) {
                for node in nodes {
                    let line_start = node
                        .payload
                        .get("line_start")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as u32;
                    let line_end = node
                        .payload
                        .get("line_end")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(line_start as u64) as u32;

                    let overlaps = changed_lines
                        .iter()
                        .any(|&l| l >= line_start && l <= line_end);
                    if overlaps && seen_symbols.insert(node.id.clone()) {
                        mapping.changed_symbols.push(node.id.clone());
                    }
                }
            }
        }

        // Find containing symbols (parents of changed symbols via CONTAINS edges)
        let changed_set: HashSet<&str> =
            mapping.changed_symbols.iter().map(|s| s.as_str()).collect();
        for node in &all_nodes {
            if !node.id.starts_with("sym:") || changed_set.contains(node.id.as_str()) {
                continue;
            }
            // Check if this symbol contains any changed symbol
            let edges = graph.get_edges(&node.id).unwrap_or_default();
            let contains_changed = edges.iter().any(|e| {
                e.relationship == RelationshipType::Contains && changed_set.contains(e.dst.as_str())
            });
            if contains_changed && seen_symbols.insert(node.id.clone()) {
                mapping.containing_symbols.push(node.id.clone());
            }
        }

        Ok(mapping)
    }

    /// Compute the blast radius for a diff: changed symbols, dependents, risk score,
    /// relevant memories, and missing change detection.
    pub fn blast_radius(
        &self,
        diff: &str,
        depth: usize,
    ) -> Result<BlastRadiusReport, CodememError> {
        let mapping = self.diff_to_symbols(diff)?;
        let graph = self.lock_graph()?;

        let mut changed_infos = Vec::new();
        let mut direct_deps = Vec::new();
        let mut transitive_deps = Vec::new();
        let mut affected_files: HashSet<String> = HashSet::new();
        let mut affected_modules: HashSet<String> = HashSet::new();
        let mut seen: HashSet<String> = HashSet::new();
        let mut risk_score: f64 = 0.0;

        // Collect changed symbol info + their PageRank for risk scoring
        for sym_id in &mapping.changed_symbols {
            if let Some(info) = node_to_symbol_info(&**graph, sym_id) {
                risk_score += info.pagerank;
                if let Some(ref fp) = info.file_path {
                    affected_files.insert(fp.clone());
                }
                seen.insert(sym_id.clone());
                changed_infos.push(info);
            }
        }
        for sym_id in &mapping.containing_symbols {
            if let Some(info) = node_to_symbol_info(&**graph, sym_id) {
                if let Some(ref fp) = info.file_path {
                    affected_files.insert(fp.clone());
                }
                seen.insert(sym_id.clone());
                changed_infos.push(info);
            }
        }

        // BFS from changed symbols to find dependents
        let all_changed: Vec<&str> = mapping
            .changed_symbols
            .iter()
            .chain(mapping.containing_symbols.iter())
            .map(|s| s.as_str())
            .collect();

        for &start_id in &all_changed {
            // Get direct dependents (1-hop incoming edges: who CALLS/IMPORTS this symbol?)
            let edges = graph.get_edges(start_id).unwrap_or_default();
            for edge in &edges {
                // Incoming edges: other symbols that depend on this one
                let dependent_id = if edge.dst == start_id {
                    &edge.src
                } else {
                    continue; // outgoing edge, skip
                };
                if !dependent_id.starts_with("sym:") || !seen.insert(dependent_id.clone()) {
                    continue;
                }
                if matches!(
                    edge.relationship,
                    RelationshipType::Calls
                        | RelationshipType::Imports
                        | RelationshipType::Inherits
                        | RelationshipType::Implements
                        | RelationshipType::Overrides
                ) {
                    if let Some(info) = node_to_symbol_info(&**graph, dependent_id) {
                        if let Some(ref fp) = info.file_path {
                            affected_files.insert(fp.clone());
                        }
                        direct_deps.push(info);
                    }
                }
            }
        }

        // Transitive dependents (2+ hops) via iterative incoming-edge traversal.
        // BFS follows outgoing edges (wrong direction for "who depends on me?").
        // Instead, walk incoming edges layer by layer.
        if depth > 1 {
            let mut frontier: Vec<String> = direct_deps.iter().map(|d| d.id.clone()).collect();
            for _ in 1..depth {
                let mut next_frontier = Vec::new();
                for node_id in &frontier {
                    let edges = graph.get_edges(node_id).unwrap_or_default();
                    for edge in &edges {
                        // Only follow incoming dependency edges
                        if edge.dst != *node_id {
                            continue;
                        }
                        if !matches!(
                            edge.relationship,
                            RelationshipType::Calls
                                | RelationshipType::Imports
                                | RelationshipType::Inherits
                                | RelationshipType::Implements
                                | RelationshipType::Overrides
                        ) {
                            continue;
                        }
                        let dep_id = &edge.src;
                        if !dep_id.starts_with("sym:") || !seen.insert(dep_id.clone()) {
                            continue;
                        }
                        if let Some(info) = node_to_symbol_info(&**graph, dep_id) {
                            if let Some(ref fp) = info.file_path {
                                affected_files.insert(fp.clone());
                            }
                            if info.kind == "Module" {
                                affected_modules.insert(info.id.clone());
                            }
                            next_frontier.push(dep_id.clone());
                            transitive_deps.push(info);
                        }
                    }
                }
                if next_frontier.is_empty() {
                    break;
                }
                frontier = next_frontier;
            }
        }

        // Detect affected modules from all symbols
        for info in changed_infos.iter().chain(direct_deps.iter()) {
            if info.kind == "Module" {
                affected_modules.insert(info.id.clone());
            }
        }

        // Risk score: Σ(pagerank) + log(transitive_count + 1)
        // Additive so that diffs touching zero-pagerank symbols (common when
        // centrality hasn't been computed or symbols have no edges) still get
        // a nonzero risk score from their dependent count.
        let transitive_count = direct_deps.len() + transitive_deps.len();
        risk_score += (transitive_count as f64 + 1.0).ln();

        // Drop graph lock before accessing storage
        drop(graph);

        // Find relevant memories connected to changed symbols
        let mut relevant_memories = Vec::new();
        for sym_id in mapping
            .changed_symbols
            .iter()
            .chain(mapping.containing_symbols.iter())
            .take(20)
        {
            if let Ok(results) = self.get_node_memories(sym_id, 1, None) {
                for r in &results {
                    relevant_memories.push(MemorySnippet::from(&r.memory));
                }
            }
        }
        // Dedup memories by ID
        let mut seen_mem_ids: HashSet<String> = HashSet::new();
        relevant_memories.retain(|m| seen_mem_ids.insert(m.id.clone()));

        // Missing change detection: find symbols with similar caller patterns
        let graph = self.lock_graph()?;
        let missing_changes = detect_missing_changes(&**graph, &mapping.changed_symbols, &seen);

        let affected_files: Vec<String> = affected_files.into_iter().collect();
        let affected_modules: Vec<String> = affected_modules.into_iter().collect();

        Ok(BlastRadiusReport {
            changed_symbols: changed_infos,
            direct_dependents: direct_deps,
            transitive_dependents: transitive_deps,
            affected_files,
            affected_modules,
            risk_score,
            missing_changes,
            relevant_memories,
        })
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────

fn node_to_symbol_info(graph: &dyn GraphBackend, node_id: &str) -> Option<SymbolInfo> {
    let node = graph.get_node(node_id).ok()??;
    Some(SymbolInfo {
        id: node.id.clone(),
        label: node.label.clone(),
        kind: node.kind.to_string(),
        file_path: node
            .payload
            .get("file_path")
            .and_then(|v| v.as_str())
            .map(String::from),
        line_start: node
            .payload
            .get("line_start")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32),
        pagerank: graph.get_pagerank(&node.id),
    })
}

/// Detect potentially missing changes: symbols that share callers with changed symbols
/// but aren't in the diff.
fn detect_missing_changes(
    graph: &dyn GraphBackend,
    changed_symbols: &[String],
    already_in_diff: &HashSet<String>,
) -> Vec<MissingChange> {
    let mut missing = Vec::new();

    // For each changed symbol, find its callers. Then find what else those callers call.
    // If a sibling is called by the same callers but not in the diff, flag it.
    let mut caller_sets: HashMap<String, HashSet<String>> = HashMap::new();

    for sym_id in changed_symbols {
        let edges = graph.get_edges(sym_id).unwrap_or_default();
        let callers: HashSet<String> = edges
            .iter()
            .filter(|e| e.dst == *sym_id && e.relationship == RelationshipType::Calls)
            .map(|e| e.src.clone())
            .collect();
        if !callers.is_empty() {
            caller_sets.insert(sym_id.clone(), callers);
        }
    }

    // Find siblings: other symbols called by the same callers
    let mut sibling_counts: HashMap<String, usize> = HashMap::new();
    for callers in caller_sets.values() {
        for caller_id in callers {
            let edges = graph.get_edges(caller_id).unwrap_or_default();
            for edge in &edges {
                if edge.src == *caller_id
                    && edge.relationship == RelationshipType::Calls
                    && edge.dst.starts_with("sym:")
                    && !already_in_diff.contains(&edge.dst)
                {
                    *sibling_counts.entry(edge.dst.clone()).or_insert(0) += 1;
                }
            }
        }
    }

    // Flag siblings that share multiple callers with changed symbols
    let threshold = (changed_symbols.len() / 2).max(2);
    for (sibling, count) in &sibling_counts {
        if *count >= threshold {
            missing.push(MissingChange {
                symbol: sibling.clone(),
                reason: format!(
                    "shares {} callers with {} changed symbols",
                    count,
                    changed_symbols.len()
                ),
            });
        }
    }

    missing
}

#[cfg(test)]
#[path = "tests/review_tests.rs"]
mod tests;
