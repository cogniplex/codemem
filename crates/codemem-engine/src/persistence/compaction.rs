use crate::CodememEngine;
use codemem_core::{GraphBackend, GraphNode, NodeKind, RelationshipType};
use std::collections::{HashMap, HashSet};

// ── Chunk compaction weights ────────────────────────────────────────────────
// Normal weights (when memories exist):
const CHUNK_W_CENTRALITY: f64 = 0.3;
const CHUNK_W_PARENT: f64 = 0.2;
const CHUNK_W_MEMORY: f64 = 0.3;
const CHUNK_W_SIZE: f64 = 0.2;
// Cold-start weights (no memories → redistribute memory weight):
const CHUNK_COLD_W_CENTRALITY: f64 = 0.4;
const CHUNK_COLD_W_PARENT: f64 = 0.3;
const CHUNK_COLD_W_SIZE: f64 = 0.3;

// ── Symbol compaction weights ───────────────────────────────────────────────
// Normal weights:
const SYM_W_CALLS: f64 = 0.30;
const SYM_W_VISIBILITY: f64 = 0.20;
const SYM_W_KIND: f64 = 0.15;
const SYM_W_MEMORY: f64 = 0.20;
const SYM_W_SIZE: f64 = 0.15;
// Cold-start weights:
const SYM_COLD_W_CALLS: f64 = 0.40;
const SYM_COLD_W_VISIBILITY: f64 = 0.20;
const SYM_COLD_W_KIND: f64 = 0.15;
const SYM_COLD_W_SIZE: f64 = 0.25;

/// Check whether a graph node has any edge linking it to a memory node
/// (i.e. an edge whose other endpoint is not a code-structural ID).
fn has_memory_link_edge(graph: &dyn GraphBackend, node_id: &str) -> bool {
    graph
        .get_edges(node_id)
        .map(|edges| {
            edges.iter().any(|e| {
                let other = if e.src == node_id { &e.dst } else { &e.src };
                !other.starts_with("sym:")
                    && !other.starts_with("file:")
                    && !other.starts_with("chunk:")
                    && !other.starts_with("pkg:")
                    && !other.starts_with("contains:")
                    && !other.starts_with("ref:")
            })
        })
        .unwrap_or(false)
}

impl CodememEngine {
    // ── Graph Compaction ────────────────────────────────────────────────────

    /// Compact chunk and symbol graph-nodes after indexing.
    /// Returns (chunks_pruned, symbols_pruned).
    pub fn compact_graph(&self, seen_files: &HashSet<String>) -> (usize, usize) {
        let mut graph = match self.lock_graph() {
            Ok(g) => g,
            Err(_) => return (0, 0),
        };

        // Fetch all nodes once and share between both compaction passes.
        let all_nodes = graph.get_all_nodes();
        let chunks_pruned = self.compact_chunks(seen_files, &mut **graph, &all_nodes);
        let symbols_pruned = self.compact_symbols(seen_files, &mut **graph, &all_nodes);

        if chunks_pruned > 0 || symbols_pruned > 0 {
            // compute_centrality: updates node.centrality with degree centrality.
            // recompute_centrality: caches PageRank + betweenness for hybrid scoring.
            // Both are needed — they populate different data used by different scoring paths.
            graph.compute_centrality();
            graph.recompute_centrality();
        }

        (chunks_pruned, symbols_pruned)
    }

    /// Pass 1: Score and prune low-value chunks, transferring line ranges to parent symbols.
    ///
    /// Scoring weights adjust on cold start: when no memories exist yet, the
    /// `memory_link_score` weight (normally 0.3) is redistributed to the other
    /// factors so compaction still produces meaningful rankings.
    fn compact_chunks(
        &self,
        seen_files: &HashSet<String>,
        graph: &mut dyn codemem_core::GraphBackend,
        all_nodes: &[GraphNode],
    ) -> usize {
        let max_chunks_per_file = self.config.chunking.max_retained_chunks_per_file;
        let chunk_score_threshold = self.config.chunking.min_chunk_score_threshold;

        // Cold-start detection: if no memories exist, memory_link_score is always 0
        // and its weight should be redistributed to other factors.
        let has_memories = self
            .storage
            .list_memory_ids()
            .map(|ids| !ids.is_empty())
            .unwrap_or(false);
        let (w_centrality, w_parent, w_memory, w_size) = if has_memories {
            (
                CHUNK_W_CENTRALITY,
                CHUNK_W_PARENT,
                CHUNK_W_MEMORY,
                CHUNK_W_SIZE,
            )
        } else {
            // Redistribute memory_link weight when no memories exist
            (
                CHUNK_COLD_W_CENTRALITY,
                CHUNK_COLD_W_PARENT,
                0.0,
                CHUNK_COLD_W_SIZE,
            )
        };

        let mut chunks_by_file: HashMap<String, Vec<(String, f64)>> = HashMap::new();

        let mut max_degree: f64 = 1.0;
        let mut max_non_ws: f64 = 1.0;

        let chunk_nodes: Vec<&GraphNode> = all_nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Chunk)
            .collect();

        for node in &chunk_nodes {
            let degree = graph
                .get_edges(&node.id)
                .map(|edges| edges.len() as f64)
                .unwrap_or(0.0);
            max_degree = max_degree.max(degree);

            let non_ws = node
                .payload
                .get("non_ws_chars")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            max_non_ws = max_non_ws.max(non_ws);
        }

        for node in &chunk_nodes {
            let file_path = match node.payload.get("file_path").and_then(|v| v.as_str()) {
                Some(fp) => fp.to_string(),
                None => continue,
            };
            if !seen_files.contains(&file_path) {
                continue;
            }

            let degree = graph
                .get_edges(&node.id)
                .map(|edges| edges.len() as f64)
                .unwrap_or(0.0);
            let centrality_rank = degree / max_degree;

            let has_symbol_parent = if node.payload.contains_key("parent_symbol") {
                1.0
            } else {
                0.0
            };

            let memory_link_score = if has_memories && has_memory_link_edge(graph, &node.id) {
                1.0
            } else {
                0.0
            };

            let non_ws = node
                .payload
                .get("non_ws_chars")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let non_ws_rank = non_ws / max_non_ws;

            let chunk_score = centrality_rank * w_centrality
                + has_symbol_parent * w_parent
                + memory_link_score * w_memory
                + non_ws_rank * w_size;

            chunks_by_file
                .entry(file_path)
                .or_default()
                .push((node.id.clone(), chunk_score));
        }

        let mut symbol_count_by_file: HashMap<String, usize> = HashMap::new();
        for node in all_nodes {
            if matches!(
                node.kind,
                NodeKind::Function
                    | NodeKind::Method
                    | NodeKind::Class
                    | NodeKind::Interface
                    | NodeKind::Type
                    | NodeKind::Constant
                    | NodeKind::Test
            ) {
                if let Some(fp) = node.payload.get("file_path").and_then(|v| v.as_str()) {
                    *symbol_count_by_file.entry(fp.to_string()).or_default() += 1;
                }
            }
        }

        let mut chunks_pruned = 0usize;
        for (file_path, mut chunks) in chunks_by_file {
            chunks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let sym_count = symbol_count_by_file.get(&file_path).copied().unwrap_or(0);
            let k = max_chunks_per_file.min(chunks.len()).max(3.max(sym_count));

            for (i, (chunk_id, score)) in chunks.iter().enumerate() {
                // Prune aggressively: remove if beyond the top-k slots OR below the
                // quality threshold. Using || ensures both caps are enforced independently
                // (keep at most k chunks, and never keep any chunk below threshold).
                if i >= k || *score < chunk_score_threshold {
                    self.transfer_chunk_ranges_to_parent(graph, chunk_id);

                    if let Err(e) = self.storage.delete_graph_edges_for_node(chunk_id) {
                        tracing::warn!("Failed to delete graph edges for chunk {chunk_id}: {e}");
                    }
                    if let Err(e) = self.storage.delete_graph_node(chunk_id) {
                        tracing::warn!("Failed to delete graph node for chunk {chunk_id}: {e}");
                    }
                    if let Err(e) = graph.remove_node(chunk_id) {
                        tracing::warn!("Failed to remove chunk {chunk_id} from graph: {e}");
                    }
                    chunks_pruned += 1;
                }
            }
        }

        chunks_pruned
    }

    /// When pruning a chunk, transfer its line range to the parent symbol node.
    fn transfer_chunk_ranges_to_parent(
        &self,
        graph: &mut dyn codemem_core::GraphBackend,
        chunk_id: &str,
    ) {
        if let Ok(Some(chunk_node)) = graph.get_node(chunk_id) {
            if let Some(parent_sym) = chunk_node
                .payload
                .get("parent_symbol")
                .and_then(|v| v.as_str())
            {
                let parent_id = format!("sym:{parent_sym}");
                if let Ok(Some(mut parent_node)) = graph.get_node(&parent_id) {
                    let line_start = chunk_node
                        .payload
                        .get("line_start")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let line_end = chunk_node
                        .payload
                        .get("line_end")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let ranges = parent_node
                        .payload
                        .entry("covered_ranges".to_string())
                        .or_insert_with(|| serde_json::json!([]));
                    if let Some(arr) = ranges.as_array_mut() {
                        arr.push(serde_json::json!([line_start, line_end]));
                    }
                    let count = parent_node
                        .payload
                        .entry("pruned_chunk_count".to_string())
                        .or_insert_with(|| serde_json::json!(0));
                    if let Some(n) = count.as_u64() {
                        *count = serde_json::json!(n + 1);
                    }
                    let _ = self.storage.insert_graph_node(&parent_node);
                    let _ = graph.add_node(parent_node);
                }
            }
        }
    }

    /// Pass 2: Score and prune low-value symbol nodes, transferring ranges to parent files.
    ///
    /// Like chunk compaction, scoring weights adjust on cold start: when no memories
    /// exist yet, the `memory_link_val` weight (normally 0.20) is redistributed to
    /// call connectivity and code size factors.
    fn compact_symbols(
        &self,
        seen_files: &HashSet<String>,
        graph: &mut dyn codemem_core::GraphBackend,
        all_nodes: &[GraphNode],
    ) -> usize {
        let max_syms_per_file = self.config.chunking.max_retained_symbols_per_file;
        let sym_score_threshold = self.config.chunking.min_symbol_score_threshold;

        // Cold-start: redistribute memory_link weight when no memories exist
        let has_memories = self
            .storage
            .list_memory_ids()
            .map(|ids| !ids.is_empty())
            .unwrap_or(false);
        let (w_calls, w_vis, w_kind, w_mem, w_size) = if has_memories {
            (
                SYM_W_CALLS,
                SYM_W_VISIBILITY,
                SYM_W_KIND,
                SYM_W_MEMORY,
                SYM_W_SIZE,
            )
        } else {
            // Redistribute memory weight to calls and code size
            (
                SYM_COLD_W_CALLS,
                SYM_COLD_W_VISIBILITY,
                SYM_COLD_W_KIND,
                0.0,
                SYM_COLD_W_SIZE,
            )
        };

        // Only compact ast-grep symbols. SCIP-sourced symbols (both explicit and
        // synthetic containment nodes) should not be pruned by heuristic scoring.
        let sym_nodes: Vec<&GraphNode> = all_nodes
            .iter()
            .filter(|n| {
                n.id.starts_with("sym:")
                    && !matches!(
                        n.payload.get("source").and_then(|v| v.as_str()),
                        Some("scip" | "scip-synthetic")
                    )
            })
            .collect();

        let mut max_calls_degree: f64 = 1.0;
        let mut max_code_size: f64 = 1.0;

        for node in &sym_nodes {
            let calls_degree = graph
                .get_edges(&node.id)
                .map(|edges| {
                    edges
                        .iter()
                        .filter(|e| e.relationship == RelationshipType::Calls)
                        .count() as f64
                })
                .unwrap_or(0.0);
            max_calls_degree = max_calls_degree.max(calls_degree);

            let line_start = node
                .payload
                .get("line_start")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let line_end = node
                .payload
                .get("line_end")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let code_size = (line_end - line_start).max(0.0);
            max_code_size = max_code_size.max(code_size);
        }

        let mut syms_by_file: HashMap<String, Vec<(String, f64, bool, bool)>> = HashMap::new();

        for node in &sym_nodes {
            let file_path = match node.payload.get("file_path").and_then(|v| v.as_str()) {
                Some(fp) => fp.to_string(),
                None => continue,
            };
            if !seen_files.contains(&file_path) {
                continue;
            }

            let calls_degree = graph
                .get_edges(&node.id)
                .map(|edges| {
                    edges
                        .iter()
                        .filter(|e| e.relationship == RelationshipType::Calls)
                        .count() as f64
                })
                .unwrap_or(0.0);
            let call_connectivity = calls_degree / max_calls_degree;

            let visibility_score = match node
                .payload
                .get("visibility")
                .and_then(|v| v.as_str())
                .unwrap_or("private")
            {
                "public" => 1.0,
                "crate" => 0.5,
                _ => 0.0,
            };

            let kind_score = match node.kind {
                NodeKind::Class | NodeKind::Interface => 1.0,
                NodeKind::Module => 1.0,
                NodeKind::Function | NodeKind::Method => 0.6,
                NodeKind::Test => 0.3,
                NodeKind::Constant => 0.1,
                _ => 0.5,
            };

            let mem_linked = has_memories && has_memory_link_edge(graph, &node.id);
            let memory_link_val = if mem_linked { 1.0 } else { 0.0 };

            let line_start = node
                .payload
                .get("line_start")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let line_end = node
                .payload
                .get("line_end")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let code_size = (line_end - line_start).max(0.0);
            let code_size_rank = code_size / max_code_size;

            let symbol_score = call_connectivity * w_calls
                + visibility_score * w_vis
                + kind_score * w_kind
                + memory_link_val * w_mem
                + code_size_rank * w_size;

            let is_structural = matches!(
                node.kind,
                NodeKind::Class | NodeKind::Interface | NodeKind::Module
            );

            syms_by_file.entry(file_path).or_default().push((
                node.id.clone(),
                symbol_score,
                is_structural,
                mem_linked,
            ));
        }

        let mut symbols_pruned = 0usize;
        for (_file_path, mut syms) in syms_by_file {
            syms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let public_count = syms
                .iter()
                .filter(|(id, ..)| {
                    graph
                        .get_node(id)
                        .ok()
                        .flatten()
                        .and_then(|n| {
                            n.payload
                                .get("visibility")
                                .and_then(|v| v.as_str())
                                .map(|v| v == "public")
                        })
                        .unwrap_or(false)
                })
                .count();
            let k = max_syms_per_file.max(public_count);

            for (i, (sym_id, score, is_structural, mem_linked)) in syms.iter().enumerate() {
                if *is_structural || *mem_linked {
                    continue;
                }
                if i < k && *score >= sym_score_threshold {
                    continue;
                }

                self.transfer_symbol_ranges_to_file(graph, sym_id);

                if let Err(e) = self.storage.delete_graph_edges_for_node(sym_id) {
                    tracing::warn!("Failed to delete graph edges for symbol {sym_id}: {e}");
                }
                if let Err(e) = self.storage.delete_graph_node(sym_id) {
                    tracing::warn!("Failed to delete graph node for symbol {sym_id}: {e}");
                }
                if let Err(e) = graph.remove_node(sym_id) {
                    tracing::warn!("Failed to remove symbol {sym_id} from graph: {e}");
                }
                symbols_pruned += 1;
            }
        }

        symbols_pruned
    }

    /// When pruning a symbol, transfer its line range to the parent file node.
    fn transfer_symbol_ranges_to_file(
        &self,
        graph: &mut dyn codemem_core::GraphBackend,
        sym_id: &str,
    ) {
        if let Ok(Some(sym_node)) = graph.get_node(sym_id) {
            if let Some(fp) = sym_node.payload.get("file_path").and_then(|v| v.as_str()) {
                let file_id = format!("file:{fp}");
                if let Ok(Some(mut file_node)) = graph.get_node(&file_id) {
                    let line_start = sym_node
                        .payload
                        .get("line_start")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let line_end = sym_node
                        .payload
                        .get("line_end")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    if line_end > line_start {
                        let ranges = file_node
                            .payload
                            .entry("pruned_symbol_ranges".to_string())
                            .or_insert_with(|| serde_json::json!([]));
                        if let Some(arr) = ranges.as_array_mut() {
                            arr.push(serde_json::json!([line_start, line_end]));
                        }
                        let _ = self.storage.insert_graph_node(&file_node);
                        let _ = graph.add_node(file_node);
                    }
                }
            }
        }
    }
}
