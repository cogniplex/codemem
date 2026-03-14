use crate::index::{CodeChunk, ResolvedEdge, Symbol};
use crate::scoring;
use crate::CodememEngine;
use codemem_core::MemoryType;

impl CodememEngine {
    // ── Contextual Enrichment ────────────────────────────────────────────────

    /// Build contextual text for a memory node.
    ///
    /// NOTE: Acquires the graph lock on each call. For batch operations,
    /// consider passing a pre-acquired guard or caching results.
    pub fn enrich_memory_text(
        &self,
        content: &str,
        memory_type: MemoryType,
        tags: &[String],
        namespace: Option<&str>,
        node_id: Option<&str>,
    ) -> String {
        let mut ctx = String::new();
        ctx.push_str(&format!("[{}]", memory_type));

        if let Some(ns) = namespace {
            ctx.push_str(&format!(" [namespace:{}]", ns));
        }

        if !tags.is_empty() {
            ctx.push_str(&format!(" [tags:{}]", tags.join(",")));
        }

        if let Some(nid) = node_id {
            let graph = match self.lock_graph() {
                Ok(g) => g,
                Err(_) => return format!("{ctx}\n{content}"),
            };
            if let Ok(edges) = graph.get_edges(nid) {
                let mut rels: Vec<String> = Vec::new();
                for edge in edges.iter().take(8) {
                    let other = if edge.src == nid {
                        &edge.dst
                    } else {
                        &edge.src
                    };
                    let label = graph
                        .get_node(other)
                        .ok()
                        .flatten()
                        .map(|n| n.label.clone())
                        .unwrap_or_else(|| other.to_string());
                    let dir = if edge.src == nid { "->" } else { "<-" };
                    rels.push(format!("{dir} {} ({})", label, edge.relationship));
                }
                if !rels.is_empty() {
                    ctx.push_str(&format!("\nRelated: {}", rels.join("; ")));
                }
            }
        }

        format!("{ctx}\n{content}")
    }

    /// Build contextual text for a code symbol.
    pub fn enrich_symbol_text(&self, sym: &Symbol, edges: &[ResolvedEdge]) -> String {
        let mut ctx = String::new();
        ctx.push_str(&format!("[{} {}]", sym.visibility, sym.kind));
        ctx.push_str(&format!(" File: {}", sym.file_path));

        if let Some(ref parent) = sym.parent {
            ctx.push_str(&format!(" Parent: {}", parent));
        }

        let related: Vec<String> = edges
            .iter()
            .filter(|e| {
                e.source_qualified_name == sym.qualified_name
                    || e.target_qualified_name == sym.qualified_name
            })
            .take(8)
            .map(|e| {
                if e.source_qualified_name == sym.qualified_name {
                    format!("-> {} ({})", e.target_qualified_name, e.relationship)
                } else {
                    format!("<- {} ({})", e.source_qualified_name, e.relationship)
                }
            })
            .collect();
        if !related.is_empty() {
            ctx.push_str(&format!("\nRelated: {}", related.join("; ")));
        }

        let mut body = format!("{}: {}", sym.qualified_name, sym.signature);
        if let Some(ref doc) = sym.doc_comment {
            body.push('\n');
            body.push_str(doc);
        }

        format!("{ctx}\n{body}")
    }

    /// Build contextual text for a code chunk before embedding.
    pub fn enrich_chunk_text(&self, chunk: &CodeChunk) -> String {
        let mut ctx = String::new();
        ctx.push_str(&format!("[chunk:{}]", chunk.node_kind));
        ctx.push_str(&format!(" File: {}", chunk.file_path));
        ctx.push_str(&format!(" Lines: {}-{}", chunk.line_start, chunk.line_end));
        if let Some(ref parent) = chunk.parent_symbol {
            ctx.push_str(&format!(" Parent: {}", parent));
        }

        let body = scoring::truncate_content(&chunk.text, 4000);

        format!("{ctx}\n{body}")
    }
}
