//! API surface analysis: public vs private symbol counts per file.

use super::EnrichResult;
use crate::CodememEngine;
use codemem_core::{CodememError, NodeKind};
use serde_json::json;
use std::collections::HashMap;

impl CodememEngine {
    /// Analyze the public API surface of each module/package.
    ///
    /// Counts public vs private symbols per file and stores Insight memories
    /// summarizing the public API.
    pub fn enrich_api_surface(
        &self,
        namespace: Option<&str>,
    ) -> Result<EnrichResult, CodememError> {
        let all_nodes = {
            let graph = self.lock_graph()?;
            graph.get_all_nodes()
        };

        // Count public vs private symbols per file
        struct ApiStats {
            public: Vec<String>,
            private_count: usize,
        }
        let mut file_api: HashMap<String, ApiStats> = HashMap::new();

        for node in &all_nodes {
            if !matches!(
                node.kind,
                NodeKind::Function
                    | NodeKind::Method
                    | NodeKind::Class
                    | NodeKind::Interface
                    | NodeKind::Type
                    | NodeKind::Constant
            ) {
                continue;
            }
            let file_path = match node.payload.get("file_path").and_then(|v| v.as_str()) {
                Some(fp) => fp.to_string(),
                None => continue,
            };
            let visibility = node
                .payload
                .get("visibility")
                .and_then(|v| v.as_str())
                .unwrap_or("private");

            let stats = file_api.entry(file_path).or_insert(ApiStats {
                public: Vec::new(),
                private_count: 0,
            });
            if visibility == "public" {
                stats.public.push(node.label.clone());
            } else {
                stats.private_count += 1;
            }
        }

        let mut insights_stored = 0;
        let mut total_public = 0usize;
        let mut total_private = 0usize;

        for (file_path, stats) in &file_api {
            total_public += stats.public.len();
            total_private += stats.private_count;

            if stats.public.is_empty() {
                continue;
            }
            let names: Vec<&str> = stats.public.iter().take(15).map(|s| s.as_str()).collect();
            let suffix = if stats.public.len() > 15 {
                format!(" (and {} more)", stats.public.len() - 15)
            } else {
                String::new()
            };
            let ratio = stats.public.len() as f64
                / (stats.public.len() + stats.private_count).max(1) as f64;
            let content = format!(
                "API surface: {} — {} public, {} private (ratio {:.0}%). Exports: {}{}",
                file_path,
                stats.public.len(),
                stats.private_count,
                ratio * 100.0,
                names.join(", "),
                suffix
            );
            let importance = if ratio > 0.8 { 0.6 } else { 0.4 };
            if self
                .store_insight(
                    &content,
                    "api",
                    &[],
                    importance,
                    namespace,
                    &[format!("file:{file_path}")],
                )
                .is_some()
            {
                insights_stored += 1;
            }
        }

        self.save_index();

        Ok(EnrichResult {
            insights_stored,
            details: json!({
                "files_analyzed": file_api.len(),
                "total_public_symbols": total_public,
                "total_private_symbols": total_private,
                "insights_stored": insights_stored,
            }),
        })
    }
}
