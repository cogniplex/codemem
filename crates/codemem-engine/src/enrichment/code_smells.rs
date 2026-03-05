//! Code smell detection: long functions, many parameters, deep nesting, long files.

use super::{resolve_path, EnrichResult};
use crate::CodememEngine;
use codemem_core::{CodememError, NodeKind};
use serde_json::json;
use std::collections::HashMap;
use std::path::Path;

impl CodememEngine {
    /// Detect common code smells: long functions (>50 lines), too many parameters (>5),
    /// deep nesting (>4 levels), and long files (>500 lines).
    ///
    /// Stores findings as Pattern memories with importance 0.5.
    pub fn enrich_code_smells(
        &self,
        namespace: Option<&str>,
        project_root: Option<&Path>,
    ) -> Result<EnrichResult, CodememError> {
        let all_nodes = {
            let graph = self.lock_graph()?;
            graph.get_all_nodes()
        };

        let mut smells_stored = 0;

        // Check functions/methods for long bodies and deep nesting
        let mut file_cache: HashMap<String, Vec<String>> = HashMap::new();

        for node in &all_nodes {
            if !matches!(node.kind, NodeKind::Function | NodeKind::Method) {
                continue;
            }
            let file_path = match node.payload.get("file_path").and_then(|v| v.as_str()) {
                Some(fp) => fp.to_string(),
                None => continue,
            };
            let line_start = node
                .payload
                .get("line_start")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
            let line_end = node
                .payload
                .get("line_end")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;

            let fn_length = line_end.saturating_sub(line_start);

            // Long function (>50 lines)
            if fn_length > 50 {
                let content = format!(
                    "Code smell: Long function {} ({} lines) in {} — consider splitting",
                    node.label, fn_length, file_path
                );
                if self
                    .store_pattern_memory(&content, namespace, std::slice::from_ref(&node.id))
                    .is_some()
                {
                    smells_stored += 1;
                }
            }

            // Check parameter count from signature
            let signature = node
                .payload
                .get("signature")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if let Some(params_str) = signature
                .split('(')
                .nth(1)
                .and_then(|s| s.split(')').next())
            {
                let param_count = if params_str.trim().is_empty() {
                    0
                } else {
                    params_str.split(',').count()
                };
                if param_count > 5 {
                    let content = format!(
                        "Code smell: {} has {} parameters in {} — consider using a struct",
                        node.label, param_count, file_path
                    );
                    if self
                        .store_pattern_memory(&content, namespace, std::slice::from_ref(&node.id))
                        .is_some()
                    {
                        smells_stored += 1;
                    }
                }
            }

            // Check nesting depth
            if fn_length > 0 {
                let lines = file_cache.entry(file_path.clone()).or_insert_with(|| {
                    std::fs::read_to_string(resolve_path(&file_path, project_root))
                        .unwrap_or_default()
                        .lines()
                        .map(String::from)
                        .collect()
                });

                let start = line_start.saturating_sub(1);
                let end = line_end.min(lines.len());
                if start < end {
                    let mut max_depth = 0usize;
                    let mut depth = 0usize;
                    for line in &lines[start..end] {
                        for ch in line.chars() {
                            match ch {
                                '{' => {
                                    depth += 1;
                                    max_depth = max_depth.max(depth);
                                }
                                '}' => depth = depth.saturating_sub(1),
                                _ => {}
                            }
                        }
                    }
                    if max_depth > 4 {
                        let content = format!(
                            "Code smell: Deep nesting ({} levels) in {} in {} — consider extracting",
                            max_depth, node.label, file_path
                        );
                        if self
                            .store_pattern_memory(
                                &content,
                                namespace,
                                std::slice::from_ref(&node.id),
                            )
                            .is_some()
                        {
                            smells_stored += 1;
                        }
                    }
                }
            }
        }

        // Check for long files (>500 lines)
        for node in &all_nodes {
            if node.kind != NodeKind::File {
                continue;
            }
            let file_path = &node.label;
            let line_count = file_cache
                .get(file_path)
                .map(|lines| lines.len())
                .unwrap_or_else(|| {
                    std::fs::read_to_string(resolve_path(file_path, project_root))
                        .map(|s| s.lines().count())
                        .unwrap_or(0)
                });
            if line_count > 500 {
                let content = format!(
                    "Code smell: Long file {} ({} lines) — consider splitting into modules",
                    file_path, line_count
                );
                if self
                    .store_pattern_memory(&content, namespace, std::slice::from_ref(&node.id))
                    .is_some()
                {
                    smells_stored += 1;
                }
            }
        }

        self.save_index();

        Ok(EnrichResult {
            insights_stored: smells_stored,
            details: json!({
                "smells_detected": smells_stored,
            }),
        })
    }
}
