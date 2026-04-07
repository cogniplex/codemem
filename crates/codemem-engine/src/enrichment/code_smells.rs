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
    /// Stores findings as Pattern memories with importance 0.5 (0.3 for test files).
    /// Skips non-source files (docs, markdown, etc.). Caps total smells at 50 to
    /// avoid flooding the memory store with low-value findings.
    pub fn enrich_code_smells(
        &self,
        namespace: Option<&str>,
        project_root: Option<&Path>,
    ) -> Result<EnrichResult, CodememError> {
        let all_nodes = {
            let graph = self.lock_graph()?;
            graph.get_all_nodes()
        };

        // Collect all smells with a severity score, then keep only top N.
        let mut candidates: Vec<(String, f64, Vec<String>)> = Vec::new();

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

            // Skip non-source files
            if is_non_source_file(&file_path) {
                continue;
            }

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

            // Guard against underflow: if line_end < line_start, skip
            if line_end <= line_start {
                continue;
            }
            let fn_length = line_end - line_start;

            // Sanity cap: no function is >100K lines
            if fn_length > 100_000 {
                continue;
            }

            let is_test = is_test_file(&file_path);

            // Long function (>50 lines)
            if fn_length > 50 {
                let content = format!(
                    "Code smell: Long function {} ({} lines) in {} — consider splitting",
                    node.label, fn_length, file_path
                );
                let severity = if is_test {
                    0.3
                } else {
                    0.5 + (fn_length as f64 / 500.0).min(0.3)
                };
                candidates.push((content, severity, vec![node.id.clone()]));
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
                    let severity = if is_test { 0.3 } else { 0.5 };
                    candidates.push((content, severity, vec![node.id.clone()]));
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
                        let severity = if is_test {
                            0.3
                        } else {
                            0.5 + (max_depth as f64 / 20.0).min(0.3)
                        };
                        candidates.push((content, severity, vec![node.id.clone()]));
                    }
                }
            }
        }

        // Check for long files (>500 lines) — source files only
        for node in &all_nodes {
            if node.kind != NodeKind::File {
                continue;
            }
            let file_path = &node.label;

            if is_non_source_file(file_path) {
                continue;
            }

            let line_count = file_cache
                .get(file_path.as_str())
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
                let is_test = is_test_file(file_path);
                let severity = if is_test { 0.3 } else { 0.5 };
                candidates.push((content, severity, vec![node.id.clone()]));
            }
        }

        // Sort by severity (highest first) and cap at 50 to avoid flooding
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let max_smells = self.config.enrichment.max_code_smells;
        candidates.truncate(max_smells);

        let mut smells_stored = 0;
        for (content, _severity, links) in &candidates {
            if self
                .store_pattern_memory(content, namespace, links)
                .is_some()
            {
                smells_stored += 1;
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

/// Check if a file path is a test file.
fn is_test_file(path: &str) -> bool {
    path.contains("/tests/")
        || path.contains("/test/")
        || path.contains("_test.")
        || path.contains("_tests.")
        || path.contains(".test.")
        || path.contains(".spec.")
        || path.ends_with("_test.rs")
        || path.ends_with("_tests.rs")
}

/// Check if a file is non-source (docs, config, generated, etc.)
fn is_non_source_file(path: &str) -> bool {
    let lower = path.to_lowercase();
    lower.ends_with(".md")
        || lower.ends_with(".txt")
        || lower.ends_with(".json")
        || lower.ends_with(".yaml")
        || lower.ends_with(".yml")
        || lower.ends_with(".toml")
        || lower.ends_with(".lock")
        || lower.ends_with(".svg")
        || lower.ends_with(".css")
        || lower.contains("/node_modules/")
        || lower.contains("/target/")
        || lower.contains("/dist/")
}
