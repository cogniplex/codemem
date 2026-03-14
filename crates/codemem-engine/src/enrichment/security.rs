//! Security analysis: sensitive files, endpoints, security functions.

use super::EnrichResult;
use crate::CodememEngine;
use codemem_core::{CodememError, NodeKind};
use serde_json::json;

impl CodememEngine {
    /// Enrich the graph with security analysis: sensitive files, endpoints, security functions.
    pub fn enrich_security(&self, namespace: Option<&str>) -> Result<EnrichResult, CodememError> {
        use std::sync::LazyLock;

        static SECURITY_PATTERN: LazyLock<regex::Regex> = LazyLock::new(|| {
            regex::Regex::new(
                r"(?i)(auth|secret|key|password|token|credential|\.env|private|encrypt|decrypt|cert|permission|rbac|oauth|jwt|session|cookie|csrf|xss|injection|sanitize)"
            ).expect("valid regex")
        });

        static SECURITY_FN_PATTERN: LazyLock<regex::Regex> = LazyLock::new(|| {
            regex::Regex::new(
                r"(?i)(hash|verify|sign|encrypt|authenticate|authorize|validate_token|check_permission)"
            ).expect("valid regex")
        });

        let security_pattern = &*SECURITY_PATTERN;
        let security_fn_pattern = &*SECURITY_FN_PATTERN;

        let graph = self.lock_graph()?;
        let all_nodes = graph.get_all_nodes();
        drop(graph);

        let mut sensitive_files: Vec<String> = Vec::new();
        let mut endpoints: Vec<String> = Vec::new();
        let mut security_functions: Vec<(String, String, String)> = Vec::new();
        let mut nodes_to_annotate: Vec<(String, Vec<String>)> = Vec::new();

        for node in &all_nodes {
            match node.kind {
                NodeKind::File => {
                    if security_pattern.is_match(&node.label) {
                        sensitive_files.push(node.label.clone());
                        nodes_to_annotate.push((
                            node.id.clone(),
                            vec!["sensitive".into(), "auth_related".into()],
                        ));
                    }
                }
                NodeKind::Endpoint => {
                    endpoints.push(node.label.clone());
                    nodes_to_annotate.push((node.id.clone(), vec!["exposed_endpoint".into()]));
                }
                NodeKind::Function | NodeKind::Method => {
                    if security_fn_pattern.is_match(&node.label) {
                        let file = node
                            .payload
                            .get("file_path")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_string();
                        security_functions.push((node.label.clone(), file, node.id.clone()));
                        nodes_to_annotate.push((node.id.clone(), vec!["security_function".into()]));
                    }
                }
                _ => {}
            }
        }

        // Annotate nodes with security flags
        {
            let mut graph = self.lock_graph()?;
            for (node_id, flags) in &nodes_to_annotate {
                if let Ok(Some(mut node)) = graph.get_node(node_id) {
                    node.payload.insert("security_flags".into(), json!(flags));
                    let _ = graph.add_node(node);
                }
            }
        }

        // Store insights
        let mut insights_stored = 0;

        for file_path in &sensitive_files {
            let content = format!(
                "Sensitive file: {} — contains security-critical code (auth/credentials)",
                file_path
            );
            if self
                .store_insight(
                    &content,
                    "security",
                    &["severity:high"],
                    0.8,
                    namespace,
                    &[format!("file:{file_path}")],
                )
                .is_some()
            {
                insights_stored += 1;
            }
        }

        if !endpoints.is_empty() {
            let content = format!(
                "{} exposed API endpoints detected — review access controls",
                endpoints.len()
            );
            if self
                .store_insight(
                    &content,
                    "security",
                    &["severity:medium", "endpoints"],
                    0.7,
                    namespace,
                    &[],
                )
                .is_some()
            {
                insights_stored += 1;
            }
        }

        for (name, file, node_id) in &security_functions {
            let content = format!(
                "Security-critical function: {} in {} — ensure proper testing",
                name, file
            );
            if self
                .store_insight(
                    &content,
                    "security",
                    &["severity:medium"],
                    0.6,
                    namespace,
                    std::slice::from_ref(node_id),
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
                "sensitive_file_count": sensitive_files.len(),
                "endpoint_count": endpoints.len(),
                "security_function_count": security_functions.len(),
                "insights_stored": insights_stored,
            }),
        })
    }
}
