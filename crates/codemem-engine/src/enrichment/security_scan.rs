//! Enhanced security scanning: hardcoded credentials, SQL concatenation, unsafe blocks.

use super::{resolve_path, EnrichResult};
use crate::CodememEngine;
use codemem_core::{CodememError, NodeKind};
use serde_json::json;
use std::path::Path;

impl CodememEngine {
    /// Scan actual file contents for security issues: hardcoded credentials,
    /// SQL concatenation, unsafe blocks, etc.
    pub fn enrich_security_scan(
        &self,
        namespace: Option<&str>,
        project_root: Option<&Path>,
    ) -> Result<EnrichResult, CodememError> {
        use std::sync::LazyLock;

        static CREDENTIAL_PATTERN: LazyLock<regex::Regex> = LazyLock::new(|| {
            regex::Regex::new(
                r#"(?i)(password|secret|api_key|apikey|token|private_key)\s*[:=]\s*["'][^"']{8,}["']"#,
            )
            .expect("valid regex")
        });

        static SQL_CONCAT_PATTERN: LazyLock<regex::Regex> = LazyLock::new(|| {
            regex::Regex::new(
                r#"(?i)(SELECT|INSERT|UPDATE|DELETE|DROP)\s+.*\+\s*[a-zA-Z_]|format!\s*\(\s*"[^"]*(?:SELECT|INSERT|UPDATE|DELETE)"#,
            )
            .expect("valid regex")
        });

        static UNSAFE_PATTERN: LazyLock<regex::Regex> =
            LazyLock::new(|| regex::Regex::new(r"unsafe\s*\{").expect("valid regex"));

        let file_nodes: Vec<String> = {
            let graph = self.lock_graph()?;
            graph
                .get_all_nodes()
                .into_iter()
                .filter(|n| n.kind == NodeKind::File)
                .map(|n| n.label.clone())
                .collect()
        };

        let mut insights_stored = 0;
        let mut files_scanned = 0;

        for file_path in &file_nodes {
            let content = match std::fs::read_to_string(resolve_path(file_path, project_root)) {
                Ok(c) => c,
                Err(_) => continue,
            };
            files_scanned += 1;

            // Check for hardcoded credentials
            if CREDENTIAL_PATTERN.is_match(&content) {
                let text = format!(
                    "Security: Potential hardcoded credential in {} — use environment variables or secrets manager",
                    file_path
                );
                if self
                    .store_insight(
                        &text,
                        "security",
                        &["severity:critical", "credentials"],
                        0.95,
                        namespace,
                        &[format!("file:{file_path}")],
                    )
                    .is_some()
                {
                    insights_stored += 1;
                }
            }

            // Check for SQL concatenation
            if SQL_CONCAT_PATTERN.is_match(&content) {
                let text = format!(
                    "Security: Potential SQL injection in {} — use parameterized queries",
                    file_path
                );
                if self
                    .store_insight(
                        &text,
                        "security",
                        &["severity:critical", "sql-injection"],
                        0.9,
                        namespace,
                        &[format!("file:{file_path}")],
                    )
                    .is_some()
                {
                    insights_stored += 1;
                }
            }

            // Check for unsafe blocks (Rust-specific)
            if file_path.ends_with(".rs") {
                let unsafe_count = UNSAFE_PATTERN.find_iter(&content).count();
                if unsafe_count > 0 {
                    let text = format!(
                        "Security: {} unsafe block(s) in {} — review for memory safety",
                        unsafe_count, file_path
                    );
                    let importance = if unsafe_count > 3 { 0.8 } else { 0.6 };
                    if self
                        .store_insight(
                            &text,
                            "security",
                            &["severity:medium", "unsafe"],
                            importance,
                            namespace,
                            &[format!("file:{file_path}")],
                        )
                        .is_some()
                    {
                        insights_stored += 1;
                    }
                }
            }
        }

        self.save_index();

        Ok(EnrichResult {
            insights_stored,
            details: json!({
                "files_scanned": files_scanned,
                "insights_stored": insights_stored,
            }),
        })
    }
}
