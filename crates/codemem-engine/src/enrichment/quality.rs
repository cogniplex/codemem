//! Quality stratification: categorize enrichment insights by signal strength.

use super::EnrichResult;
use crate::CodememEngine;
use codemem_core::CodememError;
use serde_json::json;

impl CodememEngine {
    /// Categorize existing enrichment insights by signal strength and adjust importance.
    ///
    /// - Noise (< 0.3): basic counts, minor observations
    /// - Signal (0.5-0.7): moderate complexity, useful patterns
    /// - Critical (0.8-1.0): high-risk findings, security issues
    pub fn enrich_quality_stratification(
        &self,
        namespace: Option<&str>,
    ) -> Result<EnrichResult, CodememError> {
        // Query all static-analysis memories
        let all_ids = self.storage.list_memory_ids().unwrap_or_default();
        let id_refs: Vec<&str> = all_ids.iter().map(|s| s.as_str()).collect();
        let memories = self
            .storage
            .get_memories_batch(&id_refs)
            .unwrap_or_default();

        let mut noise_count = 0usize;
        let mut signal_count = 0usize;
        let mut critical_count = 0usize;
        let mut reclassified = 0usize;

        for memory in &memories {
            if !memory.tags.contains(&"static-analysis".to_string()) {
                continue;
            }
            // Apply namespace filter if specified
            if let Some(ns) = namespace {
                if memory.namespace.as_deref() != Some(ns) {
                    continue;
                }
            }

            let current_importance = memory.importance;
            let content_lower = memory.content.to_lowercase();

            // Determine signal strength based on content analysis
            let is_critical = content_lower.contains("security")
                || content_lower.contains("credential")
                || content_lower.contains("sql injection")
                || content_lower.contains("high-risk")
                || content_lower.contains("critical")
                || memory.tags.iter().any(|t| t.contains("severity:critical"));

            let is_signal = content_lower.contains("complexity")
                || content_lower.contains("untested")
                || content_lower.contains("coupling")
                || content_lower.contains("co-change")
                || content_lower.contains("architecture")
                || content_lower.contains("code smell");

            let target_importance = if is_critical {
                critical_count += 1;
                current_importance.max(0.8)
            } else if is_signal {
                signal_count += 1;
                current_importance.clamp(0.5, 0.7)
            } else {
                noise_count += 1;
                current_importance.min(0.3)
            };

            // Only update if importance actually changed
            if (target_importance - current_importance).abs() > 0.01 {
                let _ = self.storage.update_memory(
                    &memory.id,
                    &memory.content,
                    Some(target_importance),
                );
                reclassified += 1;
            }
        }

        let total = noise_count + signal_count + critical_count;

        Ok(EnrichResult {
            insights_stored: 0,
            details: json!({
                "total_analyzed": total,
                "noise": noise_count,
                "signal": signal_count,
                "critical": critical_count,
                "reclassified": reclassified,
            }),
        })
    }
}
