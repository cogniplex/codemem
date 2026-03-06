use super::ConsolidationResult;
use crate::CodememEngine;
use codemem_core::CodememError;
use serde_json::json;

impl CodememEngine {
    /// Consolidate decay: power-law decay that rewards access frequency.
    pub fn consolidate_decay(
        &self,
        threshold_days: Option<i64>,
    ) -> Result<ConsolidationResult, CodememError> {
        let threshold_days = threshold_days.unwrap_or(30);
        let now = chrono::Utc::now();
        let threshold_ts = (now - chrono::Duration::days(threshold_days)).timestamp();

        let stale = self.storage.get_stale_memories_for_decay(threshold_ts)?;

        if stale.is_empty() {
            if let Err(e) = self.storage.insert_consolidation_log("decay", 0) {
                tracing::warn!("Failed to log decay consolidation: {e}");
            }
            return Ok(ConsolidationResult {
                cycle: "decay".to_string(),
                affected: 0,
                details: json!({
                    "threshold_days": threshold_days,
                }),
            });
        }

        // Compute power-law decay: importance * 0.9^(days_since/30) * (1 + log2(max(access_count,1)) * 0.1)
        let now_ts = now.timestamp();
        let updates: Vec<(String, f64)> = stale
            .iter()
            .map(|(id, importance, access_count, last_accessed_at)| {
                let days_since = (now_ts - last_accessed_at) as f64 / 86400.0;
                let time_decay = 0.9_f64.powf(days_since / 30.0);
                let access_boost = 1.0 + ((*access_count).max(1) as f64).log2() * 0.1;
                let new_importance = (importance * time_decay * access_boost).clamp(0.0, 1.0);
                (id.clone(), new_importance)
            })
            .collect();

        let affected = self.storage.batch_update_importance(&updates)?;

        if let Err(e) = self.storage.insert_consolidation_log("decay", affected) {
            tracing::warn!("Failed to log decay consolidation: {e}");
        }

        Ok(ConsolidationResult {
            cycle: "decay".to_string(),
            affected,
            details: json!({
                "threshold_days": threshold_days,
                "algorithm": "power_law",
            }),
        })
    }
}
