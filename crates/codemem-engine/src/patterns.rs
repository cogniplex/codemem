//! Cross-session pattern detection for Codemem.
//!
//! Analyzes stored memories to detect recurring patterns like repeated searches,
//! file hotspots, decision chains, and tool preferences across sessions.

use codemem_core::{CodememError, DetectedPattern, PatternType, StorageBackend};

/// Compute log-scaled confidence from frequency, total sessions, and a recency factor.
///
/// Uses `ln(frequency) / ln(total_sessions)` as a base, scaled by `recency_factor`.
/// Returns 0.0 for zero inputs, clamped to [0.0, 1.0].
fn compute_confidence(frequency: usize, total_sessions: usize, recency_factor: f64) -> f64 {
    if frequency == 0 || total_sessions == 0 {
        return 0.0;
    }
    let base = (frequency as f64).ln() / (total_sessions as f64).ln().max(1.0);
    (base * recency_factor).min(1.0)
}

/// Detect all patterns in the memory store.
///
/// Runs multiple detectors and returns all patterns found, sorted by confidence
/// descending. The `min_frequency` parameter controls the threshold for how many
/// times a pattern must appear before it is flagged. `total_sessions` is used for
/// log-scaled confidence computation.
pub fn detect_patterns(
    storage: &dyn StorageBackend,
    namespace: Option<&str>,
    min_frequency: usize,
    total_sessions: usize,
) -> Result<Vec<DetectedPattern>, CodememError> {
    let mut patterns = Vec::new();

    patterns.extend(detect_repeated_searches(
        storage,
        namespace,
        min_frequency,
        total_sessions,
    )?);
    patterns.extend(detect_file_hotspots(
        storage,
        namespace,
        min_frequency,
        total_sessions,
    )?);
    patterns.extend(detect_decision_chains(
        storage,
        namespace,
        min_frequency,
        total_sessions,
    )?);
    patterns.extend(detect_tool_preferences(storage, namespace)?);

    // Sort by confidence descending
    patterns.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(patterns)
}

/// Detect repeated search patterns (Grep/Glob queries used multiple times).
fn detect_repeated_searches(
    storage: &dyn StorageBackend,
    namespace: Option<&str>,
    min_frequency: usize,
    total_sessions: usize,
) -> Result<Vec<DetectedPattern>, CodememError> {
    let results = storage.get_repeated_searches(min_frequency, namespace)?;

    Ok(results
        .into_iter()
        .map(|(pattern, count, memory_ids)| DetectedPattern {
            pattern_type: PatternType::RepeatedSearch,
            description: format!(
                "Search pattern '{}' used {} times across sessions",
                pattern, count
            ),
            frequency: count,
            related_memories: memory_ids,
            confidence: compute_confidence(count, total_sessions, 1.0),
        })
        .collect())
}

/// Detect file hotspots (files accessed frequently via Read/Edit/Write).
fn detect_file_hotspots(
    storage: &dyn StorageBackend,
    namespace: Option<&str>,
    min_frequency: usize,
    total_sessions: usize,
) -> Result<Vec<DetectedPattern>, CodememError> {
    let results = storage.get_file_hotspots(min_frequency, namespace)?;

    Ok(results
        .into_iter()
        .map(|(file_path, count, memory_ids)| DetectedPattern {
            pattern_type: PatternType::FileHotspot,
            description: format!(
                "File '{}' accessed {} times across sessions",
                file_path, count
            ),
            frequency: count,
            related_memories: memory_ids,
            confidence: compute_confidence(count, total_sessions, 1.0),
        })
        .collect())
}

/// Detect decision chains: files modified multiple times via Edit/Write over time.
fn detect_decision_chains(
    storage: &dyn StorageBackend,
    namespace: Option<&str>,
    min_frequency: usize,
    total_sessions: usize,
) -> Result<Vec<DetectedPattern>, CodememError> {
    let results = storage.get_decision_chains(min_frequency, namespace)?;

    Ok(results
        .into_iter()
        .map(|(file_path, count, memory_ids)| DetectedPattern {
            pattern_type: PatternType::DecisionChain,
            description: format!(
                "File '{}' modified {} times, forming a decision chain",
                file_path, count
            ),
            frequency: count,
            related_memories: memory_ids,
            confidence: compute_confidence(count, total_sessions, 1.0),
        })
        .collect())
}

/// Detect tool usage preferences by analyzing the distribution of tool usage.
fn detect_tool_preferences(
    storage: &dyn StorageBackend,
    namespace: Option<&str>,
) -> Result<Vec<DetectedPattern>, CodememError> {
    let tool_entries = storage.get_tool_usage_stats(namespace)?;

    if tool_entries.len() < 2 {
        return Ok(vec![]);
    }

    let total: usize = tool_entries.iter().map(|(_, c)| c).sum();
    if total == 0 {
        return Ok(vec![]);
    }

    Ok(tool_entries
        .into_iter()
        .map(|(tool, count)| {
            let pct = (count as f64 / total as f64 * 100.0) as usize;
            DetectedPattern {
                pattern_type: PatternType::ToolPreference,
                description: format!(
                    "Tool '{}' used {} times ({}% of all tool usage)",
                    tool, count, pct
                ),
                frequency: count,
                related_memories: vec![],
                confidence: count as f64 / total as f64,
            }
        })
        .collect())
}

/// Generate human-readable pattern insights as markdown.
pub fn generate_insights(patterns: &[DetectedPattern]) -> String {
    if patterns.is_empty() {
        return "No patterns detected yet. Keep using Codemem to build up session history."
            .to_string();
    }

    let mut md = String::from("## Cross-Session Pattern Insights\n\n");

    // File Hotspots
    let hotspots: Vec<_> = patterns
        .iter()
        .filter(|p| p.pattern_type == PatternType::FileHotspot)
        .collect();
    if !hotspots.is_empty() {
        md.push_str("### File Hotspots\n");
        md.push_str("Files you keep coming back to across sessions:\n\n");
        for p in hotspots.iter().take(10) {
            md.push_str(&format!(
                "- {} (confidence: {:.0}%)\n",
                p.description,
                p.confidence * 100.0
            ));
        }
        md.push('\n');
    }

    // Repeated Searches
    let searches: Vec<_> = patterns
        .iter()
        .filter(|p| p.pattern_type == PatternType::RepeatedSearch)
        .collect();
    if !searches.is_empty() {
        md.push_str("### Repeated Searches\n");
        md.push_str(
            "Search patterns you use repeatedly (consider creating a memory for these):\n\n",
        );
        for p in searches.iter().take(10) {
            md.push_str(&format!(
                "- {} (confidence: {:.0}%)\n",
                p.description,
                p.confidence * 100.0
            ));
        }
        md.push('\n');
    }

    // Decision Chains
    let chains: Vec<_> = patterns
        .iter()
        .filter(|p| p.pattern_type == PatternType::DecisionChain)
        .collect();
    if !chains.is_empty() {
        md.push_str("### Decision Chains\n");
        md.push_str("Files modified multiple times, suggesting evolving decisions:\n\n");
        for p in chains.iter().take(10) {
            md.push_str(&format!(
                "- {} (confidence: {:.0}%)\n",
                p.description,
                p.confidence * 100.0
            ));
        }
        md.push('\n');
    }

    // Tool Preferences
    let prefs: Vec<_> = patterns
        .iter()
        .filter(|p| p.pattern_type == PatternType::ToolPreference)
        .collect();
    if !prefs.is_empty() {
        md.push_str("### Tool Usage Distribution\n");
        for p in &prefs {
            md.push_str(&format!("- {}\n", p.description));
        }
        md.push('\n');
    }

    // Summary
    md.push_str(&format!(
        "**Total patterns detected:** {}\n",
        patterns.len()
    ));

    md
}

#[cfg(test)]
#[path = "tests/patterns_tests.rs"]
mod tests;
