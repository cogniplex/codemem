//! Cross-session pattern detection for Codemem.
//!
//! Analyzes stored memories to detect recurring patterns like repeated searches,
//! file hotspots, decision chains, and tool preferences across sessions.

use codemem_core::{CodememError, DetectedPattern, PatternType, StorageBackend};

/// Detect all patterns in the memory store.
///
/// Runs multiple detectors and returns all patterns found, sorted by confidence
/// descending. The `min_frequency` parameter controls the threshold for how many
/// times a pattern must appear before it is flagged.
pub fn detect_patterns(
    storage: &dyn StorageBackend,
    namespace: Option<&str>,
    min_frequency: usize,
) -> Result<Vec<DetectedPattern>, CodememError> {
    let mut patterns = Vec::new();

    patterns.extend(detect_repeated_searches(storage, namespace, min_frequency)?);
    patterns.extend(detect_file_hotspots(storage, namespace, min_frequency)?);
    patterns.extend(detect_decision_chains(storage, namespace, min_frequency)?);
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
            confidence: (count as f64 / 10.0).min(1.0),
        })
        .collect())
}

/// Detect file hotspots (files accessed frequently via Read/Edit/Write).
fn detect_file_hotspots(
    storage: &dyn StorageBackend,
    namespace: Option<&str>,
    min_frequency: usize,
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
            confidence: (count as f64 / 10.0).min(1.0),
        })
        .collect())
}

/// Detect decision chains: files modified multiple times via Edit/Write over time.
fn detect_decision_chains(
    storage: &dyn StorageBackend,
    namespace: Option<&str>,
    min_frequency: usize,
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
            confidence: (count as f64 / 8.0).min(1.0),
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
mod tests {
    use super::*;
    use codemem_core::MemoryNode;
    use codemem_core::MemoryType;
    use codemem_storage::Storage;
    use std::collections::HashMap;

    fn make_memory(content: &str, tool: &str, extra_metadata: Vec<(&str, &str)>) -> MemoryNode {
        let now = chrono::Utc::now();
        let mut metadata = HashMap::new();
        metadata.insert(
            "tool".to_string(),
            serde_json::Value::String(tool.to_string()),
        );
        for (k, v) in extra_metadata {
            metadata.insert(k.to_string(), serde_json::Value::String(v.to_string()));
        }
        MemoryNode {
            id: uuid::Uuid::new_v4().to_string(),
            content: content.to_string(),
            memory_type: MemoryType::Context,
            importance: 0.5,
            confidence: 1.0,
            access_count: 0,
            content_hash: codemem_storage::Storage::content_hash(content),
            tags: vec![],
            metadata,
            namespace: None,
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
        }
    }

    #[test]
    fn detect_patterns_empty_db() {
        let storage = Storage::open_in_memory().unwrap();
        let patterns = detect_patterns(&storage, None, 2).unwrap();
        assert!(patterns.is_empty());
    }

    #[test]
    fn detect_repeated_search_patterns() {
        let storage = Storage::open_in_memory().unwrap();

        // Store 3 Grep searches for "error handling"
        for i in 0..3 {
            let mem = make_memory(
                &format!("grep for error handling {i}"),
                "Grep",
                vec![("pattern", "error handling")],
            );
            storage.insert_memory(&mem).unwrap();
        }

        // Store 1 Glob search (below threshold)
        let mem = make_memory("glob for rs files", "Glob", vec![("pattern", "*.rs")]);
        storage.insert_memory(&mem).unwrap();

        let patterns = detect_patterns(&storage, None, 2).unwrap();
        let searches: Vec<_> = patterns
            .iter()
            .filter(|p| p.pattern_type == PatternType::RepeatedSearch)
            .collect();

        assert_eq!(searches.len(), 1);
        assert_eq!(searches[0].frequency, 3);
        assert_eq!(searches[0].related_memories.len(), 3);
    }

    #[test]
    fn detect_file_hotspot_patterns() {
        let storage = Storage::open_in_memory().unwrap();

        // Access main.rs 4 times
        for i in 0..4 {
            let mem = make_memory(
                &format!("read main.rs {i}"),
                "Read",
                vec![("file_path", "src/main.rs")],
            );
            storage.insert_memory(&mem).unwrap();
        }

        // Access lib.rs once (below threshold)
        let mem = make_memory("read lib.rs", "Read", vec![("file_path", "src/lib.rs")]);
        storage.insert_memory(&mem).unwrap();

        let patterns = detect_patterns(&storage, None, 3).unwrap();
        let hotspots: Vec<_> = patterns
            .iter()
            .filter(|p| p.pattern_type == PatternType::FileHotspot)
            .collect();

        assert_eq!(hotspots.len(), 1);
        assert!(hotspots[0].description.contains("src/main.rs"));
        assert_eq!(hotspots[0].frequency, 4);
    }

    #[test]
    fn detect_decision_chain_patterns() {
        let storage = Storage::open_in_memory().unwrap();

        // Edit main.rs 3 times
        for i in 0..3 {
            let mem = make_memory(
                &format!("edit main.rs {i}"),
                "Edit",
                vec![("file_path", "src/main.rs")],
            );
            storage.insert_memory(&mem).unwrap();
        }

        let patterns = detect_patterns(&storage, None, 2).unwrap();
        let chains: Vec<_> = patterns
            .iter()
            .filter(|p| p.pattern_type == PatternType::DecisionChain)
            .collect();

        assert_eq!(chains.len(), 1);
        assert!(chains[0].description.contains("decision chain"));
    }

    #[test]
    fn detect_tool_preference_patterns() {
        let storage = Storage::open_in_memory().unwrap();

        // 5 reads, 2 greps
        for i in 0..5 {
            let mem = make_memory(&format!("read file {i}"), "Read", vec![]);
            storage.insert_memory(&mem).unwrap();
        }
        for i in 0..2 {
            let mem = make_memory(&format!("grep {i}"), "Grep", vec![]);
            storage.insert_memory(&mem).unwrap();
        }

        let patterns = detect_patterns(&storage, None, 1).unwrap();
        let prefs: Vec<_> = patterns
            .iter()
            .filter(|p| p.pattern_type == PatternType::ToolPreference)
            .collect();

        assert_eq!(prefs.len(), 2);
        // Most used tool should be first (sorted by confidence)
        let read_pref = prefs
            .iter()
            .find(|p| p.description.contains("Read"))
            .unwrap();
        assert_eq!(read_pref.frequency, 5);
    }

    #[test]
    fn generate_insights_empty() {
        let md = generate_insights(&[]);
        assert!(md.contains("No patterns detected"));
    }

    #[test]
    fn generate_insights_with_patterns() {
        let patterns = vec![
            DetectedPattern {
                pattern_type: PatternType::FileHotspot,
                description: "File 'src/main.rs' accessed 5 times".to_string(),
                frequency: 5,
                related_memories: vec!["a".to_string()],
                confidence: 0.5,
            },
            DetectedPattern {
                pattern_type: PatternType::RepeatedSearch,
                description: "Search pattern 'error' used 3 times".to_string(),
                frequency: 3,
                related_memories: vec!["b".to_string()],
                confidence: 0.3,
            },
        ];

        let md = generate_insights(&patterns);
        assert!(md.contains("File Hotspots"));
        assert!(md.contains("Repeated Searches"));
        assert!(md.contains("src/main.rs"));
        assert!(md.contains("**Total patterns detected:** 2"));
    }

    #[test]
    fn single_tool_no_preference_detected() {
        let storage = Storage::open_in_memory().unwrap();

        // Only 1 tool type - should return empty preferences
        let mem = make_memory("read file", "Read", vec![]);
        storage.insert_memory(&mem).unwrap();

        let patterns = detect_patterns(&storage, None, 1).unwrap();
        let prefs: Vec<_> = patterns
            .iter()
            .filter(|p| p.pattern_type == PatternType::ToolPreference)
            .collect();

        assert!(prefs.is_empty());
    }
}
