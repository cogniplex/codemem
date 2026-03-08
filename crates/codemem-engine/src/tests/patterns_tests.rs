use super::*;
use codemem_core::MemoryNode;
use codemem_storage::Storage;
use std::collections::HashMap;

fn make_memory(content: &str, tool: &str, extra_metadata: Vec<(&str, &str)>) -> MemoryNode {
    let mut metadata = HashMap::new();
    metadata.insert(
        "tool".to_string(),
        serde_json::Value::String(tool.to_string()),
    );
    for (k, v) in extra_metadata {
        metadata.insert(k.to_string(), serde_json::Value::String(v.to_string()));
    }
    let mut m = MemoryNode::test_default(content);
    m.metadata = metadata;
    m
}

#[test]
fn detect_patterns_empty_db() {
    let storage = Storage::open_in_memory().unwrap();
    let patterns = detect_patterns(&storage, None, 2, 10).unwrap();
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

    let patterns = detect_patterns(&storage, None, 2, 10).unwrap();
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

    let patterns = detect_patterns(&storage, None, 3, 10).unwrap();
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

    let patterns = detect_patterns(&storage, None, 2, 10).unwrap();
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

    let patterns = detect_patterns(&storage, None, 1, 10).unwrap();
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

    let patterns = detect_patterns(&storage, None, 1, 10).unwrap();
    let prefs: Vec<_> = patterns
        .iter()
        .filter(|p| p.pattern_type == PatternType::ToolPreference)
        .collect();

    assert!(prefs.is_empty());
}

#[test]
fn compute_confidence_log_scaled() {
    // With frequency=10, total_sessions=100: ln(10)/ln(100) ≈ 0.5
    let c = compute_confidence(10, 100, 1.0);
    assert!((c - 0.5).abs() < 0.01, "expected ~0.5, got {c}");

    // Recency factor scales the result
    let c2 = compute_confidence(10, 100, 0.5);
    assert!((c2 - 0.25).abs() < 0.01, "expected ~0.25, got {c2}");

    // Clamped to 1.0
    let c3 = compute_confidence(100, 10, 1.0);
    assert_eq!(c3, 1.0);
}

#[test]
fn compute_confidence_zero_inputs() {
    assert_eq!(compute_confidence(0, 10, 1.0), 0.0);
    assert_eq!(compute_confidence(5, 0, 1.0), 0.0);
    assert_eq!(compute_confidence(0, 0, 1.0), 0.0);
}
