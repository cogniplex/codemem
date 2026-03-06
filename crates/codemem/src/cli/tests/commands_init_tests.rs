use super::*;

// ── is_in_path tests ──────────────────────────────────────────────────────

#[test]
fn is_in_path_finds_common_binary() {
    // `ls` or `cat` should exist on any Unix-like system
    assert!(is_in_path("ls") || is_in_path("cat"));
}

#[test]
fn is_in_path_returns_false_for_nonexistent() {
    assert!(!is_in_path("__codemem_nonexistent_binary_xyz__"));
}

#[test]
fn is_in_path_empty_name_returns_false() {
    assert!(!is_in_path(""));
}

// ── DetectedAssistant struct tests ────────────────────────────────────────

#[test]
fn detected_assistant_fields_accessible() {
    let da = DetectedAssistant {
        name: "TestAssistant",
        config_dir: std::path::PathBuf::from("/tmp/test"),
        in_path: false,
    };
    assert_eq!(da.name, "TestAssistant");
    assert_eq!(da.config_dir, std::path::PathBuf::from("/tmp/test"));
    assert!(!da.in_path);
}

// ── detect_assistants smoke test ──────────────────────────────────────────

#[test]
fn detect_assistants_returns_vec() {
    // Just ensure it doesn't panic; actual results depend on the host
    let assistants = detect_assistants();
    // Each entry should have a non-empty name
    for a in &assistants {
        assert!(!a.name.is_empty());
    }
}

#[test]
fn detect_assistants_only_known_names() {
    let known = ["Claude Code", "Cursor", "Windsurf"];
    let assistants = detect_assistants();
    for a in &assistants {
        assert!(
            known.contains(&a.name),
            "Unexpected assistant name: {}",
            a.name
        );
    }
}
