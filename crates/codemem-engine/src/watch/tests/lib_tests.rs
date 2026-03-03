use super::*;

#[test]
fn test_is_watchable() {
    assert!(is_watchable(Path::new("src/main.rs")));
    assert!(is_watchable(Path::new("index.ts")));
    assert!(is_watchable(Path::new("app.py")));
    assert!(is_watchable(Path::new("main.go")));
    assert!(!is_watchable(Path::new("image.png")));
    assert!(!is_watchable(Path::new("binary.exe")));
}

#[test]
fn test_should_ignore_without_gitignore() {
    assert!(should_ignore(
        Path::new("project/node_modules/foo/bar.js"),
        None
    ));
    assert!(should_ignore(
        Path::new("project/target/debug/build.rs"),
        None
    ));
    assert!(should_ignore(Path::new(".git/config"), None));
    assert!(!should_ignore(Path::new("src/main.rs"), None));
    assert!(!should_ignore(Path::new("lib/utils.ts"), None));
}

#[test]
fn test_should_ignore_with_gitignore() {
    let dir = tempfile::tempdir().unwrap();
    let gitignore_path = dir.path().join(".gitignore");
    std::fs::write(&gitignore_path, "*.log\nsecrets/\n").unwrap();

    let gi = build_gitignore(dir.path()).unwrap();

    // Matches .gitignore pattern (*.log)
    assert!(should_ignore(&dir.path().join("debug.log"), Some(&gi)));

    // Matches .gitignore pattern (secrets/) -- also caught by hardcoded fallback check
    // The gitignore matcher matches against paths under the root.
    // For directory patterns, the path must be checked as a directory.
    assert!(should_ignore(
        &dir.path().join("secrets/key.txt"),
        Some(&gi)
    ));

    // Matches hardcoded IGNORE_DIRS via the fallback component check
    assert!(should_ignore(
        &dir.path().join("node_modules/foo.js"),
        Some(&gi)
    ));

    // Not ignored
    assert!(!should_ignore(&dir.path().join("src/main.rs"), Some(&gi)));
}

#[test]
fn test_build_gitignore_without_file() {
    let dir = tempfile::tempdir().unwrap();
    // No .gitignore file exists
    let gi = build_gitignore(dir.path());
    // Should still return Some since we add fallback patterns
    assert!(gi.is_some());

    // Hardcoded dirs are still caught by the should_ignore fallback
    let gi = gi.unwrap();
    assert!(should_ignore(
        &dir.path().join("node_modules/foo.js"),
        Some(&gi)
    ));
}

#[test]
fn test_detect_language() {
    assert_eq!(detect_language(Path::new("main.rs")), Some("rust"));
    assert_eq!(detect_language(Path::new("app.tsx")), Some("typescript"));
    assert_eq!(detect_language(Path::new("script.py")), Some("python"));
    assert_eq!(detect_language(Path::new("main.go")), Some("go"));
    assert_eq!(detect_language(Path::new("readme.md")), None);
}

#[test]
fn test_new_file_emits_file_created() {
    let dir = tempfile::tempdir().unwrap();
    let watcher = FileWatcher::new(dir.path()).unwrap();
    let rx = watcher.receiver();

    // Create a new watchable file after the watcher starts
    let file_path = dir.path().join("hello.rs");
    std::fs::write(&file_path, "fn main() {}").unwrap();

    // Wait for the debounced event (50ms debounce + margin)
    let event = rx.recv_timeout(Duration::from_secs(2));
    assert!(event.is_ok(), "expected a watch event for new file");
    assert!(
        matches!(event.unwrap(), WatchEvent::FileCreated(_)),
        "new file should emit FileCreated"
    );
}

#[test]
fn test_modified_file_emits_file_changed() {
    let dir = tempfile::tempdir().unwrap();

    // Create file before the watcher starts so it's not "new"
    let file_path = dir.path().join("existing.rs");
    std::fs::write(&file_path, "fn main() {}").unwrap();

    let watcher = FileWatcher::new(dir.path()).unwrap();
    let rx = watcher.receiver();

    // First touch: watcher hasn't seen it yet, so this is FileCreated
    std::fs::write(&file_path, "fn main() { println!(); }").unwrap();
    let first = rx.recv_timeout(Duration::from_secs(2));
    assert!(first.is_ok(), "expected event for first write");
    assert!(matches!(first.unwrap(), WatchEvent::FileCreated(_)));

    // Second touch: watcher has now seen it, so this should be FileChanged
    std::thread::sleep(Duration::from_millis(100));
    std::fs::write(&file_path, "fn main() { eprintln!(); }").unwrap();
    let second = rx.recv_timeout(Duration::from_secs(2));
    assert!(second.is_ok(), "expected event for second write");
    assert!(
        matches!(second.unwrap(), WatchEvent::FileChanged(_)),
        "subsequent modification should emit FileChanged"
    );
}
