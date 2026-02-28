//! codemem-watch: Real-time file watcher for Codemem.
//!
//! Uses `notify` with debouncing to detect file changes and trigger re-indexing.
//! Respects `.gitignore` and common ignore patterns.

use crossbeam_channel::Receiver;
use ignore::gitignore::{Gitignore, GitignoreBuilder};
use notify_debouncer_mini::{new_debouncer, DebouncedEventKind};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// Events emitted by the file watcher.
#[derive(Debug, Clone)]
pub enum WatchEvent {
    FileChanged(PathBuf),
    FileCreated(PathBuf),
    FileDeleted(PathBuf),
}

/// Default ignore directory names.
const IGNORE_DIRS: &[&str] = &[
    "node_modules",
    "target",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".next",
    "vendor",
    ".cargo",
];

/// Watchable file extensions (code files).
const WATCHABLE_EXTENSIONS: &[&str] = &[
    "rs", "ts", "tsx", "js", "jsx", "py", "go", "c", "cpp", "cc", "cxx", "h", "hpp", "java", "rb",
    "cs", "kt", "kts", "swift", "php", "scala", "sc", "tf", "hcl", "tfvars", "toml", "json",
    "yaml", "yml",
];

/// Check if a file extension is watchable.
pub fn is_watchable(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| WATCHABLE_EXTENSIONS.contains(&ext))
        .unwrap_or(false)
}

/// Check if a path should be ignored.
///
/// Uses the provided `Gitignore` matcher first (checking the full path and
/// each ancestor directory), then falls back to the hardcoded `IGNORE_DIRS`
/// list for paths not covered by `.gitignore`.
pub fn should_ignore(path: &Path, gitignore: Option<&Gitignore>) -> bool {
    if let Some(gi) = gitignore {
        // Check the file itself
        if gi.matched(path, path.is_dir()).is_ignore() {
            return true;
        }
        // Check each ancestor directory against the gitignore
        let mut current = path.to_path_buf();
        while current.pop() {
            if gi.matched(&current, true).is_ignore() {
                return true;
            }
        }
    }
    // Fallback to hardcoded dirs
    for component in path.components() {
        if let std::path::Component::Normal(name) = component {
            if let Some(name_str) = name.to_str() {
                if IGNORE_DIRS.contains(&name_str) {
                    return true;
                }
            }
        }
    }
    false
}

/// Build a `Gitignore` matcher from a project root.
///
/// Reads `.gitignore` if present, and also adds the hardcoded `IGNORE_DIRS`
/// as fallback patterns.
pub fn build_gitignore(root: &Path) -> Option<Gitignore> {
    let mut builder = GitignoreBuilder::new(root);
    // Add .gitignore if it exists
    if let Some(err) = builder.add(root.join(".gitignore")) {
        tracing::debug!("No .gitignore found: {err}");
    }
    // Add fallback patterns (use glob-style to match as directories anywhere)
    for dir in IGNORE_DIRS {
        let _ = builder.add_line(None, &format!("{dir}/"));
    }
    builder.build().ok()
}

/// Detect programming language from file extension.
pub fn detect_language(path: &Path) -> Option<&'static str> {
    path.extension()
        .and_then(|ext| ext.to_str())
        .and_then(|ext| match ext {
            "rs" => Some("rust"),
            "ts" | "tsx" => Some("typescript"),
            "js" | "jsx" => Some("javascript"),
            "py" => Some("python"),
            "go" => Some("go"),
            "c" | "h" => Some("c"),
            "cpp" | "cc" | "cxx" | "hpp" => Some("cpp"),
            "java" => Some("java"),
            "rb" => Some("ruby"),
            "cs" => Some("csharp"),
            "kt" | "kts" => Some("kotlin"),
            "swift" => Some("swift"),
            "php" => Some("php"),
            "scala" | "sc" => Some("scala"),
            "tf" | "hcl" | "tfvars" => Some("hcl"),
            _ => None,
        })
}

/// File watcher that monitors a directory for changes with 50ms debouncing.
pub struct FileWatcher {
    _debouncer: notify_debouncer_mini::Debouncer<notify::RecommendedWatcher>,
    receiver: Receiver<WatchEvent>,
    #[allow(dead_code)]
    gitignore: Arc<Option<Gitignore>>,
}

impl FileWatcher {
    /// Create a new file watcher for the given root directory.
    pub fn new(root: &Path) -> Result<Self, codemem_core::CodememError> {
        let (tx, rx) = crossbeam_channel::unbounded::<WatchEvent>();
        let event_tx = tx;

        let gitignore = Arc::new(build_gitignore(root));
        let gi_clone = Arc::clone(&gitignore);

        // Track files we've already seen so we can distinguish create vs modify.
        let known_files = std::sync::Mutex::new(HashSet::<PathBuf>::new());

        let mut debouncer = new_debouncer(
            Duration::from_millis(50),
            move |res: Result<Vec<notify_debouncer_mini::DebouncedEvent>, notify::Error>| match res
            {
                Ok(events) => {
                    let mut seen = HashSet::new();
                    for event in events {
                        let path = event.path;
                        if !seen.insert(path.clone()) {
                            continue;
                        }
                        if should_ignore(&path, gi_clone.as_ref().as_ref()) || !is_watchable(&path)
                        {
                            continue;
                        }
                        let watch_event = match event.kind {
                            DebouncedEventKind::Any => {
                                if path.exists() {
                                    if let Ok(mut known) = known_files.lock() {
                                        if known.insert(path.clone()) {
                                            WatchEvent::FileCreated(path)
                                        } else {
                                            WatchEvent::FileChanged(path)
                                        }
                                    } else {
                                        WatchEvent::FileChanged(path)
                                    }
                                } else {
                                    if let Ok(mut known) = known_files.lock() {
                                        known.remove(&path);
                                    }
                                    WatchEvent::FileDeleted(path)
                                }
                            }
                            DebouncedEventKind::AnyContinuous => WatchEvent::FileChanged(path),
                            _ => WatchEvent::FileChanged(path),
                        };
                        let _ = event_tx.send(watch_event);
                    }
                }
                Err(e) => {
                    tracing::error!("Watch error: {e}");
                }
            },
        )
        .map_err(|e| {
            codemem_core::CodememError::Io(std::io::Error::other(format!(
                "Failed to create debouncer: {e}"
            )))
        })?;

        debouncer
            .watcher()
            .watch(root, notify::RecursiveMode::Recursive)
            .map_err(|e| {
                codemem_core::CodememError::Io(std::io::Error::other(format!(
                    "Failed to watch {}: {e}",
                    root.display()
                )))
            })?;

        tracing::info!("Watching {} for changes", root.display());

        Ok(Self {
            _debouncer: debouncer,
            receiver: rx,
            gitignore,
        })
    }

    /// Get the receiver for watch events.
    pub fn receiver(&self) -> &Receiver<WatchEvent> {
        &self.receiver
    }
}

#[cfg(test)]
mod tests {
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
}
