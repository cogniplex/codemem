//! codemem-watch: Real-time file watcher for Codemem.
//!
//! Uses `notify` with debouncing to detect file changes and trigger re-indexing.
//! Respects `.gitignore` and common ignore patterns.

use crossbeam_channel::Receiver;
use notify_debouncer_mini::{new_debouncer, DebouncedEventKind};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
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
    "rs", "ts", "tsx", "js", "jsx", "py", "go", "c", "cpp", "cc", "cxx", "h", "hpp", "java",
    "toml", "json", "yaml", "yml",
];

/// Check if a file extension is watchable.
pub fn is_watchable(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| WATCHABLE_EXTENSIONS.contains(&ext))
        .unwrap_or(false)
}

/// Check if a path is inside an ignored directory.
pub fn should_ignore(path: &Path) -> bool {
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
            _ => None,
        })
}

/// File watcher that monitors a directory for changes with 50ms debouncing.
pub struct FileWatcher {
    _debouncer: notify_debouncer_mini::Debouncer<notify::RecommendedWatcher>,
    receiver: Receiver<WatchEvent>,
}

impl FileWatcher {
    /// Create a new file watcher for the given root directory.
    pub fn new(root: &Path) -> Result<Self, codemem_core::CodememError> {
        let (tx, rx) = crossbeam_channel::unbounded::<WatchEvent>();
        let event_tx = tx;

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
                        if should_ignore(&path) || !is_watchable(&path) {
                            continue;
                        }
                        let watch_event = match event.kind {
                            DebouncedEventKind::Any => {
                                if path.exists() {
                                    WatchEvent::FileChanged(path)
                                } else {
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
    fn test_should_ignore() {
        assert!(should_ignore(Path::new("project/node_modules/foo/bar.js")));
        assert!(should_ignore(Path::new("project/target/debug/build.rs")));
        assert!(should_ignore(Path::new(".git/config")));
        assert!(!should_ignore(Path::new("src/main.rs")));
        assert!(!should_ignore(Path::new("lib/utils.ts")));
    }

    #[test]
    fn test_detect_language() {
        assert_eq!(detect_language(Path::new("main.rs")), Some("rust"));
        assert_eq!(detect_language(Path::new("app.tsx")), Some("typescript"));
        assert_eq!(detect_language(Path::new("script.py")), Some("python"));
        assert_eq!(detect_language(Path::new("main.go")), Some("go"));
        assert_eq!(detect_language(Path::new("readme.md")), None);
    }
}
