//! Watch module: Real-time file watcher for Codemem.
//!
//! Uses `notify` with debouncing to detect file changes and trigger re-indexing.
//! Respects `.gitignore` and common ignore patterns.

use crossbeam_channel::Receiver;
use ignore::gitignore::{Gitignore, GitignoreBuilder};
use notify_debouncer_mini::new_debouncer;
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
pub(crate) fn is_watchable(path: &Path) -> bool {
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
///
/// `is_dir` indicates whether the path is a directory. Pass `false` for file
/// events from the watcher to avoid a redundant `stat` syscall per event.
pub(crate) fn should_ignore(path: &Path, gitignore: Option<&Gitignore>, is_dir: bool) -> bool {
    if let Some(gi) = gitignore {
        // Check the file itself
        if gi.matched(path, is_dir).is_ignore() {
            return true;
        }
        // Check each ancestor directory against the gitignore.
        // NOTE: This traverses all the way to the filesystem root. In practice
        // this is harmless because gitignore patterns only match relative to the
        // gitignore root, but a future improvement could accept the project root
        // and stop traversal there.
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
pub(crate) fn build_gitignore(root: &Path) -> Option<Gitignore> {
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
                        // Watcher events are always files, so pass is_dir=false
                        // to avoid a stat syscall per event.
                        if should_ignore(&path, gi_clone.as_ref().as_ref(), false)
                            || !is_watchable(&path)
                        {
                            continue;
                        }
                        // Determine event type from filesystem state + known-files
                        // set rather than the debouncer event kind, which varies
                        // across platforms (FSEvents on macOS vs inotify on Linux).
                        let watch_event = if path.exists() {
                            if let Ok(mut known) = known_files.lock() {
                                // Prevent unbounded growth: clear when exceeding 50K entries.
                                // After clearing, all subsequent files will appear as "created"
                                // until the set repopulates, which is acceptable.
                                if known.len() > 50_000 {
                                    known.clear();
                                }
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
#[path = "tests/lib_tests.rs"]
mod tests;
