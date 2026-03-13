//! Cross-process lockfile to prevent concurrent index operations on the same namespace.
//!
//! Lock files live at `~/.codemem/locks/{namespace}.lock` and contain the PID of
//! the owning process. Uses `O_CREAT | O_EXCL` (via `create_new`) for atomic
//! creation, avoiding the TOCTOU race of check-then-create. Stale locks (dead PID)
//! are reclaimed with a remove-and-retry.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

/// RAII guard that removes the lock file on drop.
pub struct IndexLock {
    path: PathBuf,
}

impl Drop for IndexLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Try to acquire an exclusive index lock for the given namespace.
///
/// Returns `Ok(guard)` on success. The lock is released when the guard is dropped.
/// Returns `Err` if another live process holds the lock.
pub fn try_acquire(namespace: &str) -> Result<IndexLock, String> {
    let dir = lock_dir();
    fs::create_dir_all(&dir).map_err(|e| format!("Failed to create lock dir: {e}"))?;

    let path = dir.join(format!("{namespace}.lock"));

    try_create_lock(&path, namespace)
}

/// Atomically create the lock file. If it already exists, inspect the PID:
/// - alive  → return Err (another process is indexing)
/// - dead   → remove stale lock and retry once
fn try_create_lock(path: &PathBuf, namespace: &str) -> Result<IndexLock, String> {
    match write_lock_atomic(path) {
        Ok(()) => Ok(IndexLock { path: path.clone() }),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            // Lock exists — check if the owning PID is still alive
            let contents = fs::read_to_string(path).unwrap_or_default();
            let pid = contents.trim().parse::<u32>().unwrap_or(0);

            if pid > 0 && is_process_alive(pid) {
                return Err(format!(
                    "Already indexing '{namespace}' (pid {pid}). \
                     If this is stale, remove {}",
                    path.display()
                ));
            }

            // Stale lock — remove and retry once
            let _ = fs::remove_file(path);
            write_lock_atomic(path)
                .map(|()| IndexLock { path: path.clone() })
                .map_err(|e| format!("Failed to acquire lock after stale removal: {e}"))
        }
        Err(e) => Err(format!("Failed to create lock file {}: {e}", path.display())),
    }
}

/// Write PID to the lock file using `create_new` (O_CREAT | O_EXCL) for atomicity.
fn write_lock_atomic(path: &PathBuf) -> std::io::Result<()> {
    let mut f = OpenOptions::new().write(true).create_new(true).open(path)?;
    write!(f, "{}", std::process::id())?;
    Ok(())
}

fn lock_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".codemem")
        .join("locks")
}

#[cfg(unix)]
fn is_process_alive(pid: u32) -> bool {
    // kill(pid, 0) checks existence without sending a signal
    unsafe { libc::kill(pid as i32, 0) == 0 }
}

#[cfg(not(unix))]
fn is_process_alive(_pid: u32) -> bool {
    // Conservative: assume alive on non-unix to avoid clobbering
    true
}
