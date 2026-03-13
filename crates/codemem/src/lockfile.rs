//! Cross-process lockfile to prevent concurrent index operations on the same namespace.
//!
//! Lock files live at `~/.codemem/locks/{namespace}.lock` and contain the PID of
//! the owning process. Stale locks (dead PID) are automatically reclaimed.

use std::fs;
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
    let my_pid = std::process::id();

    if let Ok(contents) = fs::read_to_string(&path) {
        if let Ok(pid) = contents.trim().parse::<u32>() {
            if is_process_alive(pid) {
                return Err(format!(
                    "Already indexing '{namespace}' (pid {pid}). \
                     If this is stale, remove {}",
                    path.display()
                ));
            }
            // Stale lock — previous process died without cleanup
            let _ = fs::remove_file(&path);
        }
    }

    let mut f = fs::File::create(&path)
        .map_err(|e| format!("Failed to create lock file {}: {e}", path.display()))?;
    write!(f, "{my_pid}").map_err(|e| format!("Failed to write lock file: {e}"))?;

    Ok(IndexLock { path })
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
