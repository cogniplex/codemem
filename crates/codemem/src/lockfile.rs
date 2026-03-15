//! Cross-process lockfile to prevent concurrent index operations on the same namespace.
//! Lock files live at `~/.codemem/locks/{namespace}.lock` and contain the owner PID.
//! Uses `O_CREAT | O_EXCL` for atomic creation; stale locks (dead PID) are reclaimed.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

pub struct IndexLock {
    path: PathBuf,
}

impl Drop for IndexLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Acquire an exclusive index lock for `namespace`. Returns a guard that releases
/// the lock on drop. Errors if another live process holds the lock.
pub fn try_acquire(namespace: &str) -> Result<IndexLock, String> {
    let dir = lock_dir();
    fs::create_dir_all(&dir).map_err(|e| format!("Failed to create lock dir: {e}"))?;

    let safe_name: String = namespace
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let path = dir.join(format!("{safe_name}.lock"));

    try_create_lock(&path, namespace)
}

fn try_create_lock(path: &std::path::Path, namespace: &str) -> Result<IndexLock, String> {
    match write_lock_atomic(path) {
        Ok(()) => Ok(IndexLock {
            path: path.to_path_buf(),
        }),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            let contents = fs::read_to_string(path).unwrap_or_default();
            let pid = contents.trim().parse::<u32>().unwrap_or(0);

            if pid > 0 && is_process_alive(pid) {
                return Err(format!(
                    "Already indexing '{namespace}' (pid {pid}). \
                     If this is stale, remove {}",
                    path.display()
                ));
            }

            // Stale lock from a dead process — reclaim atomically.
            // Use rename-over to avoid a TOCTOU window: write our PID to a
            // temp file, then atomically rename it over the stale lock.
            // This prevents two processes from both detecting the stale lock,
            // deleting it, and both re-creating it.
            let tmp_path = path.with_extension("lock.tmp");
            write_lock_to(&tmp_path)
                .and_then(|()| fs::rename(&tmp_path, path))
                .map(|()| IndexLock {
                    path: path.to_path_buf(),
                })
                .map_err(|e| {
                    let _ = fs::remove_file(&tmp_path);
                    format!("Failed to reclaim stale lock: {e}")
                })
        }
        Err(e) => Err(format!(
            "Failed to create lock file {}: {e}",
            path.display()
        )),
    }
}

fn write_lock_atomic(path: &std::path::Path) -> std::io::Result<()> {
    let mut f = OpenOptions::new().write(true).create_new(true).open(path)?;
    write!(f, "{}", std::process::id())?;
    Ok(())
}

/// Write PID to a file (overwriting if it exists). Used for the temp file
/// in the stale-lock reclaim path.
fn write_lock_to(path: &std::path::Path) -> std::io::Result<()> {
    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
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
    match i32::try_from(pid) {
        Ok(pid_i32) => unsafe { libc::kill(pid_i32, 0) == 0 },
        Err(_) => false,
    }
}

#[cfg(not(unix))]
fn is_process_alive(_pid: u32) -> bool {
    true
}
