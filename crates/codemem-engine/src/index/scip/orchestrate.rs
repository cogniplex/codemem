//! SCIP indexer orchestration: auto-detect project languages and available SCIP indexers,
//! run them, and merge the resulting `.scip` files.

use std::path::{Path, PathBuf};
use std::process::Command;

use codemem_core::{CodememError, ScipConfig};

use super::{parse_scip_bytes, ScipReadResult};

/// Language detected from a manifest file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScipLanguage {
    Rust,
    TypeScript,
    Python,
    Java,
    Go,
    CSharp,
    Ruby,
    Php,
    Dart,
}

impl ScipLanguage {
    /// The binary name to search for on PATH.
    fn indexer_binary(&self) -> &'static str {
        match self {
            Self::Rust => "rust-analyzer",
            Self::TypeScript => "scip-typescript",
            Self::Python => "scip-python",
            Self::Java => "scip-java",
            Self::Go => "scip-go",
            Self::CSharp => "scip-dotnet",
            Self::Ruby => "scip-ruby",
            Self::Php => "scip-php",
            Self::Dart => "scip-dart",
        }
    }

    /// Default arguments for the indexer when no config override is provided.
    fn default_args(&self) -> Vec<&'static str> {
        match self {
            Self::Rust => vec!["scip", "."],
            Self::TypeScript => vec!["index"],
            Self::Python => vec!["index", "."],
            Self::Java => vec!["index"],
            Self::Go => vec![],
            Self::CSharp => vec!["index"],
            Self::Ruby => vec![],
            Self::Php => vec!["index"],
            Self::Dart => vec![],
        }
    }

    /// Default output filename for this language's indexer.
    fn default_output_file(&self) -> &'static str {
        // All SCIP indexers write to the same default filename.
        "index.scip"
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::TypeScript => "typescript",
            Self::Python => "python",
            Self::Java => "java",
            Self::Go => "go",
            Self::CSharp => "csharp",
            Self::Ruby => "ruby",
            Self::Php => "php",
            Self::Dart => "dart",
        }
    }
}

/// Manifest file patterns that indicate a project language.
const MANIFEST_LANGUAGES: &[(&str, ScipLanguage)] = &[
    ("Cargo.toml", ScipLanguage::Rust),
    ("package.json", ScipLanguage::TypeScript),
    ("tsconfig.json", ScipLanguage::TypeScript),
    ("pyproject.toml", ScipLanguage::Python),
    ("setup.py", ScipLanguage::Python),
    ("setup.cfg", ScipLanguage::Python),
    ("go.mod", ScipLanguage::Go),
    ("pom.xml", ScipLanguage::Java),
    ("build.gradle", ScipLanguage::Java),
    ("build.gradle.kts", ScipLanguage::Java),
    ("pubspec.yaml", ScipLanguage::Dart),
    ("Gemfile", ScipLanguage::Ruby),
    ("composer.json", ScipLanguage::Php),
];

/// Result of running SCIP indexers.
#[derive(Debug)]
pub struct OrchestrationResult {
    /// The merged SCIP read result (definitions, references, externals, covered files).
    pub scip_result: ScipReadResult,
    /// Languages for which indexers ran successfully.
    pub indexed_languages: Vec<ScipLanguage>,
    /// Languages for which indexers were available but failed.
    pub failed_languages: Vec<(ScipLanguage, String)>,
}

impl OrchestrationResult {
    /// Create an empty result with no definitions, references, or indexed languages.
    fn empty(project_root: &Path) -> Self {
        Self {
            scip_result: ScipReadResult {
                project_root: project_root.to_string_lossy().to_string(),
                definitions: Vec::new(),
                references: Vec::new(),
                externals: Vec::new(),
                covered_files: Vec::new(),
            },
            indexed_languages: Vec::new(),
            failed_languages: Vec::new(),
        }
    }
}

/// Orchestrates SCIP indexer detection and execution.
pub struct ScipOrchestrator {
    config: ScipConfig,
}

impl ScipOrchestrator {
    pub fn new(config: ScipConfig) -> Self {
        Self { config }
    }

    /// Run the full orchestration pipeline: detect → run → merge.
    pub fn run(
        &self,
        project_root: &Path,
        namespace: &str,
    ) -> Result<OrchestrationResult, CodememError> {
        // Phase 1: Detect languages from manifests.
        let detected_languages = self.detect_languages(project_root);
        if detected_languages.is_empty() {
            return Ok(OrchestrationResult::empty(project_root));
        }

        // Phase 2: Determine which indexers are available.
        let available = self.detect_available_indexers(&detected_languages);
        if available.is_empty() {
            tracing::info!("No SCIP indexers found on PATH for detected languages");
            return Ok(OrchestrationResult::empty(project_root));
        }

        // Phase 3: Run indexers and collect .scip files.
        let mut indexed_languages = Vec::new();
        let mut failed_languages = Vec::new();
        let mut scip_files: Vec<PathBuf> = Vec::new();

        let temp_dir = tempfile::tempdir().map_err(|e| {
            CodememError::ScipOrchestration(format!("Failed to create temp dir: {e}"))
        })?;

        // Resolve cache dir once (None if caching disabled or home dir unavailable).
        let cache_dir = if self.config.cache_index {
            scip_cache_dir(namespace)
        } else {
            None
        };

        for lang in &available {
            // Check cache first if enabled.
            if let Some(ref cache) = cache_dir {
                if let Some(status) = check_cache(cache, *lang, self.config.cache_ttl_hours) {
                    if status.valid {
                        tracing::info!(
                            "Using cached SCIP index for {} ({})",
                            lang.name(),
                            status.path.display()
                        );
                        scip_files.push(status.path);
                        indexed_languages.push(*lang);
                        continue;
                    }
                }
            }

            let output_path = temp_dir.path().join(format!("index-{}.scip", lang.name()));

            match self.run_indexer(*lang, project_root, &output_path, namespace) {
                Ok(()) => {
                    // Find the actual .scip file (either at output_path or default location).
                    let scip_path = if output_path.exists() {
                        output_path
                    } else {
                        let default_path = project_root.join(lang.default_output_file());
                        if default_path.exists() {
                            default_path
                        } else {
                            failed_languages.push((
                                *lang,
                                "Indexer exited successfully but produced no .scip file"
                                    .to_string(),
                            ));
                            continue;
                        }
                    };

                    // Save to cache for future runs.
                    if let Some(ref cache) = cache_dir {
                        save_to_cache(cache, *lang, &scip_path);
                    }

                    scip_files.push(scip_path);
                    indexed_languages.push(*lang);
                }
                Err(e) => {
                    tracing::warn!("SCIP indexer for {} failed: {}", lang.name(), e);
                    failed_languages.push((*lang, e.to_string()));
                }
            }
        }

        // Phase 4: Parse and merge all .scip files.
        let scip_result = self.merge_scip_files(&scip_files, project_root)?;

        Ok(OrchestrationResult {
            scip_result,
            indexed_languages,
            failed_languages,
        })
    }

    /// Detect which languages are used in the project by scanning for manifest files.
    pub fn detect_languages(&self, project_root: &Path) -> Vec<ScipLanguage> {
        let mut found = std::collections::HashSet::new();

        let walker = ignore::WalkBuilder::new(project_root)
            .hidden(true)
            .git_ignore(true)
            .git_global(true)
            .git_exclude(true)
            .max_depth(Some(3)) // Don't recurse too deep for manifest detection
            .build();

        for entry in walker.flatten() {
            if !entry.file_type().is_some_and(|ft| ft.is_file()) {
                continue;
            }
            let file_name = entry
                .path()
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");

            for &(manifest, lang) in MANIFEST_LANGUAGES {
                if file_name == manifest {
                    found.insert(lang);
                }
            }

            // .csproj files by extension
            if file_name.ends_with(".csproj") {
                found.insert(ScipLanguage::CSharp);
            }
        }

        found.into_iter().collect()
    }

    /// Check which indexers are available on PATH or configured with explicit commands.
    pub fn detect_available_indexers(&self, languages: &[ScipLanguage]) -> Vec<ScipLanguage> {
        let mut available = Vec::new();

        for &lang in languages {
            // Check if there's a config override for this language.
            if self.config_command_for(lang).is_some() {
                available.push(lang);
                continue;
            }

            // Auto-detect from PATH.
            if !self.config.auto_detect_indexers {
                continue;
            }
            if which_binary(lang.indexer_binary()).is_some() {
                available.push(lang);
            }
        }

        available
    }

    /// Run a single SCIP indexer for the given language.
    fn run_indexer(
        &self,
        lang: ScipLanguage,
        project_root: &Path,
        output_path: &Path,
        namespace: &str,
    ) -> Result<(), CodememError> {
        let (program, args) = if let Some(cmd) = self.config_command_for(lang) {
            // Parse the config override command, substituting {namespace}.
            let expanded = cmd.replace("{namespace}", namespace);
            parse_shell_command(&expanded)?
        } else {
            // Resolve the absolute path to the indexer binary so child processes
            // work even when PATH doesn't include the user's shell additions
            // (e.g. when invoked from hooks running under /bin/sh).
            let binary_name = lang.indexer_binary();
            let resolved = which_binary(binary_name)
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| binary_name.to_string());
            (
                resolved,
                lang.default_args().iter().map(|s| s.to_string()).collect(),
            )
        };

        tracing::info!(
            "Running SCIP indexer for {}: {} {:?}",
            lang.name(),
            program,
            args
        );

        // Ensure the child process inherits a PATH that includes common
        // tool locations (e.g. ~/.cargo/bin, homebrew paths, nvm paths).
        // The parent process PATH may be minimal when run from hooks.
        let path_env = augmented_path();

        let output = Command::new(&program)
            .args(&args)
            .current_dir(project_root)
            .env("PATH", &path_env)
            .output()
            .map_err(|e| {
                CodememError::ScipOrchestration(format!("Failed to spawn {program}: {e}"))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(CodememError::ScipOrchestration(format!(
                "{} exited with {}: {}",
                program,
                output.status,
                stderr.trim()
            )));
        }

        // Many indexers write to index.scip in the project root by default.
        // If the output file doesn't exist yet, try to move the default output.
        if !output_path.exists() {
            let default_output = project_root.join(lang.default_output_file());
            if default_output.exists() {
                move_across_filesystems(&default_output, output_path).map_err(|e| {
                    CodememError::ScipOrchestration(format!(
                        "Failed to move {} to {}: {e}",
                        default_output.display(),
                        output_path.display()
                    ))
                })?;
            }
        }

        Ok(())
    }

    /// Get the config override command for a language, if any.
    fn config_command_for(&self, lang: ScipLanguage) -> Option<&String> {
        let cmd = match lang {
            ScipLanguage::Rust => &self.config.indexers.rust,
            ScipLanguage::TypeScript => &self.config.indexers.typescript,
            ScipLanguage::Python => &self.config.indexers.python,
            ScipLanguage::Java => &self.config.indexers.java,
            ScipLanguage::Go => &self.config.indexers.go,
            // These languages don't have config overrides in ScipIndexersConfig yet.
            ScipLanguage::CSharp | ScipLanguage::Ruby | ScipLanguage::Php | ScipLanguage::Dart => {
                return None;
            }
        };
        if cmd.is_empty() {
            None
        } else {
            Some(cmd)
        }
    }

    /// Parse and merge multiple .scip files into a single ScipReadResult.
    fn merge_scip_files(
        &self,
        paths: &[PathBuf],
        project_root: &Path,
    ) -> Result<ScipReadResult, CodememError> {
        let mut merged = ScipReadResult {
            project_root: project_root.to_string_lossy().to_string(),
            definitions: Vec::new(),
            references: Vec::new(),
            externals: Vec::new(),
            covered_files: Vec::new(),
        };

        for path in paths {
            let bytes = std::fs::read(path).map_err(|e| {
                CodememError::ScipOrchestration(format!("Failed to read {}: {e}", path.display()))
            })?;
            let result = parse_scip_bytes(&bytes).map_err(CodememError::ScipOrchestration)?;
            merged.definitions.extend(result.definitions);
            merged.references.extend(result.references);
            merged.externals.extend(result.externals);
            merged.covered_files.extend(result.covered_files);
        }

        // Dedup covered files (multiple indexers might cover overlapping files).
        merged.covered_files.sort();
        merged.covered_files.dedup();

        Ok(merged)
    }
}

/// Move a file from `src` to `dst`, falling back to copy+delete when the two
/// paths live on different filesystems. `std::fs::rename` maps to the
/// `rename(2)` syscall which returns `EXDEV` ("Invalid cross-device link")
/// across mount points — common on Linux when `/tmp` is `tmpfs` and the
/// project lives under `$HOME` on another filesystem.
fn move_across_filesystems(src: &Path, dst: &Path) -> std::io::Result<()> {
    match std::fs::rename(src, dst) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::CrossesDevices => {
            std::fs::copy(src, dst)?;
            std::fs::remove_file(src)?;
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Check if a binary is available on PATH.
fn which_binary(name: &str) -> Option<PathBuf> {
    which::which(name).ok()
}

/// Build an augmented PATH that includes common tool directories.
/// Useful when the current process was spawned by /bin/sh which
/// doesn't source shell profiles (~/.zshrc, ~/.bashrc).
fn augmented_path() -> String {
    let current = std::env::var("PATH").unwrap_or_default();
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/tmp"));

    let extra_dirs = [
        home.join(".cargo/bin"),
        home.join(".local/bin"),
        home.join(".nvm/current/bin"),
        PathBuf::from("/usr/local/bin"),
        PathBuf::from("/opt/homebrew/bin"),
    ];

    let mut parts: Vec<String> = vec![current];
    for dir in &extra_dirs {
        if dir.is_dir() {
            parts.push(dir.display().to_string());
        }
    }
    parts.join(":")
}

/// Parse a shell command string into (program, args).
///
/// Simple whitespace splitting — does not handle quoted strings.
fn parse_shell_command(cmd: &str) -> Result<(String, Vec<String>), CodememError> {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    if parts.is_empty() {
        return Err(CodememError::ScipOrchestration(
            "Empty command string".to_string(),
        ));
    }
    let program = parts[0].to_string();
    let args = parts[1..].iter().map(|s| s.to_string()).collect();
    Ok((program, args))
}

/// Result of checking SCIP cache validity.
pub struct CacheStatus {
    /// Path to the cached .scip file.
    pub path: PathBuf,
    /// Whether the cache is still valid (within TTL).
    pub valid: bool,
}

/// Resolve the SCIP cache directory for a given namespace.
/// Returns `~/.codemem/scip-cache/{namespace}/`, creating it if needed.
fn scip_cache_dir(namespace: &str) -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    let dir = home.join(".codemem").join("scip-cache").join(namespace);
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Check if a cached SCIP index exists in `cache_dir` and is within the TTL.
pub fn check_cache(cache_dir: &Path, lang: ScipLanguage, ttl_hours: u64) -> Option<CacheStatus> {
    let cache_path = cache_dir.join(format!("index-{}.scip", lang.name()));
    if !cache_path.exists() {
        return None;
    }

    let metadata = std::fs::metadata(&cache_path).ok()?;
    let modified = metadata.modified().ok()?;
    let age = modified.elapsed().ok()?;
    let valid = age.as_secs() < ttl_hours * 3600;

    Some(CacheStatus {
        path: cache_path,
        valid,
    })
}

/// Save a .scip file to the given cache directory for future runs.
fn save_to_cache(cache_dir: &Path, lang: ScipLanguage, source_path: &Path) {
    let cache_path = cache_dir.join(format!("index-{}.scip", lang.name()));
    if let Err(e) = std::fs::copy(source_path, &cache_path) {
        tracing::warn!("Failed to cache SCIP index for {}: {e}", lang.name());
    }
}

#[cfg(test)]
#[path = "../tests/scip_orchestrate_tests.rs"]
mod tests;
