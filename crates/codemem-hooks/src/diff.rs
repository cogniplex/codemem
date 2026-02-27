//! Diff computation and semantic summarization for code edits.
//!
//! Uses the `similar` crate to compute line-level diffs, then applies
//! heuristic rules to generate human-readable semantic summaries.

use similar::{ChangeTag, TextDiff};

/// Summary of a diff between old and new content.
#[derive(Debug, Clone)]
pub struct DiffSummary {
    pub file_path: String,
    pub change_type: ChangeType,
    pub lines_added: usize,
    pub lines_removed: usize,
    pub hunks: Vec<DiffHunk>,
    pub semantic_summary: String,
}

/// A contiguous region of changes.
#[derive(Debug, Clone)]
pub struct DiffHunk {
    pub added: Vec<String>,
    pub removed: Vec<String>,
}

/// The kind of change detected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangeType {
    Added,
    Modified,
    Deleted,
}

impl std::fmt::Display for ChangeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChangeType::Added => write!(f, "added"),
            ChangeType::Modified => write!(f, "modified"),
            ChangeType::Deleted => write!(f, "deleted"),
        }
    }
}

/// Compute a line-level diff between old and new content.
pub fn compute_diff(file_path: &str, old_content: &str, new_content: &str) -> DiffSummary {
    let change_type = if old_content.is_empty() && !new_content.is_empty() {
        ChangeType::Added
    } else if !old_content.is_empty() && new_content.is_empty() {
        ChangeType::Deleted
    } else {
        ChangeType::Modified
    };

    let text_diff = TextDiff::from_lines(old_content, new_content);

    let mut lines_added: usize = 0;
    let mut lines_removed: usize = 0;
    let mut hunks: Vec<DiffHunk> = Vec::new();

    for group in text_diff.grouped_ops(3) {
        let mut hunk = DiffHunk {
            added: Vec::new(),
            removed: Vec::new(),
        };

        for op in &group {
            for change in text_diff.iter_changes(op) {
                match change.tag() {
                    ChangeTag::Insert => {
                        lines_added += 1;
                        hunk.added.push(change.value().to_string());
                    }
                    ChangeTag::Delete => {
                        lines_removed += 1;
                        hunk.removed.push(change.value().to_string());
                    }
                    ChangeTag::Equal => {}
                }
            }
        }

        if !hunk.added.is_empty() || !hunk.removed.is_empty() {
            hunks.push(hunk);
        }
    }

    let mut summary = DiffSummary {
        file_path: file_path.to_string(),
        change_type,
        lines_added,
        lines_removed,
        hunks,
        semantic_summary: String::new(),
    };

    summary.semantic_summary = generate_semantic_summary(&summary);
    summary
}

/// Generate a heuristic-based human-readable summary of the diff.
pub fn generate_semantic_summary(diff: &DiffSummary) -> String {
    let mut parts: Vec<String> = Vec::new();

    let all_added: Vec<&str> = diff
        .hunks
        .iter()
        .flat_map(|h| h.added.iter().map(|s| s.trim()))
        .collect();
    let all_removed: Vec<&str> = diff
        .hunks
        .iter()
        .flat_map(|h| h.removed.iter().map(|s| s.trim()))
        .collect();

    // Detect function additions/removals
    let fn_patterns = ["fn ", "def ", "function ", "func ", "async fn "];
    let added_fns: Vec<&str> = all_added
        .iter()
        .filter(|line| fn_patterns.iter().any(|p| line.contains(p)))
        .copied()
        .collect();
    let removed_fns: Vec<&str> = all_removed
        .iter()
        .filter(|line| fn_patterns.iter().any(|p| line.contains(p)))
        .copied()
        .collect();

    for line in &added_fns {
        if let Some(name) = extract_fn_name(line) {
            parts.push(format!("Added function {name}"));
        }
    }
    for line in &removed_fns {
        if let Some(name) = extract_fn_name(line) {
            let was_readded = added_fns
                .iter()
                .any(|a| extract_fn_name(a) == Some(name.clone()));
            if !was_readded {
                parts.push(format!("Removed function {name}"));
            }
        }
    }

    // Detect import changes
    let import_patterns = ["use ", "import ", "from ", "require("];
    let added_imports = all_added
        .iter()
        .any(|line| import_patterns.iter().any(|p| line.contains(p)));
    let removed_imports = all_removed
        .iter()
        .any(|line| import_patterns.iter().any(|p| line.contains(p)));
    if added_imports || removed_imports {
        parts.push("Updated imports".to_string());
    }

    // Detect error handling
    let error_patterns = ["Result", "Error", "unwrap", "expect", "try {", "catch"];
    let added_error = all_added
        .iter()
        .any(|line| error_patterns.iter().any(|p| line.contains(p)));
    let removed_error = all_removed
        .iter()
        .any(|line| error_patterns.iter().any(|p| line.contains(p)));
    if added_error && !removed_error {
        parts.push("Added error handling".to_string());
    }

    // Detect type definitions
    let type_patterns = ["struct ", "class ", "enum ", "trait ", "interface "];
    for line in &all_added {
        if type_patterns.iter().any(|p| line.contains(p)) {
            if let Some(name) = extract_type_name(line) {
                parts.push(format!("Added type {name}"));
            }
        }
    }

    if parts.is_empty() {
        let total = diff.lines_added + diff.lines_removed;
        format!("Modified {} lines in {}", total, diff.file_path)
    } else {
        parts.join("; ")
    }
}

fn extract_fn_name(line: &str) -> Option<String> {
    let trimmed = line.trim();
    for prefix in &[
        "export async function ",
        "export function ",
        "async function ",
        "function ",
        "async fn ",
        "pub async fn ",
        "pub fn ",
        "pub(crate) fn ",
        "fn ",
        "def ",
        "func ",
    ] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                return Some(name);
            }
        }
    }
    None
}

fn extract_type_name(line: &str) -> Option<String> {
    for prefix in &["struct ", "class ", "enum ", "trait ", "interface "] {
        if let Some(rest) = line.split(prefix).nth(1) {
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                return Some(name);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_simple_edit() {
        let old = "fn main() {\n    println!(\"hello\");\n}\n";
        let new = "fn main() {\n    println!(\"world\");\n}\n";
        let summary = compute_diff("src/main.rs", old, new);
        assert_eq!(summary.change_type, ChangeType::Modified);
        assert_eq!(summary.lines_added, 1);
        assert_eq!(summary.lines_removed, 1);
    }

    #[test]
    fn semantic_summary_function_addition() {
        let old = "// module\n";
        let new = "// module\nfn new_helper() {\n    todo!()\n}\n";
        let summary = compute_diff("src/lib.rs", old, new);
        assert!(summary
            .semantic_summary
            .contains("Added function new_helper"));
    }

    #[test]
    fn semantic_summary_function_removal() {
        let old = "fn helper() {\n    todo!()\n}\nfn main() {}\n";
        let new = "fn main() {}\n";
        let summary = compute_diff("src/lib.rs", old, new);
        assert!(summary.semantic_summary.contains("Removed function helper"));
    }

    #[test]
    fn semantic_summary_import_changes() {
        let old = "use std::io;\nfn main() {}\n";
        let new = "use std::io;\nuse std::fs;\nfn main() {}\n";
        let summary = compute_diff("src/main.rs", old, new);
        assert!(summary.semantic_summary.contains("Updated imports"));
    }

    #[test]
    fn semantic_summary_type_addition() {
        let old = "// types\n";
        let new = "// types\nstruct Config {\n    name: String,\n}\n";
        let summary = compute_diff("src/types.rs", old, new);
        assert!(summary.semantic_summary.contains("Added type Config"));
    }

    #[test]
    fn empty_diff() {
        let content = "fn main() {}\n";
        let summary = compute_diff("src/main.rs", content, content);
        assert_eq!(summary.lines_added, 0);
        assert_eq!(summary.lines_removed, 0);
    }

    #[test]
    fn change_type_added() {
        let summary = compute_diff("new.rs", "", "fn new() {}\n");
        assert_eq!(summary.change_type, ChangeType::Added);
    }

    #[test]
    fn change_type_deleted() {
        let summary = compute_diff("old.rs", "fn old() {}\n", "");
        assert_eq!(summary.change_type, ChangeType::Deleted);
    }

    #[test]
    fn extract_fn_name_works() {
        assert_eq!(extract_fn_name("fn hello("), Some("hello".to_string()));
        assert_eq!(
            extract_fn_name("async fn fetch_data()"),
            Some("fetch_data".to_string())
        );
        assert_eq!(
            extract_fn_name("def process(x):"),
            Some("process".to_string())
        );
        assert_eq!(extract_fn_name("no function here"), None);
    }

    #[test]
    fn extract_type_name_works() {
        assert_eq!(
            extract_type_name("struct MyStruct {"),
            Some("MyStruct".to_string())
        );
        assert_eq!(extract_type_name("enum Color {"), Some("Color".to_string()));
        assert_eq!(
            extract_type_name("trait Display {"),
            Some("Display".to_string())
        );
        assert_eq!(extract_type_name("no type here"), None);
    }
}
