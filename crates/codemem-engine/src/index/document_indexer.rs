//! Document indexing for non-code files: Markdown and YAML.
//!
//! Markdown files are split on `##` / `###` headings into sections, each
//! becoming one `DocumentNode`. YAML files are split on top-level named
//! resources (documents that carry a `name` key at any nesting level under a
//! recognizable `metadata` or root key). Falls back to a single whole-file
//! node when no structure is found.

/// A named section extracted from a document file.
#[derive(Debug, Clone)]
pub struct DocumentNode {
    /// File this node came from (relative to the indexed root).
    pub file_path: String,
    /// 0-based index of this section within its file (stable across files).
    pub index_in_file: usize,
    /// Short name for the node — the heading text or YAML resource name.
    pub name: String,
    /// Full text content of the section (the heading line + body, or the full
    /// YAML block).
    pub content: String,
    /// 1-based start line within the file.
    pub line_start: usize,
    /// 1-based end line within the file (inclusive).
    pub line_end: usize,
    /// Source format.
    pub format: DocumentFormat,
}

/// Build a deterministic graph-node ID for a document section.
///
/// Format: `doc:{file_path}:{index_in_file}`
pub fn doc_node_id(doc: &DocumentNode) -> String {
    format!("doc:{}:{}", doc.file_path, doc.index_in_file)
}

/// Build the prefix for all doc node IDs belonging to a file.
///
/// Use with `delete_graph_nodes_by_prefix` to clean stale doc nodes.
pub fn doc_prefix_for_file(file_path: &str) -> String {
    format!("doc:{file_path}:")
}

/// The format of the source file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocumentFormat {
    Markdown,
    Yaml,
}

impl std::fmt::Display for DocumentFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Markdown => write!(f, "markdown"),
            Self::Yaml => write!(f, "yaml"),
        }
    }
}

/// Returns true if the extension is handled by the document indexer.
pub fn supports_document_extension(ext: &str) -> bool {
    matches!(ext, "md" | "markdown" | "yml" | "yaml")
}

/// Parse a document file and return its `DocumentNode`s.
///
/// Returns an empty vec if the content is not valid UTF-8 or is empty.
pub fn parse_document(path: &str, content: &[u8]) -> Vec<DocumentNode> {
    let Ok(source) = std::str::from_utf8(content) else {
        return Vec::new();
    };
    if source.trim().is_empty() {
        return Vec::new();
    }

    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    match ext {
        "yml" | "yaml" => parse_yaml(path, source),
        "md" | "markdown" => parse_markdown(path, source),
        _ => Vec::new(),
    }
}

// ── Markdown ─────────────────────────────────────────────────────────────────

/// Split Markdown source on `##` / `###` headings.
///
/// Each heading starts a new section that runs until the next `##` or `###`
/// heading (or end of file). The heading line is included in the content.
/// Falls back to a single whole-file node if no `##` / `###` headings exist.
///
/// Per CommonMark, headings may have up to 3 leading spaces and a space or tab
/// after the hash run.
fn parse_markdown(path: &str, source: &str) -> Vec<DocumentNode> {
    let lines: Vec<&str> = source.lines().collect();
    let total = lines.len();

    // Collect positions of all ## / ### headings.
    let mut cuts: Vec<(usize, String)> = Vec::new(); // (0-based line index, heading text)
    for (i, line) in lines.iter().enumerate() {
        // CommonMark: up to 3 leading spaces are allowed before the heading.
        let stripped = line.trim_start_matches(' ');
        let leading_spaces = line.len() - stripped.len();
        if leading_spaces > 3 {
            continue;
        }
        let after_hashes = stripped.trim_start_matches('#');
        let hashes = stripped.len() - after_hashes.len();
        // Accept ## or ### followed by a space or tab (CommonMark requirement).
        if (hashes == 2 || hashes == 3)
            && (after_hashes.starts_with(' ') || after_hashes.starts_with('\t'))
        {
            cuts.push((i, after_hashes.trim().to_string()));
        }
    }

    if cuts.is_empty() {
        // No section structure — whole file as one node.
        let name = heading_from_h1(&lines).unwrap_or_else(|| file_stem(path));
        return vec![DocumentNode {
            file_path: path.to_string(),
            index_in_file: 0,
            name,
            content: source.to_string(),
            line_start: 1,
            line_end: total.max(1),
            format: DocumentFormat::Markdown,
        }];
    }

    let mut nodes = Vec::with_capacity(cuts.len());
    for (idx, (start_line, heading)) in cuts.iter().enumerate() {
        let end_line = cuts
            .get(idx + 1)
            .map(|(l, _)| l.saturating_sub(1))
            .unwrap_or(total.saturating_sub(1));
        let content = lines[*start_line..=end_line].join("\n");
        nodes.push(DocumentNode {
            file_path: path.to_string(),
            index_in_file: idx,
            name: heading.clone(),
            content,
            line_start: start_line + 1,
            line_end: end_line + 1,
            format: DocumentFormat::Markdown,
        });
    }
    nodes
}

/// Extract the text of the first `# ` heading, if any.
/// Handles up to 3 leading spaces (CommonMark).
fn heading_from_h1(lines: &[&str]) -> Option<String> {
    lines.iter().find_map(|line| {
        let stripped = line.trim_start_matches(' ');
        let leading = line.len() - stripped.len();
        if leading > 3 {
            return None;
        }
        stripped
            .strip_prefix("# ")
            .or_else(|| stripped.strip_prefix("#\t"))
            .map(|t| t.trim().to_string())
    })
}

/// Return the file stem (filename without extension) as a fallback name.
fn file_stem(path: &str) -> String {
    std::path::Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(path)
        .to_string()
}

// ── YAML ──────────────────────────────────────────────────────────────────────

/// Parse YAML source into `DocumentNode`s.
///
/// Each YAML document in the stream (separated by `---`) is treated as one
/// candidate. If a document carries an identifiable name (via `metadata.name`,
/// `name`, or `id` at the top level), it becomes a named node. Otherwise the
/// whole document block is kept as a single unnamed node (using the file stem).
///
/// Falls back to a single whole-file node if `serde_yaml` fails to parse.
fn parse_yaml(path: &str, source: &str) -> Vec<DocumentNode> {
    // Split on `---` document separators to get per-document blocks with their
    // approximate line offsets.
    let doc_blocks = split_yaml_documents(source);

    if doc_blocks.is_empty() {
        return vec![whole_file_yaml_node(path, source)];
    }

    let mut nodes = Vec::new();
    let mut file_idx = 0usize;
    for (block, line_start) in doc_blocks {
        let trimmed = block.trim();
        if trimmed.is_empty() {
            continue;
        }
        let line_end = line_start + trimmed.lines().count().saturating_sub(1);
        let name = extract_yaml_name(trimmed).unwrap_or_else(|| {
            tracing::debug!("YAML block in {path} at line {line_start} has no identifiable name, using file stem");
            file_stem(path)
        });
        nodes.push(DocumentNode {
            file_path: path.to_string(),
            index_in_file: file_idx,
            name,
            content: trimmed.to_string(),
            line_start,
            line_end,
            format: DocumentFormat::Yaml,
        });
        file_idx += 1;
    }

    if nodes.is_empty() {
        vec![whole_file_yaml_node(path, source)]
    } else {
        nodes
    }
}

fn whole_file_yaml_node(path: &str, source: &str) -> DocumentNode {
    let total = source.lines().count().max(1);
    DocumentNode {
        file_path: path.to_string(),
        index_in_file: 0,
        name: file_stem(path),
        content: source.to_string(),
        line_start: 1,
        line_end: total,
        format: DocumentFormat::Yaml,
    }
}

/// Split a YAML source string on `---` separators.
///
/// Returns a vec of `(block_text, 1-based start line)` pairs.
fn split_yaml_documents(source: &str) -> Vec<(String, usize)> {
    let mut docs: Vec<(String, usize)> = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    let mut current_start = 1usize;
    let mut line_no = 0usize;

    for line in source.lines() {
        line_no += 1;
        if line.trim() == "---" {
            if !current.is_empty() {
                docs.push((current.join("\n"), current_start));
                current.clear();
            }
            current_start = line_no + 1;
        } else {
            current.push(line);
        }
    }
    if !current.is_empty() {
        docs.push((current.join("\n"), current_start));
    }
    docs
}

/// Try to extract a human-readable name from a YAML document block.
///
/// Checks (in order):
/// 1. `metadata.name` (Kubernetes-style)
/// 2. `name` at the top level
/// 3. `id` at the top level
fn extract_yaml_name(block: &str) -> Option<String> {
    let value: serde_yaml::Value = serde_yaml::from_str(block).ok()?;
    let mapping = value.as_mapping()?;

    // 1. metadata.name (Kubernetes / Flux)
    if let Some(meta) = mapping.get("metadata") {
        if let Some(name) = meta.get("name") {
            if let Some(s) = yaml_scalar_to_string(name) {
                return Some(s);
            }
        }
    }

    // 2. top-level `name`
    if let Some(name) = mapping.get("name") {
        if let Some(s) = yaml_scalar_to_string(name) {
            return Some(s);
        }
    }

    // 3. top-level `id`
    if let Some(id) = mapping.get("id") {
        if let Some(s) = yaml_scalar_to_string(id) {
            return Some(s);
        }
    }

    None
}

fn yaml_scalar_to_string(v: &serde_yaml::Value) -> Option<String> {
    match v {
        serde_yaml::Value::String(s) => Some(s.clone()),
        serde_yaml::Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markdown_splits_on_h2() {
        let src = "# Title\n\nIntro paragraph.\n\n## Section A\n\nContent A.\n\n## Section B\n\nContent B.\n";
        let nodes = parse_markdown("README.md", src);
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].name, "Section A");
        assert_eq!(nodes[1].name, "Section B");
        assert!(nodes[0].content.contains("Content A"));
        assert!(nodes[1].content.contains("Content B"));
    }

    #[test]
    fn markdown_whole_file_fallback() {
        let src = "# Title\n\nNo sections here.\n";
        let nodes = parse_markdown("doc.md", src);
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].name, "Title");
        assert_eq!(nodes[0].line_start, 1);
    }

    #[test]
    fn markdown_h3_included() {
        let src = "## Parent\n\n### Child\n\nChild content.\n";
        let nodes = parse_markdown("doc.md", src);
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].name, "Parent");
        assert_eq!(nodes[1].name, "Child");
    }

    #[test]
    fn yaml_kubernetes_name() {
        let src = "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: fastapi\nspec:\n  replicas: 1\n";
        let nodes = parse_yaml("deploy.yml", src);
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].name, "fastapi");
    }

    #[test]
    fn yaml_top_level_name() {
        let src = "name: my-service\nport: 8080\n";
        let nodes = parse_yaml("service.yml", src);
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].name, "my-service");
    }

    #[test]
    fn yaml_multi_document() {
        let src = "metadata:\n  name: alpha\n---\nmetadata:\n  name: beta\n";
        let nodes = parse_yaml("multi.yml", src);
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].name, "alpha");
        assert_eq!(nodes[1].name, "beta");
    }

    #[test]
    fn yaml_fallback_name_is_stem() {
        let src = "key: value\n";
        let nodes = parse_yaml("config.yml", src);
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].name, "config");
    }

    #[test]
    fn markdown_leading_whitespace_and_tabs() {
        // CommonMark allows up to 3 leading spaces before ATX headings.
        let src = "## Normal\n\nA.\n\n  ## Indented 2\n\nB.\n\n   ###\tTab after hashes\n\nC.\n";
        let nodes = parse_markdown("doc.md", src);
        assert_eq!(nodes.len(), 3);
        assert_eq!(nodes[0].name, "Normal");
        assert_eq!(nodes[1].name, "Indented 2");
        assert_eq!(nodes[2].name, "Tab after hashes");
    }

    #[test]
    fn markdown_4_spaces_not_heading() {
        // 4+ spaces = code block, not a heading.
        let src = "## Real heading\n\n    ## Not a heading\n";
        let nodes = parse_markdown("doc.md", src);
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].name, "Real heading");
    }

    #[test]
    fn doc_node_ids_are_per_file() {
        let nodes = parse_markdown("README.md", "## A\n\nContent A.\n\n## B\n\nContent B.\n");
        assert_eq!(doc_node_id(&nodes[0]), "doc:README.md:0");
        assert_eq!(doc_node_id(&nodes[1]), "doc:README.md:1");
    }

    #[test]
    fn supports_extension_coverage() {
        assert!(supports_document_extension("md"));
        assert!(supports_document_extension("markdown"));
        assert!(supports_document_extension("yml"));
        assert!(supports_document_extension("yaml"));
        assert!(!supports_document_extension("rs"));
        assert!(!supports_document_extension("py"));
    }
}
