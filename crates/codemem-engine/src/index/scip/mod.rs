//! SCIP integration: reader, orchestrator, and graph builder.
//!
//! - **Reader** (`mod.rs`): Parse `.scip` protobuf files into intermediate structs.
//! - **Orchestrator** (`orchestrate.rs`): Auto-detect languages and indexers, run them.

pub mod graph_builder;
pub mod orchestrate;

use codemem_core::NodeKind;
use protobuf::Message;
use scip::types::Index;

/// A symbol definition extracted from a SCIP index.
#[derive(Debug, Clone)]
pub struct ScipDefinition {
    /// Full SCIP symbol string (globally unique).
    pub scip_symbol: String,
    /// Language-appropriate qualified name (e.g., `auth::jwt::validate` for Rust).
    pub qualified_name: String,
    /// Relative file path from project root.
    pub file_path: String,
    /// Start line (0-indexed).
    pub line_start: u32,
    /// End line (0-indexed). Same as `line_start` for single-line symbols.
    pub line_end: u32,
    /// Start column (0-indexed).
    pub col_start: u32,
    /// End column (0-indexed).
    pub col_end: u32,
    /// Mapped node kind.
    pub kind: NodeKind,
    /// Hover documentation lines from `SymbolInformation.documentation`.
    pub documentation: Vec<String>,
    /// Relationships declared by this symbol.
    pub relationships: Vec<ScipRelationship>,
    /// Whether this definition is in test code.
    pub is_test: bool,
    /// Whether this definition is in generated code.
    pub is_generated: bool,
}

/// A symbol reference (non-definition occurrence) extracted from a SCIP index.
#[derive(Debug, Clone)]
pub struct ScipReference {
    /// Full SCIP symbol string being referenced.
    pub scip_symbol: String,
    /// Relative file path where the reference occurs.
    pub file_path: String,
    /// Line number (0-indexed).
    pub line: u32,
    /// Column (0-indexed).
    pub col_start: u32,
    /// End column (0-indexed).
    pub col_end: u32,
    /// Raw role bitmask from SCIP.
    pub role_bitmask: i32,
}

/// An external (dependency) symbol from the SCIP index.
#[derive(Debug, Clone)]
pub struct ScipExternal {
    /// Full SCIP symbol string.
    pub scip_symbol: String,
    /// Package manager (e.g., "cargo", "npm", "pip").
    pub package_manager: String,
    /// Package name (e.g., "serde", "@types/node").
    pub package_name: String,
    /// Package version (e.g., "1.0.0").
    pub package_version: String,
    /// Mapped node kind.
    pub kind: NodeKind,
    /// Hover documentation.
    pub documentation: Vec<String>,
}

/// A relationship declared on a `SymbolInformation`.
#[derive(Debug, Clone)]
pub struct ScipRelationship {
    /// Target symbol string.
    pub target_symbol: String,
    pub is_implementation: bool,
    pub is_type_definition: bool,
    pub is_reference: bool,
    pub is_definition: bool,
}

/// Parsed result from reading a `.scip` file.
#[derive(Debug, Clone)]
pub struct ScipReadResult {
    /// Project root from SCIP metadata.
    pub project_root: String,
    /// All symbol definitions found.
    pub definitions: Vec<ScipDefinition>,
    /// All symbol references found.
    pub references: Vec<ScipReference>,
    /// All external (dependency) symbols.
    pub externals: Vec<ScipExternal>,
    /// Set of relative file paths covered by this SCIP index.
    pub covered_files: Vec<String>,
}

// SCIP occurrence role bitmask constants.
const ROLE_DEFINITION: i32 = 0x1;
pub(crate) const ROLE_IMPORT: i32 = 0x2;
pub(crate) const ROLE_WRITE_ACCESS: i32 = 0x4;
pub(crate) const ROLE_READ_ACCESS: i32 = 0x8;
const ROLE_TEST: i32 = 0x20;
const ROLE_GENERATED: i32 = 0x10;

/// Parse SCIP protobuf bytes into intermediate structs.
pub fn parse_scip_bytes(bytes: &[u8]) -> Result<ScipReadResult, String> {
    let index = Index::parse_from_bytes(bytes)
        .map_err(|e| format!("Failed to parse SCIP protobuf: {e}"))?;

    let project_root = index
        .metadata
        .as_ref()
        .map(|m| m.project_root.clone())
        .unwrap_or_default();

    let mut definitions = Vec::new();
    let mut references = Vec::new();
    let mut covered_files = Vec::new();

    for doc in &index.documents {
        let file_path = &doc.relative_path;
        let language = &doc.language;
        covered_files.push(file_path.clone());

        let lang_sep = detect_language_separator(language);

        // Build a map of symbol string -> SymbolInformation for this document.
        let mut sym_info_map = std::collections::HashMap::new();
        for sym_info in &doc.symbols {
            if !sym_info.symbol.is_empty() {
                sym_info_map.insert(sym_info.symbol.as_str(), sym_info);
            }
        }

        for occ in &doc.occurrences {
            if occ.symbol.is_empty() || scip::symbol::is_local_symbol(&occ.symbol) {
                continue;
            }

            let (start_line, start_col, end_line, end_col) = match parse_range(&occ.range) {
                Some(r) => r,
                None => continue,
            };

            let roles = occ.symbol_roles;
            let is_def = (roles & ROLE_DEFINITION) != 0;
            let is_test = (roles & ROLE_TEST) != 0;
            let is_generated = (roles & ROLE_GENERATED) != 0;

            if is_def {
                // Early noise filter: if SymbolInformation.Kind identifies this as a
                // variable, parameter, or literal type, skip it entirely. This avoids
                // building qualified names and containment chains for noise symbols.
                if let Some(info) = sym_info_map.get(occ.symbol.as_str()) {
                    if is_noise_kind(info.kind.value()) {
                        continue;
                    }
                }

                let qualified_name = match scip_symbol_to_qualified_name(&occ.symbol, lang_sep) {
                    Some(q) => q,
                    None => continue,
                };

                // Look up SymbolInformation for kind and documentation.
                let (kind, documentation, relationships) =
                    if let Some(info) = sym_info_map.get(occ.symbol.as_str()) {
                        let kind = resolve_node_kind(info.kind.value(), &occ.symbol);
                        let docs: Vec<String> =
                            info.documentation.iter().map(|s| s.to_string()).collect();
                        let rels: Vec<ScipRelationship> = info
                            .relationships
                            .iter()
                            .map(|r| ScipRelationship {
                                target_symbol: r.symbol.clone(),
                                is_implementation: r.is_implementation,
                                is_type_definition: r.is_type_definition,
                                is_reference: r.is_reference,
                                is_definition: r.is_definition,
                            })
                            .collect();
                        (kind, docs, rels)
                    } else {
                        (infer_kind_from_symbol(&occ.symbol), Vec::new(), Vec::new())
                    };

                definitions.push(ScipDefinition {
                    scip_symbol: occ.symbol.clone(),
                    qualified_name,
                    file_path: file_path.clone(),
                    line_start: start_line,
                    line_end: end_line,
                    col_start: start_col,
                    col_end: end_col,
                    kind,
                    documentation,
                    relationships,
                    is_test,
                    is_generated,
                });
            } else {
                references.push(ScipReference {
                    scip_symbol: occ.symbol.clone(),
                    file_path: file_path.clone(),
                    line: start_line,
                    col_start: start_col,
                    col_end: end_col,
                    role_bitmask: roles,
                });
            }
        }
    }

    // Process external symbols.
    let externals = index
        .external_symbols
        .iter()
        .filter(|ext| !ext.symbol.is_empty() && !scip::symbol::is_local_symbol(&ext.symbol))
        .filter_map(|ext| {
            let parsed = scip::symbol::parse_symbol(&ext.symbol).ok()?;
            let package = parsed.package.as_ref()?;
            let kind = resolve_node_kind(ext.kind.value(), &ext.symbol);
            let documentation: Vec<String> =
                ext.documentation.iter().map(|s| s.to_string()).collect();

            Some(ScipExternal {
                scip_symbol: ext.symbol.clone(),
                package_manager: package.manager.clone(),
                package_name: package.name.clone(),
                package_version: package.version.clone(),
                kind,
                documentation,
            })
        })
        .collect();

    // SCIP occurrences only mark the identifier token, not the full body extent.
    // Infer body ranges: each definition extends to the next sibling at the same
    // nesting depth (or end-of-file). This lets find_enclosing_def_indexed() match
    // references inside function bodies to their enclosing function.
    infer_definition_extents(&mut definitions);

    Ok(ScipReadResult {
        project_root,
        definitions,
        references,
        externals,
        covered_files,
    })
}

/// Infer body extents for definitions whose `line_start == line_end` (identifier-only ranges).
///
/// Groups definitions by file, sorts by line, then counts nesting depth (based on
/// SCIP descriptor chain length). Each definition's `line_end` is set to just before
/// the next sibling at the same or shallower depth, or `u32::MAX` for the last in a file.
fn infer_definition_extents(definitions: &mut [ScipDefinition]) {
    use std::collections::HashMap;

    // Group definition indices by file (clone file_path to avoid borrowing definitions).
    let mut by_file: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, def) in definitions.iter().enumerate() {
        by_file.entry(def.file_path.clone()).or_default().push(i);
    }

    for indices in by_file.values() {
        // Sort by line_start.
        let mut sorted: Vec<usize> = indices.clone();
        sorted.sort_by_key(|&i| definitions[i].line_start);

        // Pre-compute depths to avoid re-borrowing during mutation.
        let depths: Vec<usize> = sorted
            .iter()
            .map(|&i| descriptor_depth(&definitions[i].scip_symbol))
            .collect();

        for pos in 0..sorted.len() {
            let idx = sorted[pos];
            // Skip if already has a meaningful range (multi-line identifier).
            if definitions[idx].line_end > definitions[idx].line_start {
                continue;
            }

            let my_depth = depths[pos];

            // Find the next definition at the same or shallower depth.
            // Default to u32::MAX so the last definition in a file "owns" all
            // remaining lines. This is a known trade-off: references in trailing
            // comments or whitespace will be attributed to the last function.
            // Without CST data we can't know where the body actually ends.
            let mut end_line = u32::MAX;
            for next_pos in pos + 1..sorted.len() {
                if depths[next_pos] <= my_depth {
                    end_line = definitions[sorted[next_pos]].line_start.saturating_sub(1);
                    break;
                }
            }

            definitions[idx].line_end = end_line;
        }
    }
}

/// Count the number of descriptors in a SCIP symbol to determine nesting depth.
fn descriptor_depth(scip_symbol: &str) -> usize {
    scip::symbol::parse_symbol(scip_symbol)
        .map(|p| p.descriptors.len())
        .unwrap_or(0)
}

/// Parse a SCIP occurrence range into (start_line, start_col, end_line, end_col).
///
/// 3-element: `[line, startCol, endCol]` (single-line).
/// 4-element: `[startLine, startCol, endLine, endCol]` (multi-line).
fn parse_range(range: &[i32]) -> Option<(u32, u32, u32, u32)> {
    match range.len() {
        3 => Some((
            range[0].try_into().ok()?,
            range[1].try_into().ok()?,
            range[0].try_into().ok()?,
            range[2].try_into().ok()?,
        )),
        4 => Some((
            range[0].try_into().ok()?,
            range[1].try_into().ok()?,
            range[2].try_into().ok()?,
            range[3].try_into().ok()?,
        )),
        _ => None,
    }
}

/// Extract a qualified name from a SCIP symbol string using the appropriate language separator.
///
/// Strips scheme, package manager, package name, and version — joins descriptors with the
/// language-appropriate separator (`::` for Rust, `.` for Python/TS/Java/Go).
pub fn scip_symbol_to_qualified_name(scip_symbol: &str, lang_separator: &str) -> Option<String> {
    let parsed = scip::symbol::parse_symbol(scip_symbol).ok()?;
    let parts: Vec<&str> = parsed
        .descriptors
        .iter()
        .map(|d| d.name.as_str())
        .filter(|s| !s.is_empty())
        .collect();
    if parts.is_empty() {
        return None;
    }
    Some(parts.join(lang_separator))
}

/// Detect the appropriate separator for qualified names based on language.
///
/// Returns `"::"` for Rust/C++, `"."` for everything else.
pub fn detect_language_separator(language: &str) -> &'static str {
    match language.to_lowercase().as_str() {
        "rust" | "cpp" | "c++" => "::",
        _ => ".",
    }
}

/// Map a SCIP `SymbolInformation.Kind` integer value to a codemem `NodeKind`.
///
/// Uses the `scip::types::symbol_information::Kind` enum for compile-time safety.
/// Returns `None` for `UnspecifiedKind` (0) so callers can fall back to descriptor inference.
fn scip_kind_to_node_kind(kind: i32) -> Option<NodeKind> {
    use scip::types::symbol_information::Kind;
    match kind {
        x if x == Kind::Class as i32 || x == Kind::Struct as i32 => Some(NodeKind::Class),
        x if x == Kind::Interface as i32 || x == Kind::Protocol as i32 => Some(NodeKind::Interface),
        x if x == Kind::Trait as i32 => Some(NodeKind::Trait),
        x if x == Kind::Enum as i32 => Some(NodeKind::Enum),
        x if x == Kind::EnumMember as i32 => Some(NodeKind::EnumVariant),
        x if x == Kind::Field as i32
            || x == Kind::StaticField as i32
            || x == Kind::StaticDataMember as i32 =>
        {
            Some(NodeKind::Field)
        }
        x if x == Kind::Property as i32 || x == Kind::StaticProperty as i32 => {
            Some(NodeKind::Property)
        }
        x if x == Kind::TypeParameter as i32 => Some(NodeKind::TypeParameter),
        x if x == Kind::Macro as i32 => Some(NodeKind::Macro),
        x if x == Kind::Function as i32 || x == Kind::Constructor as i32 => {
            Some(NodeKind::Function)
        }
        x if x == Kind::Method as i32
            || x == Kind::StaticMethod as i32
            || x == Kind::AbstractMethod as i32
            || x == Kind::TraitMethod as i32
            || x == Kind::ProtocolMethod as i32
            || x == Kind::PureVirtualMethod as i32
            || x == Kind::MethodSpecification as i32
            || x == Kind::Getter as i32
            || x == Kind::Setter as i32
            || x == Kind::Accessor as i32 =>
        {
            Some(NodeKind::Method)
        }
        x if x == Kind::Namespace as i32
            || x == Kind::Module as i32
            || x == Kind::PackageObject as i32 =>
        {
            Some(NodeKind::Module)
        }
        x if x == Kind::Package as i32 || x == Kind::Library as i32 => Some(NodeKind::Package),
        x if x == Kind::TypeAlias as i32
            || x == Kind::Type as i32
            || x == Kind::AssociatedType as i32 =>
        {
            Some(NodeKind::Type)
        }
        x if x == Kind::Constant as i32 || x == Kind::StaticVariable as i32 => {
            Some(NodeKind::Constant)
        }
        _ => None,
    }
}

/// Check if a `SymbolInformation.Kind` value represents a symbol that should
/// never become a graph node in a knowledge graph.
///
/// This is the primary noise filter — when the indexer provides a Kind, we trust
/// it over descriptor Suffix heuristics. Variables, parameters, literal types,
/// and other non-structural symbols are filtered here.
pub fn is_noise_kind(kind: i32) -> bool {
    use scip::types::symbol_information::Kind;
    matches!(kind,
        x if x == Kind::Variable as i32
            || x == Kind::Parameter as i32
            || x == Kind::SelfParameter as i32
            || x == Kind::ThisParameter as i32
            || x == Kind::ParameterLabel as i32
            || x == Kind::TypeParameter as i32
            // Literal/value types — not structural
            || x == Kind::Boolean as i32
            || x == Kind::Number as i32
            || x == Kind::String as i32
            || x == Kind::Null as i32
            || x == Kind::Array as i32
            || x == Kind::Object as i32
            || x == Kind::Key as i32
            || x == Kind::Pattern as i32
            // Receiver/error types
            || x == Kind::MethodReceiver as i32
            || x == Kind::Error as i32
    )
}

/// Infer `NodeKind` from the SCIP symbol's descriptor suffixes when
/// `SymbolInformation.Kind` is `UnspecifiedKind` (e.g., scip-go).
///
/// Descriptor suffix conventions (from the SCIP spec):
/// - `/` → Package/Namespace
/// - `#` → Type (struct, class, interface)
/// - `().` → Method
/// - `.` → Term (function, variable, field)
/// - `!` → Macro
pub fn infer_kind_from_symbol(scip_symbol: &str) -> NodeKind {
    let parsed = match scip::symbol::parse_symbol(scip_symbol) {
        Ok(p) => p,
        Err(_) => return NodeKind::Function,
    };
    infer_kind_from_parsed(&parsed)
}

/// Infer node kind from an already-parsed SCIP symbol, avoiding a redundant parse.
pub fn infer_kind_from_parsed(parsed: &scip::types::Symbol) -> NodeKind {
    let last = match parsed.descriptors.last() {
        Some(d) => d,
        None => return NodeKind::Function,
    };
    use scip::types::descriptor::Suffix;
    match last.suffix.enum_value() {
        Ok(Suffix::Package | Suffix::Namespace) => NodeKind::Module,
        Ok(Suffix::Type) => NodeKind::Class,
        Ok(Suffix::Method) => NodeKind::Method,
        Ok(Suffix::Macro) => NodeKind::Macro,
        Ok(Suffix::TypeParameter) => NodeKind::TypeParameter,
        Ok(Suffix::Parameter) => NodeKind::Field,
        Ok(Suffix::Term) => {
            // A Term inside a Type (class/interface/struct) is a field/property,
            // not a function. Check the parent descriptor.
            let parent = parsed.descriptors.iter().rev().nth(1);
            match parent.and_then(|d| d.suffix.enum_value().ok()) {
                Some(Suffix::Type) => NodeKind::Field,
                _ => {
                    // Module-level Term with UPPER_CASE name → Constant, not Function.
                    // SCIP classifies `const ACCOUNT_ROUTE = "/account"` as Term,
                    // but these are constants/config values, not callable functions.
                    if is_constant_name(&last.name) {
                        NodeKind::Constant
                    } else {
                        NodeKind::Function
                    }
                }
            }
        }
        _ => NodeKind::Function, // Meta, Local, UnspecifiedSuffix
    }
}

/// Check if a name looks like a constant (UPPER_CASE_WITH_UNDERSCORES or ALL_CAPS).
/// Examples: `ACCOUNT_ROUTE`, `API_URL`, `MAX_RETRIES`, `DEBUG`.
/// Counter-examples: `useState`, `handleClick`, `_build_filters`.
fn is_constant_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
        && name.chars().any(|c| c.is_ascii_uppercase())
}

/// Resolve a node kind: use the SCIP `Kind` if specified, otherwise infer from the symbol.
fn resolve_node_kind(kind: i32, scip_symbol: &str) -> NodeKind {
    scip_kind_to_node_kind(kind).unwrap_or_else(|| infer_kind_from_symbol(scip_symbol))
}

/// Check if a reference has the import role.
pub fn is_import_ref(role_bitmask: i32) -> bool {
    (role_bitmask & ROLE_IMPORT) != 0
}

/// Check if a reference has read access.
pub fn is_read_ref(role_bitmask: i32) -> bool {
    (role_bitmask & ROLE_READ_ACCESS) != 0
}

/// Check if a reference has write access.
pub fn is_write_ref(role_bitmask: i32) -> bool {
    (role_bitmask & ROLE_WRITE_ACCESS) != 0
}

#[cfg(test)]
#[path = "../tests/scip_reader_tests.rs"]
mod tests;
