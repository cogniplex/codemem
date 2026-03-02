//! Rust language extractor using tree-sitter-rust.

use crate::extractor::LanguageExtractor;
use crate::symbol::{Reference, ReferenceKind, Symbol, SymbolKind, Visibility};
use tree_sitter::{Node, Tree};

/// Rust language extractor for tree-sitter-based code indexing.
pub struct RustExtractor;

impl RustExtractor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for RustExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageExtractor for RustExtractor {
    fn language_name(&self) -> &str {
        "rust"
    }

    fn file_extensions(&self) -> &[&str] {
        &["rs"]
    }

    fn tree_sitter_language(&self) -> tree_sitter::Language {
        tree_sitter_rust::LANGUAGE.into()
    }

    fn extract_symbols(&self, tree: &Tree, source: &[u8], file_path: &str) -> Vec<Symbol> {
        let mut symbols = Vec::new();
        let root = tree.root_node();
        extract_symbols_recursive(root, source, file_path, &[], false, &mut symbols);
        symbols
    }

    fn extract_references(&self, tree: &Tree, source: &[u8], file_path: &str) -> Vec<Reference> {
        let mut references = Vec::new();
        let root = tree.root_node();
        extract_references_recursive(root, source, file_path, &[], &mut references);
        references
    }
}

// ── Symbol Extraction ─────────────────────────────────────────────────────

/// Recursively walk the AST and extract symbol definitions.
///
/// `in_impl` tracks whether we are inside an `impl` block, which determines
/// whether functions are classified as Function or Method.
fn extract_symbols_recursive(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    in_impl: bool,
    symbols: &mut Vec<Symbol>,
) {
    match node.kind() {
        "function_item" => {
            if let Some(sym) = extract_function(node, source, file_path, scope, in_impl) {
                symbols.push(sym);
            }
        }
        "struct_item" => {
            if let Some(sym) =
                extract_named_item(node, source, file_path, scope, SymbolKind::Struct)
            {
                let name = sym.name.clone();
                symbols.push(sym);
                // Recurse into struct body for nested items (rare)
                let mut new_scope = scope.to_vec();
                new_scope.push(name);
                for i in 0..node.child_count() {
                    if let Some(child) = node.child(i as u32) {
                        extract_symbols_recursive(
                            child, source, file_path, &new_scope, false, symbols,
                        );
                    }
                }
                return; // already recursed
            }
        }
        "enum_item" => {
            if let Some(sym) = extract_named_item(node, source, file_path, scope, SymbolKind::Enum)
            {
                symbols.push(sym);
            }
        }
        "trait_item" => {
            if let Some(sym) =
                extract_named_item(node, source, file_path, scope, SymbolKind::Interface)
            {
                let name = sym.name.clone();
                symbols.push(sym);
                // Recurse into trait body for method signatures
                let mut new_scope = scope.to_vec();
                new_scope.push(name);
                if let Some(body) = node.child_by_field_name("body") {
                    for i in 0..body.child_count() {
                        if let Some(child) = body.child(i as u32) {
                            // Trait methods are inside a trait, treat as impl-like scope
                            extract_symbols_recursive(
                                child, source, file_path, &new_scope, true, symbols,
                            );
                        }
                    }
                }
                return; // already recursed
            }
        }
        "impl_item" => {
            // Extract the target type name for scoping methods
            let impl_target = get_impl_target_name(node, source);
            if let Some(target) = &impl_target {
                let mut new_scope = scope.to_vec();
                new_scope.push(target.clone());
                // Recurse into impl body for methods — mark in_impl = true
                if let Some(body) = node.child_by_field_name("body") {
                    for i in 0..body.child_count() {
                        if let Some(child) = body.child(i as u32) {
                            extract_symbols_recursive(
                                child, source, file_path, &new_scope, true, symbols,
                            );
                        }
                    }
                }
                return; // already recursed
            }
        }
        "type_item" => {
            if let Some(sym) = extract_named_item(node, source, file_path, scope, SymbolKind::Type)
            {
                symbols.push(sym);
            }
        }
        "const_item" | "static_item" => {
            if let Some(sym) =
                extract_named_item(node, source, file_path, scope, SymbolKind::Constant)
            {
                symbols.push(sym);
            }
        }
        "mod_item" => {
            if let Some(sym) =
                extract_named_item(node, source, file_path, scope, SymbolKind::Module)
            {
                let name = sym.name.clone();
                symbols.push(sym);
                // Recurse into module body for nested items — NOT in_impl
                let mut new_scope = scope.to_vec();
                new_scope.push(name);
                if let Some(body) = node.child_by_field_name("body") {
                    for i in 0..body.child_count() {
                        if let Some(child) = body.child(i as u32) {
                            extract_symbols_recursive(
                                child, source, file_path, &new_scope, false, symbols,
                            );
                        }
                    }
                }
                return; // already recursed
            }
        }
        _ => {}
    }

    // Default recursion for nodes we didn't handle specially
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            extract_symbols_recursive(child, source, file_path, scope, in_impl, symbols);
        }
    }
}

/// Extract a function_item as a Symbol.
///
/// Uses `in_impl` to determine if this is a Method (inside impl/trait block)
/// or a standalone Function.
fn extract_function(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    in_impl: bool,
) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let is_test = has_test_attribute(node, source);

    let kind = if is_test {
        SymbolKind::Test
    } else if in_impl {
        SymbolKind::Method
    } else {
        SymbolKind::Function
    };

    let visibility = extract_visibility(node, source);
    let signature = extract_signature(node, source);
    let doc_comment = extract_doc_comment(node, source);
    let qualified_name = build_qualified_name(scope, &name);

    Some(Symbol {
        name,
        qualified_name,
        kind,
        signature,
        visibility,
        file_path: file_path.to_string(),
        line_start: node.start_position().row,
        line_end: node.end_position().row,
        doc_comment,
        parent: if scope.is_empty() {
            None
        } else {
            Some(scope.join("::"))
        },
    })
}

/// Extract a named item (struct, enum, trait, type, const, static, mod) as a Symbol.
fn extract_named_item(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    kind: SymbolKind,
) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let visibility = extract_visibility(node, source);
    let signature = extract_signature(node, source);
    let doc_comment = extract_doc_comment(node, source);
    let qualified_name = build_qualified_name(scope, &name);

    Some(Symbol {
        name,
        qualified_name,
        kind,
        signature,
        visibility,
        file_path: file_path.to_string(),
        line_start: node.start_position().row,
        line_end: node.end_position().row,
        doc_comment,
        parent: if scope.is_empty() {
            None
        } else {
            Some(scope.join("::"))
        },
    })
}

// ── Reference Extraction ──────────────────────────────────────────────────

/// Recursively walk the AST and extract references.
fn extract_references_recursive(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    references: &mut Vec<Reference>,
) {
    match node.kind() {
        "use_declaration" => {
            extract_use_references(node, source, file_path, scope, references);
        }
        "call_expression" => {
            if let Some(r) = extract_call_reference(node, source, file_path, scope) {
                references.push(r);
            }
        }
        "macro_invocation" => {
            if let Some(r) = extract_macro_reference(node, source, file_path, scope) {
                references.push(r);
            }
        }
        "impl_item" => {
            // Check for trait implementation reference
            if let Some(r) = extract_impl_trait_reference(node, source, file_path, scope) {
                references.push(r);
            }
            // Update scope for methods inside impl
            let impl_target = get_impl_target_name(node, source);
            if let Some(target) = &impl_target {
                let mut new_scope = scope.to_vec();
                new_scope.push(target.clone());
                if let Some(body) = node.child_by_field_name("body") {
                    for i in 0..body.child_count() {
                        if let Some(child) = body.child(i as u32) {
                            extract_references_recursive(
                                child, source, file_path, &new_scope, references,
                            );
                        }
                    }
                }
                return; // already recursed
            }
        }
        "function_item" => {
            // Update scope for references inside functions
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = node_text(name_node, source);
                let mut new_scope = scope.to_vec();
                new_scope.push(name);
                if let Some(body) = node.child_by_field_name("body") {
                    for i in 0..body.child_count() {
                        if let Some(child) = body.child(i as u32) {
                            extract_references_recursive(
                                child, source, file_path, &new_scope, references,
                            );
                        }
                    }
                }
                return; // already recursed
            }
        }
        "mod_item" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = node_text(name_node, source);
                let mut new_scope = scope.to_vec();
                new_scope.push(name);
                if let Some(body) = node.child_by_field_name("body") {
                    for i in 0..body.child_count() {
                        if let Some(child) = body.child(i as u32) {
                            extract_references_recursive(
                                child, source, file_path, &new_scope, references,
                            );
                        }
                    }
                }
                return;
            }
        }
        "trait_item" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = node_text(name_node, source);
                let mut new_scope = scope.to_vec();
                new_scope.push(name);
                if let Some(body) = node.child_by_field_name("body") {
                    for i in 0..body.child_count() {
                        if let Some(child) = body.child(i as u32) {
                            extract_references_recursive(
                                child, source, file_path, &new_scope, references,
                            );
                        }
                    }
                }
                return;
            }
        }
        _ => {}
    }

    // Default recursion
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            extract_references_recursive(child, source, file_path, scope, references);
        }
    }
}

/// Extract Import references from a `use_declaration` node.
fn extract_use_references(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    references: &mut Vec<Reference>,
) {
    let use_text = node_text(node, source);
    // Remove "use " prefix and trailing ";"
    let trimmed = use_text
        .trim_start_matches("use ")
        .trim_end_matches(';')
        .trim();

    let source_qn = if scope.is_empty() {
        file_path.to_string()
    } else {
        scope.join("::")
    };

    references.push(Reference {
        source_qualified_name: source_qn,
        target_name: trimmed.to_string(),
        kind: ReferenceKind::Import,
        file_path: file_path.to_string(),
        line: node.start_position().row,
    });
}

/// Extract a Call reference from a `call_expression` node.
fn extract_call_reference(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Reference> {
    let function_node = node.child_by_field_name("function")?;
    let function_name = node_text(function_node, source);

    let source_qn = if scope.is_empty() {
        file_path.to_string()
    } else {
        scope.join("::")
    };

    Some(Reference {
        source_qualified_name: source_qn,
        target_name: function_name,
        kind: ReferenceKind::Call,
        file_path: file_path.to_string(),
        line: node.start_position().row,
    })
}

/// Extract a Call reference from a `macro_invocation` node.
fn extract_macro_reference(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Reference> {
    let macro_node = node.child_by_field_name("macro")?;
    let macro_name = node_text(macro_node, source);

    let source_qn = if scope.is_empty() {
        file_path.to_string()
    } else {
        scope.join("::")
    };

    Some(Reference {
        source_qualified_name: source_qn,
        target_name: format!("{}!", macro_name),
        kind: ReferenceKind::Call,
        file_path: file_path.to_string(),
        line: node.start_position().row,
    })
}

/// Extract an Implements reference from an `impl_item` that implements a trait.
/// e.g., `impl Foo for Bar { ... }` -> Bar implements Foo
fn extract_impl_trait_reference(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Reference> {
    let trait_node = node.child_by_field_name("trait")?;
    let type_node = node.child_by_field_name("type")?;

    let trait_name = node_text(trait_node, source);
    let type_name = node_text(type_node, source);

    let source_qn = if scope.is_empty() {
        type_name.clone()
    } else {
        format!("{}::{}", scope.join("::"), type_name)
    };

    Some(Reference {
        source_qualified_name: source_qn,
        target_name: trait_name,
        kind: ReferenceKind::Implements,
        file_path: file_path.to_string(),
        line: node.start_position().row,
    })
}

// ── Helper Functions ──────────────────────────────────────────────────────

/// Get the text content of a tree-sitter node.
fn node_text(node: Node, source: &[u8]) -> String {
    node.utf8_text(source).unwrap_or("").to_string()
}

/// Build a qualified name from scope and name.
fn build_qualified_name(scope: &[String], name: &str) -> String {
    if scope.is_empty() {
        name.to_string()
    } else {
        format!("{}::{}", scope.join("::"), name)
    }
}

/// Extract visibility from a node by checking for a `visibility_modifier` child.
fn extract_visibility(node: Node, source: &[u8]) -> Visibility {
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            if child.kind() == "visibility_modifier" {
                let text = node_text(child, source);
                if text.contains("pub(crate)") {
                    return Visibility::Crate;
                } else if text.starts_with("pub") {
                    return Visibility::Public;
                }
            }
        }
    }
    Visibility::Private
}

/// Extract the signature of a node (text up to the first `{`, or the whole node for short items).
fn extract_signature(node: Node, source: &[u8]) -> String {
    let text = node_text(node, source);
    // Find the first `{` and take everything before it
    if let Some(pos) = text.find('{') {
        text[..pos].trim().to_string()
    } else {
        // For items without braces (e.g., type alias, const), take the whole text
        // but limit to the first line if it's multi-line
        let first_line = text.lines().next().unwrap_or(&text);
        first_line.trim_end_matches(';').trim().to_string()
    }
}

/// Extract doc comments preceding a node.
/// Looks for `///` line comments or `//!` inner doc comments immediately before the node.
fn extract_doc_comment(node: Node, source: &[u8]) -> Option<String> {
    let mut comment_nodes = Vec::new();

    // Walk preceding siblings to find consecutive doc comments
    let mut prev = node.prev_sibling();

    while let Some(sibling) = prev {
        match sibling.kind() {
            "line_comment" => {
                let text = node_text(sibling, source);
                if text.starts_with("///") || text.starts_with("//!") {
                    comment_nodes.push(text);
                    prev = sibling.prev_sibling();
                    continue;
                }
                break;
            }
            "attribute_item" | "inner_attribute_item" => {
                // Skip attributes between doc comments and the item
                prev = sibling.prev_sibling();
                continue;
            }
            _ => break,
        }
    }

    // Reverse since we collected bottom-up
    comment_nodes.reverse();

    let doc_lines: Vec<String> = comment_nodes
        .iter()
        .map(|text| {
            let line = if let Some(stripped) = text.strip_prefix("/// ") {
                stripped
            } else if let Some(stripped) = text.strip_prefix("///") {
                stripped
            } else if let Some(stripped) = text.strip_prefix("//! ") {
                stripped
            } else if let Some(stripped) = text.strip_prefix("//!") {
                stripped
            } else {
                text.as_str()
            };
            line.trim_end().to_string()
        })
        .collect();

    if doc_lines.is_empty() {
        None
    } else {
        // Trim trailing whitespace from the joined doc string
        Some(doc_lines.join("\n").trim_end().to_string())
    }
}

/// Check if a node has a `#[test]` attribute.
fn has_test_attribute(node: Node, source: &[u8]) -> bool {
    let mut prev = node.prev_sibling();
    while let Some(sibling) = prev {
        match sibling.kind() {
            "attribute_item" => {
                let text = node_text(sibling, source);
                if text.contains("test") {
                    return true;
                }
                prev = sibling.prev_sibling();
            }
            "line_comment" => {
                // Skip comments between attributes and the item
                prev = sibling.prev_sibling();
            }
            _ => break,
        }
    }
    false
}

/// Get the target type name from an `impl_item` node.
/// e.g., `impl Foo { ... }` returns Some("Foo")
/// e.g., `impl Trait for Foo { ... }` returns Some("Foo")
fn get_impl_target_name(node: Node, source: &[u8]) -> Option<String> {
    if let Some(type_node) = node.child_by_field_name("type") {
        return Some(node_text(type_node, source));
    }
    None
}

#[cfg(test)]
#[path = "tests/rust_tests.rs"]
mod tests;
