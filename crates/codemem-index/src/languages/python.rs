//! Python language extractor using tree-sitter-python.

use crate::extractor::LanguageExtractor;
use crate::symbol::{Reference, ReferenceKind, Symbol, SymbolKind, Visibility};
use tree_sitter::{Node, Tree};

/// Python language extractor for tree-sitter-based code indexing.
pub struct PythonExtractor;

impl PythonExtractor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for PythonExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageExtractor for PythonExtractor {
    fn language_name(&self) -> &str {
        "python"
    }

    fn file_extensions(&self) -> &[&str] {
        &["py"]
    }

    fn tree_sitter_language(&self) -> tree_sitter::Language {
        tree_sitter_python::LANGUAGE.into()
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

fn extract_symbols_recursive(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    in_class: bool,
    symbols: &mut Vec<Symbol>,
) {
    match node.kind() {
        "function_definition" => {
            if let Some(sym) = extract_function(node, source, file_path, scope, in_class) {
                let name = sym.name.clone();
                symbols.push(sym);
                // Recurse into function body for nested definitions
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
                return;
            }
        }
        "class_definition" => {
            if let Some(sym) = extract_class(node, source, file_path, scope) {
                let name = sym.name.clone();
                symbols.push(sym);
                // Recurse into class body for methods
                let mut new_scope = scope.to_vec();
                new_scope.push(name);
                if let Some(body) = node.child_by_field_name("body") {
                    for i in 0..body.child_count() {
                        if let Some(child) = body.child(i as u32) {
                            extract_symbols_recursive(
                                child, source, file_path, &new_scope, true, symbols,
                            );
                        }
                    }
                }
                return;
            }
        }
        "decorated_definition" => {
            // Unwrap to inner definition
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i as u32) {
                    match child.kind() {
                        "function_definition" | "class_definition" => {
                            extract_symbols_recursive(
                                child, source, file_path, scope, in_class, symbols,
                            );
                            return;
                        }
                        _ => {}
                    }
                }
            }
        }
        "expression_statement" => {
            // Top-level assignments: UPPER_CASE = value → Constant
            if !in_class && scope.is_empty() {
                if let Some(sym) = extract_constant_assignment(node, source, file_path, scope) {
                    symbols.push(sym);
                    return;
                }
            }
        }
        _ => {}
    }

    // Default recursion
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            extract_symbols_recursive(child, source, file_path, scope, in_class, symbols);
        }
    }
}

fn extract_function(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    in_class: bool,
) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let is_test =
        name.starts_with("test_") || name.starts_with("test") && has_test_decorator(node, source);

    let kind = if is_test {
        SymbolKind::Test
    } else if in_class {
        SymbolKind::Method
    } else {
        SymbolKind::Function
    };

    let visibility = python_visibility(&name);
    let signature = extract_function_signature(node, source);
    let doc_comment = extract_docstring(node, source);
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
            Some(scope.join("."))
        },
    })
}

fn extract_class(node: Node, source: &[u8], file_path: &str, scope: &[String]) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let visibility = python_visibility(&name);
    let signature = extract_class_signature(node, source);
    let doc_comment = extract_docstring(node, source);
    let qualified_name = build_qualified_name(scope, &name);

    Some(Symbol {
        name,
        qualified_name,
        kind: SymbolKind::Class,
        signature,
        visibility,
        file_path: file_path.to_string(),
        line_start: node.start_position().row,
        line_end: node.end_position().row,
        doc_comment,
        parent: if scope.is_empty() {
            None
        } else {
            Some(scope.join("."))
        },
    })
}

fn extract_constant_assignment(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Symbol> {
    // Look for assignment: NAME = value where NAME is UPPER_CASE
    let child = node.child(0)?;
    if child.kind() != "assignment" {
        return None;
    }
    let left = child.child_by_field_name("left")?;
    if left.kind() != "identifier" {
        return None;
    }
    let name = node_text(left, source);
    // Check if name is UPPER_CASE (at least 2 chars, all uppercase/underscore/digits)
    if name.len() < 2
        || !name
            .chars()
            .all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit())
    {
        return None;
    }

    let signature = node_text(child, source);
    let first_line = signature.lines().next().unwrap_or(&signature);
    let qualified_name = build_qualified_name(scope, &name);

    Some(Symbol {
        name,
        qualified_name,
        kind: SymbolKind::Constant,
        signature: first_line.to_string(),
        visibility: Visibility::Public,
        file_path: file_path.to_string(),
        line_start: node.start_position().row,
        line_end: node.end_position().row,
        doc_comment: None,
        parent: None,
    })
}

// ── Reference Extraction ──────────────────────────────────────────────────

fn extract_references_recursive(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    references: &mut Vec<Reference>,
) {
    match node.kind() {
        "import_statement" => {
            extract_import_reference(node, source, file_path, scope, references);
        }
        "import_from_statement" => {
            extract_import_from_reference(node, source, file_path, scope, references);
        }
        "call" => {
            if let Some(r) = extract_call_reference(node, source, file_path, scope) {
                references.push(r);
            }
        }
        "class_definition" => {
            // Extract base class references
            extract_class_bases(node, source, file_path, scope, references);
            // Recurse with updated scope
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
        "function_definition" => {
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
        "decorated_definition" => {
            // Recurse into inner definition
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i as u32) {
                    extract_references_recursive(child, source, file_path, scope, references);
                }
            }
            return;
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

fn extract_import_reference(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    references: &mut Vec<Reference>,
) {
    // import foo, bar, baz
    let source_qn = scope_qn(scope, file_path);
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            if child.kind() == "dotted_name" {
                references.push(Reference {
                    source_qualified_name: source_qn.clone(),
                    target_name: node_text(child, source),
                    kind: ReferenceKind::Import,
                    file_path: file_path.to_string(),
                    line: node.start_position().row,
                });
            }
        }
    }
}

fn extract_import_from_reference(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    references: &mut Vec<Reference>,
) {
    // from module import name1, name2
    let source_qn = scope_qn(scope, file_path);
    let module = node
        .child_by_field_name("module_name")
        .map(|n| node_text(n, source))
        .unwrap_or_default();

    references.push(Reference {
        source_qualified_name: source_qn,
        target_name: module,
        kind: ReferenceKind::Import,
        file_path: file_path.to_string(),
        line: node.start_position().row,
    });
}

fn extract_call_reference(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Reference> {
    let function_node = node.child_by_field_name("function")?;
    let function_name = node_text(function_node, source);
    let source_qn = scope_qn(scope, file_path);

    Some(Reference {
        source_qualified_name: source_qn,
        target_name: function_name,
        kind: ReferenceKind::Call,
        file_path: file_path.to_string(),
        line: node.start_position().row,
    })
}

fn extract_class_bases(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    references: &mut Vec<Reference>,
) {
    let name_node = match node.child_by_field_name("name") {
        Some(n) => n,
        None => return,
    };
    let class_name = node_text(name_node, source);
    let class_qn = build_qualified_name(scope, &class_name);

    // Look for superclasses in argument_list
    if let Some(superclasses) = node.child_by_field_name("superclasses") {
        for i in 0..superclasses.child_count() {
            if let Some(child) = superclasses.child(i as u32) {
                match child.kind() {
                    "identifier" | "attribute" => {
                        let base_name = node_text(child, source);
                        references.push(Reference {
                            source_qualified_name: class_qn.clone(),
                            target_name: base_name,
                            kind: ReferenceKind::Inherits,
                            file_path: file_path.to_string(),
                            line: child.start_position().row,
                        });
                    }
                    _ => {}
                }
            }
        }
    }
}

// ── Helper Functions ──────────────────────────────────────────────────────

fn node_text(node: Node, source: &[u8]) -> String {
    node.utf8_text(source).unwrap_or("").to_string()
}

fn build_qualified_name(scope: &[String], name: &str) -> String {
    if scope.is_empty() {
        name.to_string()
    } else {
        format!("{}.{}", scope.join("."), name)
    }
}

fn scope_qn(scope: &[String], file_path: &str) -> String {
    if scope.is_empty() {
        file_path.to_string()
    } else {
        scope.join(".")
    }
}

fn python_visibility(name: &str) -> Visibility {
    if (name.starts_with("__") && !name.ends_with("__")) || name.starts_with('_') {
        Visibility::Private
    } else {
        Visibility::Public
    }
}

fn extract_function_signature(node: Node, source: &[u8]) -> String {
    let text = node_text(node, source);
    // Take up to the first colon at end of def line
    if let Some(pos) = text.find(':') {
        text[..pos].trim().to_string()
    } else {
        let first_line = text.lines().next().unwrap_or(&text);
        first_line.trim_end_matches(':').trim().to_string()
    }
}

fn extract_class_signature(node: Node, source: &[u8]) -> String {
    let text = node_text(node, source);
    let first_line = text.lines().next().unwrap_or(&text);
    first_line.trim_end_matches(':').trim().to_string()
}

fn extract_docstring(node: Node, source: &[u8]) -> Option<String> {
    // Python docstrings: first statement in body that is a string expression
    let body = node.child_by_field_name("body")?;
    let first_stmt = body.child(0)?;
    if first_stmt.kind() != "expression_statement" {
        return None;
    }
    let expr = first_stmt.child(0)?;
    if expr.kind() != "string" {
        return None;
    }

    let raw = node_text(expr, source);
    // Strip triple quotes
    let stripped = raw
        .trim_start_matches("\"\"\"")
        .trim_start_matches("'''")
        .trim_end_matches("\"\"\"")
        .trim_end_matches("'''")
        .trim();
    if stripped.is_empty() {
        None
    } else {
        Some(stripped.to_string())
    }
}

fn has_test_decorator(node: Node, source: &[u8]) -> bool {
    let mut prev = node.prev_sibling();
    while let Some(sibling) = prev {
        if sibling.kind() == "decorator" {
            let text = node_text(sibling, source);
            if text.contains("test") || text.contains("pytest") {
                return true;
            }
        }
        prev = sibling.prev_sibling();
    }
    false
}

#[cfg(test)]
#[path = "tests/python_tests.rs"]
mod tests;
