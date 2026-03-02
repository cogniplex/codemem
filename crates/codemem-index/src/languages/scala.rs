//! Scala language extractor using tree-sitter-scala.

use crate::extractor::LanguageExtractor;
use crate::symbol::{Reference, ReferenceKind, Symbol, SymbolKind, Visibility};
use tree_sitter::{Node, Tree};

/// Scala language extractor for tree-sitter-based code indexing.
pub struct ScalaExtractor;

impl ScalaExtractor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ScalaExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageExtractor for ScalaExtractor {
    fn language_name(&self) -> &str {
        "scala"
    }

    fn file_extensions(&self) -> &[&str] {
        &["scala", "sc"]
    }

    fn tree_sitter_language(&self) -> tree_sitter::Language {
        tree_sitter_scala::LANGUAGE.into()
    }

    fn extract_symbols(&self, tree: &Tree, source: &[u8], file_path: &str) -> Vec<Symbol> {
        let mut symbols = Vec::new();
        let root = tree.root_node();
        extract_symbols_recursive(root, source, file_path, &[], &mut symbols);
        symbols
    }

    fn extract_references(&self, tree: &Tree, source: &[u8], file_path: &str) -> Vec<Reference> {
        let mut references = Vec::new();
        let root = tree.root_node();
        extract_references_recursive(root, source, file_path, &[], &mut references);
        references
    }
}

// -- Symbol Extraction --------------------------------------------------------

fn extract_symbols_recursive(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    symbols: &mut Vec<Symbol>,
) {
    match node.kind() {
        "class_definition" => {
            if let Some(sym) = extract_class(node, source, file_path, scope) {
                let class_name = sym.name.clone();
                symbols.push(sym);
                // Recurse into class body with updated scope
                if let Some(body) = node.child_by_field_name("body") {
                    let mut new_scope = scope.to_vec();
                    new_scope.push(class_name);
                    for i in 0..body.child_count() {
                        if let Some(child) = body.child(i as u32) {
                            extract_symbols_recursive(
                                child, source, file_path, &new_scope, symbols,
                            );
                        }
                    }
                }
                return; // Don't default-recurse
            }
        }
        "trait_definition" => {
            if let Some(sym) = extract_trait(node, source, file_path, scope) {
                let trait_name = sym.name.clone();
                symbols.push(sym);
                if let Some(body) = node.child_by_field_name("body") {
                    let mut new_scope = scope.to_vec();
                    new_scope.push(trait_name);
                    for i in 0..body.child_count() {
                        if let Some(child) = body.child(i as u32) {
                            extract_symbols_recursive(
                                child, source, file_path, &new_scope, symbols,
                            );
                        }
                    }
                }
                return;
            }
        }
        "object_definition" => {
            if let Some(sym) = extract_object(node, source, file_path, scope) {
                let obj_name = sym.name.clone();
                symbols.push(sym);
                if let Some(body) = node.child_by_field_name("body") {
                    let mut new_scope = scope.to_vec();
                    new_scope.push(obj_name);
                    for i in 0..body.child_count() {
                        if let Some(child) = body.child(i as u32) {
                            extract_symbols_recursive(
                                child, source, file_path, &new_scope, symbols,
                            );
                        }
                    }
                }
                return;
            }
        }
        "function_definition" | "function_declaration" => {
            if let Some(sym) = extract_function(node, source, file_path, scope) {
                symbols.push(sym);
            }
            return; // Don't recurse into function bodies for symbols
        }
        "val_definition" | "val_declaration" => {
            if let Some(sym) = extract_val_constant(node, source, file_path, scope) {
                symbols.push(sym);
            }
            return;
        }
        _ => {}
    }

    // Default recursion for other node types
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            extract_symbols_recursive(child, source, file_path, scope, symbols);
        }
    }
}

fn extract_class(node: Node, source: &[u8], file_path: &str, scope: &[String]) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let qualified_name = qualified(scope, &name);
    let visibility = extract_visibility(node, source);
    let signature = extract_scala_signature(node, source);
    let doc_comment = extract_scaladoc(node, source);

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
        parent: scope.last().cloned(),
    })
}

fn extract_trait(node: Node, source: &[u8], file_path: &str, scope: &[String]) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let qualified_name = qualified(scope, &name);
    let visibility = extract_visibility(node, source);
    let signature = extract_scala_signature(node, source);
    let doc_comment = extract_scaladoc(node, source);

    Some(Symbol {
        name,
        qualified_name,
        kind: SymbolKind::Interface,
        signature,
        visibility,
        file_path: file_path.to_string(),
        line_start: node.start_position().row,
        line_end: node.end_position().row,
        doc_comment,
        parent: scope.last().cloned(),
    })
}

fn extract_object(node: Node, source: &[u8], file_path: &str, scope: &[String]) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let qualified_name = qualified(scope, &name);
    let visibility = extract_visibility(node, source);
    let signature = extract_scala_signature(node, source);
    let doc_comment = extract_scaladoc(node, source);

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
        parent: scope.last().cloned(),
    })
}

fn extract_function(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let qualified_name = qualified(scope, &name);
    let visibility = extract_visibility(node, source);
    let signature = extract_scala_signature(node, source);
    let doc_comment = extract_scaladoc(node, source);

    // If inside a class/trait/object (scope is non-empty), it's a Method; otherwise Function
    let kind = if scope.is_empty() {
        SymbolKind::Function
    } else {
        SymbolKind::Method
    };

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
        parent: scope.last().cloned(),
    })
}

/// Extract a `val` definition as a Constant only if the `final` modifier is present.
fn extract_val_constant(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Symbol> {
    // Only extract vals with the `final` modifier as constants
    if !has_modifier(node, source, "final") {
        return None;
    }

    // The val_definition pattern field holds the identifier(s)
    let name_node = node.child_by_field_name("pattern")?;
    let name = node_text(name_node, source);

    let qualified_name = qualified(scope, &name);
    let visibility = extract_visibility(node, source);
    let signature = extract_scala_signature(node, source);
    let doc_comment = extract_scaladoc(node, source);

    Some(Symbol {
        name,
        qualified_name,
        kind: SymbolKind::Constant,
        signature,
        visibility,
        file_path: file_path.to_string(),
        line_start: node.start_position().row,
        line_end: node.end_position().row,
        doc_comment,
        parent: scope.last().cloned(),
    })
}

// -- Reference Extraction -----------------------------------------------------

fn extract_references_recursive(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    references: &mut Vec<Reference>,
) {
    match node.kind() {
        "import_declaration" => {
            extract_import_references(node, source, file_path, scope, references);
            return;
        }
        "call_expression" => {
            if let Some(r) = extract_call_reference(node, source, file_path, scope) {
                references.push(r);
            }
            // Still recurse into arguments etc.
        }
        "class_definition" | "trait_definition" | "object_definition" => {
            // Extract inheritance references from extends clause
            extract_extends_references(node, source, file_path, scope, references);

            // Recurse with updated scope
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = node_text(name_node, source);
                let mut new_scope = scope.to_vec();
                new_scope.push(name);
                for i in 0..node.child_count() {
                    if let Some(child) = node.child(i as u32) {
                        extract_references_recursive(
                            child, source, file_path, &new_scope, references,
                        );
                    }
                }
                return;
            }
        }
        "function_definition" | "function_declaration" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = node_text(name_node, source);
                let mut new_scope = scope.to_vec();
                new_scope.push(name);
                for i in 0..node.child_count() {
                    if let Some(child) = node.child(i as u32) {
                        extract_references_recursive(
                            child, source, file_path, &new_scope, references,
                        );
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

fn extract_import_references(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    references: &mut Vec<Reference>,
) {
    // import_declaration text is like: "import scala.collection.mutable.ListBuffer"
    // or "import scala.collection.mutable.{ListBuffer, ArrayBuffer}"
    let text = node_text(node, source);
    let import_text = text.trim();

    // Strip "import " prefix
    let path_str = match import_text.strip_prefix("import") {
        Some(rest) => rest.trim(),
        None => return,
    };

    if path_str.is_empty() {
        return;
    }

    let source_qn = if scope.is_empty() {
        file_path.to_string()
    } else {
        scope.join(".")
    };

    // Handle comma-separated imports like "import a.B, c.D"
    for segment in path_str.split(',') {
        let segment = segment.trim();
        if segment.is_empty() {
            continue;
        }
        // Handle wildcard and selector imports by taking the full text
        references.push(Reference {
            source_qualified_name: source_qn.clone(),
            target_name: segment.to_string(),
            kind: ReferenceKind::Import,
            file_path: file_path.to_string(),
            line: node.start_position().row,
        });
    }
}

fn extract_call_reference(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Reference> {
    let func_node = node.child_by_field_name("function")?;
    let target = node_text(func_node, source);

    if target.is_empty() {
        return None;
    }

    let source_qn = if scope.is_empty() {
        file_path.to_string()
    } else {
        scope.join(".")
    };

    Some(Reference {
        source_qualified_name: source_qn,
        target_name: target,
        kind: ReferenceKind::Call,
        file_path: file_path.to_string(),
        line: node.start_position().row,
    })
}

fn extract_extends_references(
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
    let qn = qualified(scope, &class_name);

    // Find the extends_clause child via the "extend" field
    let extends_node = match node.child_by_field_name("extend") {
        Some(n) => n,
        None => return,
    };

    // The extends_clause contains one or more type references
    // Walk its children looking for type identifiers
    extract_extends_types(extends_node, source, file_path, &qn, references);
}

fn extract_extends_types(
    node: Node,
    source: &[u8],
    file_path: &str,
    source_qn: &str,
    references: &mut Vec<Reference>,
) {
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            match child.kind() {
                "type_identifier" | "generic_type" | "stable_type_identifier" => {
                    let type_name = node_text(child, source);
                    if !type_name.is_empty() {
                        references.push(Reference {
                            source_qualified_name: source_qn.to_string(),
                            target_name: type_name,
                            kind: ReferenceKind::Inherits,
                            file_path: file_path.to_string(),
                            line: child.start_position().row,
                        });
                    }
                }
                // Recurse into compound type expressions
                _ => {
                    extract_extends_types(child, source, file_path, source_qn, references);
                }
            }
        }
    }
}

// -- Helper Functions ---------------------------------------------------------

fn node_text(node: Node, source: &[u8]) -> String {
    node.utf8_text(source).unwrap_or("").to_string()
}

fn qualified(scope: &[String], name: &str) -> String {
    if scope.is_empty() {
        name.to_string()
    } else {
        format!("{}.{}", scope.join("."), name)
    }
}

/// Check whether a node has a `modifiers` child containing a specific modifier keyword.
fn has_modifier(node: Node, source: &[u8], modifier: &str) -> bool {
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            if child.kind() == "modifiers" {
                for j in 0..child.child_count() {
                    if let Some(mod_child) = child.child(j as u32) {
                        let text = node_text(mod_child, source);
                        if text == modifier {
                            return true;
                        }
                    }
                }
                return false;
            }
        }
    }
    false
}

/// Check whether a node has an `access_modifier` containing `private` or `protected`.
fn extract_visibility(node: Node, source: &[u8]) -> Visibility {
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            if child.kind() == "modifiers" {
                return visibility_from_modifiers(child, source);
            }
        }
    }
    // Scala default visibility is public
    Visibility::Public
}

fn visibility_from_modifiers(modifiers_node: Node, source: &[u8]) -> Visibility {
    for i in 0..modifiers_node.child_count() {
        if let Some(child) = modifiers_node.child(i as u32) {
            if child.kind() == "access_modifier" {
                let text = node_text(child, source);
                if text.starts_with("private") {
                    return Visibility::Private;
                } else if text.starts_with("protected") {
                    return Visibility::Protected;
                }
            }
        }
    }
    // No access modifier found; Scala default is public
    Visibility::Public
}

fn extract_scala_signature(node: Node, source: &[u8]) -> String {
    let text = node_text(node, source);
    // Signature is everything up to the opening brace or equals sign (for short definitions)
    if let Some(pos) = text.find('{') {
        text[..pos].trim().to_string()
    } else if let Some(pos) = text.find('=') {
        text[..pos].trim().to_string()
    } else {
        let first_line = text.lines().next().unwrap_or(&text);
        first_line.trim().to_string()
    }
}

fn extract_scaladoc(node: Node, source: &[u8]) -> Option<String> {
    // Look for a block_comment (Scaladoc) immediately preceding this node
    let mut prev = node.prev_sibling();
    while let Some(sibling) = prev {
        match sibling.kind() {
            "block_comment" => {
                let text = node_text(sibling, source);
                if text.starts_with("/**") {
                    return Some(clean_scaladoc(&text));
                }
                return None;
            }
            "comment" => {
                // Skip line comments, look for Scaladoc before them
                prev = sibling.prev_sibling();
                continue;
            }
            _ => return None,
        }
    }
    None
}

fn clean_scaladoc(raw: &str) -> String {
    // Remove /** and */ delimiters, strip leading * from each line
    let trimmed = raw
        .strip_prefix("/**")
        .unwrap_or(raw)
        .strip_suffix("*/")
        .unwrap_or(raw);

    let lines: Vec<&str> = trimmed.lines().collect();
    let mut result_lines = Vec::new();

    for line in &lines {
        let stripped = line.trim();
        let cleaned = if let Some(rest) = stripped.strip_prefix("* ") {
            rest
        } else if stripped == "*" {
            ""
        } else {
            stripped
        };
        result_lines.push(cleaned);
    }

    // Trim leading/trailing empty lines
    while result_lines.first().is_some_and(|l| l.is_empty()) {
        result_lines.remove(0);
    }
    while result_lines.last().is_some_and(|l| l.is_empty()) {
        result_lines.pop();
    }

    result_lines.join("\n")
}


#[cfg(test)]
#[path = "tests/scala_tests.rs"]
mod tests;
