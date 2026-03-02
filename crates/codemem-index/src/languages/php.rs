//! PHP language extractor using tree-sitter-php.

use crate::extractor::LanguageExtractor;
use crate::symbol::{Reference, ReferenceKind, Symbol, SymbolKind, Visibility};
use tree_sitter::{Node, Tree};

/// PHP language extractor for tree-sitter-based code indexing.
pub struct PhpExtractor;

impl PhpExtractor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for PhpExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageExtractor for PhpExtractor {
    fn language_name(&self) -> &str {
        "php"
    }

    fn file_extensions(&self) -> &[&str] {
        &["php"]
    }

    fn tree_sitter_language(&self) -> tree_sitter::Language {
        tree_sitter_php::LANGUAGE_PHP.into()
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

// ── Symbol Extraction ─────────────────────────────────────────────────────

fn extract_symbols_recursive(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    symbols: &mut Vec<Symbol>,
) {
    match node.kind() {
        "function_definition" => {
            if let Some(sym) = extract_function(node, source, file_path, scope) {
                symbols.push(sym);
            }
            return; // Don't recurse into function bodies for symbols
        }
        "class_declaration" => {
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
        "method_declaration" => {
            if let Some(sym) = extract_method(node, source, file_path, scope) {
                symbols.push(sym);
            }
            return; // Don't recurse into method bodies for symbols
        }
        "namespace_definition" => {
            if let Some(sym) = extract_namespace(node, source, file_path, scope) {
                let ns_name = sym.name.clone();
                symbols.push(sym);
                // Recurse into namespace body with updated scope
                if let Some(body) = node.child_by_field_name("body") {
                    let mut new_scope = scope.to_vec();
                    new_scope.push(ns_name);
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
        "interface_declaration" => {
            if let Some(sym) = extract_interface(node, source, file_path, scope) {
                let iface_name = sym.name.clone();
                symbols.push(sym);
                if let Some(body) = node.child_by_field_name("body") {
                    let mut new_scope = scope.to_vec();
                    new_scope.push(iface_name);
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
        "trait_declaration" => {
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
        _ => {}
    }

    // Default recursion for other node types
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            extract_symbols_recursive(child, source, file_path, scope, symbols);
        }
    }
}

fn extract_function(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let qualified_name = qualified_ns(scope, &name);
    let visibility = Visibility::Public; // Top-level functions are always public in PHP
    let signature = extract_php_signature(node, source);
    let doc_comment = extract_phpdoc(node, source);

    Some(Symbol {
        name,
        qualified_name,
        kind: SymbolKind::Function,
        signature,
        visibility,
        file_path: file_path.to_string(),
        line_start: node.start_position().row,
        line_end: node.end_position().row,
        doc_comment,
        parent: scope.last().cloned(),
    })
}

fn extract_class(node: Node, source: &[u8], file_path: &str, scope: &[String]) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let qualified_name = qualified_ns(scope, &name);
    let visibility = Visibility::Public;
    let signature = extract_php_signature(node, source);
    let doc_comment = extract_phpdoc(node, source);

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

fn extract_method(node: Node, source: &[u8], file_path: &str, scope: &[String]) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let qualified_name = qualified_member(scope, &name);
    let visibility = extract_visibility(node, source);
    let signature = extract_php_signature(node, source);
    let doc_comment = extract_phpdoc(node, source);

    Some(Symbol {
        name,
        qualified_name,
        kind: SymbolKind::Method,
        signature,
        visibility,
        file_path: file_path.to_string(),
        line_start: node.start_position().row,
        line_end: node.end_position().row,
        doc_comment,
        parent: scope.last().cloned(),
    })
}

fn extract_namespace(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let qualified_name = qualified_ns(scope, &name);
    let visibility = Visibility::Public;
    let signature = extract_php_signature(node, source);
    let doc_comment = extract_phpdoc(node, source);

    Some(Symbol {
        name,
        qualified_name,
        kind: SymbolKind::Module,
        signature,
        visibility,
        file_path: file_path.to_string(),
        line_start: node.start_position().row,
        line_end: node.end_position().row,
        doc_comment,
        parent: scope.last().cloned(),
    })
}

fn extract_interface(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let qualified_name = qualified_ns(scope, &name);
    let visibility = Visibility::Public;
    let signature = extract_php_signature(node, source);
    let doc_comment = extract_phpdoc(node, source);

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

fn extract_trait(node: Node, source: &[u8], file_path: &str, scope: &[String]) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let qualified_name = qualified_ns(scope, &name);
    let visibility = Visibility::Public;
    let signature = extract_php_signature(node, source);
    let doc_comment = extract_phpdoc(node, source);

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

// ── Reference Extraction ──────────────────────────────────────────────────

fn extract_references_recursive(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    references: &mut Vec<Reference>,
) {
    match node.kind() {
        "namespace_use_declaration" => {
            extract_use_references(node, source, file_path, scope, references);
            return;
        }
        "function_call_expression" => {
            if let Some(r) = extract_call_reference(node, source, file_path, scope) {
                references.push(r);
            }
            // Still recurse into arguments etc.
        }
        "class_declaration" | "interface_declaration" | "trait_declaration" => {
            // Extract inheritance/implementation references
            extract_type_references(node, source, file_path, scope, references);

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
        "method_declaration" => {
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
        "namespace_definition" => {
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

fn extract_use_references(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    references: &mut Vec<Reference>,
) {
    // PHP use declarations: `use App\Models\User;` or `use App\Models\{User, Post};`
    // Walk children to find qualified_name or namespace_use_clause nodes
    let source_qn = if scope.is_empty() {
        file_path.to_string()
    } else {
        scope.join("\\")
    };

    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            match child.kind() {
                "qualified_name" | "name" => {
                    let use_path = node_text(child, source);
                    if !use_path.is_empty() {
                        references.push(Reference {
                            source_qualified_name: source_qn.clone(),
                            target_name: use_path,
                            kind: ReferenceKind::Import,
                            file_path: file_path.to_string(),
                            line: node.start_position().row,
                        });
                    }
                }
                "namespace_use_clause" | "namespace_use_group_clause" => {
                    // Recurse to find the qualified name inside the clause
                    for j in 0..child.child_count() {
                        if let Some(inner) = child.child(j as u32) {
                            if inner.kind() == "qualified_name" || inner.kind() == "name" {
                                let use_path = node_text(inner, source);
                                if !use_path.is_empty() {
                                    references.push(Reference {
                                        source_qualified_name: source_qn.clone(),
                                        target_name: use_path,
                                        kind: ReferenceKind::Import,
                                        file_path: file_path.to_string(),
                                        line: node.start_position().row,
                                    });
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // Fallback: if no children matched, try extracting from the full text
    if references
        .iter()
        .all(|r| r.line != node.start_position().row)
    {
        let text = node_text(node, source);
        // Parse "use Foo\Bar\Baz;" style
        let import_path = text
            .trim()
            .strip_prefix("use")
            .unwrap_or("")
            .trim()
            .strip_suffix(';')
            .unwrap_or("")
            .trim()
            .to_string();

        if !import_path.is_empty() {
            references.push(Reference {
                source_qualified_name: source_qn,
                target_name: import_path,
                kind: ReferenceKind::Import,
                file_path: file_path.to_string(),
                line: node.start_position().row,
            });
        }
    }
}

fn extract_call_reference(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Reference> {
    let func_node = node.child_by_field_name("function")?;
    let func_name = node_text(func_node, source);

    if func_name.is_empty() {
        return None;
    }

    let source_qn = if scope.is_empty() {
        file_path.to_string()
    } else {
        scope.join("::")
    };

    Some(Reference {
        source_qualified_name: source_qn,
        target_name: func_name,
        kind: ReferenceKind::Call,
        file_path: file_path.to_string(),
        line: node.start_position().row,
    })
}

fn extract_type_references(
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
    let qn = qualified_ns(scope, &class_name);

    // Look for base_clause (extends) and class_interface_clause (implements)
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            match child.kind() {
                "base_clause" => {
                    // base_clause contains the types being extended
                    for j in 0..child.child_count() {
                        if let Some(type_node) = child.child(j as u32) {
                            match type_node.kind() {
                                "qualified_name" | "name" => {
                                    let parent_name = node_text(type_node, source);
                                    if !parent_name.is_empty() {
                                        references.push(Reference {
                                            source_qualified_name: qn.clone(),
                                            target_name: parent_name,
                                            kind: ReferenceKind::Inherits,
                                            file_path: file_path.to_string(),
                                            line: type_node.start_position().row,
                                        });
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
                "class_interface_clause" => {
                    // class_interface_clause contains the interfaces being implemented
                    for j in 0..child.child_count() {
                        if let Some(type_node) = child.child(j as u32) {
                            match type_node.kind() {
                                "qualified_name" | "name" => {
                                    let iface_name = node_text(type_node, source);
                                    if !iface_name.is_empty() {
                                        references.push(Reference {
                                            source_qualified_name: qn.clone(),
                                            target_name: iface_name,
                                            kind: ReferenceKind::Implements,
                                            file_path: file_path.to_string(),
                                            line: type_node.start_position().row,
                                        });
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

// ── Helper Functions ──────────────────────────────────────────────────────

fn node_text(node: Node, source: &[u8]) -> String {
    node.utf8_text(source).unwrap_or("").to_string()
}

/// Build a qualified name using `\` as the namespace separator (PHP convention).
fn qualified_ns(scope: &[String], name: &str) -> String {
    if scope.is_empty() {
        name.to_string()
    } else {
        format!("{}\\{}", scope.join("\\"), name)
    }
}

/// Build a qualified name using `::` for class member access (PHP convention).
fn qualified_member(scope: &[String], name: &str) -> String {
    if scope.is_empty() {
        name.to_string()
    } else {
        format!("{}::{}", scope.join("\\"), name)
    }
}

/// Extract visibility from a node by checking for `visibility_modifier` children.
fn extract_visibility(node: Node, source: &[u8]) -> Visibility {
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            if child.kind() == "visibility_modifier" {
                let text = node_text(child, source);
                return match text.as_str() {
                    "private" => Visibility::Private,
                    "protected" => Visibility::Protected,
                    "public" => Visibility::Public,
                    _ => Visibility::Public,
                };
            }
        }
    }
    Visibility::Public
}

/// Extract the signature: everything up to the opening `{`.
fn extract_php_signature(node: Node, source: &[u8]) -> String {
    let text = node_text(node, source);
    if let Some(pos) = text.find('{') {
        text[..pos].trim().to_string()
    } else {
        let first_line = text.lines().next().unwrap_or(&text);
        first_line.trim().to_string()
    }
}

/// Extract PHPDoc comment (`/** ... */`) preceding the node.
fn extract_phpdoc(node: Node, source: &[u8]) -> Option<String> {
    let mut prev = node.prev_sibling();
    while let Some(sibling) = prev {
        match sibling.kind() {
            "comment" => {
                let text = node_text(sibling, source);
                if text.starts_with("/**") {
                    return Some(clean_phpdoc(&text));
                }
                // Skip non-PHPDoc comments, keep looking
                prev = sibling.prev_sibling();
                continue;
            }
            _ => return None,
        }
    }
    None
}

/// Clean up a PHPDoc comment by removing delimiters and leading `*`.
fn clean_phpdoc(raw: &str) -> String {
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
#[path = "tests/php_tests.rs"]
mod tests;
