//! Ruby language extractor using tree-sitter-ruby.

use crate::extractor::LanguageExtractor;
use crate::symbol::{Reference, ReferenceKind, Symbol, SymbolKind, Visibility};
use tree_sitter::{Node, Tree};

/// Ruby language extractor for tree-sitter-based code indexing.
pub struct RubyExtractor;

impl RubyExtractor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for RubyExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageExtractor for RubyExtractor {
    fn language_name(&self) -> &str {
        "ruby"
    }

    fn file_extensions(&self) -> &[&str] {
        &["rb"]
    }

    fn tree_sitter_language(&self) -> tree_sitter::Language {
        tree_sitter_ruby::LANGUAGE.into()
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
        "class" => {
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
        "module" => {
            if let Some(sym) = extract_module(node, source, file_path, scope) {
                let module_name = sym.name.clone();
                symbols.push(sym);
                // Recurse into module body with updated scope
                if let Some(body) = node.child_by_field_name("body") {
                    let mut new_scope = scope.to_vec();
                    new_scope.push(module_name);
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
        "method" => {
            if let Some(sym) = extract_method(node, source, file_path, scope) {
                symbols.push(sym);
            }
            return; // Don't recurse into method bodies for symbols
        }
        "singleton_method" => {
            if let Some(sym) = extract_singleton_method(node, source, file_path, scope) {
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
    let signature = extract_ruby_signature(node, source);
    let doc_comment = extract_ruby_comment(node, source);

    Some(Symbol {
        name,
        qualified_name,
        kind: SymbolKind::Class,
        signature,
        visibility: Visibility::Public,
        file_path: file_path.to_string(),
        line_start: node.start_position().row,
        line_end: node.end_position().row,
        doc_comment,
        parent: scope.last().cloned(),
    })
}

fn extract_module(node: Node, source: &[u8], file_path: &str, scope: &[String]) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let qualified_name = qualified(scope, &name);
    let signature = extract_ruby_signature(node, source);
    let doc_comment = extract_ruby_comment(node, source);

    Some(Symbol {
        name,
        qualified_name,
        kind: SymbolKind::Module,
        signature,
        visibility: Visibility::Public,
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

    let qualified_name = qualified(scope, &name);
    let signature = extract_ruby_signature(node, source);
    let doc_comment = extract_ruby_comment(node, source);

    Some(Symbol {
        name,
        qualified_name,
        kind: SymbolKind::Method,
        signature,
        visibility: Visibility::Public,
        file_path: file_path.to_string(),
        line_start: node.start_position().row,
        line_end: node.end_position().row,
        doc_comment,
        parent: scope.last().cloned(),
    })
}

fn extract_singleton_method(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let qualified_name = qualified(scope, &name);
    let signature = extract_ruby_signature(node, source);
    let doc_comment = extract_ruby_comment(node, source);

    Some(Symbol {
        name,
        qualified_name,
        kind: SymbolKind::Method,
        signature,
        visibility: Visibility::Public,
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
        "call" => {
            // Check for require/require_relative calls first
            if let Some(r) = extract_require_reference(node, source, file_path, scope) {
                references.push(r);
            } else if let Some(r) = extract_call_reference(node, source, file_path, scope) {
                references.push(r);
            }
            // Still recurse into arguments etc.
        }
        "identifier" => {
            // In Ruby, a bare identifier used as a statement (child of
            // body_statement) is a method call without arguments (e.g.,
            // `do_work`). Tree-sitter-ruby parses these as plain identifier
            // nodes rather than `call` nodes.
            if let Some(parent) = node.parent() {
                if parent.kind() == "body_statement" {
                    let name = node_text(node, source);
                    let source_qn = if scope.is_empty() {
                        file_path.to_string()
                    } else {
                        scope.join("::")
                    };
                    references.push(Reference {
                        source_qualified_name: source_qn,
                        target_name: name,
                        kind: ReferenceKind::Call,
                        file_path: file_path.to_string(),
                        line: node.start_position().row,
                    });
                }
            }
        }
        "class" | "module" => {
            // Recurse with updated scope
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = node_text(name_node, source);
                let mut new_scope = scope.to_vec();
                new_scope.push(name);

                // Extract superclass reference for classes
                if node.kind() == "class" {
                    if let Some(superclass) = node.child_by_field_name("superclass") {
                        let super_text = node_text(superclass, source);
                        let super_name = super_text.trim().to_string();
                        if !super_name.is_empty() {
                            let qn = qualified(scope, &node_text(name_node, source));
                            references.push(Reference {
                                source_qualified_name: qn,
                                target_name: super_name,
                                kind: ReferenceKind::Inherits,
                                file_path: file_path.to_string(),
                                line: superclass.start_position().row,
                            });
                        }
                    }
                }

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
        "method" | "singleton_method" => {
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

fn extract_require_reference(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Reference> {
    // Check if this is a require or require_relative call
    let method_node = node.child_by_field_name("method")?;
    let method_name = node_text(method_node, source);

    if method_name != "require" && method_name != "require_relative" {
        return None;
    }

    // Find the arguments node and extract the string argument
    let arguments = node.child_by_field_name("arguments")?;
    let arg = find_string_argument(arguments, source)?;

    let source_qn = if scope.is_empty() {
        file_path.to_string()
    } else {
        scope.join("::")
    };

    Some(Reference {
        source_qualified_name: source_qn,
        target_name: arg,
        kind: ReferenceKind::Import,
        file_path: file_path.to_string(),
        line: node.start_position().row,
    })
}

fn extract_call_reference(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Reference> {
    let method_node = node.child_by_field_name("method")?;
    let method_name = node_text(method_node, source);

    // Include the receiver for qualified calls (e.g., obj.method)
    let target = if let Some(receiver) = node.child_by_field_name("receiver") {
        let receiver_text = node_text(receiver, source);
        format!("{}.{}", receiver_text, method_name)
    } else {
        method_name
    };

    let source_qn = if scope.is_empty() {
        file_path.to_string()
    } else {
        scope.join("::")
    };

    Some(Reference {
        source_qualified_name: source_qn,
        target_name: target,
        kind: ReferenceKind::Call,
        file_path: file_path.to_string(),
        line: node.start_position().row,
    })
}

// ── Helper Functions ──────────────────────────────────────────────────────

fn node_text(node: Node, source: &[u8]) -> String {
    node.utf8_text(source).unwrap_or("").to_string()
}

fn qualified(scope: &[String], name: &str) -> String {
    if scope.is_empty() {
        name.to_string()
    } else {
        format!("{}::{}", scope.join("::"), name)
    }
}

/// Extract a string literal value from an arguments node, stripping quotes.
fn find_string_argument(node: Node, source: &[u8]) -> Option<String> {
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            match child.kind() {
                "string" => {
                    let text = node_text(child, source);
                    // Strip surrounding quotes: "foo" or 'foo'
                    let stripped = text
                        .trim_start_matches('"')
                        .trim_end_matches('"')
                        .trim_start_matches('\'')
                        .trim_end_matches('\'')
                        .to_string();
                    return Some(stripped);
                }
                "string_content" => {
                    return Some(node_text(child, source));
                }
                _ => {
                    // Recurse into nested nodes (e.g., argument_list)
                    if let Some(found) = find_string_argument(child, source) {
                        return Some(found);
                    }
                }
            }
        }
    }
    None
}

fn extract_ruby_signature(node: Node, source: &[u8]) -> String {
    let text = node_text(node, source);
    // Signature is the first line (up to newline)
    let first_line = text.lines().next().unwrap_or(&text);
    first_line.trim().to_string()
}

fn extract_ruby_comment(node: Node, source: &[u8]) -> Option<String> {
    // Look for preceding comment nodes (Ruby uses # comments).
    // First, try direct prev_sibling (works when comment is in same scope).
    let mut comments = Vec::new();
    let mut prev = node.prev_sibling();
    while let Some(sibling) = prev {
        if sibling.kind() == "comment" {
            let text = node_text(sibling, source);
            let cleaned = text.trim_start_matches('#').trim().to_string();
            comments.push(cleaned);
            prev = sibling.prev_sibling();
        } else {
            break;
        }
    }

    if !comments.is_empty() {
        // Collected bottom-to-top via prev_sibling, reverse for correct order
        comments.reverse();
        return Some(comments.join("\n"));
    }

    // If no comment found via prev_sibling, the Ruby tree-sitter grammar may
    // place the comment as a child of the parent node (e.g., a comment before
    // the first method inside a class body is stored as a child of the class
    // node, between the name and the body_statement). Walk the parent's children
    // to find comments that appear just before this node's start line.
    if let Some(parent) = node.parent() {
        if let Some(grandparent) = parent.parent() {
            let node_start_row = node.start_position().row;
            let mut gp_comments: Vec<Node> = Vec::new();
            for i in 0..grandparent.child_count() {
                if let Some(child) = grandparent.child(i as u32) {
                    if child.kind() == "comment" && child.end_position().row < node_start_row {
                        gp_comments.push(child);
                    }
                }
            }

            if !gp_comments.is_empty() {
                // Only keep comments that are contiguous and end just
                // before the method's start line (within 1 line gap).
                let mut relevant = Vec::new();
                for c in gp_comments.iter().rev() {
                    let c_end = c.end_position().row;
                    let expected_next = if relevant.is_empty() {
                        node_start_row
                    } else {
                        relevant
                            .last()
                            .map(|n: &Node| n.start_position().row)
                            .unwrap()
                    };
                    if expected_next - c_end <= 1 {
                        relevant.push(*c);
                    } else {
                        break;
                    }
                }
                relevant.reverse();
                for c in &relevant {
                    let text = node_text(*c, source);
                    let cleaned = text.trim_start_matches('#').trim().to_string();
                    comments.push(cleaned);
                }
            }
        }
    }

    if comments.is_empty() {
        None
    } else {
        Some(comments.join("\n"))
    }
}


#[cfg(test)]
#[path = "tests/ruby_tests.rs"]
mod tests;
