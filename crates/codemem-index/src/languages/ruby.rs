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

fn extract_class(
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

fn extract_module(
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

fn extract_method(
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
                        relevant.last().map(|n: &Node| n.start_position().row).unwrap()
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
mod tests {
    use super::*;
    use crate::extractor::LanguageExtractor;
    use tree_sitter::Parser;

    fn parse_ruby(source: &str) -> Tree {
        let mut parser = Parser::new();
        let lang = tree_sitter_ruby::LANGUAGE;
        parser
            .set_language(&lang.into())
            .expect("failed to set Ruby language");
        parser
            .parse(source.as_bytes(), None)
            .expect("failed to parse")
    }

    #[test]
    fn extract_ruby_class_with_methods() {
        let source = r#"
class Calculator
  def add(a, b)
    a + b
  end

  def subtract(a, b)
    a - b
  end
end
"#;
        let tree = parse_ruby(source);
        let extractor = RubyExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "calculator.rb");

        let class = symbols.iter().find(|s| s.name == "Calculator").unwrap();
        assert_eq!(class.kind, SymbolKind::Class);
        assert_eq!(class.visibility, Visibility::Public);
        assert!(
            class.signature.contains("class Calculator"),
            "signature: {}",
            class.signature
        );

        let add = symbols.iter().find(|s| s.name == "add").unwrap();
        assert_eq!(add.kind, SymbolKind::Method);
        assert_eq!(add.qualified_name, "Calculator::add");
        assert_eq!(add.parent.as_deref(), Some("Calculator"));
        assert!(
            add.signature.contains("def add(a, b)"),
            "signature: {}",
            add.signature
        );

        let subtract = symbols.iter().find(|s| s.name == "subtract").unwrap();
        assert_eq!(subtract.kind, SymbolKind::Method);
        assert_eq!(subtract.qualified_name, "Calculator::subtract");
        assert_eq!(subtract.parent.as_deref(), Some("Calculator"));
    }

    #[test]
    fn extract_ruby_module() {
        let source = r#"
module Helpers
  def format(text)
    text.strip
  end
end
"#;
        let tree = parse_ruby(source);
        let extractor = RubyExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "helpers.rb");

        let module = symbols.iter().find(|s| s.name == "Helpers").unwrap();
        assert_eq!(module.kind, SymbolKind::Module);
        assert_eq!(module.visibility, Visibility::Public);

        let method = symbols.iter().find(|s| s.name == "format").unwrap();
        assert_eq!(method.kind, SymbolKind::Method);
        assert_eq!(method.qualified_name, "Helpers::format");
        assert_eq!(method.parent.as_deref(), Some("Helpers"));
    }

    #[test]
    fn extract_ruby_singleton_method() {
        let source = r#"
class Config
  def self.default
    new
  end
end
"#;
        let tree = parse_ruby(source);
        let extractor = RubyExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "config.rb");

        let class = symbols.iter().find(|s| s.name == "Config").unwrap();
        assert_eq!(class.kind, SymbolKind::Class);

        let singleton = symbols.iter().find(|s| s.name == "default").unwrap();
        assert_eq!(singleton.kind, SymbolKind::Method);
        assert_eq!(singleton.qualified_name, "Config::default");
    }

    #[test]
    fn extract_ruby_references_calls() {
        let source = r#"
class App
  def run
    puts "hello"
    do_work
  end

  def do_work
  end
end
"#;
        let tree = parse_ruby(source);
        let extractor = RubyExtractor::new();
        let references = extractor.extract_references(&tree, source.as_bytes(), "app.rb");

        let calls: Vec<_> = references
            .iter()
            .filter(|r| r.kind == ReferenceKind::Call)
            .collect();
        assert!(
            calls.iter().any(|r| r.target_name == "puts"),
            "calls: {:#?}",
            calls
        );
        assert!(
            calls.iter().any(|r| r.target_name == "do_work"),
            "calls: {:#?}",
            calls
        );
    }

    #[test]
    fn extract_ruby_references_require() {
        let source = r#"
require "json"
require_relative "helpers"

class App
end
"#;
        let tree = parse_ruby(source);
        let extractor = RubyExtractor::new();
        let references = extractor.extract_references(&tree, source.as_bytes(), "app.rb");

        let imports: Vec<_> = references
            .iter()
            .filter(|r| r.kind == ReferenceKind::Import)
            .collect();
        assert!(
            imports.iter().any(|r| r.target_name == "json"),
            "imports: {:#?}",
            imports
        );
        assert!(
            imports.iter().any(|r| r.target_name == "helpers"),
            "imports: {:#?}",
            imports
        );
    }

    #[test]
    fn extract_ruby_inheritance() {
        let source = r#"
class MyService < BaseService
  def run
  end
end
"#;
        let tree = parse_ruby(source);
        let extractor = RubyExtractor::new();
        let references = extractor.extract_references(&tree, source.as_bytes(), "my_service.rb");

        let inherits: Vec<_> = references
            .iter()
            .filter(|r| r.kind == ReferenceKind::Inherits)
            .collect();
        assert!(
            inherits
                .iter()
                .any(|r| r.target_name.contains("BaseService")),
            "inherits: {:#?}",
            inherits
        );
    }

    #[test]
    fn extract_ruby_nested_class_in_module() {
        let source = r#"
module Api
  class UsersController
    def index
    end
  end
end
"#;
        let tree = parse_ruby(source);
        let extractor = RubyExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "users_controller.rb");

        let module = symbols.iter().find(|s| s.name == "Api").unwrap();
        assert_eq!(module.kind, SymbolKind::Module);

        let class = symbols
            .iter()
            .find(|s| s.name == "UsersController")
            .unwrap();
        assert_eq!(class.kind, SymbolKind::Class);
        assert_eq!(class.qualified_name, "Api::UsersController");
        assert_eq!(class.parent.as_deref(), Some("Api"));

        let method = symbols.iter().find(|s| s.name == "index").unwrap();
        assert_eq!(method.kind, SymbolKind::Method);
        assert_eq!(method.qualified_name, "Api::UsersController::index");
        assert_eq!(method.parent.as_deref(), Some("UsersController"));
    }

    #[test]
    fn extract_ruby_doc_comment() {
        let source = r#"
# A utility class for string operations.
# Provides helper methods for formatting.
class StringUtils
  # Formats the given input string.
  def format(input)
    input.strip
  end
end
"#;
        let tree = parse_ruby(source);
        let extractor = RubyExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "string_utils.rb");

        let class = symbols.iter().find(|s| s.name == "StringUtils").unwrap();
        let doc = class
            .doc_comment
            .as_ref()
            .expect("expected doc comment on class");
        assert!(
            doc.contains("utility class for string operations"),
            "doc: {}",
            doc
        );

        let method = symbols.iter().find(|s| s.name == "format").unwrap();
        let method_doc = method
            .doc_comment
            .as_ref()
            .expect("expected doc comment on method");
        assert!(
            method_doc.contains("Formats the given input string"),
            "method_doc: {}",
            method_doc
        );
    }

    #[test]
    fn extract_ruby_qualified_call() {
        let source = r#"
class App
  def run
    File.read("data.txt")
  end
end
"#;
        let tree = parse_ruby(source);
        let extractor = RubyExtractor::new();
        let references = extractor.extract_references(&tree, source.as_bytes(), "app.rb");

        let calls: Vec<_> = references
            .iter()
            .filter(|r| r.kind == ReferenceKind::Call)
            .collect();
        assert!(
            calls.iter().any(|r| r.target_name == "File.read"),
            "calls: {:#?}",
            calls
        );
    }

    #[test]
    fn ruby_extractor_language_name() {
        let extractor = RubyExtractor::new();
        assert_eq!(extractor.language_name(), "ruby");
        assert_eq!(extractor.file_extensions(), &["rb"]);
    }
}
