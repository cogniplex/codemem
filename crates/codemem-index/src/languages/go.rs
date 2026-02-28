//! Go language extractor using tree-sitter-go.

use crate::extractor::LanguageExtractor;
use crate::symbol::{Reference, ReferenceKind, Symbol, SymbolKind, Visibility};
use tree_sitter::{Node, Tree};

/// Go language extractor for tree-sitter-based code indexing.
pub struct GoExtractor;

impl GoExtractor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GoExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageExtractor for GoExtractor {
    fn language_name(&self) -> &str {
        "go"
    }

    fn file_extensions(&self) -> &[&str] {
        &["go"]
    }

    fn tree_sitter_language(&self) -> tree_sitter::Language {
        tree_sitter_go::LANGUAGE.into()
    }

    fn extract_symbols(&self, tree: &Tree, source: &[u8], file_path: &str) -> Vec<Symbol> {
        let mut symbols = Vec::new();
        let root = tree.root_node();
        extract_symbols_recursive(root, source, file_path, &mut symbols);
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
    symbols: &mut Vec<Symbol>,
) {
    match node.kind() {
        "function_declaration" => {
            if let Some(sym) = extract_function(node, source, file_path) {
                symbols.push(sym);
            }
        }
        "method_declaration" => {
            if let Some(sym) = extract_method(node, source, file_path) {
                symbols.push(sym);
            }
        }
        "type_declaration" => {
            extract_type_declaration(node, source, file_path, symbols);
        }
        "const_declaration" => {
            extract_const_declaration(node, source, file_path, symbols);
        }
        "var_declaration" => {
            // Only extract top-level var as constants if they look like constants
            extract_var_declaration(node, source, file_path, symbols);
        }
        _ => {}
    }

    // Recurse for children
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            extract_symbols_recursive(child, source, file_path, symbols);
        }
    }
}

fn extract_function(node: Node, source: &[u8], file_path: &str) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let is_test = name.starts_with("Test") && file_path.ends_with("_test.go");

    let kind = if is_test {
        SymbolKind::Test
    } else {
        SymbolKind::Function
    };

    let visibility = go_visibility(&name);
    let signature = extract_go_signature(node, source);
    let doc_comment = extract_go_doc_comment(node, source);

    Some(Symbol {
        name: name.clone(),
        qualified_name: name,
        kind,
        signature,
        visibility,
        file_path: file_path.to_string(),
        line_start: node.start_position().row,
        line_end: node.end_position().row,
        doc_comment,
        parent: None,
    })
}

fn extract_method(node: Node, source: &[u8], file_path: &str) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    // Get receiver type for qualified name
    let receiver_type = get_receiver_type(node, source);
    let qualified_name = if let Some(ref recv) = receiver_type {
        format!("{}.{}", recv, name)
    } else {
        name.clone()
    };

    let visibility = go_visibility(&name);
    let signature = extract_go_signature(node, source);
    let doc_comment = extract_go_doc_comment(node, source);

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
        parent: receiver_type,
    })
}

fn extract_type_declaration(node: Node, source: &[u8], file_path: &str, symbols: &mut Vec<Symbol>) {
    // type_declaration can have multiple type_spec or type_alias children
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            match child.kind() {
                "type_spec" => {
                    if let Some(sym) = extract_type_spec(child, node, source, file_path) {
                        symbols.push(sym);
                    }
                }
                "type_alias" => {
                    if let Some(sym) = extract_type_alias(child, node, source, file_path) {
                        symbols.push(sym);
                    }
                }
                _ => {}
            }
        }
    }
}

fn extract_type_alias(alias: Node, decl: Node, source: &[u8], file_path: &str) -> Option<Symbol> {
    let name_node = alias.child_by_field_name("name")?;
    let name = node_text(name_node, source);
    let visibility = go_visibility(&name);
    let signature = node_text(alias, source);
    let first_line = signature.lines().next().unwrap_or(&signature);
    let doc_comment = extract_go_doc_comment(decl, source);

    Some(Symbol {
        name: name.clone(),
        qualified_name: name,
        kind: SymbolKind::Type,
        signature: format!("type {}", first_line.trim()),
        visibility,
        file_path: file_path.to_string(),
        line_start: alias.start_position().row,
        line_end: alias.end_position().row,
        doc_comment,
        parent: None,
    })
}

fn extract_type_spec(spec: Node, decl: Node, source: &[u8], file_path: &str) -> Option<Symbol> {
    let name_node = spec.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let type_node = spec.child_by_field_name("type")?;
    let kind = match type_node.kind() {
        "struct_type" => SymbolKind::Struct,
        "interface_type" => SymbolKind::Interface,
        _ => SymbolKind::Type,
    };

    let visibility = go_visibility(&name);
    let signature = format!(
        "type {} {}",
        name,
        node_text(type_node, source).lines().next().unwrap_or("")
    );
    let doc_comment = extract_go_doc_comment(decl, source);

    Some(Symbol {
        name: name.clone(),
        qualified_name: name,
        kind,
        signature: signature.trim().to_string(),
        visibility,
        file_path: file_path.to_string(),
        line_start: spec.start_position().row,
        line_end: spec.end_position().row,
        doc_comment,
        parent: None,
    })
}

fn extract_const_declaration(
    node: Node,
    source: &[u8],
    file_path: &str,
    symbols: &mut Vec<Symbol>,
) {
    let doc_comment = extract_go_doc_comment(node, source);
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            if child.kind() == "const_spec" {
                if let Some(name_node) = child.child_by_field_name("name") {
                    let name = node_text(name_node, source);
                    let visibility = go_visibility(&name);
                    let sig = node_text(child, source);
                    let first_line = sig.lines().next().unwrap_or(&sig);

                    symbols.push(Symbol {
                        name: name.clone(),
                        qualified_name: name,
                        kind: SymbolKind::Constant,
                        signature: first_line.trim().to_string(),
                        visibility,
                        file_path: file_path.to_string(),
                        line_start: child.start_position().row,
                        line_end: child.end_position().row,
                        doc_comment: doc_comment.clone(),
                        parent: None,
                    });
                }
            }
        }
    }
}

fn extract_var_declaration(node: Node, source: &[u8], file_path: &str, symbols: &mut Vec<Symbol>) {
    // Only extract package-level vars that look like they could be important
    // (exported names)
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            if child.kind() == "var_spec" {
                if let Some(name_node) = child.child_by_field_name("name") {
                    let name = node_text(name_node, source);
                    if name.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
                        let sig = node_text(child, source);
                        let first_line = sig.lines().next().unwrap_or(&sig);

                        symbols.push(Symbol {
                            name: name.clone(),
                            qualified_name: name,
                            kind: SymbolKind::Constant,
                            signature: first_line.trim().to_string(),
                            visibility: Visibility::Public,
                            file_path: file_path.to_string(),
                            line_start: child.start_position().row,
                            line_end: child.end_position().row,
                            doc_comment: None,
                            parent: None,
                        });
                    }
                }
            }
        }
    }
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
        "import_declaration" => {
            extract_import_references(node, source, file_path, scope, references);
        }
        "call_expression" => {
            if let Some(r) = extract_call_reference(node, source, file_path, scope) {
                references.push(r);
            }
        }
        "function_declaration" => {
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
        "method_declaration" => {
            let recv = get_receiver_type(node, source);
            let name_node = node.child_by_field_name("name");
            if let Some(nn) = name_node {
                let name = node_text(nn, source);
                let qn = if let Some(ref r) = recv {
                    format!("{}.{}", r, name)
                } else {
                    name
                };
                let mut new_scope = scope.to_vec();
                new_scope.push(qn);
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

fn extract_import_references(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    references: &mut Vec<Reference>,
) {
    let source_qn = if scope.is_empty() {
        file_path.to_string()
    } else {
        scope.join(".")
    };

    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            if child.kind() == "import_spec" {
                if let Some(path_node) = child.child_by_field_name("path") {
                    let path_text = node_text(path_node, source);
                    let clean = path_text.trim_matches('"');
                    references.push(Reference {
                        source_qualified_name: source_qn.clone(),
                        target_name: clean.to_string(),
                        kind: ReferenceKind::Import,
                        file_path: file_path.to_string(),
                        line: child.start_position().row,
                    });
                }
            }
            // Handle import_spec_list (grouped imports)
            if child.kind() == "import_spec_list" {
                for j in 0..child.child_count() {
                    if let Some(spec) = child.child(j as u32) {
                        if spec.kind() == "import_spec" {
                            if let Some(path_node) = spec.child_by_field_name("path") {
                                let path_text = node_text(path_node, source);
                                let clean = path_text.trim_matches('"');
                                references.push(Reference {
                                    source_qualified_name: source_qn.clone(),
                                    target_name: clean.to_string(),
                                    kind: ReferenceKind::Import,
                                    file_path: file_path.to_string(),
                                    line: spec.start_position().row,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
}

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
        scope.join(".")
    };

    Some(Reference {
        source_qualified_name: source_qn,
        target_name: function_name,
        kind: ReferenceKind::Call,
        file_path: file_path.to_string(),
        line: node.start_position().row,
    })
}

// ── Helper Functions ──────────────────────────────────────────────────────

fn node_text(node: Node, source: &[u8]) -> String {
    node.utf8_text(source).unwrap_or("").to_string()
}

fn go_visibility(name: &str) -> Visibility {
    if name.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
        Visibility::Public
    } else {
        Visibility::Private
    }
}

fn extract_go_signature(node: Node, source: &[u8]) -> String {
    let text = node_text(node, source);
    if let Some(pos) = text.find('{') {
        text[..pos].trim().to_string()
    } else {
        let first_line = text.lines().next().unwrap_or(&text);
        first_line.trim().to_string()
    }
}

fn extract_go_doc_comment(node: Node, source: &[u8]) -> Option<String> {
    let mut comment_lines = Vec::new();
    let mut prev = node.prev_sibling();

    while let Some(sibling) = prev {
        if sibling.kind() == "comment" {
            let text = node_text(sibling, source);
            if text.starts_with("//") {
                let stripped = text
                    .strip_prefix("// ")
                    .or_else(|| text.strip_prefix("//"))
                    .unwrap_or(&text);
                comment_lines.push(stripped.trim_end().to_string());
                prev = sibling.prev_sibling();
                continue;
            }
        }
        break;
    }

    comment_lines.reverse();
    if comment_lines.is_empty() {
        None
    } else {
        Some(comment_lines.join("\n"))
    }
}

fn get_receiver_type(node: Node, source: &[u8]) -> Option<String> {
    let receiver = node.child_by_field_name("receiver")?;
    // The receiver is a parameter_list like `(s *Server)` or `(s Server)`
    // We want the type name
    for i in 0..receiver.child_count() {
        if let Some(child) = receiver.child(i as u32) {
            if child.kind() == "parameter_declaration" {
                if let Some(type_node) = child.child_by_field_name("type") {
                    let type_text = node_text(type_node, source);
                    // Strip pointer prefix
                    let clean = type_text.trim_start_matches('*');
                    return Some(clean.to_string());
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extractor::LanguageExtractor;
    use tree_sitter::Parser;

    fn parse_go(source: &str) -> Tree {
        let mut parser = Parser::new();
        let lang = tree_sitter_go::LANGUAGE;
        parser
            .set_language(&lang.into())
            .expect("failed to set Go language");
        parser
            .parse(source.as_bytes(), None)
            .expect("failed to parse")
    }

    #[test]
    fn extract_go_function() {
        let source = r#"package main

// Add adds two integers.
func Add(a int, b int) int {
	return a + b
}
"#;
        let tree = parse_go(source);
        let extractor = GoExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "main.go");

        let add = symbols.iter().find(|s| s.name == "Add").unwrap();
        assert_eq!(add.kind, SymbolKind::Function);
        assert_eq!(add.visibility, Visibility::Public);
        assert!(add.signature.contains("func Add(a int, b int) int"));
        assert_eq!(add.doc_comment.as_deref(), Some("Add adds two integers."));
    }

    #[test]
    fn extract_go_method() {
        let source = r#"package main

type Server struct{}

// Start starts the server.
func (s *Server) Start() error {
	return nil
}
"#;
        let tree = parse_go(source);
        let extractor = GoExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "main.go");

        let start = symbols.iter().find(|s| s.name == "Start").unwrap();
        assert_eq!(start.kind, SymbolKind::Method);
        assert_eq!(start.qualified_name, "Server.Start");
        assert_eq!(start.parent.as_deref(), Some("Server"));
        assert_eq!(start.visibility, Visibility::Public);
    }

    #[test]
    fn extract_go_struct() {
        let source = r#"package main

type Config struct {
	Debug bool
	Port  int
}
"#;
        let tree = parse_go(source);
        let extractor = GoExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "main.go");

        let config = symbols.iter().find(|s| s.name == "Config").unwrap();
        assert_eq!(config.kind, SymbolKind::Struct);
        assert_eq!(config.visibility, Visibility::Public);
    }

    #[test]
    fn extract_go_interface() {
        let source = r#"package main

type Reader interface {
	Read(p []byte) (n int, err error)
}
"#;
        let tree = parse_go(source);
        let extractor = GoExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "main.go");

        let reader = symbols.iter().find(|s| s.name == "Reader").unwrap();
        assert_eq!(reader.kind, SymbolKind::Interface);
    }

    #[test]
    fn extract_go_imports() {
        let source = r#"package main

import (
	"fmt"
	"os"
	"net/http"
)
"#;
        let tree = parse_go(source);
        let extractor = GoExtractor::new();
        let references = extractor.extract_references(&tree, source.as_bytes(), "main.go");

        let imports: Vec<_> = references
            .iter()
            .filter(|r| r.kind == ReferenceKind::Import)
            .collect();
        assert!(
            imports.iter().any(|r| r.target_name == "fmt"),
            "imports: {:#?}",
            imports
        );
        assert!(
            imports.iter().any(|r| r.target_name == "os"),
            "imports: {:#?}",
            imports
        );
        assert!(
            imports.iter().any(|r| r.target_name == "net/http"),
            "imports: {:#?}",
            imports
        );
    }

    #[test]
    fn extract_go_constants() {
        let source = r#"package main

const MaxSize = 1024

const (
	DefaultPort = 8080
	Version     = "1.0"
)
"#;
        let tree = parse_go(source);
        let extractor = GoExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "main.go");

        let constants: Vec<_> = symbols
            .iter()
            .filter(|s| s.kind == SymbolKind::Constant)
            .collect();
        assert!(
            constants.iter().any(|s| s.name == "MaxSize"),
            "Expected MaxSize, got: {:#?}",
            constants
        );
        assert!(
            constants.iter().any(|s| s.name == "DefaultPort"),
            "Expected DefaultPort, got: {:#?}",
            constants
        );
    }

    #[test]
    fn extract_go_test_function() {
        let source = r#"package main

func TestAdd(t *testing.T) {
	if Add(1, 2) != 3 {
		t.Fatal("wrong")
	}
}
"#;
        let tree = parse_go(source);
        let extractor = GoExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "main_test.go");

        let test_fn = symbols.iter().find(|s| s.name == "TestAdd").unwrap();
        assert_eq!(test_fn.kind, SymbolKind::Test);
    }

    #[test]
    fn extract_go_visibility() {
        let source = r#"package main

func PublicFunc() {}
func privateFunc() {}
"#;
        let tree = parse_go(source);
        let extractor = GoExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "main.go");

        let public = symbols.iter().find(|s| s.name == "PublicFunc").unwrap();
        assert_eq!(public.visibility, Visibility::Public);

        let private = symbols.iter().find(|s| s.name == "privateFunc").unwrap();
        assert_eq!(private.visibility, Visibility::Private);
    }

    #[test]
    fn extract_go_call_references() {
        let source = r#"package main

func caller() {
	x := foo()
	bar(x)
}
"#;
        let tree = parse_go(source);
        let extractor = GoExtractor::new();
        let references = extractor.extract_references(&tree, source.as_bytes(), "main.go");

        let calls: Vec<_> = references
            .iter()
            .filter(|r| r.kind == ReferenceKind::Call)
            .collect();
        assert!(
            calls.iter().any(|r| r.target_name == "foo"),
            "calls: {:#?}",
            calls
        );
        assert!(
            calls.iter().any(|r| r.target_name == "bar"),
            "calls: {:#?}",
            calls
        );
    }

    #[test]
    fn extract_go_type_alias() {
        let source = r#"package main

type ID string
type Result = int
"#;
        let tree = parse_go(source);
        let extractor = GoExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "main.go");

        let id = symbols.iter().find(|s| s.name == "ID").unwrap();
        assert_eq!(id.kind, SymbolKind::Type);

        let result = symbols.iter().find(|s| s.name == "Result").unwrap();
        assert_eq!(result.kind, SymbolKind::Type);
    }
}
