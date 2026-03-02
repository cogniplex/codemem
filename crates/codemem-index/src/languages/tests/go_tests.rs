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
