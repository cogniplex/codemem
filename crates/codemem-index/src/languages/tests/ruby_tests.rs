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
