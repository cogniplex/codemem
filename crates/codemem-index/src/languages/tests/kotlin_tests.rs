use super::*;
use crate::extractor::LanguageExtractor;
use tree_sitter::Parser;

fn parse_kotlin(source: &str) -> Tree {
    let mut parser = Parser::new();
    let lang: tree_sitter::Language = tree_sitter_kotlin_ng::LANGUAGE.into();
    parser
        .set_language(&lang)
        .expect("failed to set Kotlin language");
    parser
        .parse(source.as_bytes(), None)
        .expect("failed to parse")
}

#[test]
fn extract_class_with_functions() {
    let source = r#"
class Calculator {
    fun add(a: Int, b: Int): Int {
        return a + b
    }

    fun subtract(a: Int, b: Int): Int {
        return a - b
    }
}
"#;
    let tree = parse_kotlin(source);
    let extractor = KotlinExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Calculator.kt");

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
    assert_eq!(add.qualified_name, "Calculator.add");
    assert_eq!(add.parent.as_deref(), Some("Calculator"));
    assert!(
        add.signature.contains("fun add"),
        "signature: {}",
        add.signature
    );

    let subtract = symbols.iter().find(|s| s.name == "subtract").unwrap();
    assert_eq!(subtract.kind, SymbolKind::Method);
    assert_eq!(subtract.qualified_name, "Calculator.subtract");
}

#[test]
fn extract_object_declaration() {
    let source = r#"
object Logger {
    fun log(message: String) {
        println(message)
    }
}
"#;
    let tree = parse_kotlin(source);
    let extractor = KotlinExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Logger.kt");

    let obj = symbols.iter().find(|s| s.name == "Logger").unwrap();
    assert_eq!(obj.kind, SymbolKind::Class);
    assert_eq!(obj.visibility, Visibility::Public);
    assert!(
        obj.signature.contains("object Logger"),
        "signature: {}",
        obj.signature
    );

    let log_fn = symbols.iter().find(|s| s.name == "log").unwrap();
    assert_eq!(log_fn.kind, SymbolKind::Method);
    assert_eq!(log_fn.qualified_name, "Logger.log");
    assert_eq!(log_fn.parent.as_deref(), Some("Logger"));
}

#[test]
fn extract_imports() {
    let source = r#"
import com.example.models.User
import com.example.services.AuthService
import kotlin.collections.List

class App
"#;
    let tree = parse_kotlin(source);
    let extractor = KotlinExtractor::new();
    let references = extractor.extract_references(&tree, source.as_bytes(), "App.kt");

    let imports: Vec<_> = references
        .iter()
        .filter(|r| r.kind == ReferenceKind::Import)
        .collect();
    assert!(
        imports
            .iter()
            .any(|r| r.target_name.contains("com.example.models.User")),
        "imports: {:#?}",
        imports
    );
    assert!(
        imports
            .iter()
            .any(|r| r.target_name.contains("com.example.services.AuthService")),
        "imports: {:#?}",
        imports
    );
    assert!(
        imports
            .iter()
            .any(|r| r.target_name.contains("kotlin.collections.List")),
        "imports: {:#?}",
        imports
    );
}

#[test]
fn extract_inheritance() {
    let source = r#"
open class Animal(val name: String)

class Dog(name: String) : Animal(name) {
    fun bark() {}
}
"#;
    let tree = parse_kotlin(source);
    let extractor = KotlinExtractor::new();
    let references = extractor.extract_references(&tree, source.as_bytes(), "Dog.kt");

    let inherits: Vec<_> = references
        .iter()
        .filter(|r| r.kind == ReferenceKind::Inherits)
        .collect();
    assert!(
        inherits.iter().any(|r| r.target_name.contains("Animal")),
        "inherits: {:#?}",
        inherits
    );
}
