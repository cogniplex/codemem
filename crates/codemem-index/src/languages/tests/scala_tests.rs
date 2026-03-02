use super::*;
use crate::extractor::LanguageExtractor;
use tree_sitter::Parser;

fn parse_scala(source: &str) -> Tree {
    let mut parser = Parser::new();
    let lang = tree_sitter_scala::LANGUAGE;
    parser
        .set_language(&lang.into())
        .expect("failed to set Scala language");
    parser
        .parse(source.as_bytes(), None)
        .expect("failed to parse")
}

#[test]
fn extract_class_with_methods() {
    let source = r#"
class Calculator {
  def add(a: Int, b: Int): Int = a + b

  def subtract(a: Int, b: Int): Int = a - b
}
"#;
    let tree = parse_scala(source);
    let extractor = ScalaExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Calculator.scala");

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

    let subtract = symbols.iter().find(|s| s.name == "subtract").unwrap();
    assert_eq!(subtract.kind, SymbolKind::Method);
    assert_eq!(subtract.qualified_name, "Calculator.subtract");
}

#[test]
fn extract_trait() {
    let source = r#"
trait Repository {
  def save(entity: Any): Unit

  def findById(id: String): Option[Any]
}
"#;
    let tree = parse_scala(source);
    let extractor = ScalaExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Repository.scala");

    let trait_sym = symbols.iter().find(|s| s.name == "Repository").unwrap();
    assert_eq!(trait_sym.kind, SymbolKind::Interface);
    assert_eq!(trait_sym.visibility, Visibility::Public);
    assert!(
        trait_sym.signature.contains("trait Repository"),
        "signature: {}",
        trait_sym.signature
    );
}

#[test]
fn extract_object() {
    let source = r#"
object AppConfig {
  def defaultPort: Int = 8080

  def appName: String = "MyApp"
}
"#;
    let tree = parse_scala(source);
    let extractor = ScalaExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "AppConfig.scala");

    let obj = symbols.iter().find(|s| s.name == "AppConfig").unwrap();
    assert_eq!(obj.kind, SymbolKind::Class);
    assert_eq!(obj.visibility, Visibility::Public);
    assert!(
        obj.signature.contains("object AppConfig"),
        "signature: {}",
        obj.signature
    );

    let method = symbols.iter().find(|s| s.name == "defaultPort").unwrap();
    assert_eq!(method.kind, SymbolKind::Method);
    assert_eq!(method.qualified_name, "AppConfig.defaultPort");
}

#[test]
fn extract_imports() {
    let source = r#"
import scala.collection.mutable.ListBuffer
import java.util.UUID

class App {}
"#;
    let tree = parse_scala(source);
    let extractor = ScalaExtractor::new();
    let references = extractor.extract_references(&tree, source.as_bytes(), "App.scala");

    let imports: Vec<_> = references
        .iter()
        .filter(|r| r.kind == ReferenceKind::Import)
        .collect();
    assert!(
        imports.iter().any(|r| r
            .target_name
            .contains("scala.collection.mutable.ListBuffer")),
        "imports: {:#?}",
        imports
    );
    assert!(
        imports
            .iter()
            .any(|r| r.target_name.contains("java.util.UUID")),
        "imports: {:#?}",
        imports
    );
}

#[test]
fn extract_inheritance() {
    let source = r#"
class MyService extends BaseService with Serializable {
  def run(): Unit = {}
}
"#;
    let tree = parse_scala(source);
    let extractor = ScalaExtractor::new();
    let references = extractor.extract_references(&tree, source.as_bytes(), "MyService.scala");

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
    assert!(
        inherits
            .iter()
            .any(|r| r.target_name.contains("Serializable")),
        "inherits: {:#?}",
        inherits
    );
}

#[test]
fn extract_visibility() {
    let source = r#"
class Example {
  def publicMethod(): Unit = {}
  private def privateMethod(): Unit = {}
  protected def protectedMethod(): Unit = {}
}
"#;
    let tree = parse_scala(source);
    let extractor = ScalaExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Example.scala");

    let public_m = symbols.iter().find(|s| s.name == "publicMethod").unwrap();
    assert_eq!(public_m.visibility, Visibility::Public);

    let private_m = symbols.iter().find(|s| s.name == "privateMethod").unwrap();
    assert_eq!(private_m.visibility, Visibility::Private);

    let protected_m = symbols
        .iter()
        .find(|s| s.name == "protectedMethod")
        .unwrap();
    assert_eq!(protected_m.visibility, Visibility::Protected);
}

#[test]
fn extract_scaladoc_comment() {
    let source = r#"
/**
 * A utility class for string operations.
 * Provides helper methods for formatting.
 */
class StringUtils {
  /**
   * Formats the given input string.
   * @param input the raw string
   * @return the formatted string
   */
  def format(input: String): String = input.trim
}
"#;
    let tree = parse_scala(source);
    let extractor = ScalaExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "StringUtils.scala");

    let class = symbols.iter().find(|s| s.name == "StringUtils").unwrap();
    let doc = class
        .doc_comment
        .as_ref()
        .expect("expected Scaladoc on class");
    assert!(
        doc.contains("utility class for string operations"),
        "doc: {}",
        doc
    );

    let method = symbols.iter().find(|s| s.name == "format").unwrap();
    let method_doc = method
        .doc_comment
        .as_ref()
        .expect("expected Scaladoc on method");
    assert!(
        method_doc.contains("Formats the given input string"),
        "method_doc: {}",
        method_doc
    );
}

#[test]
fn extract_final_val_constant() {
    let source = r#"
object Config {
  final val MaxSize: Int = 1024
  final val Version: String = "1.0"
  val mutableField: Int = 0
}
"#;
    let tree = parse_scala(source);
    let extractor = ScalaExtractor::new();
    let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Config.scala");

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
        constants.iter().any(|s| s.name == "Version"),
        "Expected Version, got: {:#?}",
        constants
    );
    // mutableField should NOT appear as a constant (no final modifier)
    assert!(
        !constants.iter().any(|s| s.name == "mutableField"),
        "mutableField should not be a constant"
    );
}

#[test]
fn extract_call_references() {
    let source = r#"
object App {
  def run(): Unit = {
    println("hello")
    doWork()
  }

  def doWork(): Unit = {}
}
"#;
    let tree = parse_scala(source);
    let extractor = ScalaExtractor::new();
    let references = extractor.extract_references(&tree, source.as_bytes(), "App.scala");

    let calls: Vec<_> = references
        .iter()
        .filter(|r| r.kind == ReferenceKind::Call)
        .collect();
    assert!(
        calls.iter().any(|r| r.target_name.contains("println")),
        "calls: {:#?}",
        calls
    );
    assert!(
        calls.iter().any(|r| r.target_name == "doWork"),
        "calls: {:#?}",
        calls
    );
}
