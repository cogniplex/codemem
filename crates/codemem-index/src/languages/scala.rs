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

fn extract_class(
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

fn extract_trait(
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

fn extract_object(
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
mod tests {
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
            imports
                .iter()
                .any(|r| r.target_name.contains("scala.collection.mutable.ListBuffer")),
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
        let references =
            extractor.extract_references(&tree, source.as_bytes(), "MyService.scala");

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

        let public_m = symbols
            .iter()
            .find(|s| s.name == "publicMethod")
            .unwrap();
        assert_eq!(public_m.visibility, Visibility::Public);

        let private_m = symbols
            .iter()
            .find(|s| s.name == "privateMethod")
            .unwrap();
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
}
