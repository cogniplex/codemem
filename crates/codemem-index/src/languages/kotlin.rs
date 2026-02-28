//! Kotlin language extractor using tree-sitter-kotlin.

use crate::extractor::LanguageExtractor;
use crate::symbol::{Reference, ReferenceKind, Symbol, SymbolKind, Visibility};
use tree_sitter::{Node, Tree};

/// Kotlin language extractor for tree-sitter-based code indexing.
pub struct KotlinExtractor;

impl KotlinExtractor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for KotlinExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageExtractor for KotlinExtractor {
    fn language_name(&self) -> &str {
        "kotlin"
    }

    fn file_extensions(&self) -> &[&str] {
        &["kt", "kts"]
    }

    fn tree_sitter_language(&self) -> tree_sitter::Language {
        tree_sitter_kotlin_ng::LANGUAGE.into()
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
        "class_declaration" => {
            if let Some(sym) = extract_class(node, source, file_path, scope) {
                let class_name = sym.name.clone();
                symbols.push(sym);
                // Recurse into class body with updated scope
                if let Some(body) = find_child_by_kind(node, "class_body") {
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
        "object_declaration" => {
            if let Some(sym) = extract_object(node, source, file_path, scope) {
                let obj_name = sym.name.clone();
                symbols.push(sym);
                // Recurse into object body with updated scope
                if let Some(body) = find_child_by_kind(node, "class_body") {
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
        "function_declaration" => {
            if let Some(sym) = extract_function(node, source, file_path, scope) {
                symbols.push(sym);
            }
            return; // Don't recurse into function bodies for symbols
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
    let name = find_type_identifier(node, source)?;

    let qualified_name = qualified(scope, &name);
    let visibility = extract_visibility(node, source);
    let signature = extract_kotlin_signature(node, source);
    let doc_comment = extract_kdoc(node, source);

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

fn extract_object(node: Node, source: &[u8], file_path: &str, scope: &[String]) -> Option<Symbol> {
    let name = find_type_identifier(node, source)?;

    let qualified_name = qualified(scope, &name);
    let visibility = extract_visibility(node, source);
    let signature = extract_kotlin_signature(node, source);
    let doc_comment = extract_kdoc(node, source);

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
    let name = find_simple_identifier(node, source)?;

    let qualified_name = qualified(scope, &name);
    let visibility = extract_visibility(node, source);
    let signature = extract_kotlin_signature(node, source);
    let doc_comment = extract_kdoc(node, source);

    // If inside a class/object (scope is non-empty), it's a Method; otherwise Function
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

// ── Reference Extraction ──────────────────────────────────────────────────

fn extract_references_recursive(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    references: &mut Vec<Reference>,
) {
    match node.kind() {
        "import_header" | "import" => {
            if let Some(r) = extract_import_reference(node, source, file_path, scope) {
                references.push(r);
            }
            return;
        }
        "call_expression" => {
            if let Some(r) = extract_call_reference(node, source, file_path, scope) {
                references.push(r);
            }
            // Still recurse into arguments etc.
        }
        "delegation_specifier" => {
            if let Some(r) = extract_inheritance_reference(node, source, file_path, scope) {
                references.push(r);
            }
            return;
        }
        "class_declaration" | "object_declaration" => {
            // Recurse with updated scope
            let name = find_type_identifier(node, source);
            if let Some(name) = name {
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
        "function_declaration" => {
            let name = find_simple_identifier(node, source);
            if let Some(name) = name {
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

fn extract_import_reference(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Reference> {
    // import / import_header contains a child with the import path.
    // tree-sitter-kotlin-ng uses `qualified_identifier` containing `identifier` children
    // separated by `.` nodes. Other grammars may use `identifier` or `scoped_identifier`.
    let mut import_path = String::new();
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            let kind = child.kind();
            if kind == "qualified_identifier" || kind == "identifier" || kind == "scoped_identifier"
            {
                import_path = node_text(child, source);
                break;
            }
        }
    }

    // Fallback: parse from the full node text
    if import_path.is_empty() {
        let text = node_text(node, source);
        import_path = text.trim().strip_prefix("import")?.trim().to_string();
    }

    if import_path.is_empty() {
        return None;
    }

    let source_qn = if scope.is_empty() {
        file_path.to_string()
    } else {
        scope.join(".")
    };

    Some(Reference {
        source_qualified_name: source_qn,
        target_name: import_path,
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
    // call_expression has a first child that is the function being called
    // It could be a simple_identifier, navigation_expression, etc.
    let func_node = node.child(0)?;
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

fn extract_inheritance_reference(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Reference> {
    // delegation_specifier contains the supertype (e.g., "BaseClass" or "Interface()")
    // Extract the type name, stripping any constructor call parentheses
    let text = node_text(node, source);
    let type_name = text.split('(').next().unwrap_or(&text).trim().to_string();

    if type_name.is_empty() {
        return None;
    }

    let source_qn = if scope.is_empty() {
        file_path.to_string()
    } else {
        scope.join(".")
    };

    Some(Reference {
        source_qualified_name: source_qn,
        target_name: type_name,
        kind: ReferenceKind::Inherits,
        file_path: file_path.to_string(),
        line: node.start_position().row,
    })
}

// ── Helper Functions ──────────────────────────────────────────────────────

fn node_text(node: Node, source: &[u8]) -> String {
    node.utf8_text(source).unwrap_or("").to_string()
}

/// Find the first direct child of `node` whose kind matches `kind`.
fn find_child_by_kind<'a>(node: Node<'a>, kind: &str) -> Option<Node<'a>> {
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            if child.kind() == kind {
                return Some(child);
            }
        }
    }
    None
}

fn qualified(scope: &[String], name: &str) -> String {
    if scope.is_empty() {
        name.to_string()
    } else {
        format!("{}.{}", scope.join("."), name)
    }
}

/// Find the name of a class or object declaration.
/// In tree-sitter-kotlin-ng the name is an `identifier` node accessible
/// via the `name` field (or as a direct child of kind `identifier`).
fn find_type_identifier(node: Node, source: &[u8]) -> Option<String> {
    // Prefer the named field first (tree-sitter-kotlin-ng uses `name:`)
    if let Some(name_node) = node.child_by_field_name("name") {
        let text = node_text(name_node, source);
        if !text.is_empty() {
            return Some(text);
        }
    }
    // Fallback: scan for type_identifier or identifier child
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            if child.kind() == "type_identifier" || child.kind() == "identifier" {
                let text = node_text(child, source);
                if !text.is_empty() {
                    return Some(text);
                }
            }
        }
    }
    None
}

/// Find the name of a function declaration.
/// In tree-sitter-kotlin-ng the name is an `identifier` node accessible
/// via the `name` field (or as a direct child of kind `simple_identifier` / `identifier`).
fn find_simple_identifier(node: Node, source: &[u8]) -> Option<String> {
    // Prefer the named field first (tree-sitter-kotlin-ng uses `name:`)
    if let Some(name_node) = node.child_by_field_name("name") {
        let text = node_text(name_node, source);
        if !text.is_empty() {
            return Some(text);
        }
    }
    // Fallback: scan for simple_identifier or identifier child
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            if child.kind() == "simple_identifier" || child.kind() == "identifier" {
                let text = node_text(child, source);
                if !text.is_empty() {
                    return Some(text);
                }
            }
        }
    }
    None
}

/// Extract visibility from Kotlin modifier keywords.
/// Kotlin defaults to public when no modifier is specified.
fn extract_visibility(node: Node, source: &[u8]) -> Visibility {
    let modifiers = collect_modifiers(node, source);
    if modifiers.iter().any(|m| m == "private") {
        Visibility::Private
    } else if modifiers.iter().any(|m| m == "protected") {
        Visibility::Protected
    } else if modifiers.iter().any(|m| m == "internal") {
        Visibility::Crate
    } else {
        // Kotlin default visibility is public
        Visibility::Public
    }
}

/// Collect modifier keywords from a node's `modifiers` child.
fn collect_modifiers(node: Node, source: &[u8]) -> Vec<String> {
    let mut modifiers = Vec::new();
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            let kind = child.kind();
            if kind == "modifiers" || kind == "modifier" {
                collect_modifier_tokens(child, source, &mut modifiers);
            }
            // Also check for visibility_modifier and inheritance_modifier directly
            if kind == "visibility_modifier"
                || kind == "inheritance_modifier"
                || kind == "member_modifier"
            {
                let text = node_text(child, source);
                if !modifiers.contains(&text) {
                    modifiers.push(text);
                }
            }
        }
    }
    modifiers
}

fn collect_modifier_tokens(node: Node, source: &[u8], modifiers: &mut Vec<String>) {
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            let kind = child.kind();
            match kind {
                "visibility_modifier"
                | "inheritance_modifier"
                | "member_modifier"
                | "class_modifier"
                | "function_modifier"
                | "property_modifier"
                | "parameter_modifier"
                | "platform_modifier" => {
                    let text = node_text(child, source);
                    if !text.is_empty() && !modifiers.contains(&text) {
                        modifiers.push(text);
                    }
                }
                "modifier" => {
                    // Recurse into nested modifier nodes
                    collect_modifier_tokens(child, source, modifiers);
                }
                _ => {
                    // Some grammars put keyword text directly
                    let text = node_text(child, source);
                    let keywords = [
                        "public",
                        "private",
                        "protected",
                        "internal",
                        "open",
                        "abstract",
                        "sealed",
                        "data",
                        "override",
                        "final",
                    ];
                    if keywords.contains(&text.as_str()) && !modifiers.contains(&text) {
                        modifiers.push(text);
                    }
                }
            }
        }
    }
}

/// Extract signature text: everything up to `{` or the full first line.
fn extract_kotlin_signature(node: Node, source: &[u8]) -> String {
    let text = node_text(node, source);
    if let Some(pos) = text.find('{') {
        text[..pos].trim().to_string()
    } else {
        let first_line = text.lines().next().unwrap_or(&text);
        first_line.trim().to_string()
    }
}

/// Extract KDoc (/** ... */) comment preceding a node.
fn extract_kdoc(node: Node, source: &[u8]) -> Option<String> {
    let mut prev = node.prev_sibling();
    while let Some(sibling) = prev {
        match sibling.kind() {
            "multiline_comment" => {
                let text = node_text(sibling, source);
                if text.starts_with("/**") {
                    return Some(clean_kdoc(&text));
                }
                return None;
            }
            "line_comment" => {
                // Skip line comments, look for KDoc before them
                prev = sibling.prev_sibling();
                continue;
            }
            _ => return None,
        }
    }
    None
}

/// Clean KDoc comment text by removing delimiters and leading asterisks.
fn clean_kdoc(raw: &str) -> String {
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
}
