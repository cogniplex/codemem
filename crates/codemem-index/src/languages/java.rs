//! Java language extractor using tree-sitter-java.

use crate::extractor::LanguageExtractor;
use crate::symbol::{Reference, ReferenceKind, Symbol, SymbolKind, Visibility};
use tree_sitter::{Node, Tree};

/// Java language extractor for tree-sitter-based code indexing.
pub struct JavaExtractor;

impl JavaExtractor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for JavaExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageExtractor for JavaExtractor {
    fn language_name(&self) -> &str {
        "java"
    }

    fn file_extensions(&self) -> &[&str] {
        &["java"]
    }

    fn tree_sitter_language(&self) -> tree_sitter::Language {
        tree_sitter_java::LANGUAGE.into()
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
        "enum_declaration" => {
            if let Some(sym) = extract_enum(node, source, file_path, scope) {
                let enum_name = sym.name.clone();
                symbols.push(sym);
                if let Some(body) = node.child_by_field_name("body") {
                    let mut new_scope = scope.to_vec();
                    new_scope.push(enum_name);
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
        "annotation_type_declaration" => {
            if let Some(sym) = extract_annotation_type(node, source, file_path, scope) {
                symbols.push(sym);
                return;
            }
        }
        "method_declaration" => {
            if let Some(sym) = extract_method(node, source, file_path, scope) {
                symbols.push(sym);
            }
            return; // Don't recurse into method bodies for symbols
        }
        "constructor_declaration" => {
            if let Some(sym) = extract_constructor(node, source, file_path, scope) {
                symbols.push(sym);
            }
            return;
        }
        "field_declaration" => {
            extract_field(node, source, file_path, scope, symbols);
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
    let visibility = extract_visibility(node, source);
    let signature = extract_java_signature(node, source);
    let doc_comment = extract_javadoc(node, source);

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

fn extract_interface(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let qualified_name = qualified(scope, &name);
    let visibility = extract_visibility(node, source);
    let signature = extract_java_signature(node, source);
    let doc_comment = extract_javadoc(node, source);

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

fn extract_enum(node: Node, source: &[u8], file_path: &str, scope: &[String]) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let qualified_name = qualified(scope, &name);
    let visibility = extract_visibility(node, source);
    let signature = extract_java_signature(node, source);
    let doc_comment = extract_javadoc(node, source);

    Some(Symbol {
        name,
        qualified_name,
        kind: SymbolKind::Enum,
        signature,
        visibility,
        file_path: file_path.to_string(),
        line_start: node.start_position().row,
        line_end: node.end_position().row,
        doc_comment,
        parent: scope.last().cloned(),
    })
}

fn extract_annotation_type(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let qualified_name = qualified(scope, &name);
    let visibility = extract_visibility(node, source);
    let signature = extract_java_signature(node, source);
    let doc_comment = extract_javadoc(node, source);

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

fn extract_method(node: Node, source: &[u8], file_path: &str, scope: &[String]) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let qualified_name = qualified(scope, &name);
    let visibility = extract_visibility(node, source);
    let signature = extract_java_signature(node, source);
    let doc_comment = extract_javadoc(node, source);

    // Determine if this is a test method
    let is_test = has_test_annotation(node, source)
        || (name.starts_with("test") && file_path.ends_with("Test.java"));

    // If inside a class (scope is non-empty), it's a Method; otherwise Function
    let kind = if is_test {
        SymbolKind::Test
    } else if scope.is_empty() {
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

fn extract_constructor(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let qualified_name = qualified(scope, &name);
    let visibility = extract_visibility(node, source);
    let signature = extract_java_signature(node, source);
    let doc_comment = extract_javadoc(node, source);

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

fn extract_field(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    symbols: &mut Vec<Symbol>,
) {
    let modifiers = collect_modifiers(node, source);
    let is_static = modifiers.iter().any(|m| m == "static");
    let is_final = modifiers.iter().any(|m| m == "final");

    // Only extract static final fields as constants
    if !(is_static && is_final) {
        return;
    }

    let visibility = visibility_from_modifiers(&modifiers);
    let doc_comment = extract_javadoc(node, source);

    // Find variable_declarator children to get the field name(s)
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            if child.kind() == "variable_declarator" {
                if let Some(name_node) = child.child_by_field_name("name") {
                    let name = node_text(name_node, source);
                    let qualified_name = qualified(scope, &name);
                    let sig_text = node_text(node, source);
                    let signature = sig_text.trim_end_matches(';').trim().to_string();

                    symbols.push(Symbol {
                        name,
                        qualified_name,
                        kind: SymbolKind::Constant,
                        signature,
                        visibility,
                        file_path: file_path.to_string(),
                        line_start: node.start_position().row,
                        line_end: node.end_position().row,
                        doc_comment: doc_comment.clone(),
                        parent: scope.last().cloned(),
                    });
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
            if let Some(r) = extract_import_reference(node, source, file_path, scope) {
                references.push(r);
            }
            return;
        }
        "method_invocation" => {
            if let Some(r) = extract_call_reference(node, source, file_path, scope) {
                references.push(r);
            }
            // Still recurse into arguments etc.
        }
        "class_declaration" | "interface_declaration" | "enum_declaration" => {
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
        "method_declaration" | "constructor_declaration" => {
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

fn extract_import_reference(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Reference> {
    // import_declaration text is like: "import java.util.List;"
    let text = node_text(node, source);
    let import_path = text
        .trim()
        .strip_prefix("import")?
        .trim()
        .strip_suffix(';')
        .unwrap_or("")
        .trim()
        .trim_start_matches("static ")
        .to_string();

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
    let name_node = node.child_by_field_name("name")?;
    let method_name = node_text(name_node, source);

    // Optionally include the object for qualified calls (e.g., obj.method)
    let target = if let Some(obj_node) = node.child_by_field_name("object") {
        let obj_text = node_text(obj_node, source);
        format!("{}.{}", obj_text, method_name)
    } else {
        method_name
    };

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
    let qn = qualified(scope, &class_name);

    // Check for superclass (extends)
    if let Some(superclass) = node.child_by_field_name("superclass") {
        // superclass node contains the type_identifier
        let super_text = node_text(superclass, source);
        // The text may include "extends " prefix depending on grammar version
        let super_name = super_text.trim().to_string();
        if !super_name.is_empty() {
            references.push(Reference {
                source_qualified_name: qn.clone(),
                target_name: super_name,
                kind: ReferenceKind::Inherits,
                file_path: file_path.to_string(),
                line: superclass.start_position().row,
            });
        }
    }

    // Check for interfaces (implements / extends for interfaces)
    if let Some(interfaces) = node.child_by_field_name("interfaces") {
        // super_interfaces node: iterate children for type_identifier / type_list
        extract_interface_list(interfaces, source, file_path, &qn, references);
    }
}

fn extract_interface_list(
    node: Node,
    source: &[u8],
    file_path: &str,
    source_qn: &str,
    references: &mut Vec<Reference>,
) {
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            match child.kind() {
                "type_identifier" | "generic_type" | "scoped_type_identifier" => {
                    let iface_name = node_text(child, source);
                    references.push(Reference {
                        source_qualified_name: source_qn.to_string(),
                        target_name: iface_name,
                        kind: ReferenceKind::Implements,
                        file_path: file_path.to_string(),
                        line: child.start_position().row,
                    });
                }
                "type_list" => {
                    // Recurse into type_list
                    extract_interface_list(child, source, file_path, source_qn, references);
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

fn qualified(scope: &[String], name: &str) -> String {
    if scope.is_empty() {
        name.to_string()
    } else {
        format!("{}.{}", scope.join("."), name)
    }
}

/// Collect modifier keywords from a node's `modifiers` child.
fn collect_modifiers(node: Node, source: &[u8]) -> Vec<String> {
    let mut modifiers = Vec::new();
    if let Some(mods) = node.child_by_field_name("modifiers") {
        for i in 0..mods.child_count() {
            if let Some(child) = mods.child(i as u32) {
                let text = node_text(child, source);
                modifiers.push(text);
            }
        }
    }
    // Also check unnamed children that might be modifier keywords
    // (some grammar versions place modifiers as direct children)
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            if child.kind() == "modifiers" {
                for j in 0..child.child_count() {
                    if let Some(mod_child) = child.child(j as u32) {
                        let text = node_text(mod_child, source);
                        if !modifiers.contains(&text) {
                            modifiers.push(text);
                        }
                    }
                }
            }
        }
    }
    modifiers
}

fn visibility_from_modifiers(modifiers: &[String]) -> Visibility {
    if modifiers.iter().any(|m| m == "public") || modifiers.iter().any(|m| m == "protected") {
        Visibility::Public
    } else if modifiers.iter().any(|m| m == "private") {
        Visibility::Private
    } else {
        // Package-private (default) → treat as Private
        Visibility::Private
    }
}

fn extract_visibility(node: Node, source: &[u8]) -> Visibility {
    let modifiers = collect_modifiers(node, source);
    visibility_from_modifiers(&modifiers)
}

fn extract_java_signature(node: Node, source: &[u8]) -> String {
    let text = node_text(node, source);
    // Signature is everything up to the opening brace
    if let Some(pos) = text.find('{') {
        text[..pos].trim().to_string()
    } else {
        let first_line = text.lines().next().unwrap_or(&text);
        first_line.trim().to_string()
    }
}

fn extract_javadoc(node: Node, source: &[u8]) -> Option<String> {
    // Look for a block_comment (Javadoc) immediately preceding this node
    let mut prev = node.prev_sibling();
    while let Some(sibling) = prev {
        match sibling.kind() {
            "block_comment" => {
                let text = node_text(sibling, source);
                if text.starts_with("/**") {
                    return Some(clean_javadoc(&text));
                }
                return None;
            }
            "line_comment" => {
                // Skip line comments, look for Javadoc before them
                prev = sibling.prev_sibling();
                continue;
            }
            _ => return None,
        }
    }
    None
}

fn clean_javadoc(raw: &str) -> String {
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

/// Check if a method has a `@Test` annotation in its modifiers.
fn has_test_annotation(node: Node, source: &[u8]) -> bool {
    if let Some(mods) = node.child_by_field_name("modifiers") {
        for i in 0..mods.child_count() {
            if let Some(child) = mods.child(i as u32) {
                if child.kind() == "marker_annotation" || child.kind() == "annotation" {
                    let text = node_text(child, source);
                    if text == "@Test" || text.starts_with("@Test(") {
                        return true;
                    }
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extractor::LanguageExtractor;
    use tree_sitter::Parser;

    fn parse_java(source: &str) -> Tree {
        let mut parser = Parser::new();
        let lang = tree_sitter_java::LANGUAGE;
        parser
            .set_language(&lang.into())
            .expect("failed to set Java language");
        parser
            .parse(source.as_bytes(), None)
            .expect("failed to parse")
    }

    #[test]
    fn extract_java_class() {
        let source = r#"
public class MyService {
    public void run() {}
}
"#;
        let tree = parse_java(source);
        let extractor = JavaExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "MyService.java");

        let class = symbols.iter().find(|s| s.name == "MyService").unwrap();
        assert_eq!(class.kind, SymbolKind::Class);
        assert_eq!(class.visibility, Visibility::Public);
        assert!(
            class.signature.contains("class MyService"),
            "signature: {}",
            class.signature
        );
    }

    #[test]
    fn extract_java_interface() {
        let source = r#"
public interface Repository {
    void save(Object entity);
}
"#;
        let tree = parse_java(source);
        let extractor = JavaExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Repository.java");

        let iface = symbols.iter().find(|s| s.name == "Repository").unwrap();
        assert_eq!(iface.kind, SymbolKind::Interface);
        assert_eq!(iface.visibility, Visibility::Public);
    }

    #[test]
    fn extract_java_enum() {
        let source = r#"
public enum Color {
    RED, GREEN, BLUE
}
"#;
        let tree = parse_java(source);
        let extractor = JavaExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Color.java");

        let en = symbols.iter().find(|s| s.name == "Color").unwrap();
        assert_eq!(en.kind, SymbolKind::Enum);
        assert_eq!(en.visibility, Visibility::Public);
    }

    #[test]
    fn extract_java_method() {
        let source = r#"
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
"#;
        let tree = parse_java(source);
        let extractor = JavaExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Calculator.java");

        let method = symbols.iter().find(|s| s.name == "add").unwrap();
        assert_eq!(method.kind, SymbolKind::Method);
        assert_eq!(method.qualified_name, "Calculator.add");
        assert_eq!(method.parent.as_deref(), Some("Calculator"));
        assert!(
            method.signature.contains("int add(int a, int b)"),
            "signature: {}",
            method.signature
        );
    }

    #[test]
    fn extract_java_constructor() {
        let source = r#"
public class Server {
    private int port;

    public Server(int port) {
        this.port = port;
    }
}
"#;
        let tree = parse_java(source);
        let extractor = JavaExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Server.java");

        let ctor = symbols
            .iter()
            .find(|s| s.name == "Server" && s.kind == SymbolKind::Method)
            .unwrap();
        assert_eq!(ctor.qualified_name, "Server.Server");
        assert!(
            ctor.signature.contains("Server(int port)"),
            "signature: {}",
            ctor.signature
        );
        assert_eq!(ctor.parent.as_deref(), Some("Server"));
    }

    #[test]
    fn extract_java_static_constant() {
        let source = r#"
public class Config {
    public static final int MAX_SIZE = 1024;
    public static final String VERSION = "1.0";
    private int mutableField = 0;
}
"#;
        let tree = parse_java(source);
        let extractor = JavaExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Config.java");

        let constants: Vec<_> = symbols
            .iter()
            .filter(|s| s.kind == SymbolKind::Constant)
            .collect();
        assert!(
            constants.iter().any(|s| s.name == "MAX_SIZE"),
            "Expected MAX_SIZE, got: {:#?}",
            constants
        );
        assert!(
            constants.iter().any(|s| s.name == "VERSION"),
            "Expected VERSION, got: {:#?}",
            constants
        );
        // mutableField should NOT appear as a constant
        assert!(
            !constants.iter().any(|s| s.name == "mutableField"),
            "mutableField should not be a constant"
        );
    }

    #[test]
    fn extract_java_imports() {
        let source = r#"
import java.util.List;
import java.util.Map;
import java.io.IOException;

public class App {}
"#;
        let tree = parse_java(source);
        let extractor = JavaExtractor::new();
        let references = extractor.extract_references(&tree, source.as_bytes(), "App.java");

        let imports: Vec<_> = references
            .iter()
            .filter(|r| r.kind == ReferenceKind::Import)
            .collect();
        assert!(
            imports.iter().any(|r| r.target_name == "java.util.List"),
            "imports: {:#?}",
            imports
        );
        assert!(
            imports.iter().any(|r| r.target_name == "java.util.Map"),
            "imports: {:#?}",
            imports
        );
        assert!(
            imports
                .iter()
                .any(|r| r.target_name == "java.io.IOException"),
            "imports: {:#?}",
            imports
        );
    }

    #[test]
    fn extract_java_inheritance() {
        let source = r#"
public class MyService extends BaseService implements Runnable, Serializable {
    public void run() {}
}
"#;
        let tree = parse_java(source);
        let extractor = JavaExtractor::new();
        let references = extractor.extract_references(&tree, source.as_bytes(), "MyService.java");

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

        let implements: Vec<_> = references
            .iter()
            .filter(|r| r.kind == ReferenceKind::Implements)
            .collect();
        assert!(
            implements.iter().any(|r| r.target_name == "Runnable"),
            "implements: {:#?}",
            implements
        );
        assert!(
            implements.iter().any(|r| r.target_name == "Serializable"),
            "implements: {:#?}",
            implements
        );
    }

    #[test]
    fn extract_java_test_method() {
        let source = r#"
import org.junit.Test;

public class CalculatorTest {
    @Test
    public void testAdd() {
        assert(1 + 1 == 2);
    }

    public void helperMethod() {}
}
"#;
        let tree = parse_java(source);
        let extractor = JavaExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "CalculatorTest.java");

        let test_method = symbols.iter().find(|s| s.name == "testAdd").unwrap();
        assert_eq!(test_method.kind, SymbolKind::Test);

        let helper = symbols.iter().find(|s| s.name == "helperMethod").unwrap();
        assert_eq!(helper.kind, SymbolKind::Method);
    }

    #[test]
    fn extract_java_visibility() {
        let source = r#"
public class Example {
    public void publicMethod() {}
    private void privateMethod() {}
    protected void protectedMethod() {}
    void packagePrivateMethod() {}
}
"#;
        let tree = parse_java(source);
        let extractor = JavaExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Example.java");

        let public_m = symbols.iter().find(|s| s.name == "publicMethod").unwrap();
        assert_eq!(public_m.visibility, Visibility::Public);

        let private_m = symbols.iter().find(|s| s.name == "privateMethod").unwrap();
        assert_eq!(private_m.visibility, Visibility::Private);

        let protected_m = symbols
            .iter()
            .find(|s| s.name == "protectedMethod")
            .unwrap();
        assert_eq!(protected_m.visibility, Visibility::Public); // protected → Public

        let pkg_m = symbols
            .iter()
            .find(|s| s.name == "packagePrivateMethod")
            .unwrap();
        assert_eq!(pkg_m.visibility, Visibility::Private); // package-private → Private
    }

    #[test]
    fn extract_java_javadoc() {
        let source = r#"
/**
 * A utility class for string operations.
 * Provides helper methods for formatting.
 */
public class StringUtils {
    /**
     * Formats the given input string.
     * @param input the raw string
     * @return the formatted string
     */
    public String format(String input) {
        return input.trim();
    }
}
"#;
        let tree = parse_java(source);
        let extractor = JavaExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "StringUtils.java");

        let class = symbols.iter().find(|s| s.name == "StringUtils").unwrap();
        let doc = class
            .doc_comment
            .as_ref()
            .expect("expected Javadoc on class");
        assert!(
            doc.contains("utility class for string operations"),
            "doc: {}",
            doc
        );

        let method = symbols.iter().find(|s| s.name == "format").unwrap();
        let method_doc = method
            .doc_comment
            .as_ref()
            .expect("expected Javadoc on method");
        assert!(
            method_doc.contains("Formats the given input string"),
            "method_doc: {}",
            method_doc
        );
    }

    #[test]
    fn extract_java_method_invocation_references() {
        let source = r#"
public class App {
    public void run() {
        System.out.println("hello");
        doWork();
    }

    private void doWork() {}
}
"#;
        let tree = parse_java(source);
        let extractor = JavaExtractor::new();
        let references = extractor.extract_references(&tree, source.as_bytes(), "App.java");

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
