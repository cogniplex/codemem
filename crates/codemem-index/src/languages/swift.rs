//! Swift language extractor using tree-sitter-swift.

use crate::extractor::LanguageExtractor;
use crate::symbol::{Reference, ReferenceKind, Symbol, SymbolKind, Visibility};
use tree_sitter::{Node, Tree};

/// Swift language extractor for tree-sitter-based code indexing.
pub struct SwiftExtractor;

impl SwiftExtractor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SwiftExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageExtractor for SwiftExtractor {
    fn language_name(&self) -> &str {
        "swift"
    }

    fn file_extensions(&self) -> &[&str] {
        &["swift"]
    }

    fn tree_sitter_language(&self) -> tree_sitter::Language {
        tree_sitter_swift::LANGUAGE.into()
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
        // tree-sitter-swift v0.7.x uses "class_declaration" for class, struct,
        // and enum.  We disambiguate via the keyword child node ("class",
        // "struct", or "enum").
        "class_declaration" => {
            let extract_fn = match class_declaration_keyword(node) {
                Some("struct") => {
                    extract_struct as fn(Node, &[u8], &str, &[String]) -> Option<Symbol>
                }
                Some("enum") => extract_enum,
                _ => extract_class,
            };
            if let Some(sym) = extract_fn(node, source, file_path, scope) {
                let type_name = sym.name.clone();
                symbols.push(sym);
                let mut new_scope = scope.to_vec();
                new_scope.push(type_name);
                recurse_into_body(node, source, file_path, &new_scope, symbols);
                return; // Don't default-recurse
            }
        }
        "protocol_declaration" => {
            if let Some(sym) = extract_protocol(node, source, file_path, scope) {
                let proto_name = sym.name.clone();
                symbols.push(sym);
                let mut new_scope = scope.to_vec();
                new_scope.push(proto_name);
                recurse_into_body(node, source, file_path, &new_scope, symbols);
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

/// Recurse into the body of a type declaration (class, struct, protocol, enum).
fn recurse_into_body(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    symbols: &mut Vec<Symbol>,
) {
    // tree-sitter-swift uses "class_body", "protocol_body", etc.,
    // but also may have a generic "body" field. Try both approaches.
    if let Some(body) = node.child_by_field_name("body") {
        for i in 0..body.child_count() {
            if let Some(child) = body.child(i as u32) {
                extract_symbols_recursive(child, source, file_path, scope, symbols);
            }
        }
        return;
    }
    // Fallback: look for children that are body-like nodes
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            let kind = child.kind();
            if kind.ends_with("_body") {
                for j in 0..child.child_count() {
                    if let Some(grandchild) = child.child(j as u32) {
                        extract_symbols_recursive(grandchild, source, file_path, scope, symbols);
                    }
                }
            }
        }
    }
}

fn extract_class(node: Node, source: &[u8], file_path: &str, scope: &[String]) -> Option<Symbol> {
    let name = find_declaration_name(node, source)?;
    let qualified_name = qualified(scope, &name);
    let visibility = extract_visibility(node, source);
    let signature = extract_swift_signature(node, source);
    let doc_comment = extract_doc_comment(node, source);

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

fn extract_protocol(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Symbol> {
    let name = find_declaration_name(node, source)?;
    let qualified_name = qualified(scope, &name);
    let visibility = extract_visibility(node, source);
    let signature = extract_swift_signature(node, source);
    let doc_comment = extract_doc_comment(node, source);

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

fn extract_struct(node: Node, source: &[u8], file_path: &str, scope: &[String]) -> Option<Symbol> {
    let name = find_declaration_name(node, source)?;
    let qualified_name = qualified(scope, &name);
    let visibility = extract_visibility(node, source);
    let signature = extract_swift_signature(node, source);
    let doc_comment = extract_doc_comment(node, source);

    Some(Symbol {
        name,
        qualified_name,
        kind: SymbolKind::Struct,
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
    let name = find_declaration_name(node, source)?;
    let qualified_name = qualified(scope, &name);
    let visibility = extract_visibility(node, source);
    let signature = extract_swift_signature(node, source);
    let doc_comment = extract_doc_comment(node, source);

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

fn extract_function(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Symbol> {
    let name = find_declaration_name(node, source)?;
    let qualified_name = qualified(scope, &name);
    let visibility = extract_visibility(node, source);
    let signature = extract_swift_signature(node, source);
    let doc_comment = extract_doc_comment(node, source);

    // If inside a class/struct/protocol/enum scope, it's a Method; otherwise Function
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
        "import_declaration" => {
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
        "class_declaration" => {
            // tree-sitter-swift v0.7.x uses class_declaration for class/struct/enum
            // Extract inheritance references
            extract_inheritance_references(node, source, file_path, scope, references);

            // Recurse with updated scope
            if let Some(name) = find_declaration_name(node, source) {
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
        "protocol_declaration" => {
            // Protocols can also inherit from other protocols
            extract_inheritance_references(node, source, file_path, scope, references);

            if let Some(name) = find_declaration_name(node, source) {
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
            if let Some(name) = find_declaration_name(node, source) {
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
    // import_declaration text is like: "import Foundation" or "import UIKit"
    let text = node_text(node, source);
    let import_path = text.trim().strip_prefix("import")?.trim().to_string();

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
    // call_expression typically has a "function" child in tree-sitter-swift
    // Try to get the callable name from the first child or field
    let callee = node
        .child_by_field_name("function")
        .or_else(|| node.child(0))?;
    let target = node_text(callee, source);

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

fn extract_inheritance_references(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    references: &mut Vec<Reference>,
) {
    let name = match find_declaration_name(node, source) {
        Some(n) => n,
        None => return,
    };
    let qn = qualified(scope, &name);

    // Look for type_inheritance_clause or inheritance_type children
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            let kind = child.kind();
            if kind == "type_inheritance_clause" || kind == "inheritance_clause" {
                // Inside the inheritance clause, look for type identifiers
                extract_inherited_types(child, source, file_path, &qn, references);
            }
        }
    }
}

fn extract_inherited_types(
    node: Node,
    source: &[u8],
    file_path: &str,
    source_qn: &str,
    references: &mut Vec<Reference>,
) {
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            match child.kind() {
                "type_identifier" | "user_type" | "simple_identifier" => {
                    let type_name = node_text(child, source);
                    if !type_name.is_empty() && type_name != ":" && type_name != "," {
                        references.push(Reference {
                            source_qualified_name: source_qn.to_string(),
                            target_name: type_name,
                            kind: ReferenceKind::Inherits,
                            file_path: file_path.to_string(),
                            line: child.start_position().row,
                        });
                    }
                }
                // Recurse into nested inheritance nodes
                _ => {
                    extract_inherited_types(child, source, file_path, source_qn, references);
                }
            }
        }
    }
}

// ── Helper Functions ──────────────────────────────────────────────────────

/// Determine whether a `class_declaration` node actually represents a class,
/// struct, or enum.  tree-sitter-swift v0.7.x uses `class_declaration` for all
/// three; the first unnamed child carries the keyword (`"class"`, `"struct"`,
/// or `"enum"`).
fn class_declaration_keyword(node: Node) -> Option<&'static str> {
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            match child.kind() {
                "struct" => return Some("struct"),
                "enum" => return Some("enum"),
                "class" => return Some("class"),
                _ => {}
            }
        }
    }
    None
}

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

/// Find the name of a declaration node.
///
/// tree-sitter-swift may use a "name" field or place the identifier as a direct child.
fn find_declaration_name(node: Node, source: &[u8]) -> Option<String> {
    // Try the "name" field first (common in many grammars)
    if let Some(name_node) = node.child_by_field_name("name") {
        let name = node_text(name_node, source);
        if !name.is_empty() {
            return Some(name);
        }
    }

    // Fallback: look for the first type_identifier or simple_identifier child
    // that comes after the keyword (class, struct, func, etc.)
    let mut past_keyword = false;
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            let kind = child.kind();
            // Skip the keyword itself
            if kind == "class"
                || kind == "struct"
                || kind == "protocol"
                || kind == "enum"
                || kind == "func"
            {
                past_keyword = true;
                continue;
            }
            if past_keyword
                && (kind == "type_identifier"
                    || kind == "simple_identifier"
                    || kind == "identifier")
            {
                let name = node_text(child, source);
                if !name.is_empty() {
                    return Some(name);
                }
            }
        }
    }

    None
}

/// Extract visibility from Swift access-level modifiers.
///
/// Swift has: `public`, `open`, `internal`, `fileprivate`, `private`.
/// We map `public` and `open` to Visibility::Public, `private` and `fileprivate`
/// to Visibility::Private, and `internal` (the default) to Visibility::Private.
fn extract_visibility(node: Node, source: &[u8]) -> Visibility {
    // Check for modifiers field
    if let Some(mods) = node.child_by_field_name("modifiers") {
        return visibility_from_modifier_node(mods, source);
    }

    // Fallback: scan direct children for modifier nodes
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            let kind = child.kind();
            if kind == "modifiers"
                || kind == "modifier"
                || kind == "visibility_modifier"
                || kind == "access_level_modifier"
            {
                return visibility_from_modifier_node(child, source);
            }
        }
    }

    // Default in Swift is `internal`, which we map to Private
    Visibility::Private
}

fn visibility_from_modifier_node(node: Node, source: &[u8]) -> Visibility {
    let text = node_text(node, source);

    // Check the full modifier text for access level keywords
    if text.contains("public") || text.contains("open") {
        return Visibility::Public;
    }
    if text.contains("private") || text.contains("fileprivate") {
        return Visibility::Private;
    }
    if text.contains("internal") {
        return Visibility::Private;
    }

    // Also recurse into children in case modifiers is a container
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            let child_text = node_text(child, source);
            if child_text.contains("public") || child_text.contains("open") {
                return Visibility::Public;
            }
            if child_text.contains("private") || child_text.contains("fileprivate") {
                return Visibility::Private;
            }
        }
    }

    Visibility::Private
}

/// Extract signature: everything up to the opening `{`.
fn extract_swift_signature(node: Node, source: &[u8]) -> String {
    let text = node_text(node, source);
    if let Some(pos) = text.find('{') {
        text[..pos].trim().to_string()
    } else {
        let first_line = text.lines().next().unwrap_or(&text);
        first_line.trim().to_string()
    }
}

/// Extract doc comments preceding a node.
///
/// Swift uses `///` for line doc comments and `/** ... */` for block doc comments.
fn extract_doc_comment(node: Node, source: &[u8]) -> Option<String> {
    let mut prev = node.prev_sibling();
    let mut doc_lines: Vec<String> = Vec::new();

    while let Some(sibling) = prev {
        match sibling.kind() {
            "comment" | "line_comment" => {
                let text = node_text(sibling, source);
                let trimmed = text.trim();
                if let Some(rest) = trimmed.strip_prefix("///") {
                    // Collect triple-slash doc comment lines (in reverse order)
                    doc_lines.push(rest.trim().to_string());
                    prev = sibling.prev_sibling();
                    continue;
                }
                // Regular comment, skip and keep looking
                prev = sibling.prev_sibling();
                continue;
            }
            "block_comment" | "multiline_comment" => {
                let text = node_text(sibling, source);
                if text.starts_with("/**") {
                    return Some(clean_block_doc_comment(&text));
                }
                return None;
            }
            _ => break,
        }
    }

    if doc_lines.is_empty() {
        return None;
    }

    // Reverse because we collected them bottom-up
    doc_lines.reverse();
    Some(doc_lines.join("\n"))
}

/// Clean a block doc comment (/** ... */).
fn clean_block_doc_comment(raw: &str) -> String {
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

    fn parse_swift(source: &str) -> Tree {
        let mut parser = Parser::new();
        let lang = tree_sitter_swift::LANGUAGE.into();
        parser
            .set_language(&lang)
            .expect("failed to set Swift language");
        parser
            .parse(source.as_bytes(), None)
            .expect("failed to parse")
    }

    #[test]
    fn extract_swift_class_with_methods() {
        let source = r#"
public class UserService {
    public func findUser(id: Int) -> User? {
        return nil
    }

    private func validate(name: String) -> Bool {
        return !name.isEmpty
    }
}
"#;
        let tree = parse_swift(source);
        let extractor = SwiftExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "UserService.swift");

        let class = symbols.iter().find(|s| s.name == "UserService");
        assert!(
            class.is_some(),
            "Expected class UserService, got symbols: {:#?}",
            symbols
        );
        let class = class.unwrap();
        assert_eq!(class.kind, SymbolKind::Class);
        assert!(
            class.signature.contains("class UserService"),
            "signature: {}",
            class.signature
        );

        let find_user = symbols.iter().find(|s| s.name == "findUser");
        assert!(
            find_user.is_some(),
            "Expected method findUser, got symbols: {:#?}",
            symbols
        );
        let find_user = find_user.unwrap();
        assert_eq!(find_user.kind, SymbolKind::Method);
        assert_eq!(find_user.qualified_name, "UserService.findUser");
        assert_eq!(find_user.parent.as_deref(), Some("UserService"));

        let validate = symbols.iter().find(|s| s.name == "validate");
        assert!(
            validate.is_some(),
            "Expected method validate, got symbols: {:#?}",
            symbols
        );
        let validate = validate.unwrap();
        assert_eq!(validate.kind, SymbolKind::Method);
        assert_eq!(validate.qualified_name, "UserService.validate");
    }

    #[test]
    fn extract_swift_struct() {
        let source = r#"
public struct Point {
    var x: Double
    var y: Double

    func distance(to other: Point) -> Double {
        let dx = x - other.x
        let dy = y - other.y
        return (dx * dx + dy * dy).squareRoot()
    }
}
"#;
        let tree = parse_swift(source);
        let extractor = SwiftExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Point.swift");

        let s = symbols.iter().find(|s| s.name == "Point");
        assert!(
            s.is_some(),
            "Expected struct Point, got symbols: {:#?}",
            symbols
        );
        let s = s.unwrap();
        assert_eq!(s.kind, SymbolKind::Struct);
        assert!(
            s.signature.contains("struct Point"),
            "signature: {}",
            s.signature
        );

        let distance = symbols.iter().find(|sym| sym.name == "distance");
        assert!(
            distance.is_some(),
            "Expected method distance, got symbols: {:#?}",
            symbols
        );
        let distance = distance.unwrap();
        assert_eq!(distance.kind, SymbolKind::Method);
        assert_eq!(distance.qualified_name, "Point.distance");
        assert_eq!(distance.parent.as_deref(), Some("Point"));
    }

    #[test]
    fn extract_swift_protocol() {
        let source = r#"
public protocol Drawable {
    func draw()
    func resize(width: Int, height: Int)
}
"#;
        let tree = parse_swift(source);
        let extractor = SwiftExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Drawable.swift");

        let proto = symbols.iter().find(|s| s.name == "Drawable");
        assert!(
            proto.is_some(),
            "Expected protocol Drawable, got symbols: {:#?}",
            symbols
        );
        let proto = proto.unwrap();
        assert_eq!(proto.kind, SymbolKind::Interface);
        assert!(
            proto.signature.contains("protocol Drawable"),
            "signature: {}",
            proto.signature
        );
    }

    #[test]
    fn extract_swift_enum() {
        let source = r#"
public enum Direction {
    case north
    case south
    case east
    case west
}
"#;
        let tree = parse_swift(source);
        let extractor = SwiftExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Direction.swift");

        let en = symbols.iter().find(|s| s.name == "Direction");
        assert!(
            en.is_some(),
            "Expected enum Direction, got symbols: {:#?}",
            symbols
        );
        let en = en.unwrap();
        assert_eq!(en.kind, SymbolKind::Enum);
        assert!(
            en.signature.contains("enum Direction"),
            "signature: {}",
            en.signature
        );
    }

    #[test]
    fn extract_swift_imports() {
        let source = r#"
import Foundation
import UIKit
import SwiftUI

class App {}
"#;
        let tree = parse_swift(source);
        let extractor = SwiftExtractor::new();
        let references = extractor.extract_references(&tree, source.as_bytes(), "App.swift");

        let imports: Vec<_> = references
            .iter()
            .filter(|r| r.kind == ReferenceKind::Import)
            .collect();
        assert!(
            imports.iter().any(|r| r.target_name == "Foundation"),
            "imports: {:#?}",
            imports
        );
        assert!(
            imports.iter().any(|r| r.target_name == "UIKit"),
            "imports: {:#?}",
            imports
        );
        assert!(
            imports.iter().any(|r| r.target_name == "SwiftUI"),
            "imports: {:#?}",
            imports
        );
    }

    #[test]
    fn extract_swift_free_function() {
        let source = r#"
func greet(name: String) -> String {
    return "Hello, \(name)!"
}
"#;
        let tree = parse_swift(source);
        let extractor = SwiftExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Greet.swift");

        let func = symbols.iter().find(|s| s.name == "greet");
        assert!(
            func.is_some(),
            "Expected function greet, got symbols: {:#?}",
            symbols
        );
        let func = func.unwrap();
        assert_eq!(func.kind, SymbolKind::Function);
        assert!(func.parent.is_none());
    }
}
