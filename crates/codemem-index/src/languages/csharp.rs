//! C# language extractor using tree-sitter-c-sharp.

use crate::extractor::LanguageExtractor;
use crate::symbol::{Reference, ReferenceKind, Symbol, SymbolKind, Visibility};
use tree_sitter::{Node, Tree};

/// C# language extractor for tree-sitter-based code indexing.
pub struct CSharpExtractor;

impl CSharpExtractor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CSharpExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageExtractor for CSharpExtractor {
    fn language_name(&self) -> &str {
        "csharp"
    }

    fn file_extensions(&self) -> &[&str] {
        &["cs"]
    }

    fn tree_sitter_language(&self) -> tree_sitter::Language {
        tree_sitter_c_sharp::LANGUAGE.into()
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
        "namespace_declaration" => {
            if let Some(sym) = extract_namespace(node, source, file_path, scope) {
                let ns_name = sym.name.clone();
                symbols.push(sym);
                // Recurse into namespace body with updated scope
                if let Some(body) = node.child_by_field_name("body") {
                    let mut new_scope = scope.to_vec();
                    new_scope.push(ns_name);
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
                return;
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
                symbols.push(sym);
                return;
            }
        }
        "struct_declaration" => {
            if let Some(sym) = extract_struct(node, source, file_path, scope) {
                let struct_name = sym.name.clone();
                symbols.push(sym);
                if let Some(body) = node.child_by_field_name("body") {
                    let mut new_scope = scope.to_vec();
                    new_scope.push(struct_name);
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
        _ => {}
    }

    // Default recursion for other node types
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            extract_symbols_recursive(child, source, file_path, scope, symbols);
        }
    }
}

fn extract_namespace(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let qualified_name = qualified(scope, &name);
    let signature = extract_csharp_signature(node, source);
    let doc_comment = extract_xml_doc(node, source);

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

fn extract_class(node: Node, source: &[u8], file_path: &str, scope: &[String]) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let qualified_name = qualified(scope, &name);
    let visibility = extract_visibility(node, source);
    let signature = extract_csharp_signature(node, source);
    let doc_comment = extract_xml_doc(node, source);

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
    let signature = extract_csharp_signature(node, source);
    let doc_comment = extract_xml_doc(node, source);

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
    let signature = extract_csharp_signature(node, source);
    let doc_comment = extract_xml_doc(node, source);

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

fn extract_struct(node: Node, source: &[u8], file_path: &str, scope: &[String]) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let qualified_name = qualified(scope, &name);
    let visibility = extract_visibility(node, source);
    let signature = extract_csharp_signature(node, source);
    let doc_comment = extract_xml_doc(node, source);

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

fn extract_method(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let qualified_name = qualified(scope, &name);
    let visibility = extract_visibility(node, source);
    let signature = extract_csharp_signature(node, source);
    let doc_comment = extract_xml_doc(node, source);

    // If inside a class/struct (scope is non-empty), it's a Method; otherwise Function
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
    let signature = extract_csharp_signature(node, source);
    let doc_comment = extract_xml_doc(node, source);

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

// -- Reference Extraction -----------------------------------------------------

fn extract_references_recursive(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    references: &mut Vec<Reference>,
) {
    match node.kind() {
        "using_directive" => {
            if let Some(r) = extract_using_reference(node, source, file_path, scope) {
                references.push(r);
            }
            return;
        }
        "invocation_expression" => {
            if let Some(r) = extract_call_reference(node, source, file_path, scope) {
                references.push(r);
            }
            // Still recurse into arguments etc.
        }
        "class_declaration" | "struct_declaration" | "interface_declaration" => {
            // Extract inheritance/implementation references from base_list
            extract_base_list_references(node, source, file_path, scope, references);

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
        "namespace_declaration" => {
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

fn extract_using_reference(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Reference> {
    // using_directive text is like: "using System.Collections.Generic;"
    // or "global using System;" or "using static System.Math;"
    let text = node_text(node, source);
    let mut remainder = text.trim();

    // Strip optional "global" prefix
    if let Some(rest) = remainder.strip_prefix("global") {
        remainder = rest.trim();
    }

    // Strip required "using" keyword
    remainder = remainder.strip_prefix("using")?.trim();

    // Strip optional "static" keyword
    if let Some(rest) = remainder.strip_prefix("static") {
        remainder = rest.trim();
    }

    // Strip trailing semicolon
    remainder = remainder.strip_suffix(';').unwrap_or(remainder).trim();

    if remainder.is_empty() {
        return None;
    }

    // Skip alias using directives (e.g., "using Alias = Type")
    if remainder.contains('=') {
        return None;
    }

    let import_path = remainder.to_string();

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
    // invocation_expression has field "function" and "arguments"
    let func_node = node.child_by_field_name("function")?;
    let func_text = node_text(func_node, source);

    let source_qn = if scope.is_empty() {
        file_path.to_string()
    } else {
        scope.join(".")
    };

    Some(Reference {
        source_qualified_name: source_qn,
        target_name: func_text,
        kind: ReferenceKind::Call,
        file_path: file_path.to_string(),
        line: node.start_position().row,
    })
}

fn extract_base_list_references(
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
    let type_name = node_text(name_node, source);
    let qn = qualified(scope, &type_name);

    let is_interface = node.kind() == "interface_declaration";

    // Find base_list child node
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            if child.kind() == "base_list" {
                extract_base_types(child, source, file_path, &qn, is_interface, references);
            }
        }
    }
}

fn extract_base_types(
    node: Node,
    source: &[u8],
    file_path: &str,
    source_qn: &str,
    is_interface: bool,
    references: &mut Vec<Reference>,
) {
    // In C#, the base_list contains types separated by commas after ':'
    // The first type for a class could be a base class or interface.
    // For interfaces, all base types are interfaces (Inherits).
    // For classes, we treat the first as Inherits if it looks like a class,
    // but since we can't reliably distinguish without type resolution,
    // we use a heuristic: types starting with 'I' followed by uppercase
    // are treated as interfaces (Implements), others as Inherits for the first,
    // and Implements for the rest.
    let mut first = true;
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            match child.kind() {
                "identifier" | "qualified_name" | "generic_name" => {
                    let type_text = node_text(child, source);
                    let kind = if is_interface {
                        // Interface extending other interfaces
                        ReferenceKind::Inherits
                    } else if first && !looks_like_interface(&type_text) {
                        // First base type that doesn't look like interface -> Inherits
                        ReferenceKind::Inherits
                    } else {
                        ReferenceKind::Implements
                    };
                    references.push(Reference {
                        source_qualified_name: source_qn.to_string(),
                        target_name: type_text,
                        kind,
                        file_path: file_path.to_string(),
                        line: child.start_position().row,
                    });
                    first = false;
                }
                _ => {
                    // Recurse into nested nodes (e.g., simple_base_type wrapping identifier)
                    extract_base_types(child, source, file_path, source_qn, is_interface, references);
                    if child.kind() != ":" && child.kind() != "," {
                        first = false;
                    }
                }
            }
        }
    }
}

/// Heuristic: C# interfaces conventionally start with 'I' followed by uppercase.
fn looks_like_interface(name: &str) -> bool {
    let name = name.split('<').next().unwrap_or(name);
    let name = name.split('.').next_back().unwrap_or(name);
    name.starts_with('I')
        && name.len() > 1
        && name.chars().nth(1).is_some_and(|c| c.is_uppercase())
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

/// Collect modifier keywords from a node's direct `modifier` children.
fn collect_modifiers(node: Node, source: &[u8]) -> Vec<String> {
    let mut modifiers = Vec::new();
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            if child.kind() == "modifier" {
                let text = node_text(child, source);
                modifiers.push(text);
            }
        }
    }
    modifiers
}

fn visibility_from_modifiers(modifiers: &[String]) -> Visibility {
    if modifiers.iter().any(|m| m == "public") {
        Visibility::Public
    } else if modifiers.iter().any(|m| m == "protected") {
        Visibility::Protected
    } else if modifiers.iter().any(|m| m == "internal") {
        // internal is roughly equivalent to crate visibility
        Visibility::Public
    } else if modifiers.iter().any(|m| m == "private") {
        Visibility::Private
    } else {
        // C# default is private for members, internal for top-level types
        Visibility::Private
    }
}

fn extract_visibility(node: Node, source: &[u8]) -> Visibility {
    let modifiers = collect_modifiers(node, source);
    visibility_from_modifiers(&modifiers)
}

fn extract_csharp_signature(node: Node, source: &[u8]) -> String {
    let text = node_text(node, source);
    // Signature is everything up to the opening brace
    if let Some(pos) = text.find('{') {
        text[..pos].trim().to_string()
    } else {
        let first_line = text.lines().next().unwrap_or(&text);
        first_line.trim().to_string()
    }
}

/// Extract XML documentation comments (/// ...) preceding a node.
fn extract_xml_doc(node: Node, source: &[u8]) -> Option<String> {
    let mut comment_lines = Vec::new();
    let mut prev = node.prev_sibling();

    // Collect consecutive comment nodes preceding this declaration
    while let Some(sibling) = prev {
        match sibling.kind() {
            "comment" => {
                let text = node_text(sibling, source);
                if text.starts_with("///") {
                    comment_lines.push(text);
                    prev = sibling.prev_sibling();
                    continue;
                }
                // Not an XML doc comment, stop
                break;
            }
            _ => break,
        }
    }

    if comment_lines.is_empty() {
        return None;
    }

    // Lines were collected in reverse order (bottom to top)
    comment_lines.reverse();

    let cleaned: Vec<&str> = comment_lines
        .iter()
        .map(|line| {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("/// ") {
                rest
            } else if let Some(rest) = trimmed.strip_prefix("///") {
                rest
            } else {
                trimmed
            }
        })
        .collect();

    // Trim leading/trailing empty lines
    let mut result: Vec<&str> = cleaned;
    while result.first().is_some_and(|l| l.is_empty()) {
        result.remove(0);
    }
    while result.last().is_some_and(|l| l.is_empty()) {
        result.pop();
    }

    if result.is_empty() {
        None
    } else {
        Some(result.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extractor::LanguageExtractor;
    use tree_sitter::Parser;

    fn parse_csharp(source: &str) -> Tree {
        let mut parser = Parser::new();
        let lang = tree_sitter_c_sharp::LANGUAGE;
        parser
            .set_language(&lang.into())
            .expect("failed to set C# language");
        parser
            .parse(source.as_bytes(), None)
            .expect("failed to parse")
    }

    #[test]
    fn extract_csharp_class_with_methods() {
        let source = r#"
public class MyService {
    public void Run() {}
    private int Calculate(int x) { return x * 2; }
}
"#;
        let tree = parse_csharp(source);
        let extractor = CSharpExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "MyService.cs");

        let class = symbols.iter().find(|s| s.name == "MyService").unwrap();
        assert_eq!(class.kind, SymbolKind::Class);
        assert_eq!(class.visibility, Visibility::Public);
        assert!(
            class.signature.contains("class MyService"),
            "signature: {}",
            class.signature
        );

        let run = symbols.iter().find(|s| s.name == "Run").unwrap();
        assert_eq!(run.kind, SymbolKind::Method);
        assert_eq!(run.qualified_name, "MyService.Run");
        assert_eq!(run.parent.as_deref(), Some("MyService"));
        assert_eq!(run.visibility, Visibility::Public);

        let calc = symbols.iter().find(|s| s.name == "Calculate").unwrap();
        assert_eq!(calc.kind, SymbolKind::Method);
        assert_eq!(calc.qualified_name, "MyService.Calculate");
        assert_eq!(calc.visibility, Visibility::Private);
    }

    #[test]
    fn extract_csharp_interface() {
        let source = r#"
public interface IRepository {
    void Save(object entity);
    object FindById(int id);
}
"#;
        let tree = parse_csharp(source);
        let extractor = CSharpExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "IRepository.cs");

        let iface = symbols.iter().find(|s| s.name == "IRepository").unwrap();
        assert_eq!(iface.kind, SymbolKind::Interface);
        assert_eq!(iface.visibility, Visibility::Public);
    }

    #[test]
    fn extract_csharp_namespace() {
        let source = r#"
namespace MyApp.Services {
    public class UserService {
        public void Create() {}
    }
}
"#;
        let tree = parse_csharp(source);
        let extractor = CSharpExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "UserService.cs");

        let ns = symbols
            .iter()
            .find(|s| s.name == "MyApp.Services")
            .unwrap();
        assert_eq!(ns.kind, SymbolKind::Module);

        let class = symbols.iter().find(|s| s.name == "UserService").unwrap();
        assert_eq!(class.kind, SymbolKind::Class);
        assert_eq!(class.qualified_name, "MyApp.Services.UserService");
        assert_eq!(class.parent.as_deref(), Some("MyApp.Services"));

        let method = symbols.iter().find(|s| s.name == "Create").unwrap();
        assert_eq!(method.kind, SymbolKind::Method);
        assert_eq!(
            method.qualified_name,
            "MyApp.Services.UserService.Create"
        );
    }

    #[test]
    fn extract_csharp_using_directives() {
        let source = r#"
using System;
using System.Collections.Generic;
using System.Linq;

public class App {}
"#;
        let tree = parse_csharp(source);
        let extractor = CSharpExtractor::new();
        let references = extractor.extract_references(&tree, source.as_bytes(), "App.cs");

        let imports: Vec<_> = references
            .iter()
            .filter(|r| r.kind == ReferenceKind::Import)
            .collect();
        assert!(
            imports.iter().any(|r| r.target_name == "System"),
            "imports: {:#?}",
            imports
        );
        assert!(
            imports
                .iter()
                .any(|r| r.target_name == "System.Collections.Generic"),
            "imports: {:#?}",
            imports
        );
        assert!(
            imports.iter().any(|r| r.target_name == "System.Linq"),
            "imports: {:#?}",
            imports
        );
    }

    #[test]
    fn extract_csharp_inheritance() {
        let source = r#"
public class MyService : BaseService, IRunnable, IDisposable {
    public void Run() {}
}
"#;
        let tree = parse_csharp(source);
        let extractor = CSharpExtractor::new();
        let references = extractor.extract_references(&tree, source.as_bytes(), "MyService.cs");

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
            implements.iter().any(|r| r.target_name == "IRunnable"),
            "implements: {:#?}",
            implements
        );
        assert!(
            implements.iter().any(|r| r.target_name == "IDisposable"),
            "implements: {:#?}",
            implements
        );
    }

    #[test]
    fn extract_csharp_enum() {
        let source = r#"
public enum Color {
    Red,
    Green,
    Blue
}
"#;
        let tree = parse_csharp(source);
        let extractor = CSharpExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Color.cs");

        let en = symbols.iter().find(|s| s.name == "Color").unwrap();
        assert_eq!(en.kind, SymbolKind::Enum);
        assert_eq!(en.visibility, Visibility::Public);
    }

    #[test]
    fn extract_csharp_struct() {
        let source = r#"
public struct Point {
    public int X;
    public int Y;
}
"#;
        let tree = parse_csharp(source);
        let extractor = CSharpExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Point.cs");

        let st = symbols.iter().find(|s| s.name == "Point").unwrap();
        assert_eq!(st.kind, SymbolKind::Struct);
        assert_eq!(st.visibility, Visibility::Public);
    }

    #[test]
    fn extract_csharp_constructor() {
        let source = r#"
public class Server {
    private int port;

    public Server(int port) {
        this.port = port;
    }
}
"#;
        let tree = parse_csharp(source);
        let extractor = CSharpExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Server.cs");

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
    fn extract_csharp_xml_doc_comment() {
        let source = r#"
/// <summary>
/// A utility class for string operations.
/// Provides helper methods for formatting.
/// </summary>
public class StringUtils {
    /// <summary>
    /// Formats the given input string.
    /// </summary>
    public string Format(string input) {
        return input.Trim();
    }
}
"#;
        let tree = parse_csharp(source);
        let extractor = CSharpExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "StringUtils.cs");

        let class = symbols.iter().find(|s| s.name == "StringUtils").unwrap();
        let doc = class
            .doc_comment
            .as_ref()
            .expect("expected XML doc on class");
        assert!(
            doc.contains("utility class for string operations"),
            "doc: {}",
            doc
        );

        let method = symbols.iter().find(|s| s.name == "Format").unwrap();
        let method_doc = method
            .doc_comment
            .as_ref()
            .expect("expected XML doc on method");
        assert!(
            method_doc.contains("Formats the given input string"),
            "method_doc: {}",
            method_doc
        );
    }

    #[test]
    fn extract_csharp_visibility() {
        let source = r#"
public class Example {
    public void PublicMethod() {}
    private void PrivateMethod() {}
    protected void ProtectedMethod() {}
    internal void InternalMethod() {}
    void DefaultMethod() {}
}
"#;
        let tree = parse_csharp(source);
        let extractor = CSharpExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "Example.cs");

        let public_m = symbols
            .iter()
            .find(|s| s.name == "PublicMethod")
            .unwrap();
        assert_eq!(public_m.visibility, Visibility::Public);

        let private_m = symbols
            .iter()
            .find(|s| s.name == "PrivateMethod")
            .unwrap();
        assert_eq!(private_m.visibility, Visibility::Private);

        let protected_m = symbols
            .iter()
            .find(|s| s.name == "ProtectedMethod")
            .unwrap();
        assert_eq!(protected_m.visibility, Visibility::Protected);

        let internal_m = symbols
            .iter()
            .find(|s| s.name == "InternalMethod")
            .unwrap();
        assert_eq!(internal_m.visibility, Visibility::Public); // internal -> Public

        let default_m = symbols
            .iter()
            .find(|s| s.name == "DefaultMethod")
            .unwrap();
        assert_eq!(default_m.visibility, Visibility::Private); // default -> Private
    }

    #[test]
    fn extract_csharp_invocation_references() {
        let source = r#"
public class App {
    public void Run() {
        Console.WriteLine("hello");
        DoWork();
    }

    private void DoWork() {}
}
"#;
        let tree = parse_csharp(source);
        let extractor = CSharpExtractor::new();
        let references = extractor.extract_references(&tree, source.as_bytes(), "App.cs");

        let calls: Vec<_> = references
            .iter()
            .filter(|r| r.kind == ReferenceKind::Call)
            .collect();
        assert!(
            calls
                .iter()
                .any(|r| r.target_name.contains("WriteLine")),
            "calls: {:#?}",
            calls
        );
        assert!(
            calls.iter().any(|r| r.target_name == "DoWork"),
            "calls: {:#?}",
            calls
        );
    }
}
