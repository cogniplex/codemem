//! C/C++ language extractor using tree-sitter-cpp.
//!
//! The tree-sitter-cpp grammar is a superset that covers both C and C++,
//! so this single extractor handles `.c`, `.h`, `.cpp`, `.hpp`, `.cc`,
//! `.cxx`, and `.hxx` files.

use crate::extractor::LanguageExtractor;
use crate::symbol::{Reference, ReferenceKind, Symbol, SymbolKind, Visibility};
use tree_sitter::{Node, Tree};

/// C/C++ language extractor for tree-sitter-based code indexing.
pub struct CppExtractor;

impl CppExtractor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CppExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageExtractor for CppExtractor {
    fn language_name(&self) -> &str {
        "cpp"
    }

    fn file_extensions(&self) -> &[&str] {
        &["c", "h", "cpp", "hpp", "cc", "cxx", "hxx"]
    }

    fn tree_sitter_language(&self) -> tree_sitter::Language {
        tree_sitter_cpp::LANGUAGE.into()
    }

    fn extract_symbols(&self, tree: &Tree, source: &[u8], file_path: &str) -> Vec<Symbol> {
        let mut symbols = Vec::new();
        let root = tree.root_node();
        extract_symbols_recursive(
            root,
            source,
            file_path,
            &[],
            Visibility::Public,
            false,
            &mut symbols,
        );
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

/// Recursively walk the AST and extract symbol definitions.
///
/// `scope` tracks the current namespace/class nesting for qualified names.
/// `current_visibility` tracks the current access specifier in class bodies.
/// `in_class` is true when we are inside a class or struct body.
fn extract_symbols_recursive(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    current_visibility: Visibility,
    in_class: bool,
    symbols: &mut Vec<Symbol>,
) {
    match node.kind() {
        "function_definition" => {
            if let Some(sym) =
                extract_function(node, source, file_path, scope, current_visibility, in_class)
            {
                symbols.push(sym);
            }
        }
        "class_specifier" => {
            extract_class_or_struct(node, source, file_path, scope, SymbolKind::Class, symbols);
            // Don't recurse further; extract_class_or_struct handles children.
            return;
        }
        "struct_specifier" => {
            extract_class_or_struct(node, source, file_path, scope, SymbolKind::Struct, symbols);
            return;
        }
        "enum_specifier" => {
            if let Some(sym) = extract_enum(node, source, file_path, scope) {
                symbols.push(sym);
            }
        }
        "namespace_definition" => {
            extract_namespace(node, source, file_path, scope, symbols);
            return;
        }
        "template_declaration" => {
            // A template wraps another declaration. Recurse into the child
            // declaration so that the inner class/function is extracted.
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i as u32) {
                    extract_symbols_recursive(
                        child,
                        source,
                        file_path,
                        scope,
                        current_visibility,
                        in_class,
                        symbols,
                    );
                }
            }
            return;
        }
        "type_alias_declaration" | "alias_declaration" => {
            if let Some(sym) = extract_type_alias(node, source, file_path, scope) {
                symbols.push(sym);
            }
        }
        "preproc_def" => {
            if let Some(sym) = extract_preproc_def(node, source, file_path, scope) {
                symbols.push(sym);
            }
        }
        _ => {}
    }

    // Default recursion for children.
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            extract_symbols_recursive(
                child,
                source,
                file_path,
                scope,
                current_visibility,
                in_class,
                symbols,
            );
        }
    }
}

fn extract_function(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    current_visibility: Visibility,
    in_class: bool,
) -> Option<Symbol> {
    let declarator = node.child_by_field_name("declarator")?;
    let name = extract_declarator_name(declarator, source)?;

    // Determine if this is a test function.
    let is_test = name.starts_with("test_") || name.starts_with("Test") || name.contains("TEST");

    let kind = if is_test {
        SymbolKind::Test
    } else if in_class {
        SymbolKind::Method
    } else {
        SymbolKind::Function
    };

    // Visibility: check for `static` storage class in C (file-scoped).
    let visibility = if has_static_specifier(node, source) {
        Visibility::Private
    } else {
        current_visibility
    };

    let qualified_name = build_qualified_name(scope, &name);
    let signature = extract_cpp_signature(node, source);
    let doc_comment = extract_cpp_doc_comment(node, source);

    let parent = if in_class {
        scope.last().cloned()
    } else {
        None
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
        parent,
    })
}

fn extract_class_or_struct(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    kind: SymbolKind,
    symbols: &mut Vec<Symbol>,
) {
    let name = if let Some(name_node) = node.child_by_field_name("name") {
        node_text(name_node, source)
    } else {
        // Anonymous struct/class - skip it.
        return;
    };

    let qualified_name = build_qualified_name(scope, &name);
    let signature = extract_class_signature(node, source, &name, kind);
    let doc_comment = extract_cpp_doc_comment(node, source);

    symbols.push(Symbol {
        name: name.clone(),
        qualified_name: qualified_name.clone(),
        kind,
        signature,
        visibility: Visibility::Public,
        file_path: file_path.to_string(),
        line_start: node.start_position().row,
        line_end: node.end_position().row,
        doc_comment,
        parent: scope.last().cloned(),
    });

    // Now recurse into the class body to find methods, nested types, etc.
    if let Some(body) = node.child_by_field_name("body") {
        let mut new_scope = scope.to_vec();
        new_scope.push(name);

        // Default visibility: private for class, public for struct.
        let mut vis = match kind {
            SymbolKind::Class => Visibility::Private,
            _ => Visibility::Public,
        };

        for i in 0..body.child_count() {
            if let Some(child) = body.child(i as u32) {
                if child.kind() == "access_specifier" {
                    vis = parse_access_specifier(child, source);
                } else {
                    extract_symbols_recursive(
                        child, source, file_path, &new_scope, vis, true, // in_class = true
                        symbols,
                    );
                }
            }
        }
    }
}

fn extract_enum(node: Node, source: &[u8], file_path: &str, scope: &[String]) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);
    let qualified_name = build_qualified_name(scope, &name);
    let doc_comment = extract_cpp_doc_comment(node, source);

    let signature = {
        let text = node_text(node, source);
        if let Some(pos) = text.find('{') {
            text[..pos].trim().to_string()
        } else {
            text.lines().next().unwrap_or(&text).trim().to_string()
        }
    };

    Some(Symbol {
        name,
        qualified_name,
        kind: SymbolKind::Enum,
        signature,
        visibility: Visibility::Public,
        file_path: file_path.to_string(),
        line_start: node.start_position().row,
        line_end: node.end_position().row,
        doc_comment,
        parent: scope.last().cloned(),
    })
}

fn extract_namespace(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    symbols: &mut Vec<Symbol>,
) {
    let name = if let Some(name_node) = node.child_by_field_name("name") {
        node_text(name_node, source)
    } else {
        // Anonymous namespace.
        String::new()
    };

    if !name.is_empty() {
        let qualified_name = build_qualified_name(scope, &name);
        let doc_comment = extract_cpp_doc_comment(node, source);

        symbols.push(Symbol {
            name: name.clone(),
            qualified_name,
            kind: SymbolKind::Module,
            signature: format!("namespace {}", name),
            visibility: Visibility::Public,
            file_path: file_path.to_string(),
            line_start: node.start_position().row,
            line_end: node.end_position().row,
            doc_comment,
            parent: scope.last().cloned(),
        });
    }

    // Recurse into namespace body.
    if let Some(body) = node.child_by_field_name("body") {
        let mut new_scope = scope.to_vec();
        if !name.is_empty() {
            new_scope.push(name);
        }
        for i in 0..body.child_count() {
            if let Some(child) = body.child(i as u32) {
                extract_symbols_recursive(
                    child,
                    source,
                    file_path,
                    &new_scope,
                    Visibility::Public,
                    false, // in_class = false (namespaces don't make functions into methods)
                    symbols,
                );
            }
        }
    }
}

fn extract_type_alias(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Symbol> {
    // `using MyType = int;` or `typedef int MyType;`
    // tree-sitter-cpp models `using X = Y;` as type_alias_declaration or alias_declaration.
    let text = node_text(node, source);
    let first_line = text.lines().next().unwrap_or(&text).trim().to_string();

    // Try to extract the name. For `using X = Y;`, tree-sitter may expose
    // a "name" field, or we can parse the text.
    let name = if let Some(name_node) = node.child_by_field_name("name") {
        node_text(name_node, source)
    } else if let Some(declarator) = node.child_by_field_name("declarator") {
        extract_declarator_name(declarator, source).unwrap_or_default()
    } else {
        // Fallback: parse from text.
        extract_name_from_using(&text)?
    };

    if name.is_empty() {
        return None;
    }

    let qualified_name = build_qualified_name(scope, &name);
    let doc_comment = extract_cpp_doc_comment(node, source);

    Some(Symbol {
        name,
        qualified_name,
        kind: SymbolKind::Type,
        signature: first_line,
        visibility: Visibility::Public,
        file_path: file_path.to_string(),
        line_start: node.start_position().row,
        line_end: node.end_position().row,
        doc_comment,
        parent: scope.last().cloned(),
    })
}

fn extract_preproc_def(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);
    let qualified_name = build_qualified_name(scope, &name);
    let text = node_text(node, source);
    let first_line = text.lines().next().unwrap_or(&text).trim().to_string();

    Some(Symbol {
        name,
        qualified_name,
        kind: SymbolKind::Constant,
        signature: first_line,
        visibility: Visibility::Public,
        file_path: file_path.to_string(),
        line_start: node.start_position().row,
        line_end: node.end_position().row,
        doc_comment: None,
        parent: None,
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
        "preproc_include" => {
            if let Some(r) = extract_include_reference(node, source, file_path, scope) {
                references.push(r);
            }
        }
        "call_expression" => {
            if let Some(r) = extract_call_reference(node, source, file_path, scope) {
                references.push(r);
            }
        }
        "base_class_clause" => {
            extract_inheritance_references(node, source, file_path, scope, references);
            return; // We handle children ourselves.
        }
        "function_definition" => {
            // Push function name onto scope for call references.
            let new_scope = if let Some(declarator) = node.child_by_field_name("declarator") {
                if let Some(name) = extract_declarator_name(declarator, source) {
                    let mut s = scope.to_vec();
                    s.push(name);
                    s
                } else {
                    scope.to_vec()
                }
            } else {
                scope.to_vec()
            };
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i as u32) {
                    extract_references_recursive(child, source, file_path, &new_scope, references);
                }
            }
            return;
        }
        "class_specifier" | "struct_specifier" => {
            // Push class/struct name onto scope.
            let new_scope = if let Some(name_node) = node.child_by_field_name("name") {
                let name = node_text(name_node, source);
                let mut s = scope.to_vec();
                s.push(name);
                s
            } else {
                scope.to_vec()
            };
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i as u32) {
                    extract_references_recursive(child, source, file_path, &new_scope, references);
                }
            }
            return;
        }
        "namespace_definition" => {
            let new_scope = if let Some(name_node) = node.child_by_field_name("name") {
                let name = node_text(name_node, source);
                let mut s = scope.to_vec();
                s.push(name);
                s
            } else {
                scope.to_vec()
            };
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i as u32) {
                    extract_references_recursive(child, source, file_path, &new_scope, references);
                }
            }
            return;
        }
        _ => {}
    }

    // Default recursion.
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            extract_references_recursive(child, source, file_path, scope, references);
        }
    }
}

fn extract_include_reference(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Reference> {
    let source_qn = if scope.is_empty() {
        file_path.to_string()
    } else {
        scope.join("::")
    };

    // The include path can be a system_lib_string like `<stdio.h>` or
    // a string_literal like `"myheader.h"`.
    let path_node = node.child_by_field_name("path")?;
    let path_text = node_text(path_node, source);
    let clean = path_text
        .trim_matches('"')
        .trim_start_matches('<')
        .trim_end_matches('>');

    Some(Reference {
        source_qualified_name: source_qn,
        target_name: clean.to_string(),
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
    let function_node = node.child_by_field_name("function")?;
    let function_name = node_text(function_node, source);
    let source_qn = if scope.is_empty() {
        file_path.to_string()
    } else {
        scope.join("::")
    };

    Some(Reference {
        source_qualified_name: source_qn,
        target_name: function_name,
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
    let source_qn = if scope.is_empty() {
        file_path.to_string()
    } else {
        scope.join("::")
    };

    // base_class_clause children include type_identifier nodes for base classes.
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            match child.kind() {
                "type_identifier" | "qualified_identifier" | "template_type" => {
                    let base_name = node_text(child, source);
                    references.push(Reference {
                        source_qualified_name: source_qn.clone(),
                        target_name: base_name,
                        kind: ReferenceKind::Inherits,
                        file_path: file_path.to_string(),
                        line: child.start_position().row,
                    });
                }
                // Recurse into access-qualified base specifiers, e.g. `public Base`.
                _ => {
                    for j in 0..child.child_count() {
                        if let Some(gc) = child.child(j as u32) {
                            if gc.kind() == "type_identifier"
                                || gc.kind() == "qualified_identifier"
                                || gc.kind() == "template_type"
                            {
                                let base_name = node_text(gc, source);
                                references.push(Reference {
                                    source_qualified_name: source_qn.clone(),
                                    target_name: base_name,
                                    kind: ReferenceKind::Inherits,
                                    file_path: file_path.to_string(),
                                    line: gc.start_position().row,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
}

// ── Helper Functions ──────────────────────────────────────────────────────

fn node_text(node: Node, source: &[u8]) -> String {
    node.utf8_text(source).unwrap_or("").to_string()
}

/// Build a `::` separated qualified name from scope + name.
fn build_qualified_name(scope: &[String], name: &str) -> String {
    if scope.is_empty() {
        name.to_string()
    } else {
        format!("{}::{}", scope.join("::"), name)
    }
}

/// Extract the function/method name from a declarator node.
///
/// Handles `function_declarator`, `pointer_declarator`, and
/// `qualified_identifier` (e.g., `ClassName::method`).
fn extract_declarator_name(node: Node, source: &[u8]) -> Option<String> {
    match node.kind() {
        "function_declarator" => {
            if let Some(decl) = node.child_by_field_name("declarator") {
                // Could be an identifier or a qualified_identifier.
                return extract_declarator_name(decl, source);
            }
            None
        }
        "pointer_declarator" | "reference_declarator" => {
            // Skip the `*` or `&` and get the inner declarator.
            if let Some(decl) = node.child_by_field_name("declarator") {
                return extract_declarator_name(decl, source);
            }
            None
        }
        "parenthesized_declarator" => {
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i as u32) {
                    if child.kind() != "(" && child.kind() != ")" {
                        return extract_declarator_name(child, source);
                    }
                }
            }
            None
        }
        "qualified_identifier" => {
            // e.g., `ClassName::method` - return just the rightmost name.
            if let Some(name_node) = node.child_by_field_name("name") {
                Some(node_text(name_node, source))
            } else {
                Some(node_text(node, source))
            }
        }
        "identifier" | "field_identifier" | "type_identifier" | "destructor_name"
        | "operator_name" => Some(node_text(node, source)),
        _ => {
            // Fallback: try the full text.
            let text = node_text(node, source).trim().to_string();
            if text.is_empty() {
                None
            } else {
                Some(text)
            }
        }
    }
}

/// Check if a function definition has `static` storage class specifier.
fn has_static_specifier(node: Node, source: &[u8]) -> bool {
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            if child.kind() == "storage_class_specifier" {
                let text = node_text(child, source);
                if text == "static" {
                    return true;
                }
            }
        }
    }
    false
}

/// Parse an access specifier node (`public:`, `private:`, `protected:`).
fn parse_access_specifier(node: Node, source: &[u8]) -> Visibility {
    let text = node_text(node, source);
    if text.contains("public") {
        Visibility::Public
    } else if text.contains("protected") {
        Visibility::Protected
    } else {
        Visibility::Private
    }
}

/// Extract the function signature (up to the opening brace).
fn extract_cpp_signature(node: Node, source: &[u8]) -> String {
    let text = node_text(node, source);
    if let Some(pos) = text.find('{') {
        text[..pos].trim().to_string()
    } else {
        text.lines().next().unwrap_or(&text).trim().to_string()
    }
}

/// Extract class/struct signature including base class clause.
fn extract_class_signature(node: Node, source: &[u8], name: &str, kind: SymbolKind) -> String {
    let keyword = match kind {
        SymbolKind::Class => "class",
        _ => "struct",
    };

    // Try to build from the source text up to the body.
    let text = node_text(node, source);
    if let Some(pos) = text.find('{') {
        text[..pos].trim().to_string()
    } else {
        format!("{} {}", keyword, name)
    }
}

/// Extract doc comments preceding a node (`//` line comments or `/* */` blocks).
fn extract_cpp_doc_comment(node: Node, source: &[u8]) -> Option<String> {
    let mut comment_lines = Vec::new();
    let mut prev = node.prev_sibling();

    while let Some(sibling) = prev {
        if sibling.kind() == "comment" {
            let text = node_text(sibling, source);
            if text.starts_with("//") {
                let stripped = text
                    .strip_prefix("/// ")
                    .or_else(|| text.strip_prefix("///"))
                    .or_else(|| text.strip_prefix("// "))
                    .or_else(|| text.strip_prefix("//"))
                    .unwrap_or(&text);
                comment_lines.push(stripped.trim_end().to_string());
                prev = sibling.prev_sibling();
                continue;
            } else if text.starts_with("/*") {
                // Block comment: strip `/*` and `*/` and leading `*` on each line.
                let inner = text
                    .strip_prefix("/*")
                    .and_then(|s| s.strip_suffix("*/"))
                    .unwrap_or(&text);
                for line in inner.lines() {
                    let trimmed = line.trim().trim_start_matches('*').trim_start();
                    if !trimmed.is_empty() {
                        comment_lines.push(trimmed.to_string());
                    }
                }
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

/// Parse a name from a `using X = Y;` declaration text.
fn extract_name_from_using(text: &str) -> Option<String> {
    // `using MyType = int;`
    let text = text.trim();
    if let Some(rest) = text.strip_prefix("using ") {
        if let Some(eq_pos) = rest.find('=') {
            let name = rest[..eq_pos].trim();
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
    }
    None
}


#[cfg(test)]
#[path = "tests/cpp_tests.rs"]
mod tests;
