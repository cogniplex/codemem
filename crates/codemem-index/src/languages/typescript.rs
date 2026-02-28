//! TypeScript/TSX language extractor using tree-sitter-typescript.

use crate::extractor::LanguageExtractor;
use crate::symbol::{Reference, ReferenceKind, Symbol, SymbolKind, Visibility};
use tree_sitter::{Node, Tree};

/// TypeScript/TSX language extractor for tree-sitter-based code indexing.
///
/// Uses the TSX grammar (a superset of TypeScript) so that both `.ts` and `.tsx`
/// files are handled correctly by a single extractor.
pub struct TypeScriptExtractor;

impl TypeScriptExtractor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TypeScriptExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageExtractor for TypeScriptExtractor {
    fn language_name(&self) -> &str {
        "typescript"
    }

    fn file_extensions(&self) -> &[&str] {
        &["ts", "tsx"]
    }

    fn tree_sitter_language(&self) -> tree_sitter::Language {
        // TSX grammar is a superset of TypeScript, handles both .ts and .tsx
        tree_sitter_typescript::LANGUAGE_TSX.into()
    }

    fn extract_symbols(&self, tree: &Tree, source: &[u8], file_path: &str) -> Vec<Symbol> {
        let mut symbols = Vec::new();
        let root = tree.root_node();
        extract_symbols_recursive(root, source, file_path, &[], false, &mut symbols);
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
/// `in_class` tracks whether we are inside a class body, which determines
/// whether functions are classified as Function or Method.
fn extract_symbols_recursive(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    in_class: bool,
    symbols: &mut Vec<Symbol>,
) {
    match node.kind() {
        "function_declaration" => {
            if let Some(sym) = extract_function_declaration(node, source, file_path, scope, false) {
                symbols.push(sym);
            }
        }
        "class_declaration" => {
            let exported = is_exported(node);
            if let Some(sym) = extract_class_declaration(node, source, file_path, scope, exported) {
                let name = sym.name.clone();
                symbols.push(sym);
                // Recurse into class body for methods
                let mut new_scope = scope.to_vec();
                new_scope.push(name);
                if let Some(body) = node.child_by_field_name("body") {
                    for i in 0..body.child_count() {
                        if let Some(child) = body.child(i as u32) {
                            extract_symbols_recursive(
                                child, source, file_path, &new_scope, true, symbols,
                            );
                        }
                    }
                }
                return; // already recursed
            }
        }
        "interface_declaration" => {
            if let Some(sym) = extract_interface_declaration(node, source, file_path, scope) {
                symbols.push(sym);
            }
        }
        "type_alias_declaration" => {
            if let Some(sym) = extract_type_alias(node, source, file_path, scope) {
                symbols.push(sym);
            }
        }
        "enum_declaration" => {
            if let Some(sym) = extract_enum_declaration(node, source, file_path, scope) {
                symbols.push(sym);
            }
        }
        "method_definition" => {
            if let Some(sym) = extract_method_definition(node, source, file_path, scope) {
                symbols.push(sym);
            }
        }
        "public_field_definition" => {
            // Class methods defined as arrow functions in class fields
            // e.g., `myMethod = () => { ... }`
            // Only extract if the value is an arrow function
            if has_arrow_function_value(node) {
                if let Some(sym) = extract_class_field_arrow(node, source, file_path, scope) {
                    symbols.push(sym);
                }
            }
        }
        "export_statement" => {
            // Exported declarations: `export function ...`, `export class ...`, etc.
            // The actual declaration is a child; recurse into it with export context.
            let has_declaration =
                extract_exported_declaration(node, source, file_path, scope, symbols);
            if has_declaration {
                return; // already handled children
            }
            // If it's `export default ...` or `export { ... }`, fall through
        }
        "lexical_declaration" => {
            // `const foo = () => { ... }` or `const foo = function() { ... }`
            // Only at module scope (not inside a class)
            if !in_class {
                extract_lexical_arrow_functions(node, source, file_path, scope, symbols);
            }
        }
        "module" | "internal_module" => {
            // TypeScript namespace: `namespace Foo { ... }` or `module Foo { ... }`
            if let Some(sym) = extract_namespace(node, source, file_path, scope) {
                let name = sym.name.clone();
                symbols.push(sym);
                let mut new_scope = scope.to_vec();
                new_scope.push(name);
                if let Some(body) = node.child_by_field_name("body") {
                    for i in 0..body.child_count() {
                        if let Some(child) = body.child(i as u32) {
                            extract_symbols_recursive(
                                child, source, file_path, &new_scope, false, symbols,
                            );
                        }
                    }
                }
                return; // already recursed
            }
        }
        _ => {}
    }

    // Default recursion for nodes we didn't handle specially
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            extract_symbols_recursive(child, source, file_path, scope, in_class, symbols);
        }
    }
}

/// Extract a function_declaration as a Symbol.
fn extract_function_declaration(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    exported: bool,
) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let visibility = if exported || is_exported(node) {
        Visibility::Public
    } else {
        Visibility::Private
    };

    let signature = extract_signature(node, source);
    let doc_comment = extract_jsdoc_comment(node, source);
    let qualified_name = build_qualified_name(scope, &name);

    Some(Symbol {
        name,
        qualified_name,
        kind: SymbolKind::Function,
        signature,
        visibility,
        file_path: file_path.to_string(),
        line_start: node.start_position().row,
        line_end: node.end_position().row,
        doc_comment,
        parent: if scope.is_empty() {
            None
        } else {
            Some(scope.join("."))
        },
    })
}

/// Extract a class_declaration as a Symbol.
fn extract_class_declaration(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    exported: bool,
) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let visibility = if exported || is_exported(node) {
        Visibility::Public
    } else {
        Visibility::Private
    };

    let signature = extract_signature(node, source);
    let doc_comment = extract_jsdoc_comment(node, source);
    let qualified_name = build_qualified_name(scope, &name);

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
        parent: if scope.is_empty() {
            None
        } else {
            Some(scope.join("."))
        },
    })
}

/// Extract an interface_declaration as a Symbol.
fn extract_interface_declaration(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let visibility = if is_exported(node) {
        Visibility::Public
    } else {
        Visibility::Private
    };

    let signature = extract_signature(node, source);
    let doc_comment = extract_jsdoc_comment(node, source);
    let qualified_name = build_qualified_name(scope, &name);

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
        parent: if scope.is_empty() {
            None
        } else {
            Some(scope.join("."))
        },
    })
}

/// Extract a type_alias_declaration as a Symbol.
fn extract_type_alias(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let visibility = if is_exported(node) {
        Visibility::Public
    } else {
        Visibility::Private
    };

    let signature = extract_signature(node, source);
    let doc_comment = extract_jsdoc_comment(node, source);
    let qualified_name = build_qualified_name(scope, &name);

    Some(Symbol {
        name,
        qualified_name,
        kind: SymbolKind::Type,
        signature,
        visibility,
        file_path: file_path.to_string(),
        line_start: node.start_position().row,
        line_end: node.end_position().row,
        doc_comment,
        parent: if scope.is_empty() {
            None
        } else {
            Some(scope.join("."))
        },
    })
}

/// Extract an enum_declaration as a Symbol.
fn extract_enum_declaration(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let visibility = if is_exported(node) {
        Visibility::Public
    } else {
        Visibility::Private
    };

    let signature = extract_signature(node, source);
    let doc_comment = extract_jsdoc_comment(node, source);
    let qualified_name = build_qualified_name(scope, &name);

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
        parent: if scope.is_empty() {
            None
        } else {
            Some(scope.join("."))
        },
    })
}

/// Extract a method_definition (inside a class body) as a Symbol.
fn extract_method_definition(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    // Check for access modifiers: private, protected, public
    let visibility = extract_ts_member_visibility(node, source);

    let signature = extract_signature(node, source);
    let doc_comment = extract_jsdoc_comment(node, source);
    let qualified_name = build_qualified_name(scope, &name);

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
        parent: if scope.is_empty() {
            None
        } else {
            Some(scope.join("."))
        },
    })
}

/// Extract a class field that is an arrow function as a Method symbol.
/// e.g., `myMethod = () => { ... }`
fn extract_class_field_arrow(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let visibility = extract_ts_member_visibility(node, source);
    let signature = extract_signature(node, source);
    let doc_comment = extract_jsdoc_comment(node, source);
    let qualified_name = build_qualified_name(scope, &name);

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
        parent: if scope.is_empty() {
            None
        } else {
            Some(scope.join("."))
        },
    })
}

/// Extract a namespace/module declaration as a Module symbol.
fn extract_namespace(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
) -> Option<Symbol> {
    let name_node = node.child_by_field_name("name")?;
    let name = node_text(name_node, source);

    let visibility = if is_exported(node) {
        Visibility::Public
    } else {
        Visibility::Private
    };

    let signature = extract_signature(node, source);
    let doc_comment = extract_jsdoc_comment(node, source);
    let qualified_name = build_qualified_name(scope, &name);

    Some(Symbol {
        name,
        qualified_name,
        kind: SymbolKind::Module,
        signature,
        visibility,
        file_path: file_path.to_string(),
        line_start: node.start_position().row,
        line_end: node.end_position().row,
        doc_comment,
        parent: if scope.is_empty() {
            None
        } else {
            Some(scope.join("."))
        },
    })
}

/// Handle `export_statement` by extracting the inner declaration with exported visibility.
/// Returns true if a declaration child was found and handled.
fn extract_exported_declaration(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    symbols: &mut Vec<Symbol>,
) -> bool {
    // Look for the declaration child inside the export_statement
    let mut found = false;
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            match child.kind() {
                "function_declaration" => {
                    if let Some(sym) =
                        extract_function_declaration(child, source, file_path, scope, true)
                    {
                        symbols.push(sym);
                    }
                    found = true;
                }
                "class_declaration" => {
                    if let Some(sym) =
                        extract_class_declaration(child, source, file_path, scope, true)
                    {
                        let name = sym.name.clone();
                        symbols.push(sym);
                        // Recurse into class body
                        let mut new_scope = scope.to_vec();
                        new_scope.push(name);
                        if let Some(body) = child.child_by_field_name("body") {
                            for j in 0..body.child_count() {
                                if let Some(body_child) = body.child(j as u32) {
                                    extract_symbols_recursive(
                                        body_child, source, file_path, &new_scope, true, symbols,
                                    );
                                }
                            }
                        }
                    }
                    found = true;
                }
                "interface_declaration" => {
                    if let Some(mut sym) =
                        extract_interface_declaration(child, source, file_path, scope)
                    {
                        sym.visibility = Visibility::Public;
                        symbols.push(sym);
                    }
                    found = true;
                }
                "type_alias_declaration" => {
                    if let Some(mut sym) = extract_type_alias(child, source, file_path, scope) {
                        sym.visibility = Visibility::Public;
                        symbols.push(sym);
                    }
                    found = true;
                }
                "enum_declaration" => {
                    if let Some(mut sym) = extract_enum_declaration(child, source, file_path, scope)
                    {
                        sym.visibility = Visibility::Public;
                        symbols.push(sym);
                    }
                    found = true;
                }
                "lexical_declaration" => {
                    // `export const foo = () => { ... }`
                    extract_lexical_arrow_functions_exported(
                        child, source, file_path, scope, symbols,
                    );
                    found = true;
                }
                "module" | "internal_module" => {
                    if let Some(mut sym) = extract_namespace(child, source, file_path, scope) {
                        sym.visibility = Visibility::Public;
                        let name = sym.name.clone();
                        symbols.push(sym);
                        let mut new_scope = scope.to_vec();
                        new_scope.push(name);
                        if let Some(body) = child.child_by_field_name("body") {
                            for j in 0..body.child_count() {
                                if let Some(body_child) = body.child(j as u32) {
                                    extract_symbols_recursive(
                                        body_child, source, file_path, &new_scope, false, symbols,
                                    );
                                }
                            }
                        }
                    }
                    found = true;
                }
                _ => {}
            }
        }
    }
    found
}

/// Extract arrow functions assigned to const/let at module scope.
/// e.g., `const foo = (x: number): number => x * 2`
fn extract_lexical_arrow_functions(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    symbols: &mut Vec<Symbol>,
) {
    extract_lexical_arrow_functions_inner(node, source, file_path, scope, false, symbols);
}

/// Extract arrow functions assigned to exported const/let at module scope.
fn extract_lexical_arrow_functions_exported(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    symbols: &mut Vec<Symbol>,
) {
    extract_lexical_arrow_functions_inner(node, source, file_path, scope, true, symbols);
}

fn extract_lexical_arrow_functions_inner(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    exported: bool,
    symbols: &mut Vec<Symbol>,
) {
    // lexical_declaration has variable_declarator children
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            if child.kind() == "variable_declarator" {
                let value = child.child_by_field_name("value");
                let is_fn = value.is_some_and(|v| {
                    v.kind() == "arrow_function" || v.kind() == "function_expression"
                });

                if is_fn {
                    if let Some(name_node) = child.child_by_field_name("name") {
                        let name = node_text(name_node, source);
                        let visibility = if exported {
                            Visibility::Public
                        } else {
                            Visibility::Private
                        };
                        let signature = extract_signature(node, source);
                        let doc_comment = extract_jsdoc_comment(node, source);
                        let qualified_name = build_qualified_name(scope, &name);

                        symbols.push(Symbol {
                            name,
                            qualified_name,
                            kind: SymbolKind::Function,
                            signature,
                            visibility,
                            file_path: file_path.to_string(),
                            line_start: node.start_position().row,
                            line_end: node.end_position().row,
                            doc_comment,
                            parent: if scope.is_empty() {
                                None
                            } else {
                                Some(scope.join("."))
                            },
                        });
                    }
                } else if value.is_none() || !is_fn {
                    // Check if it's a simple const (not a function) — extract as Constant
                    // Only for non-function values
                    if let Some(name_node) = child.child_by_field_name("name") {
                        // Only extract named identifiers (not destructuring patterns)
                        if name_node.kind() == "identifier" {
                            let text = node_text(node, source);
                            let is_const = text.starts_with("const ");
                            if is_const && value.is_some() {
                                // It's a constant, not a function — skip for now
                                // (we focus on function-like symbols)
                            }
                        }
                    }
                }
            }
        }
    }
}

// ── Reference Extraction ──────────────────────────────────────────────────

/// Recursively walk the AST and extract references.
fn extract_references_recursive(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    references: &mut Vec<Reference>,
) {
    match node.kind() {
        "import_statement" => {
            extract_import_references(node, source, file_path, scope, references);
        }
        "call_expression" => {
            if let Some(r) = extract_call_reference(node, source, file_path, scope) {
                references.push(r);
            }
        }
        "class_declaration" => {
            // Extract heritage (extends, implements)
            extract_class_heritage_references(node, source, file_path, scope, references);
            // Update scope for references inside class
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = node_text(name_node, source);
                let mut new_scope = scope.to_vec();
                new_scope.push(name);
                if let Some(body) = node.child_by_field_name("body") {
                    for i in 0..body.child_count() {
                        if let Some(child) = body.child(i as u32) {
                            extract_references_recursive(
                                child, source, file_path, &new_scope, references,
                            );
                        }
                    }
                }
                return; // already recursed
            }
        }
        "function_declaration" => {
            // Update scope for references inside functions
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = node_text(name_node, source);
                let mut new_scope = scope.to_vec();
                new_scope.push(name);
                if let Some(body) = node.child_by_field_name("body") {
                    for i in 0..body.child_count() {
                        if let Some(child) = body.child(i as u32) {
                            extract_references_recursive(
                                child, source, file_path, &new_scope, references,
                            );
                        }
                    }
                }
                return; // already recursed
            }
        }
        "method_definition" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = node_text(name_node, source);
                let mut new_scope = scope.to_vec();
                new_scope.push(name);
                if let Some(body) = node.child_by_field_name("body") {
                    for i in 0..body.child_count() {
                        if let Some(child) = body.child(i as u32) {
                            extract_references_recursive(
                                child, source, file_path, &new_scope, references,
                            );
                        }
                    }
                }
                return; // already recursed
            }
        }
        "export_statement" => {
            // Recurse into exported declarations for heritage/references
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i as u32) {
                    extract_references_recursive(child, source, file_path, scope, references);
                }
            }
            return; // already recursed
        }
        "module" | "internal_module" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = node_text(name_node, source);
                let mut new_scope = scope.to_vec();
                new_scope.push(name);
                if let Some(body) = node.child_by_field_name("body") {
                    for i in 0..body.child_count() {
                        if let Some(child) = body.child(i as u32) {
                            extract_references_recursive(
                                child, source, file_path, &new_scope, references,
                            );
                        }
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

/// Extract Import references from an `import_statement` node.
fn extract_import_references(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    references: &mut Vec<Reference>,
) {
    // Extract the source/module path from the import statement
    let source_module = node
        .child_by_field_name("source")
        .map(|n| node_text(n, source))
        .map(|s| s.trim_matches('\'').trim_matches('"').to_string());

    let source_qn = if scope.is_empty() {
        file_path.to_string()
    } else {
        scope.join(".")
    };

    if let Some(module_path) = source_module {
        references.push(Reference {
            source_qualified_name: source_qn,
            target_name: module_path,
            kind: ReferenceKind::Import,
            file_path: file_path.to_string(),
            line: node.start_position().row,
        });
    }
}

/// Extract a Call reference from a `call_expression` node.
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
        scope.join(".")
    };

    Some(Reference {
        source_qualified_name: source_qn,
        target_name: function_name,
        kind: ReferenceKind::Call,
        file_path: file_path.to_string(),
        line: node.start_position().row,
    })
}

/// Extract Inherits/Implements references from class heritage clauses.
fn extract_class_heritage_references(
    node: Node,
    source: &[u8],
    file_path: &str,
    scope: &[String],
    references: &mut Vec<Reference>,
) {
    let class_name = node
        .child_by_field_name("name")
        .map(|n| node_text(n, source))
        .unwrap_or_default();

    let source_qn = if scope.is_empty() {
        class_name.clone()
    } else {
        format!("{}.{}", scope.join("."), class_name)
    };

    // Iterate over children looking for class_heritage nodes
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            if child.kind() == "class_heritage" {
                // class_heritage contains extends_clause and/or implements_clause
                for j in 0..child.child_count() {
                    if let Some(heritage_child) = child.child(j as u32) {
                        match heritage_child.kind() {
                            "extends_clause" => {
                                // `extends BaseClass`
                                extract_heritage_type_refs(
                                    heritage_child,
                                    source,
                                    file_path,
                                    &source_qn,
                                    ReferenceKind::Inherits,
                                    references,
                                );
                            }
                            "implements_clause" => {
                                // `implements Interface1, Interface2`
                                extract_heritage_type_refs(
                                    heritage_child,
                                    source,
                                    file_path,
                                    &source_qn,
                                    ReferenceKind::Implements,
                                    references,
                                );
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }
}

/// Extract type references from extends/implements clauses.
fn extract_heritage_type_refs(
    clause_node: Node,
    source: &[u8],
    file_path: &str,
    source_qn: &str,
    kind: ReferenceKind,
    references: &mut Vec<Reference>,
) {
    for i in 0..clause_node.child_count() {
        if let Some(child) = clause_node.child(i as u32) {
            // The type names appear as identifier or member_expression children
            // Skip keywords like "extends", "implements", and commas
            match child.kind() {
                "identifier" | "type_identifier" | "member_expression" | "generic_type" => {
                    let type_name = if child.kind() == "generic_type" {
                        // For generic types like `Base<T>`, extract just the name
                        child
                            .child_by_field_name("name")
                            .map(|n| node_text(n, source))
                            .unwrap_or_else(|| node_text(child, source))
                    } else {
                        node_text(child, source)
                    };

                    if !type_name.is_empty() {
                        references.push(Reference {
                            source_qualified_name: source_qn.to_string(),
                            target_name: type_name,
                            kind,
                            file_path: file_path.to_string(),
                            line: child.start_position().row,
                        });
                    }
                }
                _ => {}
            }
        }
    }
}

// ── Helper Functions ──────────────────────────────────────────────────────

/// Get the text content of a tree-sitter node.
fn node_text(node: Node, source: &[u8]) -> String {
    node.utf8_text(source).unwrap_or("").to_string()
}

/// Build a qualified name from scope and name, using `.` as separator (TypeScript convention).
fn build_qualified_name(scope: &[String], name: &str) -> String {
    if scope.is_empty() {
        name.to_string()
    } else {
        format!("{}.{}", scope.join("."), name)
    }
}

/// Check if a node is exported (has an ancestor `export_statement` or starts with `export`).
fn is_exported(node: Node) -> bool {
    // Check if parent is an export_statement
    if let Some(parent) = node.parent() {
        if parent.kind() == "export_statement" {
            return true;
        }
    }
    false
}

/// Extract TypeScript member visibility (public/private/protected keywords).
fn extract_ts_member_visibility(node: Node, source: &[u8]) -> Visibility {
    // Check for accessibility_modifier child
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i as u32) {
            if child.kind() == "accessibility_modifier" {
                let text = node_text(child, source);
                return match text.as_str() {
                    "private" => Visibility::Private,
                    "protected" => Visibility::Protected,
                    "public" => Visibility::Public,
                    _ => Visibility::Public,
                };
            }
        }
    }
    // Default for class members is Public in TypeScript
    Visibility::Public
}

/// Extract the signature of a node (text up to the first `{`, or the whole node for short items).
fn extract_signature(node: Node, source: &[u8]) -> String {
    let text = node_text(node, source);
    // Find the first `{` and take everything before it
    if let Some(pos) = text.find('{') {
        text[..pos].trim().to_string()
    } else {
        // For items without braces (e.g., type alias), take the whole text
        // but limit to the first line if it's multi-line
        let first_line = text.lines().next().unwrap_or(&text);
        first_line.trim_end_matches(';').trim().to_string()
    }
}

/// Extract JSDoc comments (/** ... */) preceding a node.
fn extract_jsdoc_comment(node: Node, source: &[u8]) -> Option<String> {
    let mut comment_nodes = Vec::new();

    // Walk preceding siblings to find JSDoc comments.
    // If the node is inside an export_statement, the JSDoc comment is a sibling
    // of the export_statement, not of the declaration itself.
    let effective_node = if let Some(parent) = node.parent() {
        if parent.kind() == "export_statement" {
            parent
        } else {
            node
        }
    } else {
        node
    };
    let mut prev = effective_node.prev_sibling();

    while let Some(sibling) = prev {
        match sibling.kind() {
            "comment" => {
                let text = node_text(sibling, source);
                if text.starts_with("/**") {
                    comment_nodes.push(text);
                    prev = sibling.prev_sibling();
                    continue;
                }
                // Regular // comment: not JSDoc, stop
                break;
            }
            "decorator" => {
                // Skip decorators between JSDoc and the item
                prev = sibling.prev_sibling();
                continue;
            }
            _ => break,
        }
    }

    // Reverse since we collected bottom-up
    comment_nodes.reverse();

    if comment_nodes.is_empty() {
        return None;
    }

    // Parse JSDoc content: strip `/**`, `*/`, and leading `*` from lines
    let mut doc_lines = Vec::new();
    for comment_text in &comment_nodes {
        let trimmed = comment_text
            .trim_start_matches("/**")
            .trim_end_matches("*/")
            .trim();

        for line in trimmed.lines() {
            let line = line.trim();
            let line = if let Some(stripped) = line.strip_prefix("* ") {
                stripped
            } else if let Some(stripped) = line.strip_prefix('*') {
                stripped
            } else {
                line
            };
            let line = line.trim_end();
            if !line.is_empty() {
                doc_lines.push(line.to_string());
            }
        }
    }

    if doc_lines.is_empty() {
        None
    } else {
        Some(doc_lines.join("\n").trim_end().to_string())
    }
}

/// Check if a public_field_definition has an arrow_function value.
fn has_arrow_function_value(node: Node) -> bool {
    node.child_by_field_name("value")
        .is_some_and(|v| v.kind() == "arrow_function" || v.kind() == "function_expression")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extractor::LanguageExtractor;
    use tree_sitter::Parser;

    fn parse_ts(source: &str) -> Tree {
        let mut parser = Parser::new();
        let lang = tree_sitter_typescript::LANGUAGE_TSX;
        parser
            .set_language(&lang.into())
            .expect("failed to set TypeScript/TSX language");
        parser
            .parse(source.as_bytes(), None)
            .expect("failed to parse")
    }

    #[test]
    fn extract_ts_function() {
        let source = r#"
/** Adds two numbers. */
export function add(a: number, b: number): number {
    return a + b;
}
"#;
        let tree = parse_ts(source);
        let extractor = TypeScriptExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.ts");

        assert_eq!(symbols.len(), 1, "Expected 1 symbol, got: {:#?}", symbols);
        let sym = &symbols[0];
        assert_eq!(sym.name, "add");
        assert_eq!(sym.qualified_name, "add");
        assert_eq!(sym.kind, SymbolKind::Function);
        assert_eq!(sym.visibility, Visibility::Public);
        assert!(
            sym.signature.contains("function add"),
            "Signature should contain 'function add', got: {}",
            sym.signature
        );
        assert_eq!(sym.doc_comment.as_deref(), Some("Adds two numbers."));
        assert!(sym.parent.is_none());
    }

    #[test]
    fn extract_ts_class_and_methods() {
        let source = r#"
export class Greeter {
    private name: string;

    constructor(name: string) {
        this.name = name;
    }

    public greet(): string {
        return `Hello, ${this.name}`;
    }

    private internal(): void {}
}
"#;
        let tree = parse_ts(source);
        let extractor = TypeScriptExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.ts");

        // Should have: Class(Greeter), Method(constructor), Method(greet), Method(internal)
        let greeter = symbols.iter().find(|s| s.name == "Greeter");
        assert!(
            greeter.is_some(),
            "Expected Greeter class, got: {:#?}",
            symbols
        );
        let greeter = greeter.unwrap();
        assert_eq!(greeter.kind, SymbolKind::Class);
        assert_eq!(greeter.visibility, Visibility::Public);

        let greet = symbols.iter().find(|s| s.name == "greet");
        assert!(
            greet.is_some(),
            "Expected greet method, got: {:#?}",
            symbols
        );
        let greet = greet.unwrap();
        assert_eq!(greet.kind, SymbolKind::Method);
        assert_eq!(greet.visibility, Visibility::Public);
        assert_eq!(greet.qualified_name, "Greeter.greet");
        assert_eq!(greet.parent.as_deref(), Some("Greeter"));

        let internal = symbols.iter().find(|s| s.name == "internal");
        assert!(
            internal.is_some(),
            "Expected internal method, got: {:#?}",
            symbols
        );
        assert_eq!(internal.unwrap().visibility, Visibility::Private);
    }

    #[test]
    fn extract_ts_interface() {
        let source = r#"
/** Represents a shape. */
export interface Shape {
    area(): number;
    perimeter(): number;
}
"#;
        let tree = parse_ts(source);
        let extractor = TypeScriptExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.ts");

        let shape = symbols.iter().find(|s| s.name == "Shape");
        assert!(
            shape.is_some(),
            "Expected Shape interface, got: {:#?}",
            symbols
        );
        let shape = shape.unwrap();
        assert_eq!(shape.kind, SymbolKind::Interface);
        assert_eq!(shape.visibility, Visibility::Public);
        assert_eq!(shape.doc_comment.as_deref(), Some("Represents a shape."));
    }

    #[test]
    fn extract_ts_type_alias_and_enum() {
        let source = r#"
export type Result<T> = { ok: true; value: T } | { ok: false; error: string };

export enum Color {
    Red,
    Green,
    Blue,
}
"#;
        let tree = parse_ts(source);
        let extractor = TypeScriptExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.ts");

        assert!(
            symbols
                .iter()
                .any(|s| s.name == "Result" && s.kind == SymbolKind::Type),
            "Expected Result type alias, got: {:#?}",
            symbols
        );
        assert!(
            symbols
                .iter()
                .any(|s| s.name == "Color" && s.kind == SymbolKind::Enum),
            "Expected Color enum, got: {:#?}",
            symbols
        );
    }

    #[test]
    fn extract_ts_imports() {
        let source = r#"
import { useState, useEffect } from 'react';
import * as path from 'path';
import express from 'express';
"#;
        let tree = parse_ts(source);
        let extractor = TypeScriptExtractor::new();
        let references = extractor.extract_references(&tree, source.as_bytes(), "test.ts");

        let imports: Vec<_> = references
            .iter()
            .filter(|r| r.kind == ReferenceKind::Import)
            .collect();
        assert_eq!(
            imports.len(),
            3,
            "Expected 3 import refs, got: {:#?}",
            imports
        );
        assert!(imports.iter().any(|r| r.target_name == "react"));
        assert!(imports.iter().any(|r| r.target_name == "path"));
        assert!(imports.iter().any(|r| r.target_name == "express"));
    }

    #[test]
    fn extract_ts_class_heritage() {
        let source = r#"
class Animal {
    name: string;
}

interface Swimmer {
    swim(): void;
}

class Dog extends Animal implements Swimmer {
    swim(): void {}
}
"#;
        let tree = parse_ts(source);
        let extractor = TypeScriptExtractor::new();
        let references = extractor.extract_references(&tree, source.as_bytes(), "test.ts");

        let inherits: Vec<_> = references
            .iter()
            .filter(|r| r.kind == ReferenceKind::Inherits)
            .collect();
        assert!(
            inherits
                .iter()
                .any(|r| r.target_name == "Animal" && r.source_qualified_name == "Dog"),
            "Expected Dog extends Animal, got: {:#?}",
            inherits
        );

        let implements: Vec<_> = references
            .iter()
            .filter(|r| r.kind == ReferenceKind::Implements)
            .collect();
        assert!(
            implements
                .iter()
                .any(|r| r.target_name == "Swimmer" && r.source_qualified_name == "Dog"),
            "Expected Dog implements Swimmer, got: {:#?}",
            implements
        );
    }

    #[test]
    fn extract_ts_call_references() {
        let source = r#"
function caller() {
    const x = foo();
    bar(x);
    console.log("hello");
}
"#;
        let tree = parse_ts(source);
        let extractor = TypeScriptExtractor::new();
        let references = extractor.extract_references(&tree, source.as_bytes(), "test.ts");

        let calls: Vec<_> = references
            .iter()
            .filter(|r| r.kind == ReferenceKind::Call)
            .collect();
        assert!(
            calls.iter().any(|r| r.target_name == "foo"),
            "Expected call to foo, got: {:#?}",
            calls
        );
        assert!(
            calls.iter().any(|r| r.target_name == "bar"),
            "Expected call to bar, got: {:#?}",
            calls
        );
        assert!(
            calls.iter().any(|r| r.target_name == "console.log"),
            "Expected call to console.log, got: {:#?}",
            calls
        );
    }

    #[test]
    fn extract_ts_arrow_function() {
        let source = r#"
export const multiply = (a: number, b: number): number => a * b;
"#;
        let tree = parse_ts(source);
        let extractor = TypeScriptExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.ts");

        let multiply = symbols.iter().find(|s| s.name == "multiply");
        assert!(
            multiply.is_some(),
            "Expected multiply arrow function, got: {:#?}",
            symbols
        );
        let multiply = multiply.unwrap();
        assert_eq!(multiply.kind, SymbolKind::Function);
        assert_eq!(multiply.visibility, Visibility::Public);
    }

    #[test]
    fn extract_ts_non_exported_function() {
        let source = r#"
function privateHelper(): void {
    // ...
}
"#;
        let tree = parse_ts(source);
        let extractor = TypeScriptExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.ts");

        assert_eq!(symbols.len(), 1, "Expected 1 symbol, got: {:#?}", symbols);
        assert_eq!(symbols[0].name, "privateHelper");
        assert_eq!(symbols[0].visibility, Visibility::Private);
    }

    #[test]
    fn extract_ts_jsdoc_multiline() {
        let source = r#"
/**
 * Processes the given data.
 * @param data - The input data
 * @returns The processed result
 */
export function process(data: string): string {
    return data;
}
"#;
        let tree = parse_ts(source);
        let extractor = TypeScriptExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.ts");

        assert_eq!(symbols.len(), 1);
        let doc = symbols[0].doc_comment.as_deref().unwrap();
        assert!(
            doc.contains("Processes the given data."),
            "Expected JSDoc content, got: {}",
            doc
        );
        assert!(doc.contains("@param data"));
        assert!(doc.contains("@returns"));
    }

    #[test]
    fn extract_ts_namespace() {
        let source = r#"
export namespace Validation {
    export function validate(s: string): boolean {
        return s.length > 0;
    }
}
"#;
        let tree = parse_ts(source);
        let extractor = TypeScriptExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.ts");

        let ns = symbols
            .iter()
            .find(|s| s.name == "Validation" && s.kind == SymbolKind::Module);
        assert!(
            ns.is_some(),
            "Expected Validation namespace, got: {:#?}",
            symbols
        );
        assert_eq!(ns.unwrap().visibility, Visibility::Public);

        let validate = symbols.iter().find(|s| s.name == "validate");
        assert!(
            validate.is_some(),
            "Expected validate function, got: {:#?}",
            symbols
        );
        let validate = validate.unwrap();
        assert_eq!(validate.qualified_name, "Validation.validate");
        assert_eq!(validate.kind, SymbolKind::Function);
        assert_eq!(validate.parent.as_deref(), Some("Validation"));
    }

    #[test]
    fn extract_ts_protected_method() {
        let source = r#"
class Base {
    protected helper(): void {}
}
"#;
        let tree = parse_ts(source);
        let extractor = TypeScriptExtractor::new();
        let symbols = extractor.extract_symbols(&tree, source.as_bytes(), "test.ts");

        let helper = symbols.iter().find(|s| s.name == "helper");
        assert!(
            helper.is_some(),
            "Expected helper method, got: {:#?}",
            symbols
        );
        assert_eq!(helper.unwrap().visibility, Visibility::Protected);
    }
}
