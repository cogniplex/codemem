//! Unified AST extraction engine using ast-grep.
//!
//! Replaces per-language tree-sitter extractors with a single engine driven
//! by YAML rules. Language-specific behavior (visibility, doc comments, etc.)
//! is handled by shared helpers keyed on language name.

mod references;
mod symbols;
mod visibility;

use crate::index::rule_loader::{LanguageRules, ReferenceRule, ScopeContainerRule, SymbolRule};
use crate::index::symbol::{Reference, ReferenceKind, Symbol, SymbolKind, Visibility};
use ast_grep_core::tree_sitter::LanguageExt;
use ast_grep_core::tree_sitter::StrDoc;
use ast_grep_core::{Doc, Node};
use ast_grep_language::SupportLang;
use std::borrow::Cow;

/// Type alias for ast-grep nodes parameterized on SupportLang.
pub type SgNode<'r> = Node<'r, StrDoc<SupportLang>>;

/// The unified extraction engine.
pub struct AstGrepEngine {
    languages: Vec<LanguageRules>,
}

impl AstGrepEngine {
    /// Create a new engine with all language rules loaded.
    pub fn new() -> Self {
        let languages = crate::index::rule_loader::load_all_rules();
        Self { languages }
    }

    /// Look up the rules for a given file extension.
    pub fn find_language(&self, ext: &str) -> Option<&LanguageRules> {
        self.languages
            .iter()
            .find(|lr| lr.extensions.contains(&ext))
    }

    /// List all file extensions we can handle.
    pub fn supported_extensions(&self) -> Vec<&str> {
        self.languages
            .iter()
            .flat_map(|lr| lr.extensions.iter().copied())
            .collect()
    }

    /// Check if a given extension is supported.
    pub fn supports_extension(&self, ext: &str) -> bool {
        self.languages.iter().any(|lr| lr.extensions.contains(&ext))
    }

    /// Get the language name for a given extension.
    pub fn language_name(&self, ext: &str) -> Option<&str> {
        self.find_language(ext).map(|lr| lr.name)
    }

    pub fn extract_symbols(
        &self,
        lang: &LanguageRules,
        source: &str,
        file_path: &str,
    ) -> Vec<Symbol> {
        let root = lang.lang.ast_grep(source);
        let root_node = root.root();
        let mut symbols = Vec::new();
        self.extract_symbols_recursive(
            lang,
            &root_node,
            source,
            file_path,
            &[],
            false,
            &mut symbols,
        );
        symbols
    }

    /// Parse source code and extract references.
    pub fn extract_references(
        &self,
        lang: &LanguageRules,
        source: &str,
        file_path: &str,
    ) -> Vec<Reference> {
        let root = lang.lang.ast_grep(source);
        let root_node = root.root();
        let mut references = Vec::new();
        self.extract_references_recursive(
            lang,
            &root_node,
            source,
            file_path,
            &[],
            &mut references,
        );
        references
    }

    // ── Symbol Extraction ─────────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    fn extract_symbols_recursive<D: Doc>(
        &self,
        lang: &LanguageRules,
        node: &Node<'_, D>,
        source: &str,
        file_path: &str,
        scope: &[String],
        in_method_scope: bool,
        symbols: &mut Vec<Symbol>,
    ) where
        D::Lang: ast_grep_core::Language,
    {
        let kind: Cow<'_, str> = node.kind();
        let kind_str = kind.as_ref();

        // Check if this is an unwrap node (e.g., decorated_definition, export_statement)
        if lang.symbol_unwrap_set.contains(kind_str) {
            for child in node.children() {
                self.extract_symbols_recursive(
                    lang,
                    &child,
                    source,
                    file_path,
                    scope,
                    in_method_scope,
                    symbols,
                );
            }
            return;
        }

        // Check if this is a scope container
        if let Some(&sc_idx) = lang.symbol_scope_index.get(kind_str) {
            let sc = &lang.symbol_scope_containers[sc_idx];
            if let Some(scope_name) = self.get_scope_name(lang, sc, node, source) {
                let mut new_scope = scope.to_vec();
                new_scope.push(scope_name);
                let new_method_scope = sc.is_method_scope;

                // Recurse into the body
                if let Some(body) = self.get_scope_body(sc, node) {
                    for child in body.children() {
                        self.extract_symbols_recursive(
                            lang,
                            &child,
                            source,
                            file_path,
                            &new_scope,
                            new_method_scope,
                            symbols,
                        );
                    }
                } else {
                    // No body field found, recurse into all children
                    for child in node.children() {
                        self.extract_symbols_recursive(
                            lang,
                            &child,
                            source,
                            file_path,
                            &new_scope,
                            new_method_scope,
                            symbols,
                        );
                    }
                }
                // The scope container itself might also be a symbol
                // (e.g., trait_item is both a scope container and an interface symbol)
            }
        }

        // Check if this matches any symbol rules
        if let Some(rule_indices) = lang.symbol_index.get(kind_str) {
            for &rule_idx in rule_indices {
                let rule = &lang.symbol_rules[rule_idx];

                // Handle multi-symbol special cases (e.g. Go type/const/var declarations)
                if let Some(ref special) = rule.special {
                    let multi = self
                        .handle_special_symbol_multi(lang, special, node, source, file_path, scope);
                    if !multi.is_empty() {
                        symbols.extend(multi);
                        return;
                    }
                }

                if let Some(sym) = self.extract_symbol_from_rule(
                    lang,
                    rule,
                    node,
                    source,
                    file_path,
                    scope,
                    in_method_scope,
                ) {
                    let name = sym.name.clone();
                    let is_scope = rule.is_scope;
                    symbols.push(sym);

                    // If this symbol creates a scope, recurse into it
                    if is_scope {
                        let mut new_scope = scope.to_vec();
                        new_scope.push(name);
                        if let Some(body) = node.field("body") {
                            for child in body.children() {
                                self.extract_symbols_recursive(
                                    lang,
                                    &child,
                                    source,
                                    file_path,
                                    &new_scope,
                                    in_method_scope,
                                    symbols,
                                );
                            }
                        }
                    }
                    return; // handled this node
                }
            }
        }

        // If this node was already handled as a scope container (which recursed),
        // and it was also a symbol (handled above), we've already returned.
        // If it was only a scope container (not a symbol), we already recursed.
        if lang.symbol_scope_index.contains_key(kind_str) {
            return; // already recursed above
        }

        // Default: recurse into children
        for child in node.children() {
            self.extract_symbols_recursive(
                lang,
                &child,
                source,
                file_path,
                scope,
                in_method_scope,
                symbols,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn extract_symbol_from_rule<D: Doc>(
        &self,
        lang: &LanguageRules,
        rule: &SymbolRule,
        node: &Node<'_, D>,
        source: &str,
        file_path: &str,
        scope: &[String],
        in_method_scope: bool,
    ) -> Option<Symbol>
    where
        D::Lang: ast_grep_core::Language,
    {
        // Handle special cases first
        if let Some(ref special) = rule.special {
            return self.handle_special_symbol(lang, special, node, source, file_path, scope);
        }

        // Extract name
        let name = self.get_node_field_text(node, &rule.name_field)?;
        if name.is_empty() {
            return None;
        }

        // Determine symbol kind
        let base_kind = parse_symbol_kind(&rule.symbol_kind)?;
        let is_test = self.detect_test(lang.name, node, source, file_path, &name);
        let kind = if is_test {
            SymbolKind::Test
        } else if rule.method_when_scoped && in_method_scope {
            SymbolKind::Method
        } else {
            base_kind
        };

        let visibility = self.detect_visibility(lang.name, node, source, &name);
        let signature = self.extract_signature(lang.name, node, source);
        let doc_comment = self.extract_doc_comment(lang.name, node, source);

        Some(build_symbol(
            name,
            kind,
            signature,
            visibility,
            doc_comment,
            file_path,
            node.start_pos().line(),
            node.end_pos().line(),
            scope,
            lang.scope_separator,
        ))
    }

    // ── Reference Extraction ──────────────────────────────────────────

    fn extract_references_recursive<D: Doc>(
        &self,
        lang: &LanguageRules,
        node: &Node<'_, D>,
        source: &str,
        file_path: &str,
        scope: &[String],
        references: &mut Vec<Reference>,
    ) where
        D::Lang: ast_grep_core::Language,
    {
        let kind: Cow<'_, str> = node.kind();
        let kind_str = kind.as_ref();

        // Check unwrap nodes
        if lang.reference_unwrap_set.contains(kind_str) {
            for child in node.children() {
                self.extract_references_recursive(
                    lang, &child, source, file_path, scope, references,
                );
            }
            return;
        }

        // Check scope containers
        if let Some(&sc_idx) = lang.reference_scope_index.get(kind_str) {
            let sc = &lang.reference_scope_containers[sc_idx];
            if let Some(scope_name) = self.get_scope_name(lang, sc, node, source) {
                let mut new_scope = scope.to_vec();
                new_scope.push(scope_name);

                // Still extract references from this node itself before recursing
                if let Some(rule_indices) = lang.reference_index.get(kind_str) {
                    for &rule_idx in rule_indices {
                        let rule = &lang.reference_rules[rule_idx];
                        self.extract_reference_from_rule(
                            lang, rule, node, source, file_path, scope, references,
                        );
                    }
                }

                if let Some(body) = self.get_scope_body(sc, node) {
                    for child in body.children() {
                        self.extract_references_recursive(
                            lang, &child, source, file_path, &new_scope, references,
                        );
                    }
                } else {
                    for child in node.children() {
                        self.extract_references_recursive(
                            lang, &child, source, file_path, &new_scope, references,
                        );
                    }
                }
                return;
            }
        }

        // Check reference rules
        if let Some(rule_indices) = lang.reference_index.get(kind_str) {
            for &rule_idx in rule_indices {
                let rule = &lang.reference_rules[rule_idx];
                self.extract_reference_from_rule(
                    lang, rule, node, source, file_path, scope, references,
                );
            }
        }

        // Default recursion
        for child in node.children() {
            self.extract_references_recursive(lang, &child, source, file_path, scope, references);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn extract_reference_from_rule<D: Doc>(
        &self,
        lang: &LanguageRules,
        rule: &ReferenceRule,
        node: &Node<'_, D>,
        source: &str,
        file_path: &str,
        scope: &[String],
        references: &mut Vec<Reference>,
    ) where
        D::Lang: ast_grep_core::Language,
    {
        // Handle special cases
        if let Some(ref special) = rule.special {
            self.handle_special_reference(
                lang, special, node, source, file_path, scope, references,
            );
            return;
        }

        let ref_kind = match parse_reference_kind(&rule.reference_kind) {
            Some(k) => k,
            None => return,
        };

        let target_name = if let Some(ref field) = rule.name_field {
            match self.get_node_field_text(node, field) {
                Some(name) => name,
                None => return,
            }
        } else {
            // Use full node text, trimmed
            let text = node.text();
            text.trim().to_string()
        };

        if target_name.is_empty() {
            return;
        }

        let source_qn = if scope.is_empty() {
            file_path.to_string()
        } else {
            scope.join(lang.scope_separator)
        };

        push_ref(
            references,
            &source_qn,
            target_name,
            ref_kind,
            file_path,
            node.start_pos().line(),
        );
    }

    // ── Node Utility Helpers ──────────────────────────────────────────

    pub(crate) fn get_node_field_text<D: Doc>(
        &self,
        node: &Node<'_, D>,
        field_name: &str,
    ) -> Option<String>
    where
        D::Lang: ast_grep_core::Language,
    {
        node.field(field_name).map(|n| n.text().to_string())
    }

    fn get_scope_name<D: Doc>(
        &self,
        lang: &LanguageRules,
        sc: &ScopeContainerRule,
        node: &Node<'_, D>,
        source: &str,
    ) -> Option<String>
    where
        D::Lang: ast_grep_core::Language,
    {
        if let Some(ref special) = sc.special {
            return self.get_special_scope_name(lang, special, node, source);
        }
        self.get_node_field_text(node, &sc.name_field)
    }

    fn get_scope_body<'a, D: Doc>(
        &self,
        sc: &ScopeContainerRule,
        node: &Node<'a, D>,
    ) -> Option<Node<'a, D>>
    where
        D::Lang: ast_grep_core::Language,
    {
        if let Some(ref special) = sc.special {
            return self.get_special_scope_body(special, node);
        }
        node.field(&sc.body_field)
    }

    fn get_special_scope_name<D: Doc>(
        &self,
        _lang: &LanguageRules,
        special: &str,
        node: &Node<'_, D>,
        _source: &str,
    ) -> Option<String>
    where
        D::Lang: ast_grep_core::Language,
    {
        match special {
            "go_method_scope" => {
                // For Go method declarations, the scope is Receiver.MethodName
                self.get_go_receiver_type(node)
            }
            "hcl_block_scope" => {
                // HCL block: combine block type and labels
                let mut parts = Vec::new();
                for child in node.children() {
                    let ck = child.kind();
                    if ck.as_ref() == "identifier" && parts.is_empty() {
                        parts.push(child.text().to_string());
                    } else if ck.as_ref() == "string_lit" {
                        parts.push(child.text().to_string().trim_matches('"').to_string());
                    }
                }
                if parts.is_empty() {
                    None
                } else {
                    Some(parts.join("."))
                }
            }
            "kotlin_scope" => self.get_node_field_text(node, "name").or_else(|| {
                for child in node.children() {
                    let ck = child.kind();
                    if ck.as_ref() == "type_identifier" || ck.as_ref() == "simple_identifier" {
                        return Some(child.text().to_string());
                    }
                }
                None
            }),
            "swift_class_scope" => {
                // Swift class/struct: find name via field or first type_identifier/identifier child
                self.get_node_field_text(node, "name").or_else(|| {
                    node.children()
                        .find(|c| {
                            let ck = c.kind();
                            ck.as_ref() == "type_identifier" || ck.as_ref() == "identifier"
                        })
                        .map(|c| c.text().to_string())
                })
            }
            "cpp_namespace_scope" => {
                // C++ namespace: may use qualified_identifier or name field
                self.get_node_field_text(node, "name").or_else(|| {
                    node.children()
                        .find(|c| {
                            let ck = c.kind();
                            ck.as_ref() == "namespace_identifier" || ck.as_ref() == "identifier"
                        })
                        .map(|c| c.text().to_string())
                })
            }
            _ => None,
        }
    }

    fn get_special_scope_body<'a, D: Doc>(
        &self,
        _special: &str,
        node: &Node<'a, D>,
    ) -> Option<Node<'a, D>>
    where
        D::Lang: ast_grep_core::Language,
    {
        node.field("body")
    }
}

impl Default for AstGrepEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── Free Functions ─────────────────────────────────────────────────────

/// Build a Symbol struct with the common scope→parent logic.
#[allow(clippy::too_many_arguments)]
fn build_symbol(
    name: String,
    kind: SymbolKind,
    signature: String,
    visibility: Visibility,
    doc_comment: Option<String>,
    file_path: &str,
    line_start: usize,
    line_end: usize,
    scope: &[String],
    scope_separator: &str,
) -> Symbol {
    Symbol {
        qualified_name: build_qualified_name(scope, &name, scope_separator),
        name,
        kind,
        signature,
        visibility,
        file_path: file_path.to_string(),
        line_start,
        line_end,
        doc_comment,
        parent: if scope.is_empty() {
            None
        } else {
            Some(scope.join(scope_separator))
        },
    }
}

/// Push a Reference onto a collection with the standard fields.
fn push_ref(
    refs: &mut Vec<Reference>,
    source_qn: &str,
    target: String,
    kind: ReferenceKind,
    file_path: &str,
    line: usize,
) {
    refs.push(Reference {
        source_qualified_name: source_qn.to_string(),
        target_name: target,
        kind,
        file_path: file_path.to_string(),
        line,
    });
}

fn build_qualified_name(scope: &[String], name: &str, separator: &str) -> String {
    if scope.is_empty() {
        name.to_string()
    } else {
        format!("{}{}{}", scope.join(separator), separator, name)
    }
}

fn parse_symbol_kind(s: &str) -> Option<SymbolKind> {
    match s {
        "function" => Some(SymbolKind::Function),
        "method" => Some(SymbolKind::Method),
        "class" => Some(SymbolKind::Class),
        "struct" => Some(SymbolKind::Struct),
        "enum" => Some(SymbolKind::Enum),
        "interface" => Some(SymbolKind::Interface),
        "type" => Some(SymbolKind::Type),
        "constant" => Some(SymbolKind::Constant),
        "module" => Some(SymbolKind::Module),
        "test" => Some(SymbolKind::Test),
        _ => None,
    }
}

fn parse_reference_kind(s: &str) -> Option<ReferenceKind> {
    match s {
        "import" => Some(ReferenceKind::Import),
        "call" => Some(ReferenceKind::Call),
        "inherits" => Some(ReferenceKind::Inherits),
        "implements" => Some(ReferenceKind::Implements),
        "type_usage" => Some(ReferenceKind::TypeUsage),
        _ => None,
    }
}

fn clean_block_doc_comment(text: &str) -> String {
    let trimmed = text.trim_start_matches("/**").trim_end_matches("*/").trim();

    let mut doc_lines = Vec::new();
    for line in trimmed.lines() {
        let line = line.trim();
        let line = line
            .strip_prefix("* ")
            .or_else(|| line.strip_prefix('*'))
            .unwrap_or(line);
        let line = line.trim_end();
        if !line.is_empty() {
            doc_lines.push(line.to_string());
        }
    }
    doc_lines.join("\n").trim_end().to_string()
}

#[cfg(test)]
#[path = "../tests/engine_tests.rs"]
mod tests;
