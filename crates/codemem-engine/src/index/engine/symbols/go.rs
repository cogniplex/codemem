//! Go language symbol extractors.

use super::super::build_symbol;
use crate::index::rule_loader::LanguageRules;
use crate::index::symbol::{Symbol, SymbolKind, Visibility};
use ast_grep_core::{Doc, Node};

impl crate::index::engine::AstGrepEngine {
    pub(super) fn extract_go_method<D: Doc>(
        &self,
        lang: &LanguageRules,
        node: &Node<'_, D>,
        source: &str,
        file_path: &str,
    ) -> Option<Symbol>
    where
        D::Lang: ast_grep_core::Language,
    {
        let name = self.get_node_field_text(node, "name")?;
        let receiver_type = self.get_go_receiver_type(node);
        let qualified_name = if let Some(ref recv) = receiver_type {
            format!("{}.{}", recv, name)
        } else {
            name.clone()
        };
        let visibility = if name.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
            Visibility::Public
        } else {
            Visibility::Private
        };
        let signature = self.extract_signature(lang.name, node, source);
        let doc_comment = self.extract_doc_comment(lang.name, node, source);

        Some(Symbol {
            name,
            qualified_name,
            kind: SymbolKind::Method,
            signature,
            visibility,
            file_path: file_path.to_string(),
            line_start: node.start_pos().line(),
            line_end: node.end_pos().line(),
            doc_comment,
            parent: receiver_type,
            parameters: Vec::new(),
            return_type: None,
            is_async: false,
            attributes: Vec::new(),
            throws: Vec::new(),
            generic_params: None,
            is_abstract: false,
        })
    }

    pub(in crate::index::engine) fn get_go_receiver_type<D: Doc>(
        &self,
        node: &Node<'_, D>,
    ) -> Option<String>
    where
        D::Lang: ast_grep_core::Language,
    {
        let receiver = node.field("receiver")?;
        for child in receiver.children() {
            if child.kind().as_ref() == "parameter_declaration" {
                if let Some(type_node) = child.field("type") {
                    let type_text = type_node.text().to_string();
                    return Some(type_text.trim_start_matches('*').to_string());
                }
            }
        }
        None
    }

    /// Extract Go type declarations: `type Foo struct { ... }`, `type Bar interface { ... }`, etc.
    /// A single `type_declaration` may contain multiple `type_spec` children (grouped `type (...)`).
    pub(super) fn extract_go_type_declaration<D: Doc>(
        &self,
        lang: &LanguageRules,
        node: &Node<'_, D>,
        source: &str,
        file_path: &str,
        scope: &[String],
    ) -> Vec<Symbol>
    where
        D::Lang: ast_grep_core::Language,
    {
        let mut symbols = Vec::new();
        for child in node.children() {
            let ck = child.kind();
            let ck_str = ck.as_ref();
            if ck_str == "type_spec" {
                if let Some(sym) = self.extract_go_type_spec(lang, &child, source, file_path, scope)
                {
                    symbols.push(sym);
                }
            } else if ck_str == "type_spec_list" {
                for spec in child.children() {
                    if spec.kind().as_ref() == "type_spec" {
                        if let Some(sym) =
                            self.extract_go_type_spec(lang, &spec, source, file_path, scope)
                        {
                            symbols.push(sym);
                        }
                    }
                }
            }
        }
        symbols
    }

    /// Extract a single Go type_spec into a Symbol.
    fn extract_go_type_spec<D: Doc>(
        &self,
        lang: &LanguageRules,
        node: &Node<'_, D>,
        source: &str,
        file_path: &str,
        scope: &[String],
    ) -> Option<Symbol>
    where
        D::Lang: ast_grep_core::Language,
    {
        let name = self.get_node_field_text(node, "name")?;
        if name.is_empty() {
            return None;
        }

        let mut kind = SymbolKind::Type;
        if let Some(type_node) = node.field("type") {
            let type_kind = type_node.kind();
            match type_kind.as_ref() {
                "struct_type" => kind = SymbolKind::Struct,
                "interface_type" => kind = SymbolKind::Interface,
                _ => {}
            }
        }

        let visibility = if name.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
            Visibility::Public
        } else {
            Visibility::Private
        };
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

    /// Extract Go const/var declarations. Each contains one or more `const_spec`/`var_spec` children.
    pub(super) fn extract_go_const_or_var<D: Doc>(
        &self,
        lang: &LanguageRules,
        node: &Node<'_, D>,
        source: &str,
        file_path: &str,
        scope: &[String],
        spec_kind: &str,
    ) -> Vec<Symbol>
    where
        D::Lang: ast_grep_core::Language,
    {
        let list_kind = format!("{spec_kind}_list");
        let mut symbols = Vec::new();
        for child in node.children() {
            let ck = child.kind();
            let ck_str = ck.as_ref();
            if ck_str == spec_kind {
                if let Some(sym) =
                    self.extract_go_const_var_spec(lang, &child, source, file_path, scope)
                {
                    symbols.push(sym);
                }
            } else if ck_str == list_kind {
                for spec in child.children() {
                    if spec.kind().as_ref() == spec_kind {
                        if let Some(sym) =
                            self.extract_go_const_var_spec(lang, &spec, source, file_path, scope)
                        {
                            symbols.push(sym);
                        }
                    }
                }
            }
        }
        symbols
    }

    /// Extract a single Go const_spec or var_spec into a Symbol.
    fn extract_go_const_var_spec<D: Doc>(
        &self,
        lang: &LanguageRules,
        node: &Node<'_, D>,
        source: &str,
        file_path: &str,
        scope: &[String],
    ) -> Option<Symbol>
    where
        D::Lang: ast_grep_core::Language,
    {
        let name = self.get_node_field_text(node, "name")?;
        if name.is_empty() {
            return None;
        }
        let visibility = if name.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
            Visibility::Public
        } else {
            Visibility::Private
        };
        let signature = self.extract_signature(lang.name, node, source);
        let doc_comment = self.extract_doc_comment(lang.name, node, source);

        Some(build_symbol(
            name,
            SymbolKind::Constant,
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
}
