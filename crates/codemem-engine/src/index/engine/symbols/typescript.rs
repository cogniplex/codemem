//! TypeScript/JavaScript symbol extractors.

use super::super::build_symbol;
use crate::index::rule_loader::LanguageRules;
use crate::index::symbol::{Symbol, SymbolKind, Visibility};
use ast_grep_core::{Doc, Node};

impl crate::index::engine::AstGrepEngine {
    pub(super) fn extract_ts_arrow_field<D: Doc>(
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
        // public_field_definition with arrow function value
        let value = node.field("value")?;
        let vk = value.kind();
        if vk.as_ref() != "arrow_function" && vk.as_ref() != "function_expression" {
            return None;
        }
        let name = self.get_node_field_text(node, "name")?;
        let visibility = self.detect_visibility(lang.name, node, source, &name);
        let signature = self.extract_signature(lang.name, node, source);
        let doc_comment = self.extract_doc_comment(lang.name, node, source);

        Some(build_symbol(
            name,
            SymbolKind::Method,
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

    pub(super) fn extract_ts_lexical_arrow<D: Doc>(
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
        for child in node.children() {
            if child.kind().as_ref() == "variable_declarator" {
                let value = child.field("value")?;
                let vk = value.kind();
                if vk.as_ref() != "arrow_function" && vk.as_ref() != "function_expression" {
                    return None;
                }
                let name = self.get_node_field_text(&child, "name")?;
                let exported = node
                    .parent()
                    .is_some_and(|p| p.kind().as_ref() == "export_statement");
                let visibility = if exported {
                    Visibility::Public
                } else {
                    Visibility::Private
                };
                let signature = self.extract_signature(lang.name, node, source);
                let doc_comment = self.extract_doc_comment(lang.name, node, source);

                return Some(build_symbol(
                    name,
                    SymbolKind::Function,
                    signature,
                    visibility,
                    doc_comment,
                    file_path,
                    node.start_pos().line(),
                    node.end_pos().line(),
                    scope,
                    lang.scope_separator,
                ));
            }
        }
        None
    }
}
