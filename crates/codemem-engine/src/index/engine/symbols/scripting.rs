//! Python, Ruby, and PHP symbol extractors.

use super::super::build_symbol;
use crate::index::rule_loader::LanguageRules;
use crate::index::symbol::{Symbol, SymbolKind, Visibility};
use ast_grep_core::{Doc, Node};

impl crate::index::engine::AstGrepEngine {
    pub(super) fn extract_python_constant<D: Doc>(
        &self,
        lang: &LanguageRules,
        node: &Node<'_, D>,
        _source: &str,
        file_path: &str,
        scope: &[String],
    ) -> Option<Symbol>
    where
        D::Lang: ast_grep_core::Language,
    {
        if !scope.is_empty() {
            return None;
        }
        let child = node.children().next()?;
        if child.kind().as_ref() != "assignment" {
            return None;
        }
        let left = child.field("left")?;
        if left.kind().as_ref() != "identifier" {
            return None;
        }
        let name = left.text().to_string();
        if name.len() < 2
            || !name
                .chars()
                .all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit())
        {
            return None;
        }
        let signature = child.text().to_string();
        let first_line = signature.lines().next().unwrap_or(&signature);

        Some(build_symbol(
            name,
            SymbolKind::Constant,
            first_line.to_string(),
            Visibility::Public,
            None,
            file_path,
            node.start_pos().line(),
            node.end_pos().line(),
            scope,
            lang.scope_separator,
        ))
    }

    /// Extract Ruby constants — only assignments where the left side starts with uppercase.
    pub(super) fn extract_ruby_constant<D: Doc>(
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
        let left = node.field("left")?;
        let name = left.text().to_string();
        // Ruby constants must start with uppercase
        if !name.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
            return None;
        }
        let signature = node.text().lines().next().unwrap_or("").trim().to_string();
        let doc_comment = self.extract_doc_comment(lang.name, node, source);

        Some(build_symbol(
            name,
            SymbolKind::Constant,
            signature,
            Visibility::Public,
            doc_comment,
            file_path,
            node.start_pos().line(),
            node.end_pos().line(),
            scope,
            lang.scope_separator,
        ))
    }

    /// Extract PHP property declarations — `$name` in property_element children.
    pub(super) fn extract_php_property<D: Doc>(
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
        // property_declaration → property_element → variable_name ($name)
        for child in node.children() {
            if child.kind().as_ref() == "property_element" {
                for vc in child.children() {
                    if vc.kind().as_ref() == "variable_name" {
                        let name = vc.text().to_string();
                        let clean_name = name.trim_start_matches('$');
                        let visibility =
                            self.detect_visibility(lang.name, node, source, clean_name);
                        let signature = node.text().lines().next().unwrap_or("").trim().to_string();
                        let doc_comment = self.extract_doc_comment(lang.name, node, source);

                        return Some(build_symbol(
                            clean_name.to_string(),
                            SymbolKind::Field,
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
            }
        }
        None
    }

    /// Extract PHP class constants — `const NAME = value`.
    pub(super) fn extract_php_const<D: Doc>(
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
        // const_declaration → const_element → name
        for child in node.children() {
            if child.kind().as_ref() == "const_element" {
                let name = self.get_node_field_text(&child, "name").or_else(|| {
                    child
                        .children()
                        .find(|c| c.kind().as_ref() == "name")
                        .map(|c| c.text().to_string())
                })?;
                let visibility = self.detect_visibility(lang.name, node, source, &name);
                let signature = node.text().lines().next().unwrap_or("").trim().to_string();
                let doc_comment = self.extract_doc_comment(lang.name, node, source);

                return Some(build_symbol(
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
                ));
            }
        }
        None
    }
}
