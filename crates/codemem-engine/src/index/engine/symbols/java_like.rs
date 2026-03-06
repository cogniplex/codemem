//! Java, C#, Kotlin, and Scala symbol extractors.

use super::super::build_symbol;
use crate::index::rule_loader::LanguageRules;
use crate::index::symbol::{Symbol, SymbolKind};
use ast_grep_core::{Doc, Node};

impl crate::index::engine::AstGrepEngine {
    pub(super) fn extract_java_static_final_field<D: Doc>(
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
        // Walk modifiers children for static/final tokens instead of substring matching
        let mut has_static = false;
        let mut has_final = false;
        for child in node.children() {
            let ck = child.kind();
            if ck.as_ref() == "modifiers" || ck.as_ref() == "modifier" {
                for modifier_child in child.children() {
                    let mk = modifier_child.kind();
                    let mt = modifier_child.text();
                    if mk.as_ref() == "static" || mt.as_ref() == "static" {
                        has_static = true;
                    }
                    if mk.as_ref() == "final" || mt.as_ref() == "final" {
                        has_final = true;
                    }
                }
            }
        }
        if !(has_static && has_final) {
            return None;
        }
        // Find the variable declarator to get the name
        let text = node.text().to_string();
        for child in node.children() {
            if child.kind().as_ref() == "variable_declarator" {
                let name = self.get_node_field_text(&child, "name")?;
                let visibility = self.detect_visibility(lang.name, node, source, &name);
                let signature = text.lines().next().unwrap_or(&text).trim().to_string();
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

    /// Extract Java instance fields (non-static-final).
    pub(super) fn extract_java_field<D: Doc>(
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
        // Find the variable declarator to get the name
        for child in node.children() {
            if child.kind().as_ref() == "variable_declarator" {
                let name = self.get_node_field_text(&child, "name")?;
                let visibility = self.detect_visibility(lang.name, node, source, &name);
                let text = node.text().to_string();
                let signature = text.lines().next().unwrap_or(&text).trim().to_string();
                let doc_comment = self.extract_doc_comment(lang.name, node, source);

                return Some(build_symbol(
                    name,
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
        None
    }

    /// Extract C# field declarations. The name lives inside `variable_declaration` > `variable_declarator`.
    pub(super) fn extract_csharp_field<D: Doc>(
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
        // C# field_declaration → variable_declaration → variable_declarator (name)
        for child in node.children() {
            if child.kind().as_ref() == "variable_declaration" {
                for vc in child.children() {
                    if vc.kind().as_ref() == "variable_declarator" {
                        let name = self.get_node_field_text(&vc, "name").or_else(|| {
                            // Fallback: first identifier child
                            vc.children()
                                .find(|c| c.kind().as_ref() == "identifier")
                                .map(|c| c.text().to_string())
                        })?;
                        let visibility = self.detect_visibility(lang.name, node, source, &name);
                        let signature = node.text().lines().next().unwrap_or("").trim().to_string();
                        let doc_comment = self.extract_doc_comment(lang.name, node, source);

                        return Some(build_symbol(
                            name,
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

    pub(super) fn extract_kotlin_symbol<D: Doc>(
        &self,
        lang: &LanguageRules,
        special: &str,
        node: &Node<'_, D>,
        source: &str,
        file_path: &str,
        scope: &[String],
    ) -> Option<Symbol>
    where
        D::Lang: ast_grep_core::Language,
    {
        let name = self.get_node_field_text(node, "name").or_else(|| {
            for child in node.children() {
                let ck = child.kind();
                if ck.as_ref() == "type_identifier" || ck.as_ref() == "simple_identifier" {
                    return Some(child.text().to_string());
                }
            }
            None
        })?;

        let kind = match special {
            "kotlin_function" => {
                if scope.is_empty() {
                    SymbolKind::Function
                } else {
                    SymbolKind::Method
                }
            }
            _ => {
                let has_interface_keyword =
                    node.children().any(|c| c.kind().as_ref() == "interface");
                if has_interface_keyword {
                    SymbolKind::Interface
                } else {
                    SymbolKind::Class
                }
            }
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

    pub(super) fn extract_scala_final_val<D: Doc>(
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
        // Only extract if has "final" modifier
        let text = node.text().to_string();
        if !text.contains("final") {
            return None;
        }
        let name = self
            .get_node_field_text(node, "pattern")
            .or_else(|| self.get_node_field_text(node, "name"))?;
        let visibility = self.detect_visibility(lang.name, node, source, &name);
        let signature = text.lines().next().unwrap_or(&text).trim().to_string();
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
