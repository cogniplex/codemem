//! HCL, Swift, and C++ symbol extractors.

use super::super::build_symbol;
use crate::index::rule_loader::LanguageRules;
use crate::index::symbol::{Symbol, SymbolKind, Visibility};
use ast_grep_core::{Doc, Node};

impl crate::index::engine::AstGrepEngine {
    pub(super) fn extract_hcl_block<D: Doc>(
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
        let mut block_type = String::new();
        let mut labels = Vec::new();

        for child in node.children() {
            let ck = child.kind();
            if ck.as_ref() == "identifier" && block_type.is_empty() {
                block_type = child.text().to_string();
            } else if ck.as_ref() == "string_lit" {
                labels.push(child.text().to_string().trim_matches('"').to_string());
            }
        }

        if block_type.is_empty() {
            return None;
        }

        let kind = match block_type.as_str() {
            "resource" | "data" => SymbolKind::Class,
            "module" | "provider" => SymbolKind::Module,
            "variable" | "output" | "locals" => SymbolKind::Constant,
            _ => SymbolKind::Module,
        };

        let mut name_parts = vec![block_type];
        name_parts.extend(labels);
        let name = name_parts.join(".");
        let doc_comment = self.extract_doc_comment(lang.name, node, source);

        Some(build_symbol(
            name,
            kind,
            node.text().lines().next().unwrap_or("").trim().to_string(),
            Visibility::Public,
            doc_comment,
            file_path,
            node.start_pos().line(),
            node.end_pos().line(),
            scope,
            lang.scope_separator,
        ))
    }

    pub(super) fn extract_hcl_attribute<D: Doc>(
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
        if scope.is_empty() {
            return None; // Only extract attributes inside blocks
        }
        let name = node.children().next()?.text().to_string();
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

    pub(super) fn extract_swift_class<D: Doc>(
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
        // Swift uses class_declaration for class/struct/enum
        let mut keyword = "";
        let mut name = String::new();
        for child in node.children() {
            let ck = child.kind();
            let ct = child.text();
            if ck.as_ref() == "class" || ct.as_ref() == "class" {
                keyword = "class";
            } else if ct.as_ref() == "struct" {
                keyword = "struct";
            } else if ct.as_ref() == "enum" {
                keyword = "enum";
            } else if (ck.as_ref() == "type_identifier" || ck.as_ref() == "identifier")
                && name.is_empty()
            {
                name = ct.to_string();
            }
        }

        if let Some(n) = node.field("name") {
            name = n.text().to_string();
        }

        if name.is_empty() {
            return None;
        }

        let kind = match keyword {
            "struct" => SymbolKind::Struct,
            "enum" => SymbolKind::Enum,
            _ => SymbolKind::Class,
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

    pub(super) fn extract_cpp_type_alias<D: Doc>(
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
        let name = self.get_node_field_text(node, "name").or_else(|| {
            // Try declarator field for typedef
            self.get_node_field_text(node, "declarator")
        })?;
        let visibility = self.detect_visibility(lang.name, node, source, &name);
        let signature = self.extract_signature(lang.name, node, source);
        let doc_comment = self.extract_doc_comment(lang.name, node, source);

        Some(build_symbol(
            name,
            SymbolKind::Type,
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

    pub(super) fn extract_cpp_define<D: Doc>(
        &self,
        _lang: &LanguageRules,
        node: &Node<'_, D>,
        _source: &str,
        file_path: &str,
        scope: &[String],
    ) -> Option<Symbol>
    where
        D::Lang: ast_grep_core::Language,
    {
        let name = self.get_node_field_text(node, "name")?;
        let text = node.text().to_string();
        let first_line = text.lines().next().unwrap_or(&text);

        Some(build_symbol(
            name,
            SymbolKind::Constant,
            first_line.trim().to_string(),
            Visibility::Public,
            None,
            file_path,
            node.start_pos().line(),
            node.end_pos().line(),
            scope,
            "::",
        ))
    }
}
