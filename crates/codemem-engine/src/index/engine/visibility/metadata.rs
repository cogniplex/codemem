//! Enhanced symbol metadata extraction: return types, attributes, async, generics, throws, abstract.

use super::extract_result_error_type;
use ast_grep_core::{Doc, Node};

impl crate::index::engine::AstGrepEngine {
    /// Enrich a Symbol with extracted metadata (parameters, return type, attributes, etc.).
    /// Only populates fields for function/method-like symbols where it makes sense.
    pub(in crate::index::engine) fn enrich_symbol_metadata<D: Doc>(
        &self,
        lang_name: &str,
        node: &Node<'_, D>,
        sym: &mut crate::index::symbol::Symbol,
    ) where
        D::Lang: ast_grep_core::Language,
    {
        use crate::index::symbol::SymbolKind;

        let is_callable = matches!(
            sym.kind,
            SymbolKind::Function | SymbolKind::Method | SymbolKind::Test | SymbolKind::Constructor
        );

        if is_callable {
            sym.parameters = self.extract_parameters(lang_name, node);
            sym.return_type = self.extract_return_type(lang_name, node);
            sym.is_async = self.detect_async(lang_name, node);
            sym.throws = self.extract_throws(lang_name, node);
            sym.is_abstract = self.detect_abstract(lang_name, node);
            sym.generic_params = self.extract_generic_params(lang_name, node);
        }

        // Attributes apply to all symbol kinds
        sym.attributes = self.extract_attributes(lang_name, node);
    }

    /// Extract return type from a function/method node.
    pub(in crate::index::engine) fn extract_return_type<D: Doc>(
        &self,
        lang_name: &str,
        node: &Node<'_, D>,
    ) -> Option<String>
    where
        D::Lang: ast_grep_core::Language,
    {
        match lang_name {
            "rust" => node.field("return_type").map(|n| {
                n.text()
                    .to_string()
                    .trim_start_matches("->")
                    .trim()
                    .to_string()
            }),
            "python" => node.field("return_type").map(|n| n.text().to_string()),
            "typescript" | "tsx" | "javascript" => node.field("return_type").map(|n| {
                n.text()
                    .to_string()
                    .trim_start_matches(':')
                    .trim()
                    .to_string()
            }),
            "go" => node.field("result").map(|n| n.text().to_string()),
            "java" | "csharp" | "kotlin" => node.field("type").map(|n| n.text().to_string()),
            _ => self.extract_return_type_from_signature(node),
        }
    }

    /// Extract attributes/decorators/annotations preceding a node.
    pub(in crate::index::engine) fn extract_attributes<D: Doc>(
        &self,
        lang_name: &str,
        node: &Node<'_, D>,
    ) -> Vec<String>
    where
        D::Lang: ast_grep_core::Language,
    {
        let mut attrs = Vec::new();
        match lang_name {
            "rust" => {
                let mut prev = node.prev();
                while let Some(sibling) = prev {
                    match sibling.kind().as_ref() {
                        "attribute_item" => {
                            attrs.push(sibling.text().to_string());
                            prev = sibling.prev();
                        }
                        "line_comment" => {
                            prev = sibling.prev();
                        }
                        _ => break,
                    }
                }
                attrs.reverse();
            }
            "python" => {
                let mut prev = node.prev();
                while let Some(sibling) = prev {
                    if sibling.kind().as_ref() == "decorator" {
                        attrs.push(sibling.text().to_string());
                        prev = sibling.prev();
                    } else {
                        break;
                    }
                }
                if let Some(parent) = node.parent() {
                    if parent.kind().as_ref() == "decorated_definition" {
                        for child in parent.children() {
                            if child.kind().as_ref() == "decorator" {
                                let text = child.text().to_string();
                                if !attrs.contains(&text) {
                                    attrs.push(text);
                                }
                            }
                        }
                    }
                }
                attrs.reverse();
            }
            "java" | "kotlin" => {
                let mut prev = node.prev();
                while let Some(sibling) = prev {
                    match sibling.kind().as_ref() {
                        "marker_annotation" | "annotation" => {
                            attrs.push(sibling.text().to_string());
                            prev = sibling.prev();
                        }
                        "block_comment" | "line_comment" | "comment" => {
                            prev = sibling.prev();
                        }
                        _ => break,
                    }
                }
                for child in node.children() {
                    let ck = child.kind();
                    if ck.as_ref() == "modifiers" {
                        for mc in child.children() {
                            if mc.kind().as_ref() == "marker_annotation"
                                || mc.kind().as_ref() == "annotation"
                            {
                                let text = mc.text().to_string();
                                if !attrs.contains(&text) {
                                    attrs.push(text);
                                }
                            }
                        }
                    }
                }
                attrs.reverse();
            }
            "csharp" => {
                let mut prev = node.prev();
                while let Some(sibling) = prev {
                    if sibling.kind().as_ref() == "attribute_list" {
                        attrs.push(sibling.text().to_string());
                        prev = sibling.prev();
                    } else {
                        break;
                    }
                }
                attrs.reverse();
            }
            "typescript" | "tsx" | "javascript" => {
                let mut prev = node.prev();
                while let Some(sibling) = prev {
                    if sibling.kind().as_ref() == "decorator" {
                        attrs.push(sibling.text().to_string());
                        prev = sibling.prev();
                    } else {
                        break;
                    }
                }
                attrs.reverse();
            }
            _ => {}
        }
        attrs
    }

    /// Detect if a function/method is async.
    pub(in crate::index::engine) fn detect_async<D: Doc>(
        &self,
        lang_name: &str,
        node: &Node<'_, D>,
    ) -> bool
    where
        D::Lang: ast_grep_core::Language,
    {
        match lang_name {
            "rust" => node.children().any(|c| c.text().as_ref() == "async"),
            "python" => {
                let kind = node.kind();
                kind.as_ref() == "async_function_definition"
                    || node.children().any(|c| c.text().as_ref() == "async")
            }
            "typescript" | "tsx" | "javascript" => {
                node.children().any(|c| c.text().as_ref() == "async")
            }
            "kotlin" => {
                for child in node.children() {
                    if child.kind().as_ref() == "modifiers" {
                        for mc in child.children() {
                            if mc.text().as_ref() == "suspend" {
                                return true;
                            }
                        }
                    }
                }
                false
            }
            "csharp" => {
                for child in node.children() {
                    let ck = child.kind();
                    if (ck.as_ref() == "modifiers" || ck.as_ref() == "modifier")
                        && child.text().contains("async")
                    {
                        return true;
                    }
                }
                false
            }
            _ => false,
        }
    }

    /// Extract generic type parameters from a node.
    pub(in crate::index::engine) fn extract_generic_params<D: Doc>(
        &self,
        lang_name: &str,
        node: &Node<'_, D>,
    ) -> Option<String>
    where
        D::Lang: ast_grep_core::Language,
    {
        match lang_name {
            "rust" | "typescript" | "tsx" | "java" | "kotlin" | "csharp" | "go" => {
                node.field("type_parameters").map(|n| n.text().to_string())
            }
            _ => None,
        }
    }

    /// Extract error/exception types from a function signature.
    pub(in crate::index::engine) fn extract_throws<D: Doc>(
        &self,
        lang_name: &str,
        node: &Node<'_, D>,
    ) -> Vec<String>
    where
        D::Lang: ast_grep_core::Language,
    {
        match lang_name {
            "rust" => {
                if let Some(ret) = node.field("return_type") {
                    let text = ret.text().to_string();
                    if let Some(result_content) = extract_result_error_type(&text) {
                        return vec![result_content];
                    }
                }
                Vec::new()
            }
            "java" | "kotlin" => {
                for child in node.children() {
                    if child.kind().as_ref() == "throws" {
                        let mut types = Vec::new();
                        for tc in child.children() {
                            let tk = tc.kind();
                            if tk.as_ref() == "type_identifier"
                                || tk.as_ref() == "scoped_type_identifier"
                            {
                                types.push(tc.text().to_string());
                            }
                        }
                        return types;
                    }
                }
                Vec::new()
            }
            "swift" => {
                let text = node.text().to_string();
                if text.contains("throws") {
                    vec!["throws".to_string()]
                } else {
                    Vec::new()
                }
            }
            _ => Vec::new(),
        }
    }

    /// Detect if a method is abstract (trait/interface method without a body).
    pub(in crate::index::engine) fn detect_abstract<D: Doc>(
        &self,
        lang_name: &str,
        node: &Node<'_, D>,
    ) -> bool
    where
        D::Lang: ast_grep_core::Language,
    {
        match lang_name {
            "rust" => {
                node.field("body").is_none() && node.text().to_string().trim_end().ends_with(';')
            }
            "java" | "csharp" => {
                for child in node.children() {
                    let ck = child.kind();
                    if (ck.as_ref() == "modifiers" || ck.as_ref() == "modifier")
                        && child.text().contains("abstract")
                    {
                        return true;
                    }
                }
                node.field("body").is_none()
            }
            "kotlin" => {
                for child in node.children() {
                    if child.kind().as_ref() == "modifiers" {
                        for mc in child.children() {
                            if mc.text().as_ref() == "abstract" {
                                return true;
                            }
                        }
                    }
                }
                false
            }
            "typescript" | "tsx" => node.children().any(|c| c.text().as_ref() == "abstract"),
            "python" => {
                let mut prev = node.prev();
                while let Some(sibling) = prev {
                    if sibling.kind().as_ref() == "decorator" {
                        if sibling.text().contains("abstractmethod") {
                            return true;
                        }
                        prev = sibling.prev();
                    } else {
                        break;
                    }
                }
                false
            }
            _ => false,
        }
    }
}
