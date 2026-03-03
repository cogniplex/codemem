//! Language-specific helpers for visibility, test detection, signatures, doc comments,
//! and enhanced symbol metadata extraction (parameters, return types, attributes, etc.).

use super::clean_block_doc_comment;
use crate::index::symbol::{Parameter, Visibility};
use ast_grep_core::{Doc, Node};

impl super::AstGrepEngine {
    pub(super) fn detect_visibility<D: Doc>(
        &self,
        lang_name: &str,
        node: &Node<'_, D>,
        _source: &str,
        name: &str,
    ) -> Visibility
    where
        D::Lang: ast_grep_core::Language,
    {
        match lang_name {
            "rust" => {
                // Check for visibility_modifier child
                for child in node.children() {
                    if child.kind().as_ref() == "visibility_modifier" {
                        let text = child.text();
                        if text.contains("pub(crate)") {
                            return Visibility::Crate;
                        } else if text.starts_with("pub") {
                            return Visibility::Public;
                        }
                    }
                }
                Visibility::Private
            }
            "python" => {
                if name.starts_with("__") && name.ends_with("__") {
                    // Dunder methods (e.g. __init__, __str__) are public
                    Visibility::Public
                } else if name.starts_with('_') {
                    Visibility::Private
                } else {
                    Visibility::Public
                }
            }
            "go" => {
                if name.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
                    Visibility::Public
                } else {
                    Visibility::Private
                }
            }
            "typescript" | "tsx" | "javascript" => {
                // Check parent for export_statement
                if let Some(parent) = node.parent() {
                    if parent.kind().as_ref() == "export_statement" {
                        return Visibility::Public;
                    }
                }
                // Check for accessibility_modifier child
                for child in node.children() {
                    if child.kind().as_ref() == "accessibility_modifier" {
                        let text = child.text();
                        return match text.as_ref() {
                            "private" => Visibility::Private,
                            "protected" => Visibility::Protected,
                            "public" => Visibility::Public,
                            _ => Visibility::Public,
                        };
                    }
                }
                Visibility::Private
            }
            "java" | "csharp" => {
                // Check modifiers children
                for child in node.children() {
                    let ck = child.kind();
                    if ck.as_ref() == "modifiers" || ck.as_ref() == "modifier" {
                        let text = child.text();
                        if text.contains("public") {
                            return Visibility::Public;
                        } else if text.contains("protected") {
                            return Visibility::Protected;
                        } else if text.contains("private") {
                            return Visibility::Private;
                        } else if text.contains("internal") {
                            return Visibility::Public; // C# internal ~ crate
                        }
                    }
                }
                Visibility::Private
            }
            "kotlin" => {
                // Walk children for visibility_modifier node
                for child in node.children() {
                    let ck = child.kind();
                    if ck.as_ref() == "visibility_modifier" || ck.as_ref() == "modifiers" {
                        for modifier_child in child.children() {
                            let mt = modifier_child.text();
                            match mt.as_ref() {
                                "private" => return Visibility::Private,
                                "protected" => return Visibility::Protected,
                                "internal" => return Visibility::Crate,
                                "public" => return Visibility::Public,
                                _ => {}
                            }
                        }
                        // Single visibility_modifier node
                        let text = child.text();
                        match text.as_ref() {
                            "private" => return Visibility::Private,
                            "protected" => return Visibility::Protected,
                            "internal" => return Visibility::Crate,
                            "public" => return Visibility::Public,
                            _ => {}
                        }
                    }
                }
                Visibility::Public
            }
            "swift" => {
                // Walk children for modifier nodes instead of text matching
                for child in node.children() {
                    let ck = child.kind();
                    if ck.as_ref() == "modifier" || ck.as_ref() == "modifiers" {
                        for modifier_child in child.children() {
                            let mt = modifier_child.text();
                            match mt.as_ref() {
                                "public" | "open" => return Visibility::Public,
                                "private" | "fileprivate" => return Visibility::Private,
                                "internal" => return Visibility::Crate,
                                _ => {}
                            }
                        }
                        let text = child.text();
                        match text.as_ref() {
                            "public" | "open" => return Visibility::Public,
                            "private" | "fileprivate" => return Visibility::Private,
                            "internal" => return Visibility::Crate,
                            _ => {}
                        }
                    }
                }
                // Swift default access level is internal
                Visibility::Private
            }
            "php" => {
                for child in node.children() {
                    if child.kind().as_ref() == "visibility_modifier" {
                        let text = child.text();
                        return match text.as_ref() {
                            "private" => Visibility::Private,
                            "protected" => Visibility::Protected,
                            "public" => Visibility::Public,
                            _ => Visibility::Public,
                        };
                    }
                }
                Visibility::Public
            }
            "scala" => {
                // Check for access_modifier in modifiers
                for child in node.children() {
                    if child.kind().as_ref() == "modifiers" {
                        let text = child.text();
                        if text.contains("private") {
                            return Visibility::Private;
                        } else if text.contains("protected") {
                            return Visibility::Protected;
                        }
                    }
                }
                Visibility::Public
            }
            "ruby" | "hcl" => Visibility::Public,
            _ => Visibility::Private,
        }
    }

    pub(super) fn detect_test<D: Doc>(
        &self,
        lang_name: &str,
        node: &Node<'_, D>,
        _source: &str,
        file_path: &str,
        name: &str,
    ) -> bool
    where
        D::Lang: ast_grep_core::Language,
    {
        match lang_name {
            "rust" => {
                // Check preceding siblings for #[test] attribute
                let mut prev = node.prev();
                while let Some(sibling) = prev {
                    match sibling.kind().as_ref() {
                        "attribute_item" => {
                            let text = sibling.text();
                            if text.contains("test") {
                                return true;
                            }
                            prev = sibling.prev();
                        }
                        "line_comment" => {
                            prev = sibling.prev();
                        }
                        _ => break,
                    }
                }
                false
            }
            "python" => name.starts_with("test_"),
            "go" => name.starts_with("Test") && file_path.ends_with("_test.go"),
            "java" => {
                // Check for @Test annotation
                let mut prev = node.prev();
                while let Some(sibling) = prev {
                    if sibling.kind().as_ref() == "marker_annotation" {
                        let text = sibling.text();
                        if text.contains("Test") {
                            return true;
                        }
                    }
                    prev = sibling.prev();
                }
                name.starts_with("test") && file_path.ends_with("Test.java")
            }
            "cpp" => name.starts_with("test_") || name.starts_with("Test") || name.contains("TEST"),
            _ => false,
        }
    }

    pub(super) fn extract_signature<D: Doc>(
        &self,
        lang_name: &str,
        node: &Node<'_, D>,
        _source: &str,
    ) -> String
    where
        D::Lang: ast_grep_core::Language,
    {
        let text = node.text().to_string();
        let delimiter = match lang_name {
            "python" => ':',
            _ => '{',
        };

        if let Some(pos) = text.find(delimiter) {
            text[..pos].trim().to_string()
        } else {
            let first_line = text.lines().next().unwrap_or(&text);
            first_line
                .trim_end_matches(';')
                .trim_end_matches(':')
                .trim()
                .to_string()
        }
    }

    pub(super) fn extract_doc_comment<D: Doc>(
        &self,
        lang_name: &str,
        node: &Node<'_, D>,
        _source: &str,
    ) -> Option<String>
    where
        D::Lang: ast_grep_core::Language,
    {
        match lang_name {
            "rust" => {
                let mut comment_nodes = Vec::new();
                let mut prev = node.prev();
                while let Some(sibling) = prev {
                    match sibling.kind().as_ref() {
                        "line_comment" => {
                            let text = sibling.text().to_string();
                            if text.starts_with("///") || text.starts_with("//!") {
                                comment_nodes.push(text);
                                prev = sibling.prev();
                                continue;
                            }
                            break;
                        }
                        "attribute_item" | "inner_attribute_item" => {
                            prev = sibling.prev();
                            continue;
                        }
                        _ => break,
                    }
                }
                comment_nodes.reverse();
                let doc_lines: Vec<String> = comment_nodes
                    .iter()
                    .map(|text| {
                        let line = text
                            .strip_prefix("/// ")
                            .or_else(|| text.strip_prefix("///"))
                            .or_else(|| text.strip_prefix("//! "))
                            .or_else(|| text.strip_prefix("//!"))
                            .unwrap_or(text);
                        line.trim_end().to_string()
                    })
                    .collect();
                if doc_lines.is_empty() {
                    None
                } else {
                    Some(doc_lines.join("\n").trim_end().to_string())
                }
            }
            "python" => {
                // Python docstrings: first statement in body
                let body = node.field("body")?;
                let first_stmt = body.children().next()?;
                if first_stmt.kind().as_ref() != "expression_statement" {
                    return None;
                }
                let expr = first_stmt.children().next()?;
                if expr.kind().as_ref() != "string" {
                    return None;
                }
                let raw = expr.text().to_string();
                let stripped = raw
                    .trim_start_matches("\"\"\"")
                    .trim_start_matches("'''")
                    .trim_end_matches("\"\"\"")
                    .trim_end_matches("'''")
                    .trim();
                if stripped.is_empty() {
                    None
                } else {
                    Some(stripped.to_string())
                }
            }
            "go" => {
                let mut comment_lines = Vec::new();
                let mut prev = node.prev();
                while let Some(sibling) = prev {
                    if sibling.kind().as_ref() == "comment" {
                        let text = sibling.text().to_string();
                        if text.starts_with("//") {
                            let stripped = text
                                .strip_prefix("// ")
                                .or_else(|| text.strip_prefix("//"))
                                .unwrap_or(&text);
                            comment_lines.push(stripped.trim_end().to_string());
                            prev = sibling.prev();
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
            "typescript" | "tsx" | "javascript" => {
                // JSDoc: /** ... */
                let effective_node = if let Some(parent) = node.parent() {
                    if parent.kind().as_ref() == "export_statement" {
                        parent
                    } else {
                        return self.extract_jsdoc_from_node(node);
                    }
                } else {
                    return self.extract_jsdoc_from_node(node);
                };
                self.extract_jsdoc_from_node(&effective_node)
            }
            "java" | "kotlin" | "scala" | "php" => {
                // JavaDoc/KDoc/ScalaDoc/PHPDoc: /** ... */ block comments
                let mut prev = node.prev();
                while let Some(sibling) = prev {
                    let sk = sibling.kind();
                    if sk.as_ref() == "block_comment" || sk.as_ref() == "multiline_comment" {
                        let text = sibling.text().to_string();
                        if text.starts_with("/**") {
                            return Some(clean_block_doc_comment(&text));
                        }
                    }
                    if sk.as_ref() == "comment" || sk.as_ref() == "line_comment" {
                        prev = sibling.prev();
                        continue;
                    }
                    break;
                }
                None
            }
            "csharp" => {
                // XML doc comments: ///
                let mut comment_nodes = Vec::new();
                let mut prev = node.prev();
                while let Some(sibling) = prev {
                    if sibling.kind().as_ref() == "comment" {
                        let text = sibling.text().to_string();
                        if text.starts_with("///") {
                            comment_nodes.push(text);
                            prev = sibling.prev();
                            continue;
                        }
                    }
                    break;
                }
                comment_nodes.reverse();
                if comment_nodes.is_empty() {
                    None
                } else {
                    let lines: Vec<String> = comment_nodes
                        .iter()
                        .map(|t| {
                            t.strip_prefix("/// ")
                                .or_else(|| t.strip_prefix("///"))
                                .unwrap_or(t)
                                .trim_end()
                                .to_string()
                        })
                        .collect();
                    Some(lines.join("\n"))
                }
            }
            "swift" => {
                let mut comment_nodes = Vec::new();
                let mut prev = node.prev();
                while let Some(sibling) = prev {
                    let sk = sibling.kind();
                    if sk.as_ref() == "comment" || sk.as_ref() == "line_comment" {
                        let text = sibling.text().to_string();
                        if text.starts_with("///") {
                            comment_nodes.push(text);
                            prev = sibling.prev();
                            continue;
                        } else if text.starts_with("/**") {
                            return Some(clean_block_doc_comment(&text));
                        }
                    }
                    break;
                }
                comment_nodes.reverse();
                if comment_nodes.is_empty() {
                    None
                } else {
                    let lines: Vec<String> = comment_nodes
                        .iter()
                        .map(|t| {
                            t.strip_prefix("/// ")
                                .or_else(|| t.strip_prefix("///"))
                                .unwrap_or(t)
                                .trim_end()
                                .to_string()
                        })
                        .collect();
                    Some(lines.join("\n"))
                }
            }
            "ruby" => {
                let mut comment_lines = Vec::new();
                let mut prev = node.prev();
                while let Some(sibling) = prev {
                    if sibling.kind().as_ref() == "comment" {
                        let text = sibling.text().to_string();
                        if text.starts_with('#') {
                            let stripped = text
                                .strip_prefix("# ")
                                .or_else(|| text.strip_prefix("#"))
                                .unwrap_or(&text);
                            comment_lines.push(stripped.trim_end().to_string());
                            prev = sibling.prev();
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
            "hcl" => {
                let mut comment_lines = Vec::new();
                let mut prev = node.prev();
                while let Some(sibling) = prev {
                    if sibling.kind().as_ref() == "comment" {
                        let text = sibling.text().to_string();
                        let stripped = text
                            .strip_prefix("// ")
                            .or_else(|| text.strip_prefix("//"))
                            .or_else(|| text.strip_prefix("# "))
                            .or_else(|| text.strip_prefix("#"))
                            .unwrap_or(&text);
                        comment_lines.push(stripped.trim_end().to_string());
                        prev = sibling.prev();
                        continue;
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
            _ => None,
        }
    }

    pub(super) fn extract_jsdoc_from_node<D: Doc>(&self, node: &Node<'_, D>) -> Option<String>
    where
        D::Lang: ast_grep_core::Language,
    {
        let mut prev = node.prev();
        while let Some(sibling) = prev {
            match sibling.kind().as_ref() {
                "comment" => {
                    let text = sibling.text().to_string();
                    if text.starts_with("/**") {
                        return Some(clean_block_doc_comment(&text));
                    }
                    break;
                }
                "decorator" => {
                    prev = sibling.prev();
                    continue;
                }
                _ => break,
            }
        }
        None
    }

    // ── Enhanced Symbol Metadata Extraction ────────────────────────────

    /// Enrich a Symbol with extracted metadata (parameters, return type, attributes, etc.).
    /// Only populates fields for function/method-like symbols where it makes sense.
    pub(super) fn enrich_symbol_metadata<D: Doc>(
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

    /// Extract parameters from a function/method node using tree-sitter children.
    pub(super) fn extract_parameters<D: Doc>(
        &self,
        lang_name: &str,
        node: &Node<'_, D>,
    ) -> Vec<Parameter>
    where
        D::Lang: ast_grep_core::Language,
    {
        match lang_name {
            "rust" => self.extract_rust_parameters(node),
            "python" => self.extract_python_parameters(node),
            "typescript" | "tsx" | "javascript" => self.extract_ts_parameters(node),
            "go" => self.extract_go_parameters(node),
            "java" | "kotlin" | "csharp" => self.extract_java_like_parameters(node),
            _ => self.extract_parameters_from_signature(node),
        }
    }

    /// Extract return type from a function/method node.
    pub(super) fn extract_return_type<D: Doc>(
        &self,
        lang_name: &str,
        node: &Node<'_, D>,
    ) -> Option<String>
    where
        D::Lang: ast_grep_core::Language,
    {
        match lang_name {
            "rust" => {
                // Rust: look for return_type field (-> Type)
                node.field("return_type").map(|n| {
                    n.text()
                        .to_string()
                        .trim_start_matches("->")
                        .trim()
                        .to_string()
                })
            }
            "python" => {
                // Python: return_type annotation after ->
                node.field("return_type").map(|n| n.text().to_string())
            }
            "typescript" | "tsx" | "javascript" => {
                // TS: return_type or type_annotation on the function
                node.field("return_type").map(|n| {
                    n.text()
                        .to_string()
                        .trim_start_matches(':')
                        .trim()
                        .to_string()
                })
            }
            "go" => {
                // Go: result field
                node.field("result").map(|n| n.text().to_string())
            }
            "java" | "csharp" | "kotlin" => {
                // Java/C#: type field on method_declaration
                node.field("type").map(|n| n.text().to_string())
            }
            _ => {
                // Fallback: try to parse from signature
                self.extract_return_type_from_signature(node)
            }
        }
    }

    /// Extract attributes/decorators/annotations preceding a node.
    pub(super) fn extract_attributes<D: Doc>(
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
                // Walk preceding sibling attribute_item nodes (#[...])
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
                // Walk preceding decorator nodes (@...)
                let mut prev = node.prev();
                while let Some(sibling) = prev {
                    if sibling.kind().as_ref() == "decorator" {
                        attrs.push(sibling.text().to_string());
                        prev = sibling.prev();
                    } else {
                        break;
                    }
                }
                // Also check parent decorated_definition
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
                // Walk preceding annotation/marker_annotation nodes
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
                // Also check modifiers children for annotations
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
                // C# attributes: [Attribute]
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
                // TS/JS decorators (@...)
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
    pub(super) fn detect_async<D: Doc>(&self, lang_name: &str, node: &Node<'_, D>) -> bool
    where
        D::Lang: ast_grep_core::Language,
    {
        match lang_name {
            "rust" => {
                // Check for "async" keyword child
                node.children().any(|c| c.text().as_ref() == "async")
            }
            "python" => {
                // async keyword or parent is async_function_definition
                let kind = node.kind();
                kind.as_ref() == "async_function_definition"
                    || node.children().any(|c| c.text().as_ref() == "async")
            }
            "typescript" | "tsx" | "javascript" => {
                // Check for "async" keyword child
                node.children().any(|c| c.text().as_ref() == "async")
            }
            "kotlin" => {
                // Check modifiers for "suspend"
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
                // Check modifiers for "async"
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
    pub(super) fn extract_generic_params<D: Doc>(
        &self,
        lang_name: &str,
        node: &Node<'_, D>,
    ) -> Option<String>
    where
        D::Lang: ast_grep_core::Language,
    {
        match lang_name {
            "rust" => {
                // Look for type_parameters child
                node.field("type_parameters").map(|n| n.text().to_string())
            }
            "typescript" | "tsx" => {
                // Look for type_parameters child
                node.field("type_parameters").map(|n| n.text().to_string())
            }
            "java" | "kotlin" | "csharp" => {
                // Look for type_parameters child
                node.field("type_parameters").map(|n| n.text().to_string())
            }
            "go" => {
                // Go 1.18+ type parameters
                node.field("type_parameters").map(|n| n.text().to_string())
            }
            _ => None,
        }
    }

    /// Extract error/exception types from a function signature.
    pub(super) fn extract_throws<D: Doc>(&self, lang_name: &str, node: &Node<'_, D>) -> Vec<String>
    where
        D::Lang: ast_grep_core::Language,
    {
        match lang_name {
            "rust" => {
                // Extract E from Result<T, E> return types
                if let Some(ret) = node.field("return_type") {
                    let text = ret.text().to_string();
                    if let Some(result_content) = extract_result_error_type(&text) {
                        return vec![result_content];
                    }
                }
                Vec::new()
            }
            "java" | "kotlin" => {
                // Look for throws_clause child
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
                // Check for "throws" keyword
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
    pub(super) fn detect_abstract<D: Doc>(&self, lang_name: &str, node: &Node<'_, D>) -> bool
    where
        D::Lang: ast_grep_core::Language,
    {
        match lang_name {
            "rust" => {
                // In Rust, trait methods without a body are abstract (have semicolon, no body)
                node.field("body").is_none() && node.text().to_string().trim_end().ends_with(';')
            }
            "java" | "csharp" => {
                // Check for "abstract" modifier
                for child in node.children() {
                    let ck = child.kind();
                    if (ck.as_ref() == "modifiers" || ck.as_ref() == "modifier")
                        && child.text().contains("abstract")
                    {
                        return true;
                    }
                }
                // Also: interface methods without body are abstract
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
            "typescript" | "tsx" => {
                // Check for "abstract" keyword child
                node.children().any(|c| c.text().as_ref() == "abstract")
            }
            "python" => {
                // Check for @abstractmethod decorator
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

    // ── Private Parameter Extraction Helpers ───────────────────────────

    fn extract_rust_parameters<D: Doc>(&self, node: &Node<'_, D>) -> Vec<Parameter>
    where
        D::Lang: ast_grep_core::Language,
    {
        let params_node = match node.field("parameters") {
            Some(n) => n,
            None => return Vec::new(),
        };
        let mut params = Vec::new();
        for child in params_node.children() {
            let ck = child.kind();
            match ck.as_ref() {
                "parameter" => {
                    let name = child
                        .field("pattern")
                        .map(|n| n.text().to_string())
                        .unwrap_or_default();
                    let type_ann = child.field("type").map(|n| n.text().to_string());
                    params.push(Parameter {
                        name,
                        type_annotation: type_ann,
                        default_value: None,
                    });
                }
                "self_parameter" => {
                    params.push(Parameter {
                        name: child.text().to_string(),
                        type_annotation: None,
                        default_value: None,
                    });
                }
                _ => {}
            }
        }
        params
    }

    fn extract_python_parameters<D: Doc>(&self, node: &Node<'_, D>) -> Vec<Parameter>
    where
        D::Lang: ast_grep_core::Language,
    {
        let params_node = match node.field("parameters") {
            Some(n) => n,
            None => return Vec::new(),
        };
        let mut params = Vec::new();
        for child in params_node.children() {
            let ck = child.kind();
            match ck.as_ref() {
                "identifier" => {
                    params.push(Parameter {
                        name: child.text().to_string(),
                        type_annotation: None,
                        default_value: None,
                    });
                }
                "typed_parameter" | "typed_default_parameter" => {
                    let name = child
                        .children()
                        .find(|c| c.kind().as_ref() == "identifier")
                        .map(|c| c.text().to_string())
                        .unwrap_or_default();
                    let type_ann = child.field("type").map(|n| n.text().to_string());
                    let default = child.field("value").map(|n| n.text().to_string());
                    params.push(Parameter {
                        name,
                        type_annotation: type_ann,
                        default_value: default,
                    });
                }
                "default_parameter" => {
                    let name = child
                        .field("name")
                        .map(|n| n.text().to_string())
                        .unwrap_or_default();
                    let default = child.field("value").map(|n| n.text().to_string());
                    params.push(Parameter {
                        name,
                        type_annotation: None,
                        default_value: default,
                    });
                }
                _ => {}
            }
        }
        params
    }

    fn extract_ts_parameters<D: Doc>(&self, node: &Node<'_, D>) -> Vec<Parameter>
    where
        D::Lang: ast_grep_core::Language,
    {
        let params_node = match node.field("parameters") {
            Some(n) => n,
            None => return Vec::new(),
        };
        let mut params = Vec::new();
        for child in params_node.children() {
            let ck = child.kind();
            match ck.as_ref() {
                "required_parameter" | "optional_parameter" => {
                    let name = child
                        .field("pattern")
                        .or_else(|| child.field("name"))
                        .map(|n| n.text().to_string())
                        .unwrap_or_default();
                    let type_ann = child.field("type").map(|n| {
                        n.text()
                            .to_string()
                            .trim_start_matches(':')
                            .trim()
                            .to_string()
                    });
                    let default = child.field("value").map(|n| n.text().to_string());
                    params.push(Parameter {
                        name,
                        type_annotation: type_ann,
                        default_value: default,
                    });
                }
                _ => {}
            }
        }
        params
    }

    fn extract_go_parameters<D: Doc>(&self, node: &Node<'_, D>) -> Vec<Parameter>
    where
        D::Lang: ast_grep_core::Language,
    {
        let params_node = match node.field("parameters") {
            Some(n) => n,
            None => return Vec::new(),
        };
        let mut params = Vec::new();
        for child in params_node.children() {
            if child.kind().as_ref() == "parameter_declaration" {
                let name = child
                    .field("name")
                    .map(|n| n.text().to_string())
                    .unwrap_or_default();
                let type_ann = child.field("type").map(|n| n.text().to_string());
                params.push(Parameter {
                    name,
                    type_annotation: type_ann,
                    default_value: None,
                });
            }
        }
        params
    }

    fn extract_java_like_parameters<D: Doc>(&self, node: &Node<'_, D>) -> Vec<Parameter>
    where
        D::Lang: ast_grep_core::Language,
    {
        // Java/Kotlin/C#: formal_parameters → formal_parameter
        let params_node = node.field("parameters").or_else(|| {
            node.children()
                .find(|c| c.kind().as_ref() == "formal_parameters")
        });
        let params_node = match params_node {
            Some(n) => n,
            None => return Vec::new(),
        };
        let mut params = Vec::new();
        for child in params_node.children() {
            let ck = child.kind();
            if ck.as_ref() == "formal_parameter" || ck.as_ref() == "spread_parameter" {
                let name = child
                    .field("name")
                    .map(|n| n.text().to_string())
                    .unwrap_or_else(|| {
                        // Fallback: last identifier child is typically the name
                        child
                            .children()
                            .filter(|c| c.kind().as_ref() == "identifier")
                            .last()
                            .map(|c| c.text().to_string())
                            .unwrap_or_default()
                    });
                let type_ann = child
                    .field("type")
                    .map(|n| n.text().to_string())
                    .or_else(|| {
                        // Fallback: first type child
                        child
                            .children()
                            .find(|c| {
                                let k = c.kind();
                                k.as_ref().contains("type") || k.as_ref() == "identifier"
                            })
                            .map(|c| c.text().to_string())
                    });
                // Avoid adding name as type if they're the same
                let type_ann = type_ann.filter(|t| t != &name);
                params.push(Parameter {
                    name,
                    type_annotation: type_ann,
                    default_value: None,
                });
            }
        }
        params
    }

    /// Fallback: extract parameters from the signature text for unsupported languages.
    fn extract_parameters_from_signature<D: Doc>(&self, node: &Node<'_, D>) -> Vec<Parameter>
    where
        D::Lang: ast_grep_core::Language,
    {
        let text = node.text().to_string();
        let open = match text.find('(') {
            Some(p) => p,
            None => return Vec::new(),
        };
        let close = match text[open..].find(')') {
            Some(p) => open + p,
            None => return Vec::new(),
        };
        let params_text = &text[open + 1..close];
        if params_text.trim().is_empty() {
            return Vec::new();
        }
        params_text
            .split(',')
            .filter_map(|p| {
                let p = p.trim();
                if p.is_empty() {
                    return None;
                }
                // Try to split "type name" or "name: type" patterns
                let parts: Vec<&str> = p.splitn(2, ':').collect();
                if parts.len() == 2 {
                    Some(Parameter {
                        name: parts[0].trim().to_string(),
                        type_annotation: Some(parts[1].trim().to_string()),
                        default_value: None,
                    })
                } else {
                    Some(Parameter {
                        name: p.to_string(),
                        type_annotation: None,
                        default_value: None,
                    })
                }
            })
            .collect()
    }

    /// Fallback: extract return type from signature text (-> Type or : Type).
    fn extract_return_type_from_signature<D: Doc>(&self, node: &Node<'_, D>) -> Option<String>
    where
        D::Lang: ast_grep_core::Language,
    {
        let text = node.text().to_string();
        // Look for -> pattern (Rust, Python, Swift)
        if let Some(pos) = text.find("->") {
            let after = text[pos + 2..].trim();
            let ret = after.split(['{', ':', '\n']).next().unwrap_or("").trim();
            if !ret.is_empty() {
                return Some(ret.to_string());
            }
        }
        None
    }
}

/// Extract the error type `E` from a Rust `Result<T, E>` return type string.
fn extract_result_error_type(text: &str) -> Option<String> {
    let trimmed = text.trim().trim_start_matches("->").trim();
    if !trimmed.starts_with("Result") {
        return None;
    }
    let open = trimmed.find('<')?;
    let close = trimmed.rfind('>')?;
    let inner = &trimmed[open + 1..close];
    // Find the comma separating T and E, respecting nested angle brackets
    let mut depth = 0i32;
    let mut comma_pos = None;
    for (i, ch) in inner.char_indices() {
        match ch {
            '<' => depth += 1,
            '>' => depth -= 1,
            ',' if depth == 0 => {
                comma_pos = Some(i);
                break;
            }
            _ => {}
        }
    }
    let error_type = if let Some(pos) = comma_pos {
        inner[pos + 1..].trim()
    } else {
        return None;
    };
    if error_type.is_empty() {
        None
    } else {
        Some(error_type.to_string())
    }
}
