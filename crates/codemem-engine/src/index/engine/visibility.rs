//! Language-specific helpers for visibility, test detection, signatures, and doc comments.

use super::clean_block_doc_comment;
use crate::index::symbol::Visibility;
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
}
