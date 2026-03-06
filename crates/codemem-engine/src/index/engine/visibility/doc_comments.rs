//! Doc comment extraction for all supported languages.

use crate::index::engine::clean_block_doc_comment;
use ast_grep_core::{Doc, Node};

impl crate::index::engine::AstGrepEngine {
    pub(in crate::index::engine) fn extract_doc_comment<D: Doc>(
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

    pub(in crate::index::engine) fn extract_jsdoc_from_node<D: Doc>(
        &self,
        node: &Node<'_, D>,
    ) -> Option<String>
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
