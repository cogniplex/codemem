//! Language-specific helpers for visibility, test detection, signatures, doc comments,
//! and enhanced symbol metadata extraction (parameters, return types, attributes, etc.).

mod doc_comments;
mod metadata;
mod parameters;

use crate::index::symbol::Visibility;
use ast_grep_core::{Doc, Node};

impl crate::index::engine::AstGrepEngine {
    pub(in crate::index::engine) fn detect_visibility<D: Doc>(
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
                if let Some(parent) = node.parent() {
                    if parent.kind().as_ref() == "export_statement" {
                        return Visibility::Public;
                    }
                }
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
                            return Visibility::Public;
                        }
                    }
                }
                Visibility::Private
            }
            "kotlin" => {
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

    pub(in crate::index::engine) fn detect_test<D: Doc>(
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

    pub(in crate::index::engine) fn extract_signature<D: Doc>(
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
}

/// Extract the error type `E` from a Rust `Result<T, E>` return type string.
pub(in crate::index::engine) fn extract_result_error_type(text: &str) -> Option<String> {
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
