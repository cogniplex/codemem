//! Parameter extraction for all supported languages.

use crate::index::symbol::Parameter;
use ast_grep_core::{Doc, Node};

impl crate::index::engine::AstGrepEngine {
    /// Extract parameters from a function/method node using tree-sitter children.
    pub(in crate::index::engine) fn extract_parameters<D: Doc>(
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
    pub(in crate::index::engine) fn extract_return_type_from_signature<D: Doc>(
        &self,
        node: &Node<'_, D>,
    ) -> Option<String>
    where
        D::Lang: ast_grep_core::Language,
    {
        let text = node.text().to_string();
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
