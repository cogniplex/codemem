//! Special-case reference handlers and language-specific reference extractors.

use super::{build_qualified_name, push_ref};
use crate::index::rule_loader::LanguageRules;
use crate::index::symbol::{Reference, ReferenceKind};
use ast_grep_core::{Doc, Node};

impl super::AstGrepEngine {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn handle_special_reference<D: Doc>(
        &self,
        lang: &LanguageRules,
        special: &str,
        node: &Node<'_, D>,
        source: &str,
        file_path: &str,
        scope: &[String],
        references: &mut Vec<Reference>,
    ) where
        D::Lang: ast_grep_core::Language,
    {
        let source_qn = if scope.is_empty() {
            file_path.to_string()
        } else {
            scope.join(lang.scope_separator)
        };

        match special {
            "rust_use" => {
                let text = node.text().to_string();
                let trimmed = text.trim_start_matches("use ").trim_end_matches(';').trim();
                // R1: Decompose grouped imports like `std::collections::{HashMap, HashSet}`
                // and nested groups like `std::{collections::HashMap, io::{Read, Write}}`
                let paths = decompose_rust_use_path(trimmed);
                let line = node.start_pos().line();
                for path in paths {
                    push_ref(
                        references,
                        &source_qn,
                        path,
                        ReferenceKind::Import,
                        file_path,
                        line,
                    );
                }
            }
            "rust_macro" => {
                if let Some(macro_node) = node.field("macro") {
                    let macro_name = macro_node.text().to_string();
                    push_ref(
                        references,
                        &source_qn,
                        format!("{}!", macro_name),
                        ReferenceKind::Call,
                        file_path,
                        node.start_pos().line(),
                    );
                }
            }
            "rust_impl_trait" => {
                if let Some(trait_node) = node.field("trait") {
                    if let Some(type_node) = node.field("type") {
                        let trait_name = trait_node.text().to_string();
                        let type_name = type_node.text().to_string();
                        let impl_source_qn = if scope.is_empty() {
                            type_name
                        } else {
                            format!(
                                "{}{}{}",
                                scope.join(lang.scope_separator),
                                lang.scope_separator,
                                type_name
                            )
                        };
                        push_ref(
                            references,
                            &impl_source_qn,
                            trait_name,
                            ReferenceKind::Implements,
                            file_path,
                            node.start_pos().line(),
                        );
                    }
                }
            }
            "python_import" => {
                for child in node.children() {
                    if child.kind().as_ref() == "dotted_name" {
                        push_ref(
                            references,
                            &source_qn,
                            child.text().to_string(),
                            ReferenceKind::Import,
                            file_path,
                            node.start_pos().line(),
                        );
                    }
                }
            }
            "python_class_bases" => {
                if let Some(name_node) = node.field("name") {
                    let class_name = name_node.text().to_string();
                    let class_qn = build_qualified_name(scope, &class_name, lang.scope_separator);
                    if let Some(superclasses) = node.field("superclasses") {
                        for child in superclasses.children() {
                            let ck = child.kind();
                            if ck.as_ref() == "identifier" || ck.as_ref() == "attribute" {
                                push_ref(
                                    references,
                                    &class_qn,
                                    child.text().to_string(),
                                    ReferenceKind::Inherits,
                                    file_path,
                                    child.start_pos().line(),
                                );
                            }
                        }
                    }
                }
            }
            "go_import" => {
                for child in node.children() {
                    if child.kind().as_ref() == "import_spec" {
                        if let Some(path_node) = child.field("path") {
                            let path_text = path_node.text().to_string();
                            let clean = path_text.trim_matches('"');
                            push_ref(
                                references,
                                &source_qn,
                                clean.to_string(),
                                ReferenceKind::Import,
                                file_path,
                                child.start_pos().line(),
                            );
                        }
                    }
                    if child.kind().as_ref() == "import_spec_list" {
                        for spec in child.children() {
                            if spec.kind().as_ref() == "import_spec" {
                                if let Some(path_node) = spec.field("path") {
                                    let path_text = path_node.text().to_string();
                                    let clean = path_text.trim_matches('"');
                                    push_ref(
                                        references,
                                        &source_qn,
                                        clean.to_string(),
                                        ReferenceKind::Import,
                                        file_path,
                                        spec.start_pos().line(),
                                    );
                                }
                            }
                        }
                    }
                }
            }
            "ts_class_heritage" => {
                self.extract_ts_class_heritage(lang, node, source, file_path, scope, references);
            }
            "java_import" => {
                let text = node.text().to_string();
                let trimmed = text
                    .trim_start_matches("import ")
                    .trim_start_matches("static ")
                    .trim_end_matches(';')
                    .trim();
                push_ref(
                    references,
                    &source_qn,
                    trimmed.to_string(),
                    ReferenceKind::Import,
                    file_path,
                    node.start_pos().line(),
                );
            }
            "java_method_invocation" => {
                if let Some(name_node) = node.field("name") {
                    push_ref(
                        references,
                        &source_qn,
                        name_node.text().to_string(),
                        ReferenceKind::Call,
                        file_path,
                        node.start_pos().line(),
                    );
                }
            }
            "java_class_heritage" => {
                self.extract_java_heritage(
                    node,
                    file_path,
                    scope,
                    lang,
                    ReferenceKind::Inherits,
                    "superclass",
                    references,
                );
            }
            "java_class_implements" => {
                self.extract_java_heritage(
                    node,
                    file_path,
                    scope,
                    lang,
                    ReferenceKind::Implements,
                    "interfaces",
                    references,
                );
            }
            "cpp_include" => {
                let text = node.text().to_string();
                let target = text
                    .trim_start_matches("#include")
                    .trim()
                    .trim_matches(|c| c == '<' || c == '>' || c == '"')
                    .trim()
                    .to_string();
                push_ref(
                    references,
                    &source_qn,
                    target,
                    ReferenceKind::Import,
                    file_path,
                    node.start_pos().line(),
                );
            }
            "cpp_base_classes" => {
                for child in node.children() {
                    let ck = child.kind();
                    if ck.as_ref() == "type_identifier" || ck.as_ref() == "qualified_identifier" {
                        push_ref(
                            references,
                            &source_qn,
                            child.text().to_string(),
                            ReferenceKind::Inherits,
                            file_path,
                            child.start_pos().line(),
                        );
                    }
                }
            }
            "csharp_using" => {
                let text = node.text().to_string();
                let trimmed = text
                    .trim_start_matches("using ")
                    .trim_start_matches("static ")
                    .trim_end_matches(';')
                    .trim();
                push_ref(
                    references,
                    &source_qn,
                    trimmed.to_string(),
                    ReferenceKind::Import,
                    file_path,
                    node.start_pos().line(),
                );
            }
            "csharp_base_list" => {
                for child in node.children() {
                    let ck = child.kind();
                    if ck.as_ref() == "identifier"
                        || ck.as_ref() == "qualified_name"
                        || ck.as_ref() == "generic_name"
                    {
                        push_ref(
                            references,
                            &source_qn,
                            child.text().to_string(),
                            ReferenceKind::Inherits,
                            file_path,
                            child.start_pos().line(),
                        );
                    }
                }
            }
            "ruby_require" => {
                if let Some(method_node) = node.field("method") {
                    let method = method_node.text().to_string();
                    if method == "require" || method == "require_relative" {
                        if let Some(args) = node.field("arguments") {
                            let text = args.text().to_string();
                            let clean = text
                                .trim_matches(|c| c == '(' || c == ')' || c == '"' || c == '\'')
                                .trim();
                            push_ref(
                                references,
                                &source_qn,
                                clean.to_string(),
                                ReferenceKind::Import,
                                file_path,
                                node.start_pos().line(),
                            );
                        }
                    }
                }
            }
            "ruby_call" => {
                if let Some(method_node) = node.field("method") {
                    let method = method_node.text().to_string();
                    if method != "require" && method != "require_relative" {
                        push_ref(
                            references,
                            &source_qn,
                            method,
                            ReferenceKind::Call,
                            file_path,
                            node.start_pos().line(),
                        );
                    }
                }
            }
            "ruby_superclass" => {
                if let Some(name_node) = node.field("name") {
                    let class_name = name_node.text().to_string();
                    let class_qn = build_qualified_name(scope, &class_name, lang.scope_separator);
                    if let Some(superclass) = node.field("superclass") {
                        push_ref(
                            references,
                            &class_qn,
                            superclass.text().to_string(),
                            ReferenceKind::Inherits,
                            file_path,
                            superclass.start_pos().line(),
                        );
                    }
                }
            }
            "ruby_include_extend" => {
                if let Some(method_node) = node.field("method") {
                    let method = method_node.text().to_string();
                    if method == "include" || method == "extend" || method == "prepend" {
                        if let Some(args) = node.field("arguments") {
                            for arg in args.children() {
                                let ak = arg.kind();
                                if ak.as_ref() == "constant"
                                    || ak.as_ref() == "scope_resolution"
                                    || ak.as_ref() == "identifier"
                                {
                                    push_ref(
                                        references,
                                        &source_qn,
                                        arg.text().to_string(),
                                        ReferenceKind::Import,
                                        file_path,
                                        arg.start_pos().line(),
                                    );
                                }
                            }
                        }
                    }
                }
            }
            "kotlin_import" => {
                let text = node.text().to_string();
                let trimmed = text.trim_start_matches("import ").trim();
                push_ref(
                    references,
                    &source_qn,
                    trimmed.to_string(),
                    ReferenceKind::Import,
                    file_path,
                    node.start_pos().line(),
                );
            }
            "kotlin_call" | "kotlin_delegation" => {
                // Try tree-sitter "function" field first, then fall back to text splitting.
                // Strip generics (e.g. "Foo<Bar>" -> "Foo") from the extracted name.
                let name = if let Some(func_node) = node.field("function") {
                    func_node.text().to_string()
                } else {
                    let text = node.text().to_string();
                    // Split on '(' to strip arguments, then '<' to strip generics
                    let before_paren = text.split('(').next().unwrap_or(&text);
                    before_paren
                        .split('<')
                        .next()
                        .unwrap_or(before_paren)
                        .trim()
                        .to_string()
                };
                let ref_kind = if special == "kotlin_delegation" {
                    ReferenceKind::Inherits
                } else {
                    ReferenceKind::Call
                };
                push_ref(
                    references,
                    &source_qn,
                    name,
                    ref_kind,
                    file_path,
                    node.start_pos().line(),
                );
            }
            "swift_import" => {
                let text = node.text().to_string();
                let trimmed = text.trim_start_matches("import ").trim();
                push_ref(
                    references,
                    &source_qn,
                    trimmed.to_string(),
                    ReferenceKind::Import,
                    file_path,
                    node.start_pos().line(),
                );
            }
            "swift_inheritance" => {
                for child in node.children() {
                    let ck = child.kind();
                    if ck.as_ref() == "type_identifier" || ck.as_ref() == "user_type" {
                        push_ref(
                            references,
                            &source_qn,
                            child.text().to_string(),
                            ReferenceKind::Inherits,
                            file_path,
                            child.start_pos().line(),
                        );
                    }
                }
            }
            "php_use" => {
                let text = node.text().to_string();
                let trimmed = text.trim_start_matches("use ").trim_end_matches(';').trim();
                push_ref(
                    references,
                    &source_qn,
                    trimmed.to_string(),
                    ReferenceKind::Import,
                    file_path,
                    node.start_pos().line(),
                );
            }
            "php_base_clause" => {
                for child in node.children() {
                    let ck = child.kind();
                    if ck.as_ref() == "name" || ck.as_ref() == "qualified_name" {
                        push_ref(
                            references,
                            &source_qn,
                            child.text().to_string(),
                            ReferenceKind::Inherits,
                            file_path,
                            child.start_pos().line(),
                        );
                    }
                }
            }
            "php_implements" => {
                for child in node.children() {
                    let ck = child.kind();
                    if ck.as_ref() == "name" || ck.as_ref() == "qualified_name" {
                        push_ref(
                            references,
                            &source_qn,
                            child.text().to_string(),
                            ReferenceKind::Implements,
                            file_path,
                            child.start_pos().line(),
                        );
                    }
                }
            }
            "php_trait_use" => {
                for child in node.children() {
                    let ck = child.kind();
                    if ck.as_ref() == "name" || ck.as_ref() == "qualified_name" {
                        push_ref(
                            references,
                            &source_qn,
                            child.text().to_string(),
                            ReferenceKind::Import,
                            file_path,
                            child.start_pos().line(),
                        );
                    }
                }
            }
            "scala_import" => {
                let text = node.text().to_string();
                let trimmed = text.trim_start_matches("import ").trim();
                push_ref(
                    references,
                    &source_qn,
                    trimmed.to_string(),
                    ReferenceKind::Import,
                    file_path,
                    node.start_pos().line(),
                );
            }
            "scala_extends" => {
                for child in node.children() {
                    let ck = child.kind();
                    if ck.as_ref() == "type_identifier"
                        || ck.as_ref() == "stable_type_identifier"
                        || ck.as_ref() == "generic_type"
                    {
                        let name = if ck.as_ref() == "generic_type" {
                            child
                                .children()
                                .next()
                                .map(|n| n.text().to_string())
                                .unwrap_or_default()
                        } else {
                            child.text().to_string()
                        };
                        if !name.is_empty() {
                            push_ref(
                                references,
                                &source_qn,
                                name,
                                ReferenceKind::Inherits,
                                file_path,
                                child.start_pos().line(),
                            );
                        }
                    }
                }
            }
            "hcl_function_call" => {
                if let Some(name_node) = node.field("function") {
                    push_ref(
                        references,
                        &source_qn,
                        name_node.text().to_string(),
                        ReferenceKind::Call,
                        file_path,
                        node.start_pos().line(),
                    );
                }
            }
            _ => {}
        }
    }

    fn extract_ts_class_heritage<D: Doc>(
        &self,
        lang: &LanguageRules,
        node: &Node<'_, D>,
        _source: &str,
        file_path: &str,
        scope: &[String],
        references: &mut Vec<Reference>,
    ) where
        D::Lang: ast_grep_core::Language,
    {
        let class_name = self.get_node_field_text(node, "name").unwrap_or_default();
        let source_qn = if scope.is_empty() {
            class_name.clone()
        } else {
            format!("{}.{}", scope.join(lang.scope_separator), class_name)
        };

        for child in node.children() {
            if child.kind().as_ref() == "class_heritage" {
                for heritage_child in child.children() {
                    let hk = heritage_child.kind();
                    match hk.as_ref() {
                        "extends_clause" => {
                            self.extract_heritage_type_refs(
                                &heritage_child,
                                file_path,
                                &source_qn,
                                ReferenceKind::Inherits,
                                references,
                            );
                        }
                        "implements_clause" => {
                            self.extract_heritage_type_refs(
                                &heritage_child,
                                file_path,
                                &source_qn,
                                ReferenceKind::Implements,
                                references,
                            );
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    fn extract_heritage_type_refs<D: Doc>(
        &self,
        clause_node: &Node<'_, D>,
        file_path: &str,
        source_qn: &str,
        kind: ReferenceKind,
        references: &mut Vec<Reference>,
    ) where
        D::Lang: ast_grep_core::Language,
    {
        for child in clause_node.children() {
            let ck = child.kind();
            match ck.as_ref() {
                "identifier" | "type_identifier" | "member_expression" => {
                    let type_name = child.text().to_string();
                    if !type_name.is_empty() {
                        push_ref(
                            references,
                            source_qn,
                            type_name,
                            kind,
                            file_path,
                            child.start_pos().line(),
                        );
                    }
                }
                "generic_type" => {
                    let type_name = child
                        .field("name")
                        .map(|n| n.text().to_string())
                        .unwrap_or_else(|| child.text().to_string());
                    if !type_name.is_empty() {
                        push_ref(
                            references,
                            source_qn,
                            type_name,
                            kind,
                            file_path,
                            child.start_pos().line(),
                        );
                    }
                }
                _ => {}
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn extract_java_heritage<D: Doc>(
        &self,
        node: &Node<'_, D>,
        file_path: &str,
        scope: &[String],
        lang: &LanguageRules,
        kind: ReferenceKind,
        field_name: &str,
        references: &mut Vec<Reference>,
    ) where
        D::Lang: ast_grep_core::Language,
    {
        let class_name = self.get_node_field_text(node, "name").unwrap_or_default();
        let source_qn = build_qualified_name(scope, &class_name, lang.scope_separator);

        if let Some(heritage) = node.field(field_name) {
            for child in heritage.children() {
                let ck = child.kind();
                if ck.as_ref() == "type_identifier"
                    || ck.as_ref() == "generic_type"
                    || ck.as_ref() == "scoped_type_identifier"
                {
                    let type_name = if ck.as_ref() == "generic_type" {
                        child
                            .children()
                            .next()
                            .map(|n| n.text().to_string())
                            .unwrap_or_default()
                    } else {
                        child.text().to_string()
                    };
                    if !type_name.is_empty() {
                        push_ref(
                            references,
                            &source_qn,
                            type_name,
                            kind,
                            file_path,
                            child.start_pos().line(),
                        );
                    }
                }
            }
        }
    }
}

/// R1: Decompose a Rust `use` path into individual fully-qualified paths.
///
/// Handles:
/// - Simple paths: `std::collections::HashMap` -> `["std::collections::HashMap"]`
/// - Grouped: `std::collections::{HashMap, HashSet}` -> `["std::collections::HashMap", "std::collections::HashSet"]`
/// - Nested: `std::{collections::HashMap, io::{Read, Write}}` -> 4 paths
/// - Self: `std::collections::{self, HashMap}` -> `["std::collections", "std::collections::HashMap"]`
fn decompose_rust_use_path(path: &str) -> Vec<String> {
    let path = path.trim();
    if path.is_empty() {
        return vec![];
    }

    // Find the first `{` that starts a group
    if let Some(brace_pos) = find_top_level_brace(path) {
        // prefix = everything before `::{`
        let prefix = path[..brace_pos].trim_end_matches("::").trim().to_string();

        // Find matching closing brace
        let inner = &path[brace_pos + 1..];
        if let Some(close_pos) = find_matching_brace(inner) {
            let group_content = &inner[..close_pos];
            let segments = split_top_level_commas(group_content);
            let mut results = Vec::new();

            for segment in segments {
                let segment = segment.trim();
                if segment.is_empty() {
                    continue;
                }
                if segment == "self" {
                    // `use foo::{self}` means `foo` itself
                    if !prefix.is_empty() {
                        results.push(prefix.clone());
                    }
                } else {
                    // Recursively decompose each segment with the prefix
                    let full = if prefix.is_empty() {
                        segment.to_string()
                    } else {
                        format!("{}::{}", prefix, segment)
                    };
                    let sub = decompose_rust_use_path(&full);
                    results.extend(sub);
                }
            }
            return results;
        }
    }

    // No group: return the path as-is
    vec![path.to_string()]
}

/// Find the position of the first `{` that is not inside another brace group.
fn find_top_level_brace(s: &str) -> Option<usize> {
    let mut depth = 0;
    for (i, ch) in s.char_indices() {
        match ch {
            '{' if depth == 0 => return Some(i),
            '{' => depth += 1,
            '}' => depth -= 1,
            _ => {}
        }
    }
    None
}

/// Find the position of the matching `}` for a string that starts right after `{`.
fn find_matching_brace(s: &str) -> Option<usize> {
    let mut depth = 0;
    for (i, ch) in s.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' if depth == 0 => return Some(i),
            '}' => depth -= 1,
            _ => {}
        }
    }
    None
}

/// Split a string by commas, but only at the top level (not inside `{}`).
fn split_top_level_commas(s: &str) -> Vec<&str> {
    let mut segments = Vec::new();
    let mut depth = 0;
    let mut start = 0;

    for (i, ch) in s.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => depth -= 1,
            ',' if depth == 0 => {
                segments.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    // Push the last segment
    if start < s.len() {
        segments.push(&s[start..]);
    }

    segments
}
