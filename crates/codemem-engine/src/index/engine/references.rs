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
                push_ref(
                    references,
                    &source_qn,
                    trimmed.to_string(),
                    ReferenceKind::Import,
                    file_path,
                    node.start_pos().line(),
                );
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
                let text = node.text().to_string();
                let name = text.split('(').next().unwrap_or(&text).trim();
                let ref_kind = if special == "kotlin_delegation" {
                    ReferenceKind::Inherits
                } else {
                    ReferenceKind::Call
                };
                push_ref(
                    references,
                    &source_qn,
                    name.to_string(),
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
