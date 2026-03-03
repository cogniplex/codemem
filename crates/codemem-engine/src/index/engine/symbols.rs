//! Special-case symbol handlers and language-specific symbol extractors.

use super::build_symbol;
use crate::index::rule_loader::LanguageRules;
use crate::index::symbol::{Symbol, SymbolKind, Visibility};
use ast_grep_core::{Doc, Node};

impl super::AstGrepEngine {
    /// Handle special cases that produce multiple symbols (e.g. Go type/const/var blocks).
    /// Returns an empty Vec for non-multi specials so the caller falls through to `handle_special_symbol`.
    pub(super) fn handle_special_symbol_multi<D: Doc>(
        &self,
        lang: &LanguageRules,
        special: &str,
        node: &Node<'_, D>,
        source: &str,
        file_path: &str,
        scope: &[String],
    ) -> Vec<Symbol>
    where
        D::Lang: ast_grep_core::Language,
    {
        match special {
            "go_type_declaration" => {
                self.extract_go_type_declaration(lang, node, source, file_path, scope)
            }
            "go_const_declaration" => {
                self.extract_go_const_or_var(lang, node, source, file_path, scope, "const_spec")
            }
            "go_var_declaration" => {
                self.extract_go_const_or_var(lang, node, source, file_path, scope, "var_spec")
            }
            _ => Vec::new(),
        }
    }

    pub(super) fn handle_special_symbol<D: Doc>(
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
        match special {
            "python_constant" => self.extract_python_constant(lang, node, source, file_path, scope),
            "go_method" => self.extract_go_method(lang, node, source, file_path),
            "go_type_declaration" => {
                // Not a single symbol — handled in extract_symbols_recursive via special dispatch
                None
            }
            "go_const_declaration" | "go_var_declaration" => None,
            "ts_arrow_field" => self.extract_ts_arrow_field(lang, node, source, file_path, scope),
            "ts_lexical_arrow" => {
                self.extract_ts_lexical_arrow(lang, node, source, file_path, scope)
            }
            "java_static_final_field" => {
                self.extract_java_static_final_field(lang, node, source, file_path, scope)
            }
            "hcl_block" => self.extract_hcl_block(lang, node, source, file_path, scope),
            "hcl_attribute" => self.extract_hcl_attribute(lang, node, source, file_path, scope),
            "swift_class_declaration" => {
                self.extract_swift_class(lang, node, source, file_path, scope)
            }
            "kotlin_class" | "kotlin_object" | "kotlin_function" => {
                self.extract_kotlin_symbol(lang, special, node, source, file_path, scope)
            }
            "scala_final_val" => self.extract_scala_final_val(lang, node, source, file_path, scope),
            "cpp_type_alias" | "cpp_alias" => {
                self.extract_cpp_type_alias(lang, node, source, file_path, scope)
            }
            "cpp_define" => self.extract_cpp_define(lang, node, source, file_path, scope),
            "csharp_field" => self.extract_csharp_field(lang, node, source, file_path, scope),
            "java_field" => self.extract_java_field(lang, node, source, file_path, scope),
            "ruby_constant_assignment" => {
                self.extract_ruby_constant(lang, node, source, file_path, scope)
            }
            "php_property" => self.extract_php_property(lang, node, source, file_path, scope),
            "php_const" => self.extract_php_const(lang, node, source, file_path, scope),
            _ => None,
        }
    }

    // ── Language-Specific Symbol Extractors ────────────────────────────

    fn extract_python_constant<D: Doc>(
        &self,
        lang: &LanguageRules,
        node: &Node<'_, D>,
        _source: &str,
        file_path: &str,
        scope: &[String],
    ) -> Option<Symbol>
    where
        D::Lang: ast_grep_core::Language,
    {
        if !scope.is_empty() {
            return None;
        }
        let child = node.children().next()?;
        if child.kind().as_ref() != "assignment" {
            return None;
        }
        let left = child.field("left")?;
        if left.kind().as_ref() != "identifier" {
            return None;
        }
        let name = left.text().to_string();
        if name.len() < 2
            || !name
                .chars()
                .all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit())
        {
            return None;
        }
        let signature = child.text().to_string();
        let first_line = signature.lines().next().unwrap_or(&signature);

        Some(build_symbol(
            name,
            SymbolKind::Constant,
            first_line.to_string(),
            Visibility::Public,
            None,
            file_path,
            node.start_pos().line(),
            node.end_pos().line(),
            scope,
            lang.scope_separator,
        ))
    }

    fn extract_go_method<D: Doc>(
        &self,
        lang: &LanguageRules,
        node: &Node<'_, D>,
        source: &str,
        file_path: &str,
    ) -> Option<Symbol>
    where
        D::Lang: ast_grep_core::Language,
    {
        let name = self.get_node_field_text(node, "name")?;
        let receiver_type = self.get_go_receiver_type(node);
        let qualified_name = if let Some(ref recv) = receiver_type {
            format!("{}.{}", recv, name)
        } else {
            name.clone()
        };
        let visibility = if name.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
            Visibility::Public
        } else {
            Visibility::Private
        };
        let signature = self.extract_signature(lang.name, node, source);
        let doc_comment = self.extract_doc_comment(lang.name, node, source);

        Some(Symbol {
            name,
            qualified_name,
            kind: SymbolKind::Method,
            signature,
            visibility,
            file_path: file_path.to_string(),
            line_start: node.start_pos().line(),
            line_end: node.end_pos().line(),
            doc_comment,
            parent: receiver_type,
        })
    }

    pub(super) fn get_go_receiver_type<D: Doc>(&self, node: &Node<'_, D>) -> Option<String>
    where
        D::Lang: ast_grep_core::Language,
    {
        let receiver = node.field("receiver")?;
        for child in receiver.children() {
            if child.kind().as_ref() == "parameter_declaration" {
                if let Some(type_node) = child.field("type") {
                    let type_text = type_node.text().to_string();
                    return Some(type_text.trim_start_matches('*').to_string());
                }
            }
        }
        None
    }

    /// Extract Go type declarations: `type Foo struct { ... }`, `type Bar interface { ... }`, etc.
    /// A single `type_declaration` may contain multiple `type_spec` children (grouped `type (...)`).
    fn extract_go_type_declaration<D: Doc>(
        &self,
        lang: &LanguageRules,
        node: &Node<'_, D>,
        source: &str,
        file_path: &str,
        scope: &[String],
    ) -> Vec<Symbol>
    where
        D::Lang: ast_grep_core::Language,
    {
        let mut symbols = Vec::new();
        for child in node.children() {
            let ck = child.kind();
            let ck_str = ck.as_ref();
            if ck_str == "type_spec" {
                // Single: type_declaration → type_spec
                if let Some(sym) = self.extract_go_type_spec(lang, &child, source, file_path, scope)
                {
                    symbols.push(sym);
                }
            } else if ck_str == "type_spec_list" {
                // Grouped: type_declaration → type_spec_list → type_spec*
                for spec in child.children() {
                    if spec.kind().as_ref() == "type_spec" {
                        if let Some(sym) =
                            self.extract_go_type_spec(lang, &spec, source, file_path, scope)
                        {
                            symbols.push(sym);
                        }
                    }
                }
            }
        }
        symbols
    }

    /// Extract a single Go type_spec into a Symbol.
    fn extract_go_type_spec<D: Doc>(
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
        let name = self.get_node_field_text(node, "name")?;
        if name.is_empty() {
            return None;
        }

        // Determine kind from the type value
        let mut kind = SymbolKind::Type;
        if let Some(type_node) = node.field("type") {
            let type_kind = type_node.kind();
            match type_kind.as_ref() {
                "struct_type" => kind = SymbolKind::Struct,
                "interface_type" => kind = SymbolKind::Interface,
                _ => {}
            }
        }

        let visibility = if name.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
            Visibility::Public
        } else {
            Visibility::Private
        };
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

    /// Extract Go const/var declarations. Each contains one or more `const_spec`/`var_spec` children.
    fn extract_go_const_or_var<D: Doc>(
        &self,
        lang: &LanguageRules,
        node: &Node<'_, D>,
        source: &str,
        file_path: &str,
        scope: &[String],
        spec_kind: &str,
    ) -> Vec<Symbol>
    where
        D::Lang: ast_grep_core::Language,
    {
        let list_kind = format!("{spec_kind}_list");
        let mut symbols = Vec::new();
        for child in node.children() {
            let ck = child.kind();
            let ck_str = ck.as_ref();
            if ck_str == spec_kind {
                // Single declaration: const_declaration → const_spec
                if let Some(sym) =
                    self.extract_go_const_var_spec(lang, &child, source, file_path, scope)
                {
                    symbols.push(sym);
                }
            } else if ck_str == list_kind {
                // Grouped declaration: const_declaration → const_spec_list → const_spec*
                for spec in child.children() {
                    if spec.kind().as_ref() == spec_kind {
                        if let Some(sym) =
                            self.extract_go_const_var_spec(lang, &spec, source, file_path, scope)
                        {
                            symbols.push(sym);
                        }
                    }
                }
            }
        }
        symbols
    }

    /// Extract a single Go const_spec or var_spec into a Symbol.
    fn extract_go_const_var_spec<D: Doc>(
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
        let name = self.get_node_field_text(node, "name")?;
        if name.is_empty() {
            return None;
        }
        let visibility = if name.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
            Visibility::Public
        } else {
            Visibility::Private
        };
        let signature = self.extract_signature(lang.name, node, source);
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

    /// Extract C# field declarations. The name lives inside `variable_declaration` > `variable_declarator`.
    fn extract_csharp_field<D: Doc>(
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
            }
        }
        None
    }

    fn extract_ts_arrow_field<D: Doc>(
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
        // public_field_definition with arrow function value
        let value = node.field("value")?;
        let vk = value.kind();
        if vk.as_ref() != "arrow_function" && vk.as_ref() != "function_expression" {
            return None;
        }
        let name = self.get_node_field_text(node, "name")?;
        let visibility = self.detect_visibility(lang.name, node, source, &name);
        let signature = self.extract_signature(lang.name, node, source);
        let doc_comment = self.extract_doc_comment(lang.name, node, source);

        Some(build_symbol(
            name,
            SymbolKind::Method,
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

    fn extract_ts_lexical_arrow<D: Doc>(
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
        for child in node.children() {
            if child.kind().as_ref() == "variable_declarator" {
                let value = child.field("value")?;
                let vk = value.kind();
                if vk.as_ref() != "arrow_function" && vk.as_ref() != "function_expression" {
                    return None;
                }
                let name = self.get_node_field_text(&child, "name")?;
                let exported = node
                    .parent()
                    .is_some_and(|p| p.kind().as_ref() == "export_statement");
                let visibility = if exported {
                    Visibility::Public
                } else {
                    Visibility::Private
                };
                let signature = self.extract_signature(lang.name, node, source);
                let doc_comment = self.extract_doc_comment(lang.name, node, source);

                return Some(build_symbol(
                    name,
                    SymbolKind::Function,
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

    fn extract_java_static_final_field<D: Doc>(
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
        let text = node.text().to_string();
        if !(text.contains("static") && text.contains("final")) {
            return None;
        }
        // Find the variable declarator to get the name
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
    fn extract_java_field<D: Doc>(
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

    /// Extract Ruby constants — only assignments where the left side starts with uppercase.
    fn extract_ruby_constant<D: Doc>(
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
        let left = node.field("left")?;
        let name = left.text().to_string();
        // Ruby constants must start with uppercase
        if !name.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
            return None;
        }
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

    /// Extract PHP property declarations — `$name` in property_element children.
    fn extract_php_property<D: Doc>(
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
        // property_declaration → property_element → variable_name ($name)
        for child in node.children() {
            if child.kind().as_ref() == "property_element" {
                for vc in child.children() {
                    if vc.kind().as_ref() == "variable_name" {
                        let name = vc.text().to_string();
                        let clean_name = name.trim_start_matches('$');
                        let visibility =
                            self.detect_visibility(lang.name, node, source, clean_name);
                        let signature = node.text().lines().next().unwrap_or("").trim().to_string();
                        let doc_comment = self.extract_doc_comment(lang.name, node, source);

                        return Some(build_symbol(
                            clean_name.to_string(),
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
            }
        }
        None
    }

    /// Extract PHP class constants — `const NAME = value`.
    fn extract_php_const<D: Doc>(
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
        // const_declaration → const_element → name
        for child in node.children() {
            if child.kind().as_ref() == "const_element" {
                let name = self.get_node_field_text(&child, "name").or_else(|| {
                    child
                        .children()
                        .find(|c| c.kind().as_ref() == "name")
                        .map(|c| c.text().to_string())
                })?;
                let visibility = self.detect_visibility(lang.name, node, source, &name);
                let signature = node.text().lines().next().unwrap_or("").trim().to_string();
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

    fn extract_hcl_block<D: Doc>(
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

    fn extract_hcl_attribute<D: Doc>(
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

    fn extract_swift_class<D: Doc>(
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

    fn extract_kotlin_symbol<D: Doc>(
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
        // Kotlin: find name via field or by scanning children (grammar quirks).
        // The Kotlin tree-sitter grammar uses `type_identifier` for class/interface names
        // and `simple_identifier` for function names.
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
                // Kotlin uses class_declaration for both classes and interfaces.
                // Check for "interface" keyword child to distinguish.
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

    fn extract_scala_final_val<D: Doc>(
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

    fn extract_cpp_type_alias<D: Doc>(
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

    fn extract_cpp_define<D: Doc>(
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
