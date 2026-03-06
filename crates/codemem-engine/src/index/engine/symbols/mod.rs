//! Special-case symbol handlers and language-specific symbol extractors.

mod go;
mod java_like;
mod other;
mod scripting;
mod typescript;

use crate::index::rule_loader::LanguageRules;
use crate::index::symbol::Symbol;
use ast_grep_core::{Doc, Node};

impl crate::index::engine::AstGrepEngine {
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
}
