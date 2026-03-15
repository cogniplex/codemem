//! Compile-time YAML rule embedding and deserialization for per-language extraction rules.

use ast_grep_language::SupportLang;
use codemem_core::CodememError;
use serde::Deserialize;
use std::collections::HashMap;

/// A symbol extraction rule from YAML.
#[derive(Debug, Clone, Deserialize)]
pub struct SymbolRule {
    pub kind: String,
    pub symbol_kind: String,
    #[serde(default = "default_name_field")]
    pub name_field: String,
    #[serde(default)]
    pub method_when_scoped: bool,
    #[serde(default)]
    pub is_scope: bool,
    #[serde(default)]
    pub special: Option<String>,
}

/// A scope container rule from YAML.
#[derive(Debug, Clone, Deserialize)]
pub struct ScopeContainerRule {
    pub kind: String,
    #[serde(default = "default_name_field")]
    pub name_field: String,
    #[serde(default = "default_body_field")]
    pub body_field: String,
    #[serde(default)]
    pub is_method_scope: bool,
    #[serde(default)]
    pub special: Option<String>,
}

/// A reference extraction rule from YAML.
#[derive(Debug, Clone, Deserialize)]
pub struct ReferenceRule {
    pub kind: String,
    pub reference_kind: String,
    #[serde(default)]
    pub name_field: Option<String>,
    #[serde(default)]
    pub special: Option<String>,
}

/// Deserialized symbols YAML file.
#[derive(Debug, Clone, Deserialize)]
pub struct SymbolRulesFile {
    pub symbols: Vec<SymbolRule>,
    #[serde(default)]
    pub scope_containers: Vec<ScopeContainerRule>,
    #[serde(default)]
    pub unwrap_nodes: Vec<String>,
}

/// Deserialized references YAML file.
#[derive(Debug, Clone, Deserialize)]
pub struct ReferenceRulesFile {
    pub references: Vec<ReferenceRule>,
    #[serde(default)]
    pub scope_containers: Vec<ScopeContainerRule>,
    #[serde(default)]
    pub unwrap_nodes: Vec<String>,
}

fn default_name_field() -> String {
    "name".to_string()
}

fn default_body_field() -> String {
    "body".to_string()
}

/// Compiled rules for a single language, ready for the engine.
pub struct LanguageRules {
    pub name: &'static str,
    pub lang: SupportLang,
    pub extensions: &'static [&'static str],
    pub scope_separator: &'static str,
    pub symbol_rules: Vec<SymbolRule>,
    pub symbol_scope_containers: Vec<ScopeContainerRule>,
    pub symbol_unwrap_nodes: Vec<String>,
    pub reference_rules: Vec<ReferenceRule>,
    pub reference_scope_containers: Vec<ScopeContainerRule>,
    pub reference_unwrap_nodes: Vec<String>,
    /// Index: node_kind → list of symbol rules
    pub symbol_index: HashMap<String, Vec<usize>>,
    /// Index: node_kind → list of reference rules
    pub reference_index: HashMap<String, Vec<usize>>,
    /// Index: node_kind → scope container index (symbols)
    pub symbol_scope_index: HashMap<String, usize>,
    /// Index: node_kind → scope container index (references)
    pub reference_scope_index: HashMap<String, usize>,
    /// Set of node kinds to unwrap (symbols)
    pub symbol_unwrap_set: std::collections::HashSet<String>,
    /// Set of node kinds to unwrap (references)
    pub reference_unwrap_set: std::collections::HashSet<String>,
}

impl LanguageRules {
    fn build_indexes(&mut self) {
        // Symbol rule index
        for (i, rule) in self.symbol_rules.iter().enumerate() {
            self.symbol_index
                .entry(rule.kind.clone())
                .or_default()
                .push(i);
        }
        // Reference rule index
        for (i, rule) in self.reference_rules.iter().enumerate() {
            self.reference_index
                .entry(rule.kind.clone())
                .or_default()
                .push(i);
        }
        // Symbol scope container index
        for (i, sc) in self.symbol_scope_containers.iter().enumerate() {
            if let Some(prev) = self.symbol_scope_index.insert(sc.kind.clone(), i) {
                debug_assert!(
                    false,
                    "Duplicate symbol scope container kind '{}': index {} overwrites {}",
                    sc.kind, i, prev
                );
            }
        }
        // Reference scope container index
        for (i, sc) in self.reference_scope_containers.iter().enumerate() {
            if let Some(prev) = self.reference_scope_index.insert(sc.kind.clone(), i) {
                debug_assert!(
                    false,
                    "Duplicate reference scope container kind '{}': index {} overwrites {}",
                    sc.kind, i, prev
                );
            }
        }
        // Unwrap sets
        self.symbol_unwrap_set = self.symbol_unwrap_nodes.iter().cloned().collect();
        self.reference_unwrap_set = self.reference_unwrap_nodes.iter().cloned().collect();
    }
}

/// Raw embedded rules before deserialization.
struct EmbeddedRules {
    name: &'static str,
    lang: SupportLang,
    extensions: &'static [&'static str],
    scope_separator: &'static str,
    symbols_yaml: &'static str,
    references_yaml: &'static str,
}

/// All language rule definitions embedded at compile time.
static LANGUAGE_RULES: &[EmbeddedRules] = &[
    EmbeddedRules {
        name: "rust",
        lang: SupportLang::Rust,
        extensions: &["rs"],
        scope_separator: "::",
        symbols_yaml: include_str!("../../rules/rust/symbols.yml"),
        references_yaml: include_str!("../../rules/rust/references.yml"),
    },
    EmbeddedRules {
        name: "typescript",
        lang: SupportLang::TypeScript,
        extensions: &["ts"],
        scope_separator: ".",
        symbols_yaml: include_str!("../../rules/typescript/symbols.yml"),
        references_yaml: include_str!("../../rules/typescript/references.yml"),
    },
    // TSX shares rules with TypeScript but uses a different SupportLang
    EmbeddedRules {
        name: "tsx",
        lang: SupportLang::Tsx,
        extensions: &["tsx", "jsx"],
        scope_separator: ".",
        symbols_yaml: include_str!("../../rules/typescript/symbols.yml"),
        references_yaml: include_str!("../../rules/typescript/references.yml"),
    },
    // JavaScript also uses TypeScript/TSX grammar
    EmbeddedRules {
        name: "javascript",
        lang: SupportLang::JavaScript,
        extensions: &["js"],
        scope_separator: ".",
        symbols_yaml: include_str!("../../rules/typescript/symbols.yml"),
        references_yaml: include_str!("../../rules/typescript/references.yml"),
    },
    EmbeddedRules {
        name: "python",
        lang: SupportLang::Python,
        extensions: &["py"],
        scope_separator: ".",
        symbols_yaml: include_str!("../../rules/python/symbols.yml"),
        references_yaml: include_str!("../../rules/python/references.yml"),
    },
    EmbeddedRules {
        name: "go",
        lang: SupportLang::Go,
        extensions: &["go"],
        scope_separator: ".",
        symbols_yaml: include_str!("../../rules/go/symbols.yml"),
        references_yaml: include_str!("../../rules/go/references.yml"),
    },
    EmbeddedRules {
        name: "java",
        lang: SupportLang::Java,
        extensions: &["java"],
        scope_separator: ".",
        symbols_yaml: include_str!("../../rules/java/symbols.yml"),
        references_yaml: include_str!("../../rules/java/references.yml"),
    },
    EmbeddedRules {
        name: "cpp",
        lang: SupportLang::Cpp,
        extensions: &["c", "h", "cpp", "hpp", "cc", "cxx", "hxx"],
        scope_separator: "::",
        symbols_yaml: include_str!("../../rules/cpp/symbols.yml"),
        references_yaml: include_str!("../../rules/cpp/references.yml"),
    },
    EmbeddedRules {
        name: "csharp",
        lang: SupportLang::CSharp,
        extensions: &["cs"],
        scope_separator: ".",
        symbols_yaml: include_str!("../../rules/csharp/symbols.yml"),
        references_yaml: include_str!("../../rules/csharp/references.yml"),
    },
    EmbeddedRules {
        name: "ruby",
        lang: SupportLang::Ruby,
        extensions: &["rb"],
        scope_separator: "::",
        symbols_yaml: include_str!("../../rules/ruby/symbols.yml"),
        references_yaml: include_str!("../../rules/ruby/references.yml"),
    },
    EmbeddedRules {
        name: "kotlin",
        lang: SupportLang::Kotlin,
        extensions: &["kt", "kts"],
        scope_separator: ".",
        symbols_yaml: include_str!("../../rules/kotlin/symbols.yml"),
        references_yaml: include_str!("../../rules/kotlin/references.yml"),
    },
    EmbeddedRules {
        name: "swift",
        lang: SupportLang::Swift,
        extensions: &["swift"],
        scope_separator: ".",
        symbols_yaml: include_str!("../../rules/swift/symbols.yml"),
        references_yaml: include_str!("../../rules/swift/references.yml"),
    },
    EmbeddedRules {
        name: "php",
        lang: SupportLang::Php,
        extensions: &["php"],
        scope_separator: "::",
        symbols_yaml: include_str!("../../rules/php/symbols.yml"),
        references_yaml: include_str!("../../rules/php/references.yml"),
    },
    EmbeddedRules {
        name: "scala",
        lang: SupportLang::Scala,
        extensions: &["scala", "sc"],
        scope_separator: ".",
        symbols_yaml: include_str!("../../rules/scala/symbols.yml"),
        references_yaml: include_str!("../../rules/scala/references.yml"),
    },
    EmbeddedRules {
        name: "hcl",
        lang: SupportLang::Hcl,
        extensions: &["tf", "hcl", "tfvars"],
        scope_separator: ".",
        symbols_yaml: include_str!("../../rules/hcl/symbols.yml"),
        references_yaml: include_str!("../../rules/hcl/references.yml"),
    },
    EmbeddedRules {
        name: "bash",
        lang: SupportLang::Bash,
        extensions: &["sh", "bash"],
        scope_separator: ".",
        symbols_yaml: include_str!("../../rules/bash/symbols.yml"),
        references_yaml: include_str!("../../rules/bash/references.yml"),
    },
];

/// Load and deserialize all language rules.
///
/// Returns an error if any embedded YAML rule file fails to deserialize.
pub fn load_all_rules() -> Result<Vec<LanguageRules>, CodememError> {
    LANGUAGE_RULES
        .iter()
        .map(|embedded| {
            let sym_file: SymbolRulesFile =
                serde_yaml::from_str(embedded.symbols_yaml).map_err(|e| {
                    CodememError::Config(format!(
                        "Failed to parse symbols.yml for {}: {}",
                        embedded.name, e
                    ))
                })?;
            let ref_file: ReferenceRulesFile = serde_yaml::from_str(embedded.references_yaml)
                .map_err(|e| {
                    CodememError::Config(format!(
                        "Failed to parse references.yml for {}: {}",
                        embedded.name, e
                    ))
                })?;

            let mut rules = LanguageRules {
                name: embedded.name,
                lang: embedded.lang,
                extensions: embedded.extensions,
                scope_separator: embedded.scope_separator,
                symbol_rules: sym_file.symbols,
                symbol_scope_containers: sym_file.scope_containers,
                symbol_unwrap_nodes: sym_file.unwrap_nodes,
                reference_rules: ref_file.references,
                reference_scope_containers: ref_file.scope_containers,
                reference_unwrap_nodes: ref_file.unwrap_nodes,
                symbol_index: HashMap::new(),
                reference_index: HashMap::new(),
                symbol_scope_index: HashMap::new(),
                reference_scope_index: HashMap::new(),
                symbol_unwrap_set: std::collections::HashSet::new(),
                reference_unwrap_set: std::collections::HashSet::new(),
            };
            rules.build_indexes();
            Ok(rules)
        })
        .collect()
}
