//! Symbol and Reference types extracted from source code via tree-sitter.

use serde::{Deserialize, Serialize};

/// A function/method parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    /// Parameter name.
    pub name: String,
    /// Type annotation, if present.
    pub type_annotation: Option<String>,
    /// Default value expression, if present.
    pub default_value: Option<String>,
}

/// A code symbol extracted from source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symbol {
    /// Simple name (e.g., "add").
    pub name: String,
    /// Fully qualified name (e.g., "module::Struct::method").
    pub qualified_name: String,
    /// What kind of symbol this is.
    pub kind: SymbolKind,
    /// Full signature text (up to the opening brace or the whole item for short items).
    pub signature: String,
    /// Visibility of the symbol.
    pub visibility: Visibility,
    /// File path where the symbol is defined.
    pub file_path: String,
    /// 0-based starting line number.
    pub line_start: usize,
    /// 0-based ending line number.
    pub line_end: usize,
    /// Documentation comment, if any (e.g., `///` or `//!` in Rust).
    pub doc_comment: Option<String>,
    /// Qualified name of the parent symbol (e.g., struct name for a method).
    pub parent: Option<String>,
    /// Extracted parameters for functions/methods.
    #[serde(default)]
    pub parameters: Vec<Parameter>,
    /// Return type annotation, if present.
    #[serde(default)]
    pub return_type: Option<String>,
    /// Whether this is an async function/method.
    #[serde(default)]
    pub is_async: bool,
    /// Attributes, decorators, or annotations (e.g., `#[derive(Debug)]`, `@Override`).
    #[serde(default)]
    pub attributes: Vec<String>,
    /// Error/exception types this symbol can throw.
    #[serde(default)]
    pub throws: Vec<String>,
    /// Generic type parameters (e.g., `<T: Display>`).
    #[serde(default)]
    pub generic_params: Option<String>,
    /// Whether this is an abstract method (trait/interface method without body).
    #[serde(default)]
    pub is_abstract: bool,
}

/// The kind of a code symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SymbolKind {
    Function,
    Method,
    Class,
    Struct,
    Enum,
    Interface, // trait in Rust
    Type,      // type alias
    Constant,
    Module,
    Test,
    Field,       // struct/class field
    Property,    // getter/setter property
    Constructor, // __init__, constructor, new
    EnumVariant, // individual enum variant/member
    Macro,       // macro_rules!, C preprocessor macro
    Decorator,   // Python decorator, Java annotation definition
}

impl std::fmt::Display for SymbolKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Function => write!(f, "function"),
            Self::Method => write!(f, "method"),
            Self::Class => write!(f, "class"),
            Self::Struct => write!(f, "struct"),
            Self::Enum => write!(f, "enum"),
            Self::Interface => write!(f, "interface"),
            Self::Type => write!(f, "type"),
            Self::Constant => write!(f, "constant"),
            Self::Module => write!(f, "module"),
            Self::Test => write!(f, "test"),
            Self::Field => write!(f, "field"),
            Self::Property => write!(f, "property"),
            Self::Constructor => write!(f, "constructor"),
            Self::EnumVariant => write!(f, "enum_variant"),
            Self::Macro => write!(f, "macro"),
            Self::Decorator => write!(f, "decorator"),
        }
    }
}

/// Visibility of a symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Visibility {
    Public,
    Private,
    Crate,     // pub(crate) in Rust
    Protected, // for languages that have it
}

impl std::fmt::Display for Visibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Public => write!(f, "public"),
            Self::Private => write!(f, "private"),
            Self::Crate => write!(f, "crate"),
            Self::Protected => write!(f, "protected"),
        }
    }
}

/// A reference from one symbol/location to another symbol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reference {
    /// Qualified name of the symbol that contains this reference.
    pub source_qualified_name: String,
    /// Name of the referenced target (may be unresolved / simple name).
    pub target_name: String,
    /// What kind of reference this is.
    pub kind: ReferenceKind,
    /// File path where the reference occurs.
    pub file_path: String,
    /// 0-based line number of the reference.
    pub line: usize,
}

/// The kind of a reference between symbols.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReferenceKind {
    Call,
    Import,
    Inherits,
    Implements,
    TypeUsage,
}

impl From<SymbolKind> for codemem_core::NodeKind {
    fn from(kind: SymbolKind) -> Self {
        match kind {
            SymbolKind::Function => codemem_core::NodeKind::Function,
            SymbolKind::Method => codemem_core::NodeKind::Method,
            SymbolKind::Class => codemem_core::NodeKind::Class,
            SymbolKind::Struct => codemem_core::NodeKind::Class,
            SymbolKind::Enum => codemem_core::NodeKind::Class,
            SymbolKind::Interface => codemem_core::NodeKind::Interface,
            SymbolKind::Type => codemem_core::NodeKind::Type,
            SymbolKind::Constant => codemem_core::NodeKind::Constant,
            SymbolKind::Module => codemem_core::NodeKind::Module,
            SymbolKind::Test => codemem_core::NodeKind::Test,
            SymbolKind::Field => codemem_core::NodeKind::Constant,
            SymbolKind::Property => codemem_core::NodeKind::Constant,
            SymbolKind::Constructor => codemem_core::NodeKind::Method,
            SymbolKind::EnumVariant => codemem_core::NodeKind::Constant,
            SymbolKind::Macro => codemem_core::NodeKind::Function,
            SymbolKind::Decorator => codemem_core::NodeKind::Function,
        }
    }
}

impl std::fmt::Display for ReferenceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Call => write!(f, "call"),
            Self::Import => write!(f, "import"),
            Self::Inherits => write!(f, "inherits"),
            Self::Implements => write!(f, "implements"),
            Self::TypeUsage => write!(f, "type_usage"),
        }
    }
}
