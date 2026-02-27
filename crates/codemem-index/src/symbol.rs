//! Symbol and Reference types extracted from source code via tree-sitter.

use serde::{Deserialize, Serialize};

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReferenceKind {
    Call,
    Import,
    Inherits,
    Implements,
    TypeUsage,
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
