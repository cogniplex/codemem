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
    Constructor, // __init__, constructor, new
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
            Self::Constructor => write!(f, "constructor"),
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
    Callback,
    Import,
    Inherits,
    Implements,
    TypeUsage,
}

impl std::str::FromStr for SymbolKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "function" => Ok(SymbolKind::Function),
            "method" => Ok(SymbolKind::Method),
            "class" => Ok(SymbolKind::Class),
            "struct" => Ok(SymbolKind::Struct),
            "enum" => Ok(SymbolKind::Enum),
            "interface" => Ok(SymbolKind::Interface),
            "type" => Ok(SymbolKind::Type),
            "constant" => Ok(SymbolKind::Constant),
            "module" => Ok(SymbolKind::Module),
            "test" => Ok(SymbolKind::Test),
            "field" => Ok(SymbolKind::Field),
            "constructor" => Ok(SymbolKind::Constructor),
            _ => Err(format!("Unknown SymbolKind: {s}")),
        }
    }
}

/// Parse a `SymbolKind` from its `Display` string (e.g. `"function"`, `"enum_variant"`).
pub fn symbol_kind_from_str(s: &str) -> Option<SymbolKind> {
    s.parse().ok()
}

/// Parse a `Visibility` from its `Display` string.
pub fn visibility_from_str(s: &str) -> Visibility {
    match s {
        "public" => Visibility::Public,
        "crate" => Visibility::Crate,
        "protected" => Visibility::Protected,
        _ => Visibility::Private,
    }
}

/// Lossy fallback: map a `NodeKind` back to a `SymbolKind`.
/// Several `SymbolKind` variants collapse into the same `NodeKind`, so this
/// mapping is not lossless (e.g. `NodeKind::Constant` could be Field/Constant).
/// Prefer `symbol_kind_from_str` with the stored
/// `"symbol_kind"` payload field for lossless round-trips.
fn symbol_kind_from_node_kind(nk: &codemem_core::NodeKind) -> SymbolKind {
    match nk {
        codemem_core::NodeKind::Function => SymbolKind::Function,
        codemem_core::NodeKind::Method => SymbolKind::Method,
        codemem_core::NodeKind::Class => SymbolKind::Class,
        codemem_core::NodeKind::Interface | codemem_core::NodeKind::Trait => SymbolKind::Interface,
        codemem_core::NodeKind::Type => SymbolKind::Type,
        codemem_core::NodeKind::Constant => SymbolKind::Constant,
        codemem_core::NodeKind::Module => SymbolKind::Module,
        codemem_core::NodeKind::Test => SymbolKind::Test,
        codemem_core::NodeKind::Enum => SymbolKind::Enum,
        codemem_core::NodeKind::Field | codemem_core::NodeKind::Property => SymbolKind::Field,
        _ => SymbolKind::Function, // safe fallback for non-symbol node kinds
    }
}

/// Reconstruct a `Symbol` from a persisted `sym:*` `GraphNode`.
///
/// Returns `None` if the node ID doesn't start with `"sym:"` or required
/// payload fields are missing.
pub fn symbol_from_graph_node(node: &codemem_core::GraphNode) -> Option<Symbol> {
    let qualified_name = node.id.strip_prefix("sym:")?.to_string();

    // Lossless kind from payload, lossy fallback from NodeKind
    let kind = node
        .payload
        .get("symbol_kind")
        .and_then(|v| v.as_str())
        .and_then(symbol_kind_from_str)
        .unwrap_or_else(|| symbol_kind_from_node_kind(&node.kind));

    let file_path = node
        .payload
        .get("file_path")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let signature = node
        .payload
        .get("signature")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let visibility = node
        .payload
        .get("visibility")
        .and_then(|v| v.as_str())
        .map(visibility_from_str)
        .unwrap_or(Visibility::Private);

    let line_start = node
        .payload
        .get("line_start")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    let line_end = node
        .payload
        .get("line_end")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    let doc_comment = node
        .payload
        .get("doc_comment")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    // Derive simple name from qualified name (last segment after "::")
    let name = qualified_name
        .rsplit("::")
        .next()
        .unwrap_or(&qualified_name)
        .to_string();

    let parameters: Vec<Parameter> = node
        .payload
        .get("parameters")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();

    let return_type = node
        .payload
        .get("return_type")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let is_async = node
        .payload
        .get("is_async")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let attributes: Vec<String> = node
        .payload
        .get("attributes")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();

    let throws: Vec<String> = node
        .payload
        .get("throws")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();

    let generic_params = node
        .payload
        .get("generic_params")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let is_abstract = node
        .payload
        .get("is_abstract")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let parent = node
        .payload
        .get("parent")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    Some(Symbol {
        name,
        qualified_name,
        kind,
        signature,
        visibility,
        file_path,
        line_start,
        line_end,
        doc_comment,
        parent,
        parameters,
        return_type,
        is_async,
        attributes,
        throws,
        generic_params,
        is_abstract,
    })
}

impl From<SymbolKind> for codemem_core::NodeKind {
    fn from(kind: SymbolKind) -> Self {
        match kind {
            SymbolKind::Function => codemem_core::NodeKind::Function,
            SymbolKind::Method => codemem_core::NodeKind::Method,
            SymbolKind::Class => codemem_core::NodeKind::Class,
            SymbolKind::Struct => codemem_core::NodeKind::Class,
            SymbolKind::Enum => codemem_core::NodeKind::Enum,
            SymbolKind::Interface => codemem_core::NodeKind::Interface,
            SymbolKind::Type => codemem_core::NodeKind::Type,
            SymbolKind::Constant => codemem_core::NodeKind::Constant,
            SymbolKind::Module => codemem_core::NodeKind::Module,
            SymbolKind::Test => codemem_core::NodeKind::Test,
            SymbolKind::Field => codemem_core::NodeKind::Field,
            SymbolKind::Constructor => codemem_core::NodeKind::Method,
        }
    }
}

impl std::fmt::Display for ReferenceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Call => write!(f, "call"),
            Self::Callback => write!(f, "callback"),
            Self::Import => write!(f, "import"),
            Self::Inherits => write!(f, "inherits"),
            Self::Implements => write!(f, "implements"),
            Self::TypeUsage => write!(f, "type_usage"),
        }
    }
}
