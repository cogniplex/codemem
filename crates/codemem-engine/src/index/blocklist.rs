//! Per-language call noise blocklist.
//!
//! Filters builtins and stdlib methods from reference extraction to prevent
//! structural noise from calls like `print`, `len`, `console.log`, `useState`, `println!`.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

/// Normalize language aliases to canonical names.
fn normalize_language(language: &str) -> &str {
    match language {
        "tsx" | "jsx" | "javascript" => "typescript",
        other => other,
    }
}

/// Returns `true` if `name` is a blocked builtin/stdlib call for the given language.
///
/// Used by the ast-grep extraction path. Only the simple (unqualified) name
/// should be passed — callers should split on the language's scope separator first.
pub fn is_blocked_call(language: &str, name: &str) -> bool {
    let lang = normalize_language(language);
    BLOCKLISTS.get(lang).is_some_and(|set| set.contains(name))
}

/// Returns `true` if the given SCIP symbol represents a blocked builtin/stdlib call.
///
/// Extracts the language from the SCIP package manager prefix and the simple
/// name from the descriptor chain.
pub fn is_blocked_call_scip(scip_symbol: &str) -> bool {
    let Some(lang) = scip_language(scip_symbol) else {
        return false;
    };
    let Some(simple) = scip_simple_name(scip_symbol) else {
        return false;
    };
    is_blocked_call(lang, simple)
}

/// Extract language from SCIP package manager prefix.
fn scip_language(symbol: &str) -> Option<&str> {
    // SCIP symbols start with "scip-<manager> " or "<manager> "
    let s = symbol.strip_prefix("scip-").unwrap_or(symbol);
    let prefix = s.split_whitespace().next()?;
    match prefix {
        "cargo" | "crates" => Some("rust"),
        "npm" | "node" => Some("typescript"),
        "pip" | "pypi" | "python" => Some("python"),
        "go" | "gomod" => Some("go"),
        "maven" | "gradle" => Some("java"),
        _ => None,
    }
}

/// Extract simple name from SCIP descriptor chain.
/// Splits on `#` and `.`, strips trailing `()`.
fn scip_simple_name(symbol: &str) -> Option<&str> {
    // Take the last segment after space-separated parts, then split descriptors
    let descriptor_part = symbol.rsplit_once(' ').map(|(_, d)| d).unwrap_or(symbol);

    // Split on `.` and `#`, take the last non-empty segment
    let last = descriptor_part.split(['.', '#']).rfind(|s| !s.is_empty())?;

    // Strip trailing "()" or "(anything)"
    let name = if let Some(idx) = last.find('(') {
        &last[..idx]
    } else {
        last
    };

    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

static BLOCKLISTS: LazyLock<HashMap<&'static str, HashSet<&'static str>>> = LazyLock::new(|| {
    let mut m = HashMap::new();

    m.insert(
        "python",
        HashSet::from([
            "print",
            "len",
            "range",
            "map",
            "filter",
            "sorted",
            "reversed",
            "list",
            "dict",
            "set",
            "str",
            "int",
            "float",
            "bool",
            "type",
            "super",
            "isinstance",
            "issubclass",
            "hasattr",
            "getattr",
            "setattr",
            "delattr",
            "open",
            "enumerate",
            "zip",
            "any",
            "all",
            "min",
            "max",
            "sum",
            "abs",
            "round",
            "repr",
            "id",
            "hash",
            "dir",
            "vars",
            "input",
            "format",
            "tuple",
            "frozenset",
            "bytes",
            "bytearray",
            "callable",
            "iter",
            "next",
            "property",
            "staticmethod",
            "classmethod",
            "append",
            "extend",
            "update",
            "pop",
            "get",
            "items",
            "keys",
            "values",
            "split",
            "join",
            "strip",
            "lstrip",
            "rstrip",
            "replace",
            "startswith",
            "endswith",
            "lower",
            "upper",
            "encode",
            "decode",
            "read",
            "write",
            "close",
            "flush",
            "seek",
        ]),
    );

    m.insert(
        "typescript",
        HashSet::from([
            "console",
            "setTimeout",
            "setInterval",
            "clearTimeout",
            "clearInterval",
            "JSON",
            "Array",
            "Object",
            "Promise",
            "Math",
            "Date",
            "Error",
            "Symbol",
            "parseInt",
            "parseFloat",
            "fetch",
            "require",
            "document",
            "window",
            "process",
            "Buffer",
            "URL",
            "URLSearchParams",
            "RegExp",
            "Map",
            "Set",
            "WeakMap",
            "WeakSet",
            "Proxy",
            "Reflect",
            "Number",
            "String",
            "Boolean",
            "log",
            "error",
            "warn",
            "info",
            "debug",
            "trace",
            "parse",
            "stringify",
            "assign",
            "freeze",
            "keys",
            "values",
            "entries",
            "isArray",
            "from",
            "of",
            "resolve",
            "reject",
            "race",
            "all",
            "allSettled",
            "floor",
            "ceil",
            "round",
            "random",
            "abs",
            "min",
            "max",
            "pow",
            "sqrt",
            "now",
            "toISOString",
            "toJSON",
            "push",
            "pop",
            "shift",
            "unshift",
            "splice",
            "slice",
            "concat",
            "map",
            "filter",
            "reduce",
            "forEach",
            "find",
            "findIndex",
            "some",
            "every",
            "includes",
            "indexOf",
            "join",
            "sort",
            "reverse",
            "flat",
            "flatMap",
            "then",
            "catch",
            "finally",
            "toString",
            "valueOf",
            "hasOwnProperty",
            "addEventListener",
            "removeEventListener",
            "preventDefault",
            "stopPropagation",
            "querySelector",
            "querySelectorAll",
            "getElementById",
            "createElement",
            "appendChild",
            "removeChild",
            "useState",
            "useEffect",
            "useRef",
            "useCallback",
            "useMemo",
            "useContext",
            "useReducer",
            "useLayoutEffect",
            "useImperativeHandle",
            "useDebugValue",
            "useId",
            "useTransition",
            "useDeferredValue",
            "useSyncExternalStore",
            "useInsertionEffect",
        ]),
    );

    m.insert(
        "rust",
        HashSet::from([
            "println!",
            "eprintln!",
            "print!",
            "eprint!",
            "format!",
            "write!",
            "writeln!",
            "vec!",
            "todo!",
            "unimplemented!",
            "unreachable!",
            "panic!",
            "assert!",
            "assert_eq!",
            "assert_ne!",
            "debug_assert!",
            "debug_assert_eq!",
            "debug_assert_ne!",
            "cfg!",
            "env!",
            "include!",
            "include_str!",
            "include_bytes!",
            "concat!",
            "stringify!",
            "file!",
            "line!",
            "column!",
            "module_path!",
            "dbg!",
            "matches!",
            "compile_error!",
            "trace!",
            "debug!",
            "info!",
            "warn!",
            "error!",
            "clone",
            "to_string",
            "to_owned",
            "into",
            "from",
            "default",
            "fmt",
            "eq",
            "ne",
            "cmp",
            "partial_cmp",
            "hash",
            "map",
            "filter",
            "and_then",
            "or_else",
            "unwrap",
            "unwrap_or",
            "unwrap_or_else",
            "unwrap_or_default",
            "expect",
            "ok",
            "err",
            "is_some",
            "is_none",
            "is_ok",
            "is_err",
            "collect",
            "iter",
            "into_iter",
            "next",
            "push",
            "pop",
            "insert",
            "remove",
            "contains",
            "get",
            "len",
            "is_empty",
        ]),
    );

    m.insert(
        "go",
        HashSet::from([
            "make", "len", "cap", "append", "copy", "delete", "close", "new", "panic", "recover",
            "print", "println", "complex", "real", "imag", "Error", "String", "Len", "Less",
            "Swap", "Read", "Write", "Close", "Seek", "Lock", "Unlock", "RLock", "RUnlock",
            "Println", "Printf", "Sprintf", "Fprintf", "Errorf",
        ]),
    );

    m.insert(
        "java",
        HashSet::from([
            "System.out.println",
            "System.out.print",
            "System.err.println",
            "toString",
            "equals",
            "hashCode",
            "compareTo",
            "clone",
            "println",
            "print",
            "printf",
            "get",
            "set",
            "add",
            "remove",
            "contains",
            "size",
            "isEmpty",
            "put",
            "containsKey",
            "containsValue",
            "keySet",
            "values",
            "entrySet",
            "length",
            "charAt",
            "substring",
            "indexOf",
            "trim",
            "split",
            "valueOf",
            "parseInt",
            "parseDouble",
            "parseLong",
            "close",
            "read",
            "write",
            "flush",
        ]),
    );

    m
});

#[cfg(test)]
#[path = "tests/blocklist_tests.rs"]
mod tests;
