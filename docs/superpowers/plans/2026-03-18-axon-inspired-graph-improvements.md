# Axon-Inspired Graph Quality Improvements

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve call graph accuracy across both ast-grep and SCIP paths by adding a call noise blocklist, callback/higher-order function detection, and a dead code detection enrichment pass — inspired by patterns from the Axon project.

**Architecture:** Four independent improvements: (1) a compile-time call blocklist that filters noise references before resolution (ast-grep path), (2) the same blocklist applied to the SCIP graph builder, (3) new YAML reference rules + special handlers for callback arguments, (4) a new dead code enrichment analyzer that uses the graph to find unreachable symbols. Each builds on existing infrastructure — YAML rules, special handlers, graph builder, enrichment module pattern.

**Tech Stack:** Rust, ast-grep, tree-sitter, YAML rules, SCIP graph builder, existing enrichment pipeline

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `crates/codemem-engine/src/index/blocklist.rs` | Create | Call noise blocklist: per-language sets of builtin/stdlib names to skip |
| `crates/codemem-engine/src/index/engine/mod.rs` | Modify | Wire blocklist filter into `extract_references_from_tree` |
| `crates/codemem-engine/src/index/mod.rs` | Modify | Add `pub mod blocklist;` |
| `crates/codemem-engine/src/index/tests/blocklist_tests.rs` | Create | Blocklist unit tests |
| `crates/codemem-engine/src/index/scip/graph_builder.rs` | Modify | Wire blocklist into SCIP edge creation (after fan-out check, ~line 487) |
| `crates/codemem-engine/src/index/tests/scip_graph_builder_tests.rs` | Modify | Add SCIP blocklist tests |
| `crates/codemem-engine/rules/python/references.yml` | Modify | Add `argument` callback rule |
| `crates/codemem-engine/rules/typescript/references.yml` | Modify | Add `arguments` callback rule |
| `crates/codemem-engine/src/index/engine/references.rs` | Modify | Add `callback_args` special handler |
| `crates/codemem-engine/src/index/symbol.rs` | Modify | Add `ReferenceKind::Callback` variant + `Display` impl |
| `crates/codemem-engine/src/index/resolver.rs` | Modify | Map `Callback` → `Calls` with 0.6 confidence cap |
| `crates/codemem-engine/src/index/tests/engine_references_tests.rs` | Modify | Add callback extraction tests |
| `crates/codemem-engine/src/index/tests/resolver_tests.rs` | Modify | Add callback confidence tests |
| `crates/codemem-engine/src/enrichment/dead_code.rs` | Create | Dead code detection enrichment analyzer |
| `crates/codemem-engine/src/enrichment/mod.rs` | Modify | Add `mod dead_code;` |
| `crates/codemem-core/src/config.rs` | Modify | Add `DeadCodeConfig` to `EnrichmentConfig` |

---

## Task 1: Call Noise Blocklist (ast-grep path)

Filter out calls to language builtins and stdlib methods that add no structural value to the graph (e.g., `print`, `len`, `console.log`, `useState`). Applied during reference extraction, before resolution.

**Files:**
- Create: `crates/codemem-engine/src/index/blocklist.rs`
- Create: `crates/codemem-engine/src/index/tests/blocklist_tests.rs`
- Modify: `crates/codemem-engine/src/index/mod.rs`
- Modify: `crates/codemem-engine/src/index/engine/mod.rs:123-131`

- [ ] **Step 1: Write failing tests for the blocklist module**

Create `crates/codemem-engine/src/index/tests/blocklist_tests.rs`:

```rust
use crate::index::blocklist::is_blocked_call;

#[test]
fn python_builtins_blocked() {
    assert!(is_blocked_call("python", "print"));
    assert!(is_blocked_call("python", "len"));
    assert!(is_blocked_call("python", "range"));
    assert!(is_blocked_call("python", "isinstance"));
    assert!(is_blocked_call("python", "enumerate"));
}

#[test]
fn python_stdlib_methods_blocked() {
    assert!(is_blocked_call("python", "append"));
    assert!(is_blocked_call("python", "strip"));
    assert!(is_blocked_call("python", "join"));
}

#[test]
fn ts_globals_blocked() {
    assert!(is_blocked_call("typescript", "console"));
    assert!(is_blocked_call("typescript", "setTimeout"));
    assert!(is_blocked_call("typescript", "parseInt"));
    assert!(is_blocked_call("typescript", "fetch"));
}

#[test]
fn ts_react_hooks_blocked() {
    assert!(is_blocked_call("typescript", "useState"));
    assert!(is_blocked_call("typescript", "useEffect"));
    assert!(is_blocked_call("typescript", "useRef"));
    assert!(is_blocked_call("typescript", "useCallback"));
}

#[test]
fn ts_dotted_methods_blocked() {
    assert!(is_blocked_call("typescript", "log"));
    assert!(is_blocked_call("typescript", "stringify"));
    assert!(is_blocked_call("typescript", "parse"));
}

#[test]
fn rust_builtins_blocked() {
    assert!(is_blocked_call("rust", "println!"));
    assert!(is_blocked_call("rust", "eprintln!"));
    assert!(is_blocked_call("rust", "format!"));
    assert!(is_blocked_call("rust", "vec!"));
    assert!(is_blocked_call("rust", "todo!"));
    assert!(is_blocked_call("rust", "unimplemented!"));
    assert!(is_blocked_call("rust", "assert!"));
    assert!(is_blocked_call("rust", "assert_eq!"));
}

#[test]
fn go_builtins_blocked() {
    assert!(is_blocked_call("go", "make"));
    assert!(is_blocked_call("go", "len"));
    assert!(is_blocked_call("go", "cap"));
    assert!(is_blocked_call("go", "append"));
    assert!(is_blocked_call("go", "panic"));
    assert!(is_blocked_call("go", "close"));
}

#[test]
fn user_symbols_not_blocked() {
    assert!(!is_blocked_call("python", "process_data"));
    assert!(!is_blocked_call("typescript", "handleSubmit"));
    assert!(!is_blocked_call("rust", "parse_config"));
    assert!(!is_blocked_call("go", "StartServer"));
}

#[test]
fn unknown_language_blocks_nothing() {
    assert!(!is_blocked_call("haskell", "print"));
}

#[test]
fn shared_languages_use_same_blocklist() {
    // TSX, JSX, JS all use the TS blocklist
    assert!(is_blocked_call("tsx", "useState"));
    assert!(is_blocked_call("javascript", "console"));
}

#[test]
fn scip_package_manager_to_language() {
    // is_blocked_call_scip uses SCIP symbol prefix to detect language
    use crate::index::blocklist::is_blocked_call_scip;
    // cargo = rust
    assert!(is_blocked_call_scip("rust-analyzer cargo std/ fmt#Display#fmt()."));
    // npm = typescript
    assert!(is_blocked_call_scip("scip-typescript npm . console#log()."));
    // pip = python
    assert!(is_blocked_call_scip("scip-python pip builtins/ print()."));
    // user symbols not blocked
    assert!(!is_blocked_call_scip("rust-analyzer cargo my-crate/ auth#validate()."));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p codemem-engine blocklist -- --nocapture`
Expected: FAIL — module `blocklist` not found

- [ ] **Step 3: Implement the blocklist module**

Create `crates/codemem-engine/src/index/blocklist.rs`:

```rust
//! Call noise blocklist: per-language sets of builtin/stdlib names to skip
//! during reference extraction. Prevents structural noise from builtins
//! like `print`, `len`, `console.log`, `useState` etc.
//!
//! Two entry points:
//! - `is_blocked_call(language, name)` — for ast-grep path (language known)
//! - `is_blocked_call_scip(scip_symbol)` — for SCIP path (language inferred from symbol)

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

/// Check if a call target name should be filtered out for a given language.
///
/// Returns `true` if the name is a known builtin/stdlib call that adds no
/// structural value to the graph. The `language` parameter should match
/// the `LanguageRules.name` field (e.g., "python", "typescript", "rust").
pub fn is_blocked_call(language: &str, name: &str) -> bool {
    let key = normalize_language(language);
    BLOCKLISTS
        .get(key)
        .is_some_and(|set| set.contains(name))
}

/// Check if a SCIP symbol string references a blocked call.
///
/// Extracts the language from the SCIP package manager prefix and the
/// simple symbol name from the descriptor, then checks the blocklist.
///
/// SCIP symbol format: `<tool> <manager> <package>/ <descriptors>`
/// e.g., `rust-analyzer cargo std/ fmt#Display#fmt().`
pub fn is_blocked_call_scip(scip_symbol: &str) -> bool {
    let Some(lang) = language_from_scip_symbol(scip_symbol) else {
        return false;
    };
    let simple = simple_name_from_scip_symbol(scip_symbol);
    is_blocked_call(lang, &simple)
}

/// Normalize language name aliases to canonical blocklist keys.
fn normalize_language(language: &str) -> &str {
    match language {
        "tsx" | "jsx" | "javascript" => "typescript",
        other => other,
    }
}

/// Extract language from SCIP symbol's package manager field.
/// SCIP symbols start with `<tool> <manager> ...`
/// Manager → language mapping: cargo→rust, npm→typescript, pip→python, go→go, maven→java
fn language_from_scip_symbol(scip_symbol: &str) -> Option<&'static str> {
    let parts: Vec<&str> = scip_symbol.splitn(3, ' ').collect();
    if parts.len() < 2 {
        return None;
    }
    match parts[1] {
        "cargo" => Some("rust"),
        "npm" => Some("typescript"),
        "pip" | "pypi" => Some("python"),
        "go" => Some("go"),
        "maven" | "gradle" => Some("java"),
        _ => None,
    }
}

/// Extract the simple (last) symbol name from a SCIP descriptor chain.
/// e.g., `rust-analyzer cargo std/ fmt#Display#fmt().` → `fmt`
/// e.g., `scip-typescript npm . console#log().` → `log`
fn simple_name_from_scip_symbol(scip_symbol: &str) -> String {
    // Find the descriptor portion (after the `<package>/` part)
    let descriptors = scip_symbol
        .rsplit('/')
        .next()
        .unwrap_or(scip_symbol)
        .trim();

    // Split on descriptor separators: # (type), . (term), () (method)
    // Take the last meaningful segment
    let mut name = String::new();
    let mut current = String::new();
    for ch in descriptors.chars() {
        match ch {
            '#' | '.' => {
                if !current.is_empty() {
                    name = std::mem::take(&mut current);
                }
            }
            '(' | ')' => {
                // Method suffix — ignore parens
            }
            _ => current.push(ch),
        }
    }
    if !current.is_empty() {
        name = current;
    }
    name
}

static BLOCKLISTS: LazyLock<HashMap<&'static str, HashSet<&'static str>>> = LazyLock::new(|| {
    let mut m = HashMap::new();

    m.insert("python", HashSet::from([
        // Builtins
        "print", "len", "range", "map", "filter", "sorted", "reversed",
        "list", "dict", "set", "str", "int", "float", "bool", "type",
        "super", "isinstance", "issubclass", "hasattr", "getattr", "setattr",
        "delattr", "open", "enumerate", "zip", "any", "all", "min", "max",
        "sum", "abs", "round", "repr", "id", "hash", "dir", "vars",
        "input", "format", "tuple", "frozenset", "bytes", "bytearray",
        "callable", "iter", "next", "property", "staticmethod", "classmethod",
        // Common method names that collide with stdlib
        "append", "extend", "update", "pop", "get", "items", "keys", "values",
        "split", "join", "strip", "lstrip", "rstrip", "replace", "startswith",
        "endswith", "lower", "upper", "encode", "decode", "read", "write",
        "close", "flush", "seek",
    ]));

    m.insert("typescript", HashSet::from([
        // Globals
        "console", "setTimeout", "setInterval", "clearTimeout", "clearInterval",
        "JSON", "Array", "Object", "Promise", "Math", "Date", "Error",
        "Symbol", "parseInt", "parseFloat", "fetch", "require",
        "document", "window", "process", "Buffer", "URL", "URLSearchParams",
        "RegExp", "Map", "Set", "WeakMap", "WeakSet", "Proxy", "Reflect",
        "Number", "String", "Boolean",
        // Dotted method names (extracted as bare names by ast-grep)
        "log", "error", "warn", "info", "debug", "trace",
        "parse", "stringify", "assign", "freeze", "keys", "values", "entries",
        "isArray", "from", "of", "resolve", "reject", "race", "all", "allSettled",
        "floor", "ceil", "round", "random", "abs", "min", "max", "pow", "sqrt",
        "now", "toISOString", "toJSON",
        "push", "pop", "shift", "unshift", "splice", "slice", "concat",
        "map", "filter", "reduce", "forEach", "find", "findIndex", "some",
        "every", "includes", "indexOf", "join", "sort", "reverse", "flat",
        "flatMap",
        "then", "catch", "finally",
        "toString", "valueOf", "hasOwnProperty",
        "addEventListener", "removeEventListener", "preventDefault",
        "stopPropagation", "querySelector", "querySelectorAll",
        "getElementById", "createElement", "appendChild", "removeChild",
        // React hooks
        "useState", "useEffect", "useRef", "useCallback", "useMemo",
        "useContext", "useReducer", "useLayoutEffect", "useImperativeHandle",
        "useDebugValue", "useId", "useTransition", "useDeferredValue",
        "useSyncExternalStore", "useInsertionEffect",
    ]));

    m.insert("rust", HashSet::from([
        // Macro calls
        "println!", "eprintln!", "print!", "eprint!",
        "format!", "write!", "writeln!",
        "vec!", "todo!", "unimplemented!", "unreachable!",
        "panic!", "assert!", "assert_eq!", "assert_ne!",
        "debug_assert!", "debug_assert_eq!", "debug_assert_ne!",
        "cfg!", "env!", "include!", "include_str!", "include_bytes!",
        "concat!", "stringify!", "file!", "line!", "column!", "module_path!",
        "dbg!", "matches!", "compile_error!",
        "trace!", "debug!", "info!", "warn!", "error!", // tracing/log macros
        // Common trait methods
        "clone", "to_string", "to_owned", "into", "from", "default",
        "fmt", "eq", "ne", "cmp", "partial_cmp", "hash",
        // Iterator/Option/Result combinators
        "map", "filter", "and_then", "or_else", "unwrap", "unwrap_or",
        "unwrap_or_else", "unwrap_or_default", "expect", "ok", "err",
        "is_some", "is_none", "is_ok", "is_err",
        "collect", "iter", "into_iter", "next",
        "push", "pop", "insert", "remove", "contains", "get", "len",
        "is_empty",
    ]));

    m.insert("go", HashSet::from([
        // Builtins
        "make", "len", "cap", "append", "copy", "delete", "close",
        "new", "panic", "recover", "print", "println",
        "complex", "real", "imag",
        // Common method names
        "Error", "String", "Len", "Less", "Swap",
        "Read", "Write", "Close", "Seek",
        "Lock", "Unlock", "RLock", "RUnlock",
        // fmt
        "Println", "Printf", "Sprintf", "Fprintf", "Errorf",
    ]));

    m.insert("java", HashSet::from([
        "System.out.println", "System.out.print", "System.err.println",
        "toString", "equals", "hashCode", "compareTo", "clone",
        "println", "print", "printf",
        "get", "set", "add", "remove", "contains", "size", "isEmpty",
        "put", "containsKey", "containsValue", "keySet", "values", "entrySet",
        "length", "charAt", "substring", "indexOf", "trim", "split",
        "valueOf", "parseInt", "parseDouble", "parseLong",
        "close", "read", "write", "flush",
    ]));

    m
});

#[cfg(test)]
#[path = "tests/blocklist_tests.rs"]
mod tests;
```

- [ ] **Step 4: Register the module**

In `crates/codemem-engine/src/index/mod.rs`, add:

```rust
pub mod blocklist;
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p codemem-engine blocklist -- --nocapture`
Expected: All tests PASS

- [ ] **Step 6: Wire blocklist into ast-grep reference extraction**

Modify `crates/codemem-engine/src/index/engine/mod.rs`. In `extract_references_from_tree` (around line 123), add blocklist filtering after R3 dedup:

```rust
// After the existing R3 dedup block (line 131), add:
// R5: Filter out noise calls (builtins, stdlib methods)
references.retain(|r| {
    if r.kind != ReferenceKind::Call {
        return true; // only filter calls, not imports/inherits
    }
    // Extract simple name from possibly qualified target
    let simple = r.target_name
        .rsplit(lang.scope_separator)
        .next()
        .unwrap_or(&r.target_name);
    !crate::index::blocklist::is_blocked_call(lang.name, simple)
});
```

- [ ] **Step 7: Write integration test for ast-grep blocklist filtering**

Add to `crates/codemem-engine/src/index/tests/engine_references_tests.rs`:

```rust
#[test]
fn python_builtin_calls_filtered() {
    let engine = AstGrepEngine::new();
    let lang = engine.find_language("py").unwrap();
    let source = r#"
def process():
    data = get_data()
    print(data)
    result = len(data)
    transformed = transform(data)
"#;
    let refs = engine.extract_references(lang, source, "test.py");
    let call_targets: Vec<&str> = refs
        .iter()
        .filter(|r| r.kind == ReferenceKind::Call)
        .map(|r| r.target_name.as_str())
        .collect();
    assert!(call_targets.contains(&"get_data"));
    assert!(call_targets.contains(&"transform"));
    assert!(!call_targets.contains(&"print"));
    assert!(!call_targets.contains(&"len"));
}
```

- [ ] **Step 8: Run all engine reference tests**

Run: `cargo test -p codemem-engine engine_references -- --nocapture`
Expected: All PASS

- [ ] **Step 9: Run clippy and full test suite**

Run: `cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`
Expected: Clean clippy, all tests pass

- [ ] **Step 10: Commit**

```bash
git add crates/codemem-engine/src/index/blocklist.rs \
       crates/codemem-engine/src/index/tests/blocklist_tests.rs \
       crates/codemem-engine/src/index/mod.rs \
       crates/codemem-engine/src/index/engine/mod.rs \
       crates/codemem-engine/src/index/tests/engine_references_tests.rs
git commit -m "feat: add per-language call noise blocklist for ast-grep extraction

Filters builtins and stdlib methods (print, len, console, useState, etc.)
from call references before resolution. Covers Python, TypeScript/JS,
Rust, Go, and Java. Includes is_blocked_call_scip() for SCIP path."
```

---

## Task 2: SCIP Path Blocklist Integration

Apply the same noise blocklist to the SCIP graph builder. SCIP symbols carry their package manager in the symbol string (e.g., `cargo`, `npm`, `pip`), so we can infer the language and check the blocklist without any config changes.

**Files:**
- Modify: `crates/codemem-engine/src/index/scip/graph_builder.rs:469-498`
- Modify: `crates/codemem-engine/src/index/tests/scip_graph_builder_tests.rs`

**Context:** The SCIP graph builder at `graph_builder.rs` creates edges from `ScipReference` objects. The current filtering is fan-out limits only (lines 469-487). The blocklist adds a second, complementary filter: even if a symbol is under the fan-out limit, if it's a known builtin (e.g., `println!` called 5 times), skip it.

- [ ] **Step 1: Write failing test for SCIP blocklist**

Add to `crates/codemem-engine/src/index/tests/scip_graph_builder_tests.rs`:

```rust
#[test]
fn scip_builtin_calls_filtered_by_blocklist() {
    // Construct a minimal ScipReadResult with a definition and references
    // to both a user symbol and a builtin. The builtin ref should be filtered.
    use crate::index::scip::{ScipDefinition, ScipReference, ScipReadResult};

    let defs = vec![
        ScipDefinition {
            scip_symbol: "rust-analyzer cargo my-crate/ auth#validate().".into(),
            qualified_name: "auth::validate".into(),
            file_path: "src/auth.rs".into(),
            line_start: 10, line_end: 20,
            col_start: 0, col_end: 0,
            kind: NodeKind::Function,
            documentation: vec![],
            relationships: vec![],
            is_test: false, is_generated: false,
        },
    ];
    let refs = vec![
        // User call — should produce an edge
        ScipReference {
            scip_symbol: "rust-analyzer cargo my-crate/ auth#validate().".into(),
            file_path: "src/main.rs".into(),
            line: 5,
            role_bitmask: 0, // generic call role
        },
        // Builtin call — should be filtered
        ScipReference {
            scip_symbol: "rust-analyzer cargo std/ fmt#Display#fmt().".into(),
            file_path: "src/main.rs".into(),
            line: 6,
            role_bitmask: 0,
        },
    ];

    let scip = ScipReadResult {
        definitions: defs,
        references: refs,
        external_symbols: vec![],
    };

    let config = ScipConfig::default();
    let (nodes, edges, _) = build_graph(&scip, &config);

    // The fmt() call to std should be filtered out
    let call_edges: Vec<_> = edges.iter()
        .filter(|e| e.relationship == RelationshipType::Calls)
        .collect();
    assert!(
        !call_edges.iter().any(|e| e.dst.contains("fmt")),
        "Builtin fmt() call should be filtered by blocklist"
    );
}
```

Note: Adapt this test to match the actual `ScipReadResult`, `ScipReference`, and `build_graph` signatures. Check existing tests in `scip_graph_builder_tests.rs` for the exact constructor patterns.

- [ ] **Step 2: Run test to confirm it fails**

Run: `cargo test -p codemem-engine scip_builtin_calls_filtered -- --nocapture`
Expected: FAIL — the edge is not filtered yet

- [ ] **Step 3: Add blocklist check to SCIP graph builder**

In `crates/codemem-engine/src/index/scip/graph_builder.rs`, after the fan-out limit check (~line 487) and before the target node resolution (~line 489), add:

```rust
        // R5: Filter noise calls using the blocklist.
        // SCIP symbols encode language via package manager prefix.
        if crate::index::blocklist::is_blocked_call_scip(&r.scip_symbol) {
            continue;
        }
```

This is a single `continue` — same pattern as the fan-out check above it. The `is_blocked_call_scip` function (from Task 1) extracts the language from the SCIP symbol prefix and the simple name from the descriptor chain.

- [ ] **Step 4: Run SCIP tests**

Run: `cargo test -p codemem-engine scip -- --nocapture`
Expected: All PASS including the new test

- [ ] **Step 5: Run full suite + clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`
Expected: Clean

- [ ] **Step 6: Commit**

```bash
git add crates/codemem-engine/src/index/scip/graph_builder.rs \
       crates/codemem-engine/src/index/tests/scip_graph_builder_tests.rs
git commit -m "feat: apply call noise blocklist to SCIP graph builder

SCIP symbols carry package manager prefix (cargo/npm/pip/go/maven) which
maps to language for blocklist lookup. Filters fmt, println, console.log
etc. from SCIP edges, complementing the existing fan-out limits."
```

---

## Task 3: Callback / Higher-Order Function Detection

Detect bare identifier arguments passed to function calls (e.g., `map(transform, items)`, `Depends(get_db)`, `arr.map(processItem)`) and emit them as `Callback` references resolved at reduced confidence.

**Files:**
- Modify: `crates/codemem-engine/src/index/symbol.rs:133-141` (add `Callback` to `ReferenceKind` + `Display` impl)
- Modify: `crates/codemem-engine/src/index/engine/mod.rs:~689` (add `"callback"` to `parse_reference_kind`)
- Modify: `crates/codemem-engine/src/index/engine/references.rs` (add `callback_args` special handler)
- Modify: `crates/codemem-engine/src/index/resolver.rs:218-241` (cap callback confidence in `resolve_all` + `resolve_all_with_unresolved`)
- Modify: `crates/codemem-engine/rules/python/references.yml`
- Modify: `crates/codemem-engine/rules/typescript/references.yml`
- Modify: `crates/codemem-engine/src/index/tests/engine_references_tests.rs`
- Modify: `crates/codemem-engine/src/index/tests/resolver_tests.rs`

- [ ] **Step 1: Add `Callback` variant to `ReferenceKind`**

In `crates/codemem-engine/src/index/symbol.rs`:

1. Add to the `ReferenceKind` enum (around line 133):

```rust
pub enum ReferenceKind {
    Call,
    Import,
    Inherits,
    Implements,
    TypeUsage,
    Callback,  // bare identifier arg passed to a function (e.g., map(transform))
}
```

2. Update the `Display` impl (search for `impl fmt::Display for ReferenceKind`) to add:

```rust
ReferenceKind::Callback => write!(f, "callback"),
```

3. Update `parse_reference_kind` in `crates/codemem-engine/src/index/engine/mod.rs` (around line 689) to add:

```rust
"callback" => ReferenceKind::Callback,
```

- [ ] **Step 2: Run cargo check — let compiler find all exhaustive match sites**

Run: `cargo check --workspace`
Expected: Compile errors from exhaustive match — this is expected and guides us to all sites needing `Callback`.

- [ ] **Step 3: Fix all exhaustive match sites**

The compiler will flag every `match` on `ReferenceKind` that doesn't handle `Callback`. For each:

- In `resolver.rs` `resolve_all` (line ~223): add `ReferenceKind::Callback => RelationshipType::Calls`
- In `resolver.rs` `resolve_all_with_unresolved` (line ~254): same
- In `extract_package_hint` (line ~301): already handled by `kind != ReferenceKind::Import` early return — `Callback` will return `None`. No change needed.
- In `engine/mod.rs` blocklist filter (Task 1): update the `r.kind != ReferenceKind::Call` check to also pass through `Callback`:

```rust
references.retain(|r| {
    if !matches!(r.kind, ReferenceKind::Call | ReferenceKind::Callback) {
        return true;
    }
    // ...blocklist check...
});
```

- Any other match sites: treat `Callback` same as `Call`.

- [ ] **Step 4: Run cargo check again**

Run: `cargo check --workspace`
Expected: Clean compile

- [ ] **Step 5: Write failing test for callback extraction**

Add to `crates/codemem-engine/src/index/tests/engine_references_tests.rs`:

```rust
#[test]
fn python_callback_args_extracted() {
    let engine = AstGrepEngine::new();
    let lang = engine.find_language("py").unwrap();
    let source = r#"
def transform(x):
    return x * 2

def process():
    result = map(transform, items)
    filtered = filter(is_valid, data)
"#;
    let refs = engine.extract_references(lang, source, "test.py");
    let callbacks: Vec<&str> = refs
        .iter()
        .filter(|r| r.kind == ReferenceKind::Callback)
        .map(|r| r.target_name.as_str())
        .collect();
    assert!(callbacks.contains(&"transform"), "map(transform) should emit callback ref");
    assert!(callbacks.contains(&"is_valid"), "filter(is_valid) should emit callback ref");
}

#[test]
fn typescript_callback_args_extracted() {
    let engine = AstGrepEngine::new();
    let lang = engine.find_language("ts").unwrap();
    let source = r#"
function processItem(item: Item): Result {
    return transform(item);
}

const results = items.map(processItem);
const handler = app.get("/api", validateAuth, handleRequest);
"#;
    let refs = engine.extract_references(lang, source, "test.ts");
    let callbacks: Vec<&str> = refs
        .iter()
        .filter(|r| r.kind == ReferenceKind::Callback)
        .map(|r| r.target_name.as_str())
        .collect();
    assert!(callbacks.contains(&"processItem"), "arr.map(processItem) should emit callback");
    assert!(callbacks.contains(&"validateAuth"), "middleware callback should emit callback");
    assert!(callbacks.contains(&"handleRequest"), "route handler should emit callback");
}
```

- [ ] **Step 6: Run tests to confirm they fail**

Run: `cargo test -p codemem-engine callback -- --nocapture`
Expected: FAIL — no callback references extracted yet

- [ ] **Step 7: Implement the `callback_args` special handler**

In `crates/codemem-engine/src/index/engine/references.rs`, add a new special handler case in `handle_special_reference`:

```rust
"callback_args" => {
    // Extract bare identifier arguments from call expressions.
    // e.g., map(transform, items) -> Callback ref to "transform"
    // e.g., app.get("/api", validateAuth) -> Callback ref to "validateAuth"
    if let Some(args_node) = node.field("arguments") {
        let line = node.start_pos().line();
        for child in args_node.children() {
            let kind = child.kind();
            // Only bare identifiers — skip literals, calls, member access, etc.
            if kind == "identifier" {
                let name = child.text().to_string();
                // Skip common non-function identifiers
                if !is_likely_callback_arg(&name) {
                    continue;
                }
                push_ref(
                    references,
                    &source_qn,
                    name,
                    ReferenceKind::Callback,
                    file_path,
                    line,
                );
            }
        }
    }
}
```

Add the helper function in the same file (outside the `impl` block):

```rust
/// Heuristic: skip names that are very unlikely to be callback functions.
/// These are typically variable names for data, not function references.
fn is_likely_callback_arg(name: &str) -> bool {
    if name.len() <= 1 {
        return false;
    }
    !matches!(
        name,
        "self" | "this" | "cls" | "args" | "kwargs"
        | "data" | "items" | "result" | "results"
        | "err" | "error" | "ctx" | "context"
        | "req" | "res" | "request" | "response"
        | "true" | "false" | "null" | "None" | "undefined"
        | "nil"
    )
}
```

- [ ] **Step 8: Update YAML rules to trigger callback extraction**

Modify `crates/codemem-engine/rules/python/references.yml` — add a second rule entry for callbacks (keep existing `call` rule as-is):

```yaml
references:
  - kind: "import_statement"
    reference_kind: "import"
    name_field: null
    special: "python_import"

  - kind: "import_from_statement"
    reference_kind: "import"
    name_field: null
    special: "python_import_from"

  - kind: "call"
    reference_kind: "call"
    name_field: "function"
    special: null

  # Callback detection: bare identifier args passed to calls
  - kind: "call"
    reference_kind: "callback"
    name_field: null
    special: "callback_args"

  - kind: "class_definition"
    reference_kind: "inherits"
    name_field: null
    special: "python_class_bases"
```

Do the same for `crates/codemem-engine/rules/typescript/references.yml`:

```yaml
references:
  - kind: "import_statement"
    reference_kind: "import"
    name_field: "source"
    special: null

  - kind: "call_expression"
    reference_kind: "call"
    name_field: "function"
    special: null

  # Callback detection: bare identifier args passed to calls
  - kind: "call_expression"
    reference_kind: "callback"
    name_field: null
    special: "callback_args"

  - kind: "class_declaration"
    reference_kind: "inherits"
    name_field: null
    special: "ts_class_heritage"
```

- [ ] **Step 9: Run callback tests**

Run: `cargo test -p codemem-engine callback -- --nocapture`
Expected: PASS

- [ ] **Step 10: Cap callback confidence in the resolver**

In `crates/codemem-engine/src/index/resolver.rs`, modify both `resolve_all` (line ~218) and `resolve_all_with_unresolved` (line ~247). After computing confidence from `resolve_with_confidence`, apply the callback cap:

```rust
// In resolve_all, after: let (target, confidence) = self.resolve_with_confidence(r)?;
let confidence = if r.kind == ReferenceKind::Callback {
    confidence.min(0.6) // callbacks are speculative
} else {
    confidence
};
```

Apply the same pattern in `resolve_all_with_unresolved`.

- [ ] **Step 11: Write resolver confidence test for callbacks**

Add to `crates/codemem-engine/src/index/tests/resolver_tests.rs`:

```rust
#[test]
fn callback_references_capped_at_0_6() {
    let mut resolver = ReferenceResolver::new();
    // Construct Symbol with all required fields (Symbol has no Default impl)
    resolver.add_symbols(&[Symbol {
        name: "transform".into(),
        qualified_name: "utils.transform".into(),
        kind: SymbolKind::Function,
        signature: String::new(),
        visibility: Visibility::Public,
        file_path: "utils.py".into(),
        line_start: 1,
        line_end: 3,
        doc_comment: None,
        parent: None,
        parameters: vec![],
        return_type: None,
        is_async: false,
        attributes: vec![],
        throws: vec![],
        generic_params: None,
        is_abstract: false,
    }]);

    let refs = vec![Reference {
        source_qualified_name: "main.process".into(),
        target_name: "transform".into(),
        kind: ReferenceKind::Callback,
        file_path: "main.py".into(),
        line: 10,
    }];

    let edges = resolver.resolve_all(&refs);
    assert_eq!(edges.len(), 1);
    assert!(edges[0].resolution_confidence <= 0.6,
        "Callback confidence should be capped at 0.6, got {}",
        edges[0].resolution_confidence);
}
```

Note: `Symbol` does not implement `Default`. All fields must be provided explicitly. Check the `Symbol` struct definition at `symbol.rs:18-60` for the complete field list.

- [ ] **Step 12: Run full test suite + clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`
Expected: Clean

- [ ] **Step 13: Commit**

```bash
git add crates/codemem-engine/src/index/symbol.rs \
       crates/codemem-engine/src/index/engine/references.rs \
       crates/codemem-engine/src/index/resolver.rs \
       crates/codemem-engine/src/index/engine/mod.rs \
       crates/codemem-engine/rules/python/references.yml \
       crates/codemem-engine/rules/typescript/references.yml \
       crates/codemem-engine/src/index/tests/engine_references_tests.rs \
       crates/codemem-engine/src/index/tests/resolver_tests.rs
git commit -m "feat: detect callback/higher-order function args as speculative CALLS edges

Bare identifier arguments (e.g., map(transform), app.get(validateAuth))
now emit Callback references resolved at max 0.6 confidence. Covers
Python and TypeScript/JS via new YAML reference rules + special handler."
```

---

## Task 4: Dead Code Detection Enrichment

New enrichment analyzer that identifies symbols with zero inbound CALLS/IMPORTS edges, applying framework-aware exemptions (decorators, exports, constructors, test functions, protocol/trait implementations).

**Files:**
- Create: `crates/codemem-engine/src/enrichment/dead_code.rs`
- Modify: `crates/codemem-engine/src/enrichment/mod.rs:6`
- Modify: `crates/codemem-core/src/config.rs`

**Important type notes:**
- The in-memory graph uses `GraphNode` (from `codemem_core::GraphNode`) with `payload: HashMap<String, serde_json::Value>`, NOT `MemoryNode`.
- Edges use `Edge` (from `codemem_core::Edge`) with fields `src`, `dst` (not `source`, `target`).
- Access the graph via `self.lock_graph()?` which returns `MutexGuard<Box<dyn GraphBackend>>`.
- `store_insight` signature is: `store_insight(&self, content, track, tags, importance, namespace, links) -> Option<String>` — no `Result`, returns `Option<String>` directly.

- [ ] **Step 1: Add `DeadCodeConfig` to config**

In `crates/codemem-core/src/config.rs`:

1. Add the new config struct:

```rust
/// Configuration for dead code detection enrichment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadCodeConfig {
    /// Enable dead code detection.
    pub enabled: bool,
    /// Decorator/attribute names that exempt a symbol from dead code detection.
    pub exempt_decorators: Vec<String>,
    /// Symbol kinds that are always exempt (e.g., constructors, tests).
    pub exempt_kinds: Vec<String>,
    /// Minimum number of symbols in the graph before running analysis.
    pub min_symbols: usize,
}

impl Default for DeadCodeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            exempt_decorators: vec![
                "app.route".into(), "router.get".into(), "router.post".into(),
                "router.put".into(), "router.delete".into(), "router.patch".into(),
                "pytest.fixture".into(), "fixture".into(),
                "click.command".into(), "click.group".into(),
                "celery.task".into(), "task".into(),
                "property".into(), "staticmethod".into(), "classmethod".into(),
                "override".into(), "abstractmethod".into(),
                "api_view".into(), "action".into(),
                "test".into(), "tokio::test".into(), "async_trait".into(),
            ],
            exempt_kinds: vec![
                "constructor".into(), "test".into(),
            ],
            min_symbols: 10,
        }
    }
}
```

2. Add to `EnrichmentConfig` struct (find it in `config.rs`):

```rust
pub dead_code: DeadCodeConfig,
```

3. Update `EnrichmentConfig`'s `Default` impl to include:

```rust
dead_code: DeadCodeConfig::default(),
```

- [ ] **Step 2: Run cargo check**

Run: `cargo check --workspace`
Expected: Clean (config is data only, no consumers yet)

- [ ] **Step 3: Write failing tests for dead code detection**

Create `crates/codemem-engine/src/enrichment/dead_code.rs` with tests first:

```rust
//! Dead code detection enrichment.

use codemem_core::{Edge, GraphNode, NodeKind, RelationshipType};
use std::collections::{HashMap, HashSet};

/// A symbol identified as potentially dead code.
#[derive(Debug)]
pub struct DeadCodeEntry {
    pub symbol_name: String,
    pub node_id: String,
    pub kind: String,
    pub file_path: Option<String>,
}

/// Placeholder — will be implemented in Step 5.
pub fn find_dead_code(
    _nodes: &[GraphNode],
    _edges: &[Edge],
    _config: &codemem_core::config::DeadCodeConfig,
) -> Vec<DeadCodeEntry> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_graph_node(name: &str, kind_str: &str, attrs: &[&str]) -> GraphNode {
        let mut payload = HashMap::new();
        payload.insert("kind".into(), serde_json::Value::String(kind_str.into()));
        if !attrs.is_empty() {
            payload.insert("attributes".into(),
                serde_json::Value::Array(
                    attrs.iter().map(|a| serde_json::Value::String(a.to_string())).collect()
                ));
        }
        GraphNode {
            id: format!("sym:{}", name),
            kind: NodeKind::Function, // simplified for tests
            label: name.to_string(),
            payload,
            centrality: 0.0,
            memory_id: None,
            namespace: Some("test".into()),
        }
    }

    fn make_edge(src: &str, dst: &str, rel: RelationshipType) -> Edge {
        Edge {
            id: format!("{}->{}:{}", src, dst, rel),
            src: format!("sym:{}", src),
            dst: format!("sym:{}", dst),
            relationship: rel,
            weight: 1.0,
            properties: HashMap::new(),
            created_at: chrono::Utc::now(),
            valid_from: None,
            valid_to: None,
        }
    }

    #[test]
    fn unreachable_function_detected() {
        let nodes = vec![
            make_graph_node("main", "function", &[]),
            make_graph_node("helper", "function", &[]),
            make_graph_node("unused_fn", "function", &[]),
        ];
        let edges = vec![
            make_edge("main", "helper", RelationshipType::Calls),
        ];
        let mut config = codemem_core::config::DeadCodeConfig::default();
        config.min_symbols = 2; // lower threshold for test
        let dead = find_dead_code(&nodes, &edges, &config);
        assert!(dead.iter().any(|d| d.symbol_name == "unused_fn"));
        assert!(!dead.iter().any(|d| d.symbol_name == "helper"));
        assert!(!dead.iter().any(|d| d.symbol_name == "main"));
    }

    #[test]
    fn decorated_symbols_exempt() {
        let nodes = vec![
            make_graph_node("index", "function", &["app.route"]),
            make_graph_node("orphan", "function", &[]),
        ];
        let edges = vec![];
        let mut config = codemem_core::config::DeadCodeConfig::default();
        config.min_symbols = 2;
        let dead = find_dead_code(&nodes, &edges, &config);
        assert!(!dead.iter().any(|d| d.symbol_name == "index"),
            "app.route decorated function should be exempt");
        assert!(dead.iter().any(|d| d.symbol_name == "orphan"));
    }

    #[test]
    fn constructors_and_tests_exempt() {
        let nodes = vec![
            make_graph_node("__init__", "constructor", &[]),
            make_graph_node("test_foo", "test", &[]),
            make_graph_node("unused", "function", &[]),
        ];
        let edges = vec![];
        let mut config = codemem_core::config::DeadCodeConfig::default();
        config.min_symbols = 2;
        let dead = find_dead_code(&nodes, &edges, &config);
        assert!(!dead.iter().any(|d| d.symbol_name == "__init__"));
        assert!(!dead.iter().any(|d| d.symbol_name == "test_foo"));
        assert!(dead.iter().any(|d| d.symbol_name == "unused"));
    }

    #[test]
    fn min_symbols_threshold_respected() {
        let nodes = vec![
            make_graph_node("only_one", "function", &[]),
        ];
        let edges = vec![];
        let config = codemem_core::config::DeadCodeConfig::default(); // min_symbols = 10
        let dead = find_dead_code(&nodes, &edges, &config);
        assert!(dead.is_empty(), "Should not report dead code below min_symbols threshold");
    }
}
```

- [ ] **Step 4: Register the module and run tests to confirm they fail**

In `crates/codemem-engine/src/enrichment/mod.rs`, add:

```rust
pub(crate) mod dead_code;
```

Run: `cargo test -p codemem-engine dead_code -- --nocapture`
Expected: FAIL — `todo!()` panics

- [ ] **Step 5: Implement `find_dead_code`**

Replace the `todo!()` placeholder in `dead_code.rs`:

```rust
/// Find dead code symbols in the graph.
///
/// A symbol is "dead" if:
/// 1. It has zero inbound CALLS, IMPORTS, INHERITS, or IMPLEMENTS edges
/// 2. It is not exempted by decorator, kind, export, or naming convention
pub fn find_dead_code(
    nodes: &[GraphNode],
    edges: &[Edge],
    config: &codemem_core::config::DeadCodeConfig,
) -> Vec<DeadCodeEntry> {
    // Only consider nodes that have a "kind" payload (symbol nodes)
    let symbol_nodes: Vec<&GraphNode> = nodes
        .iter()
        .filter(|n| n.payload.get("kind").and_then(|v| v.as_str()).is_some())
        .collect();

    if symbol_nodes.len() < config.min_symbols {
        return Vec::new();
    }

    // Build set of node IDs with inbound structural edges
    let mut referenced: HashSet<&str> = HashSet::new();
    for edge in edges {
        if matches!(
            edge.relationship,
            RelationshipType::Calls
                | RelationshipType::Imports
                | RelationshipType::Inherits
                | RelationshipType::Implements
        ) {
            referenced.insert(&edge.dst);
        }
    }

    let exempt_decorators: HashSet<&str> = config
        .exempt_decorators.iter().map(|s| s.as_str()).collect();
    let exempt_kinds: HashSet<&str> = config
        .exempt_kinds.iter().map(|s| s.as_str()).collect();

    let mut dead = Vec::new();

    for node in &symbol_nodes {
        if referenced.contains(node.id.as_str()) {
            continue;
        }

        let kind = node.payload.get("kind")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if exempt_kinds.contains(kind) {
            continue;
        }

        // Exempt by decorator/attribute
        let attrs = node.payload.get("attributes")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
            .unwrap_or_default();

        if attrs.iter().any(|a| {
            exempt_decorators.contains(a)
                || a.contains("route")
                || a.contains("endpoint")
                || a.contains("export")
                || a.contains("api")
        }) {
            continue;
        }

        // Exempt: "main" entry points
        if node.label == "main" || node.label == "Main" {
            continue;
        }

        // Exempt: dunder methods (Python protocol conformance)
        if node.label.starts_with("__") && node.label.ends_with("__") {
            continue;
        }

        // Exempt: public visibility
        if node.payload.get("visibility")
            .and_then(|v| v.as_str())
            .is_some_and(|v| v == "public")
        {
            continue;
        }

        let file_path = node.payload.get("file_path")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        dead.push(DeadCodeEntry {
            symbol_name: node.label.clone(),
            node_id: node.id.clone(),
            kind: kind.to_string(),
            file_path,
        });
    }

    dead
}
```

- [ ] **Step 6: Run dead code tests**

Run: `cargo test -p codemem-engine dead_code -- --nocapture`
Expected: All PASS

- [ ] **Step 7: Wire into the enrichment pipeline**

Add a method on `CodememEngine` in `dead_code.rs`:

```rust
use crate::CodememEngine;
use super::EnrichResult;

impl CodememEngine {
    /// Run dead code detection and store insights.
    pub fn enrich_dead_code(
        &self,
        namespace: Option<&str>,
    ) -> Result<EnrichResult, codemem_core::CodememError> {
        let config = self.config().enrichment.dead_code.clone();
        if !config.enabled {
            return Ok(EnrichResult {
                insights_stored: 0,
                details: serde_json::json!({"skipped": "disabled"}),
            });
        }

        // Gather nodes and edges from the in-memory graph
        let graph = self.lock_graph()?;
        let nodes: Vec<GraphNode> = graph.all_nodes()?;
        let edges: Vec<Edge> = graph.all_edges()?;
        drop(graph); // release lock before storing insights

        let dead = find_dead_code(&nodes, &edges, &config);

        let mut stored = 0;
        for entry in &dead {
            let content = format!(
                "Potentially dead code: `{}` ({}) has no callers or importers{}",
                entry.symbol_name,
                entry.kind,
                entry.file_path.as_deref()
                    .map(|p| format!(" in {}", p))
                    .unwrap_or_default(),
            );
            if self.store_insight(
                &content,
                "dead-code",
                &[],
                0.4, // low importance — dead code is advisory
                namespace,
                &[entry.node_id.clone()],
            ).is_some() {
                stored += 1;
            }
        }

        Ok(EnrichResult {
            insights_stored: stored,
            details: serde_json::json!({
                "dead_symbols_found": dead.len(),
                "insights_stored": stored,
            }),
        })
    }
}
```

**Important:** Check whether `GraphBackend` trait has `all_nodes()` and `all_edges()` methods. If not, the enrichment method may need to iterate the graph differently (e.g., via node/edge count + individual lookups, or by querying the storage layer). Adapt accordingly based on what the `GraphBackend` trait actually exposes.

- [ ] **Step 8: Run clippy and full test suite**

Run: `cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`
Expected: Clean

- [ ] **Step 9: Commit**

```bash
git add crates/codemem-engine/src/enrichment/dead_code.rs \
       crates/codemem-engine/src/enrichment/mod.rs \
       crates/codemem-core/src/config.rs
git commit -m "feat: add dead code detection enrichment pass

Identifies symbols with zero inbound CALLS/IMPORTS edges using GraphNode
data from the in-memory graph. Framework-aware exemptions for decorators
(routes, fixtures, tasks), constructors, tests, dunder methods, public
symbols, and main entry points. Configurable via DeadCodeConfig."
```

---

## Task 5: Integration Verification

End-to-end verification that all four features work together.

**Files:**
- No new files — this task runs existing tests + manual verification

- [ ] **Step 1: Run the full test suite**

Run: `cargo test --workspace`
Expected: All PASS

- [ ] **Step 2: Run clippy with CI flags**

Run: `RUSTFLAGS="-D warnings" cargo clippy --workspace --all-targets -- -D warnings`
Expected: Clean

- [ ] **Step 3: Run cargo fmt check**

Run: `cargo fmt --all -- --check`
Expected: Clean

- [ ] **Step 4: Verify both paths work together**

Confirm SCIP tests still pass alongside new blocklist:

Run: `cargo test -p codemem-engine scip -- --nocapture`
Expected: All PASS

Confirm ast-grep tests pass with blocklist + callbacks:

Run: `cargo test -p codemem-engine engine_references -- --nocapture`
Expected: All PASS

Confirm dead code tests pass:

Run: `cargo test -p codemem-engine dead_code -- --nocapture`
Expected: All PASS

- [ ] **Step 5: Commit final state if any fixups were needed**

Only if steps 1-4 required changes. Otherwise, skip.
