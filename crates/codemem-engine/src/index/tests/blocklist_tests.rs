use super::{is_blocked_call, is_blocked_call_scip};

// ── Python ──────────────────────────────────────────────────────────────

#[test]
fn python_builtins_blocked() {
    for name in &["print", "len", "range", "isinstance", "enumerate"] {
        assert!(
            is_blocked_call("python", name),
            "expected python builtin '{name}' to be blocked"
        );
    }
}

#[test]
fn python_stdlib_methods_blocked() {
    for name in &["append", "strip", "join"] {
        assert!(
            is_blocked_call("python", name),
            "expected python stdlib method '{name}' to be blocked"
        );
    }
}

// ── TypeScript ──────────────────────────────────────────────────────────

#[test]
fn typescript_globals_blocked() {
    for name in &["console", "setTimeout", "parseInt", "fetch"] {
        assert!(
            is_blocked_call("typescript", name),
            "expected typescript global '{name}' to be blocked"
        );
    }
}

#[test]
fn typescript_react_hooks_blocked() {
    for name in &["useState", "useEffect", "useRef", "useCallback"] {
        assert!(
            is_blocked_call("typescript", name),
            "expected React hook '{name}' to be blocked"
        );
    }
}

#[test]
fn typescript_dotted_methods_blocked() {
    for name in &["log", "stringify", "parse"] {
        assert!(
            is_blocked_call("typescript", name),
            "expected TS dotted method '{name}' to be blocked"
        );
    }
}

// ── Rust ────────────────────────────────────────────────────────────────

#[test]
fn rust_builtins_blocked() {
    for name in &[
        "println!",
        "eprintln!",
        "format!",
        "vec!",
        "todo!",
        "assert!",
        "assert_eq!",
    ] {
        assert!(
            is_blocked_call("rust", name),
            "expected rust builtin '{name}' to be blocked"
        );
    }
}

// ── Go ──────────────────────────────────────────────────────────────────

#[test]
fn go_builtins_blocked() {
    for name in &["make", "len", "cap", "append", "panic", "close"] {
        assert!(
            is_blocked_call("go", name),
            "expected go builtin '{name}' to be blocked"
        );
    }
}

// ── User symbols NOT blocked ────────────────────────────────────────────

#[test]
fn user_symbols_not_blocked() {
    assert!(!is_blocked_call("python", "process_data"));
    assert!(!is_blocked_call("typescript", "handleSubmit"));
    assert!(!is_blocked_call("rust", "parse_config"));
    assert!(!is_blocked_call("go", "StartServer"));
}

// ── Unknown language ────────────────────────────────────────────────────

#[test]
fn unknown_language_blocks_nothing() {
    assert!(!is_blocked_call("haskell", "print"));
}

// ── Shared language aliases ─────────────────────────────────────────────

#[test]
fn shared_languages_use_same_blocklist() {
    assert!(is_blocked_call("tsx", "useState"));
    assert!(is_blocked_call("javascript", "console"));
    assert!(is_blocked_call("jsx", "useEffect"));
}

// ── SCIP path ───────────────────────────────────────────────────────────

#[test]
fn scip_symbol_blocked() {
    // cargo → rust
    assert!(is_blocked_call_scip(
        "scip-cargo crate_name 1.0.0 module#clone()"
    ));
    // npm → typescript
    assert!(is_blocked_call_scip(
        "scip-npm @types/react 18.0.0 hooks.useState()"
    ));
    // pip → python
    assert!(is_blocked_call_scip(
        "scip-pip stdlib 3.11 builtins#print()"
    ));
}

#[test]
fn scip_user_symbols_not_blocked() {
    assert!(!is_blocked_call_scip(
        "scip-cargo my_crate 0.1.0 module#process_data()"
    ));
    assert!(!is_blocked_call_scip(
        "scip-npm my-lib 1.0.0 utils.handleSubmit()"
    ));
}
