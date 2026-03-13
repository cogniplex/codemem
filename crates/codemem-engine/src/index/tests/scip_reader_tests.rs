use super::*;

#[test]
fn test_scip_symbol_to_qualified_name_rust() {
    let symbol = "rust-analyzer cargo my_crate 1.0.0 auth/jwt/validate().";
    let result = scip_symbol_to_qualified_name(symbol, "::").unwrap();
    assert_eq!(result, "auth::jwt::validate");
}

#[test]
fn test_scip_symbol_to_qualified_name_python() {
    let symbol = "scip-python pip django 4.2.0 django/http/HttpRequest#GET.";
    let result = scip_symbol_to_qualified_name(symbol, ".").unwrap();
    assert_eq!(result, "django.http.HttpRequest.GET");
}

#[test]
fn test_scip_symbol_to_qualified_name_empty_descriptors() {
    // A symbol with no descriptors should return None.
    let result = scip_symbol_to_qualified_name("rust-analyzer cargo foo 1.0 ", "::");
    assert!(
        result.is_none(),
        "Expected None for symbol with no descriptors, got {result:?}"
    );
}

#[test]
fn test_detect_language_separator_rust() {
    assert_eq!(detect_language_separator("rust"), "::");
}

#[test]
fn test_detect_language_separator_python() {
    assert_eq!(detect_language_separator("python"), ".");
}

#[test]
fn test_detect_language_separator_typescript() {
    assert_eq!(detect_language_separator("typescript"), ".");
}

#[test]
fn test_detect_language_separator_cpp() {
    assert_eq!(detect_language_separator("cpp"), "::");
    assert_eq!(detect_language_separator("c++"), "::");
}

#[test]
fn test_parse_range_single_line() {
    let range = vec![10, 5, 15];
    assert_eq!(parse_range(&range), Some((10, 5, 10, 15)));
}

#[test]
fn test_parse_range_multi_line() {
    let range = vec![10, 5, 20, 15];
    assert_eq!(parse_range(&range), Some((10, 5, 20, 15)));
}

#[test]
fn test_parse_range_invalid() {
    assert_eq!(parse_range(&[]), None);
    assert_eq!(parse_range(&[1]), None);
    assert_eq!(parse_range(&[1, 2]), None);
    assert_eq!(parse_range(&[1, 2, 3, 4, 5]), None);
}

#[test]
fn test_role_bitmask_helpers() {
    assert!(is_import_ref(ROLE_IMPORT));
    assert!(is_import_ref(ROLE_IMPORT | ROLE_READ_ACCESS));
    assert!(!is_import_ref(ROLE_READ_ACCESS));

    assert!(is_read_ref(ROLE_READ_ACCESS));
    assert!(!is_read_ref(ROLE_WRITE_ACCESS));

    assert!(is_write_ref(ROLE_WRITE_ACCESS));
    assert!(!is_write_ref(ROLE_READ_ACCESS));
}

#[test]
fn test_scip_kind_to_node_kind() {
    use scip::types::symbol_information::Kind;
    assert_eq!(
        scip_kind_to_node_kind(Kind::Class as i32),
        Some(NodeKind::Class)
    );
    assert_eq!(
        scip_kind_to_node_kind(Kind::Interface as i32),
        Some(NodeKind::Interface)
    );
    assert_eq!(
        scip_kind_to_node_kind(Kind::Enum as i32),
        Some(NodeKind::Enum)
    );
    assert_eq!(
        scip_kind_to_node_kind(Kind::EnumMember as i32),
        Some(NodeKind::EnumVariant)
    );
    assert_eq!(
        scip_kind_to_node_kind(Kind::Field as i32),
        Some(NodeKind::Field)
    );
    assert_eq!(
        scip_kind_to_node_kind(Kind::Method as i32),
        Some(NodeKind::Method)
    );
    assert_eq!(
        scip_kind_to_node_kind(Kind::Function as i32),
        Some(NodeKind::Function)
    );
    assert_eq!(
        scip_kind_to_node_kind(Kind::Constructor as i32),
        Some(NodeKind::Function)
    );
    assert_eq!(
        scip_kind_to_node_kind(Kind::Macro as i32),
        Some(NodeKind::Macro)
    );
    assert_eq!(
        scip_kind_to_node_kind(Kind::Property as i32),
        Some(NodeKind::Property)
    );
    assert_eq!(
        scip_kind_to_node_kind(Kind::Namespace as i32),
        Some(NodeKind::Module)
    );
    assert_eq!(
        scip_kind_to_node_kind(Kind::Module as i32),
        Some(NodeKind::Module)
    );
    assert_eq!(
        scip_kind_to_node_kind(Kind::Package as i32),
        Some(NodeKind::Package)
    );
    assert_eq!(
        scip_kind_to_node_kind(Kind::TypeAlias as i32),
        Some(NodeKind::Type)
    );
    assert_eq!(
        scip_kind_to_node_kind(Kind::Type as i32),
        Some(NodeKind::Type)
    );
    assert_eq!(
        scip_kind_to_node_kind(Kind::Constant as i32),
        Some(NodeKind::Constant)
    );
    assert_eq!(
        scip_kind_to_node_kind(Kind::Trait as i32),
        Some(NodeKind::Trait)
    );
    // Unknown kind returns None for descriptor-based inference fallback
    assert_eq!(scip_kind_to_node_kind(999), None);
    assert_eq!(scip_kind_to_node_kind(0), None);
}

#[test]
fn test_infer_kind_from_symbol() {
    // Type descriptor (#) → Class
    assert_eq!(
        infer_kind_from_symbol("scip-go gomod example 1.0 pkg/MyStruct#"),
        NodeKind::Class
    );
    // Method descriptor (().) → Method
    assert_eq!(
        infer_kind_from_symbol("scip-go gomod example 1.0 pkg/MyStruct#DoThing()."),
        NodeKind::Method
    );
    // Package descriptor (/) → Module
    assert_eq!(
        infer_kind_from_symbol("scip-go gomod example 1.0 pkg/"),
        NodeKind::Module
    );
    // Term descriptor (.) → Function
    assert_eq!(
        infer_kind_from_symbol("scip-go gomod example 1.0 pkg/helper."),
        NodeKind::Function
    );
    // Unparseable → Function fallback
    assert_eq!(infer_kind_from_symbol(""), NodeKind::Function);
}
