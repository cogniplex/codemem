use super::*;

#[test]
fn extract_package_from_node_modules() {
    assert_eq!(
        TsServerEnricher::extract_package_from_path("node_modules/lodash/index.js"),
        Some("lodash".to_string())
    );
}

#[test]
fn extract_scoped_package() {
    assert_eq!(
        TsServerEnricher::extract_package_from_path("node_modules/@acme/shared/dist/index.js"),
        Some("@acme/shared".to_string())
    );
}

#[test]
fn extract_package_none_for_local() {
    assert_eq!(
        TsServerEnricher::extract_package_from_path("src/utils/helper.ts"),
        None
    );
}

#[test]
fn parse_tsc_diagnostics_valid() {
    let output = "src/app.ts(10,5): error TS2304: Cannot find name 'foo'.\nsrc/utils.ts(3,1): warning TS6133: 'x' is declared but never used.";
    let diags = parse_tsc_diagnostics(output);
    assert_eq!(diags.len(), 2);
    assert_eq!(diags[0].file, "src/app.ts");
    assert_eq!(diags[0].line, 10);
    assert_eq!(diags[0].code, "TS2304");
}

#[test]
fn parse_tsc_diagnostics_empty() {
    let diags = parse_tsc_diagnostics("");
    assert!(diags.is_empty());
}

#[test]
fn extract_type_annotations_from_dts_functions() {
    let dts =
        "export declare function greet<T>(name: T): string;\ndeclare const VERSION: number;\n";
    let mut annotations = Vec::new();
    TsServerEnricher::extract_type_annotations_from_dts(dts, "src/utils.ts", &mut annotations);
    assert_eq!(annotations.len(), 2);
    assert_eq!(annotations[0].symbol_name, "greet");
    assert_eq!(annotations[0].return_type, Some("string".to_string()));
    assert_eq!(annotations[0].generic_params, vec!["T"]);
    assert_eq!(annotations[1].symbol_name, "VERSION");
    assert_eq!(annotations[1].resolved_type, "number");
}
