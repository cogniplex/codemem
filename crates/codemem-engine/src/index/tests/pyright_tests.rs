use super::*;
use std::path::Path;

#[test]
fn extract_package_from_site_packages() {
    assert_eq!(
        PyrightEnricher::extract_package_from_path(
            "/usr/lib/python3.11/site-packages/requests/api.py"
        ),
        Some("requests".to_string())
    );
}

#[test]
fn extract_package_from_venv() {
    assert_eq!(
        PyrightEnricher::extract_package_from_path(
            "/home/user/.venv/lib/python3.11/site-packages/flask/app.py"
        ),
        Some("flask".to_string())
    );
}

#[test]
fn extract_package_none_for_local() {
    assert_eq!(
        PyrightEnricher::extract_package_from_path("/home/user/project/src/myapp/utils.py"),
        None
    );
}

#[test]
fn is_external_detects_site_packages() {
    assert!(PyrightEnricher::is_external_path(
        "/venv/lib/site-packages/requests/api.py"
    ));
    assert!(!PyrightEnricher::is_external_path(
        "/home/user/project/src/main.py"
    ));
}

#[test]
fn extract_generics_basic() {
    assert_eq!(extract_generics("List[User]"), vec!["User"]);
    assert_eq!(extract_generics("Dict[str, int]"), vec!["str", "int"]);
    assert!(extract_generics("str").is_empty());
}

#[test]
fn relativize_strips_prefix() {
    let result = PyrightEnricher::relativize(
        "/home/user/project/src/main.py",
        Path::new("/home/user/project"),
    );
    assert_eq!(result, "src/main.py");
}

#[test]
fn relativize_keeps_absolute_when_no_prefix() {
    let result =
        PyrightEnricher::relativize("/other/path/main.py", Path::new("/home/user/project"));
    assert_eq!(result, "/other/path/main.py");
}
